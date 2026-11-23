"""Fetch data, fit the model, predict the next fixtures and build the page.

Usage:
    python build.py            # fetch fresh data, then build out/index.html + out/data.json
    python build.py --offline  # rebuild from the data already downloaded
"""
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from data import FULL_NAME, load_all
from fetch import META, fetch_all, fetch_live
from markets import (CountModel, count_markets, goal_markets, halftime_markets, halftime_share,
                     team_count_form)
from model import DixonColes, expected_points_from_xg, outcome_probs, score_matrix
from scorers import Scorers, grade, load_live, serialize, summarize_scorers
from squad import POS as SQ_POS
from squad import Squad, load_live_full

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
LOG_PATH = os.path.join(HERE, "predictions_log.json")
TEMPLATE = os.path.join(HERE, "page", "template.html")

MODEL_VERSION = "dc-2"
# When bookmaker odds are out (a few days before kickoff), the headline prediction is
# 40% model + 60% bookmakers. Chosen with tune_data.py: the blend beat both the model
# alone and the bookmakers alone over 2025-26 and 2026-27.
BLEND_MODEL_WEIGHT = 0.4
# Extra data tested with tune_data.py on 2026-11-22 (mean RPS over 2025-26 and 2026-27, lower is better)
DATA_EXPERIMENTS = [
    {"name": "Current model (results, xG, team news)", "rps": 0.2080, "used": True},
    {"name": "+ shot-based xG for seasons without xG", "rps": 0.2081, "used": False,
     "note": "Helped last season, hurt this one. No overall gain."},
    {"name": "+ Championship games for promoted sides (best setting)", "rps": 0.2085, "used": False,
     "note": "Predicted promoted sides' games less well than a simple 'promoted team' starting level. Used only for head-to-head."},
    {"name": "Bookmakers alone", "rps": 0.2053, "used": False},
    {"name": "Model 40% + bookmakers 60%", "rps": 0.2052, "used": True,
     "note": "Best of all. Used whenever odds are published."},
]
# Line-up check: once team sheets are out, each side's expected goals move by the change in
# its line-up's attacking value against its usual line-up (weight 1.0). Tested with
# tune_squad.py on 87 games this season (gameweeks 3-11); lower RPS is better.
LINEUP_ATTACK_WEIGHT = 1.0
SQUAD_TESTS = [
    {"name": "Current model", "rps": 0.2067, "right": 39, "blend_rps": 0.2050, "blend_right": 41, "used": True},
    {"name": "+ squad from the last 3 games (known days before)", "rps": 0.2063, "right": 41,
     "blend_rps": 0.2048, "blend_right": 41, "used": False,
     "note": "Worse in gameweeks 3-7, better in 8-11. Too unsteady to use; team news covers absences instead."},
    {"name": "+ confirmed starting line-ups (about an hour before kickoff)", "rps": 0.2043, "right": 40,
     "blend_rps": 0.2039, "blend_right": 41, "used": True,
     "note": "Better in both halves of the season: sharper chances, though only one more right result. Used in the line-up check on each match."},
    {"name": "+ line-ups + defenders' quality (FPL price)", "rps": 0.2065, "right": 40,
     "blend_rps": 0.2043, "blend_right": 42, "used": False, "note": "Adding defenders made the chances worse."},
    {"name": "+ line-ups + players' recent form", "rps": 0.2040, "right": 41,
     "blend_rps": 0.2036, "blend_right": 41, "used": False, "note": "Tiny gain, within noise."},
]
# Strongest picks: the kinds of call the model gets right most often, once they clear a
# cut-off. Cut-offs were chosen on the 2025-26 season and checked on this season with
# tune_confidence.py, so they are not tuned on the games they are judged on.
STRONG_RULES = [
    {"key": "result", "market": "Result", "label": "Result (home win, draw or away win)", "min_p": 0.55},
    {"key": "dc", "market": "Double chance", "label": "Favourite or draw", "min_p": 0.80},
    {"key": "sot", "market": "More shots on target", "label": "Which team has more shots on target", "min_p": 0.65},
    {"key": "corners", "market": "More corners", "label": "Which team wins more corners", "min_p": 0.65},
]
CONF_PATH = os.path.join(HERE, "confidence_results.json")
# Oddset odds and match links, read by hand when asked (see oddset.json); never fetched automatically
ODDSET_PATH = os.path.join(HERE, "oddset.json")


def load_oddset():
    if not os.path.exists(ODDSET_PATH):
        return {}
    with open(ODDSET_PATH, encoding="utf-8") as fh:
        return json.load(fh)
# Friday odds vs closing odds (bookmaker average, margin removed), Premier League 2021-22 to
# 2025-26, 1,900 games: how much later prices would help if we could get them before kickoff.
ODDS_TIMING = {"games": 1900, "early_right": 55.3, "closing_right": 55.9, "early_rps": 0.1949, "closing_rps": 0.1936}
SETTINGS = dict(half_life=180, xg_weight=0.7, prior_sd=0.35)  # chosen with tune.py
LONG_HL = 720   # "usual level" model
SHORT_HL = 45   # "last few weeks" model

# How much an absence hurts. Replacements recover part of what a player gives.
ATTACK_LOSS = 0.5     # lose a player with 20% of team xG+xA -> attack -10%
DEFENCE_LOSS = 0.30   # lose a defender/GK with 20% of back-line minutes -> concede +6%


def r3(x):
    return round(float(x), 3)


def result_idx(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def rps(p, o):
    t = [0, 0, 0]; t[o] = 1
    c1 = p[0] - t[0]; c2 = p[0] + p[1] - t[0] - t[1]
    return 0.5 * (c1 * c1 + c2 * c2)


def market_probs(odds):
    if not odds:
        return None
    inv = [1 / x for x in odds]; s = sum(inv)
    return [x / s for x in inv]


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


# ----------------------------------------------------------------------- predictions
def predict(mod, home, away, hmult=1.0, amult=1.0, rates=None):
    lh, la = rates if rates else mod.rates(home, away, hmult, amult)
    M = mod.score_matrix(lh, la)
    pH, pD, pA = outcome_probs(M)
    flat = sorted(((M[i, j], i, j) for i in range(7) for j in range(7)), reverse=True)
    all_scores = [{"s": f"{i}-{j}", "p": r3(p)} for p, i, j in flat]
    scores = all_scores[:6]
    over25 = float(sum(M[i, j] for i in range(M.shape[0]) for j in range(M.shape[1]) if i + j >= 3))
    btts = float(M[1:, 1:].sum())
    grid = [[r3(M[i, j]) for j in range(6)] for i in range(6)]
    return {"lh": lh, "la": la, "p": [pH, pD, pA], "scores": scores,
            "over25": over25, "btts": btts, "grid": grid, "all_scores": all_scores, "M": M}


def rates_for(target, lh, la, rho):
    """Expected goals for each side that reproduce a target home/draw/away split."""
    x = np.log([lh, la])
    for _ in range(40):
        def f(v):
            pH, pD, pA = outcome_probs(score_matrix(math.exp(v[0]), math.exp(v[1]), rho))
            return np.array([pH - target[0], pA - target[2]])
        fx = f(x)
        if np.max(np.abs(fx)) < 1e-6:
            break
        J = np.zeros((2, 2))
        for k in range(2):
            d = np.zeros(2); d[k] = 1e-4
            J[:, k] = (f(x + d) - fx) / 1e-4
        step = np.clip(np.linalg.solve(J, fx), -0.5, 0.5)
        x = x - step
    return float(math.exp(x[0])), float(math.exp(x[1]))


# "balance": take only the home/away balance from the blend and keep the model's total goals.
# Same result accuracy as fitting both sides ("full"), but goal markets stay as good as the model's.
BLEND_MODE = os.environ.get("BLEND_MODE", "balance")


def rates_keep_total(target, lh, la, rho):
    """Keep the model's total expected goals; move the home/away split so that
    P(home win) - P(away win) matches the target."""
    tot = lh + la
    want = target[0] - target[2]
    lo, hi = 0.02, 0.98
    for _ in range(60):
        mid = (lo + hi) / 2
        pH, pD, pA = outcome_probs(score_matrix(tot * mid, tot * (1 - mid), rho))
        if pH - pA < want:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return tot * s, tot * (1 - s)


def blended(mod, pr, odds, home, away):
    """Blend the model's prediction with bookmaker odds; returns (prediction, used?)."""
    mk = market_probs(odds)
    if not mk:
        return pr, False
    target = [BLEND_MODEL_WEIGHT * a + (1 - BLEND_MODEL_WEIGHT) * b for a, b in zip(pr["p"], mk)]
    if BLEND_MODE == "balance":
        lh, la = rates_keep_total(target, pr["lh"], pr["la"], mod.rho)
    else:
        lh, la = rates_for(target, pr["lh"], pr["la"], mod.rho)
    return predict(mod, home, away, rates=(lh, la)), True


def pick_calls(pr, home, away):
    """Our call: the most likely result, then the most likely score *within* that result.

    The single most likely exact score is very often 1-1 even when one side is a clear
    favourite (draws pile onto one score, wins spread over 1-0, 2-0, 2-1 ...). So we show
    the best score that matches the likely result, plus scores nearly as likely."""
    pH, pD, pA = pr["p"]
    top = max(pr["p"])
    conf = "High" if top >= 0.60 else ("Medium" if top >= 0.47 else "Low")
    o = int(np.argmax(pr["p"]))
    fav = ["home", "draw", "away"][o]
    allsc = pr["all_scores"]

    def outc(sc):
        i, j = map(int, sc["s"].split("-"))
        return 0 if i > j else (1 if i == j else 2)
    call = next(sc for sc in allsc if outc(sc) == o)
    alts = [sc for sc in allsc if sc["s"] != call["s"] and sc["p"] >= 0.75 * call["p"]][:3]
    if o == 0:
        safe = (pH + pD, f"{home} win or draw")
    elif o == 2:
        safe = (pA + pD, f"{away} win or draw")
    else:
        safe = max((pH + pD, f"{home} win or draw"), (pA + pD, f"{away} win or draw"))
    return {"confidence": conf, "favourite": fav, "outcome_p": r3(top),
            "call_score": call, "most_likely_score": allsc[0],
            "alternatives": alts, "safe": {"label": safe[1], "p": r3(safe[0])} if top < 0.60 else None}


# ----------------------------------------------------------------------- availability
RETURN_RE = re.compile(r"(?:Expected back|Suspended until)\s+(\d{1,2})\s+([A-Z][a-z]{2})")
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def return_date(news, now):
    m = RETURN_RE.search(news or "")
    if not m or m.group(2) not in MONTHS:
        return None
    mon = MONTHS[m.group(2)]
    year = now.year if mon >= 7 or now.month < 7 else now.year + 1
    try:
        return datetime(year, mon, int(m.group(1)), tzinfo=timezone.utc)
    except ValueError:
        return None


def player_chance(e, gw, kickoff, current_gw, now):
    """Chance (0-100) that a player is available for this fixture, and whether he's due back by then."""
    if e["status"] == "a":
        return 100, False
    if gw == current_gw and e["chance_of_playing_this_round"] is not None:
        chance = e["chance_of_playing_this_round"]
    elif e["chance_of_playing_next_round"] is not None:
        chance = e["chance_of_playing_next_round"]
    else:
        chance = 0 if e["status"] in ("i", "s", "n", "u") else 100
    rd = return_date(e["news"], now)
    back_by_then = rd is not None and kickoff is not None and kickoff.date() >= rd.date()
    if back_by_then:
        chance = max(chance, 75)
    return chance, back_by_then


def team_availability(boot, fpl_team_id, gw, kickoff, current_gw, now):
    players = [e for e in boot["elements"] if e["team"] == fpl_team_id and e["minutes"] > 0 and e["status"] != "u"]
    contrib = {e["id"]: float(e["expected_goals"]) + 0.7 * float(e["expected_assists"]) for e in players}
    tot_contrib = sum(contrib.values()) or 1.0
    back = [e for e in players if e["element_type"] in (1, 2)]
    tot_back = sum(e["minutes"] for e in back) or 1.0
    out, att_loss, def_loss = [], 0.0, 0.0
    for e in players:
        if e["status"] == "a":
            continue
        chance, back_by_then = player_chance(e, gw, kickoff, current_gw, now)
        absent = 1 - chance / 100
        if absent <= 0:
            continue
        added = datetime.fromisoformat(e["news_added"].replace("Z", "+00:00")) if e.get("news_added") else now
        # A long absence is already partly baked into recent results, so it counts less.
        fresh = 1.0 if (now - added).days <= 28 else 0.6
        a_share = contrib[e["id"]] / tot_contrib
        d_share = e["minutes"] / tot_back if e["element_type"] in (1, 2) else 0.0
        a_imp = absent * fresh * a_share
        d_imp = absent * fresh * d_share
        att_loss += a_imp
        def_loss += d_imp
        out.append({
            "name": e["web_name"], "pos": ["", "GK", "DEF", "MID", "FWD"][e["element_type"]],
            "status": {"i": "Injured", "s": "Suspended", "d": "Doubtful", "n": "Ineligible"}.get(e["status"], e["status"]),
            "chance": chance, "news": e["news"], "back_by_then": back_by_then,
            "minutes": e["minutes"], "starts": e.get("starts"),
            "xg": r3(e["expected_goals"]), "xa": r3(e["expected_assists"]),
            "attack_share": r3(a_share), "backline_share": r3(d_share),
            "impact": r3(a_imp + d_imp),
        })
    out.sort(key=lambda x: -x["impact"])
    # fringe players (barely used, no measurable impact) are counted but not listed
    listed = [p for p in out if p["minutes"] >= 180 or p["impact"] >= 0.01]
    fringe = [p["name"] for p in out if p not in listed]
    att_mult = max(0.75, 1 - ATTACK_LOSS * att_loss)
    def_mult = min(1.20, 1 + DEFENCE_LOSS * def_loss)
    return {"players": listed, "fringe": fringe, "attack_mult": r3(att_mult), "concede_mult": r3(def_mult)}


# ----------------------------------------------------------------------- form & table
def team_games(cur, team):
    g = []
    for m in cur:
        if team not in (m["home"], m["away"]):
            continue
        home = m["home"] == team
        gf, ga = (m["hg"], m["ag"]) if home else (m["ag"], m["hg"])
        xgf, xga = (m["hxg"], m["axg"]) if home else (m["axg"], m["hxg"])
        g.append({"date": iso(m["kickoff"]), "opp": m["away"] if home else m["home"], "venue": "H" if home else "A",
                  "gf": gf, "ga": ga, "xgf": xgf, "xga": xga,
                  "res": "W" if gf > ga else ("D" if gf == ga else "L")})
    return g


def summarize(games):
    if not games:
        return None
    pts = sum(3 if x["res"] == "W" else (1 if x["res"] == "D" else 0) for x in games)
    xs = [x for x in games if x["xgf"] is not None]
    return {
        "n": len(games), "pts": pts, "ppg": r3(pts / len(games)),
        "gf": sum(x["gf"] for x in games), "ga": sum(x["ga"] for x in games),
        "xgf_pg": r3(np.mean([x["xgf"] for x in xs])) if xs else None,
        "xga_pg": r3(np.mean([x["xga"] for x in xs])) if xs else None,
        "n_xg": len(xs),
    }


def team_form(cur, team, mods):
    games = team_games(cur, team)
    xs = [x for x in games if x["xgf"] is not None]
    season = summarize(games)
    goals_minus_xg = sum(x["gf"] for x in xs) - sum(x["xgf"] for x in xs) if xs else None
    xga_minus_ga = sum(x["xga"] for x in xs) - sum(x["ga"] for x in xs) if xs else None
    xpts = sum(expected_points_from_xg(x["xgf"], x["xga"]) for x in xs) if xs else None
    pts_xs = sum(3 if x["res"] == "W" else (1 if x["res"] == "D" else 0) for x in xs)

    def net(mod):  # goal difference per game vs an average PL side on neutral ground
        return math.exp(mod.mu + mod.att[team]) - math.exp(mod.mu + mod.dfn[team])

    level = {k: r3(net(m)) for k, m in mods.items()}
    delta = level["short"] - level["long"]
    tag = ("Peak form" if delta >= 0.30 else "Above usual level" if delta >= 0.12 else
           "Slump" if delta <= -0.30 else "Below usual level" if delta <= -0.12 else "At usual level")
    return {
        "last6": games[-6:][::-1],
        "last6_sum": summarize(games[-6:]),
        "home5": summarize([x for x in games if x["venue"] == "H"][-5:]),
        "away5": summarize([x for x in games if x["venue"] == "A"][-5:]),
        "season": season,
        "goals_minus_xg": r3(goals_minus_xg) if goals_minus_xg is not None else None,
        "xga_minus_ga": r3(xga_minus_ga) if xga_minus_ga is not None else None,
        "xpts": r3(xpts) if xpts is not None else None, "pts_in_xg_games": pts_xs, "n_xg": len(xs),
        "level": level, "level_delta": r3(delta), "level_tag": tag,
    }


def league_table(cur, teams):
    t = {n: {"team": n, "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0} for n in teams}
    for m in cur:
        for side, gf, ga in ((m["home"], m["hg"], m["ag"]), (m["away"], m["ag"], m["hg"])):
            r = t[side]; r["p"] += 1; r["gf"] += gf; r["ga"] += ga
            if gf > ga: r["w"] += 1; r["pts"] += 3
            elif gf == ga: r["d"] += 1; r["pts"] += 1
            else: r["l"] += 1
    rows = sorted(t.values(), key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"], r["team"]))
    for i, r in enumerate(rows, 1):
        r["pos"] = i; r["gd"] = r["gf"] - r["ga"]
    return rows


def head_to_head(allm, a, b, n=6):
    ms = [m for m in allm if {m["home"], m["away"]} == {a, b}]
    ms.sort(key=lambda m: m["kickoff"], reverse=True)
    rows = [{"date": iso(m["kickoff"]), "home": m["home"], "away": m["away"], "hg": m["hg"], "ag": m["ag"],
             "season": m["season"], "div": m.get("div", "E0")} for m in ms[:n]]
    wa = sum(1 for m in ms[:n] if (m["home"] == a and m["hg"] > m["ag"]) or (m["away"] == a and m["ag"] > m["hg"]))
    wb = sum(1 for m in ms[:n] if (m["home"] == b and m["hg"] > m["ag"]) or (m["away"] == b and m["ag"] > m["hg"]))
    d = len(rows) - wa - wb
    return {"games": rows, "a_wins": wa, "b_wins": wb, "draws": d, "total_meetings_since_2019": len(ms)}


# ----------------------------------------------------------------------- explanation
def pct(x):
    return f"{round(100 * x)}%"


def verdict(f):
    h, a = f["home"], f["away"]
    pH, pD, pA = f["probs"]
    lines = []
    fav_p, fav = max((pH, h), (pA, a))
    if fav_p >= 0.60:
        lines.append(f"{fav} are clear favourites ({pct(fav_p)}).")
    elif fav_p >= 0.45:
        lines.append(f"{fav} edge it ({pct(fav_p)}), but it's far from certain.")
    else:
        lines.append(f"Close to a coin-flip: {h} {pct(pH)}, draw {pct(pD)}, {a} {pct(pA)}.")

    reasons = []
    w = f["waterfall"]
    for side, team in (("home", h), ("away", a)):
        inj = w[side]["own_absences"]
        if inj <= -0.07:
            names = ", ".join(p["name"] for p in f["absences"][side]["players"][:2])
            reasons.append((abs(inj), f"{team}'s missing players cost them about {abs(inj):.2f} expected goals ({names})."))
        opp = w[side]["opp_absences"]
        if opp >= 0.07:
            other = "away" if side == "home" else "home"
            names = ", ".join(p["name"] for p in f["absences"][other]["players"] if p["pos"] in ("GK", "DEF"))
            reasons.append((opp, f"{team} should find scoring a bit easier: the opposition back line is missing {names} (+{opp:.2f} xG)."))
        frm = f["form"][side]
        if frm["level_tag"] != "At usual level":
            word = {"Peak form": "are in peak form", "Slump": "are in a slump",
                    "Above usual level": "are playing above their usual level",
                    "Below usual level": "are playing below their usual level"}[frm["level_tag"]]
            reasons.append((abs(frm["level_delta"]), f"{team} {word} over the last few weeks ({frm['level_delta']:+.2f} goal difference per game)."))
        g = frm["goals_minus_xg"]
        if g is not None and g >= 3.5:
            reasons.append((g / 10, f"{team} have scored {g:.1f} more goals than their chances (xG) suggest - that rarely lasts."))
        elif g is not None and g <= -3.5:
            reasons.append((-g / 10, f"{team} have scored {-g:.1f} fewer goals than their chances (xG) suggest - due a bounce."))
    reasons.sort(key=lambda r: -r[0])
    lines += [r[1] for r in reasons[:3]]

    hh = f["h2h"]
    if len(hh["games"]) >= 4:
        if hh["a_wins"] >= 4:
            lines.append(f"Head-to-head leans {h}: {hh['a_wins']} wins in the last {len(hh['games'])} meetings.")
        elif hh["b_wins"] >= 4:
            lines.append(f"Head-to-head leans {a}: {hh['b_wins']} wins in the last {len(hh['games'])} meetings.")

    mk = f.get("market")
    if mk:
        mp = f["model_probs"]
        i, name = max(((0, h), (2, a)), key=lambda x: abs(mp[x[0]] - mk[x[0]]))
        diff = mp[i] - mk[i]
        if abs(diff) >= 0.08:
            more = "more" if diff > 0 else "less"
            lines.append(f"On its own the model rates {name} {more} highly than the bookmakers do ({pct(mp[i])} vs {pct(mk[i])}).")
    if f.get("blend_used"):
        lines.append(f"Bookmaker odds are out, so the chances shown blend the model ({round(100 * BLEND_MODEL_WEIGHT)}%) with the bookmakers ({round(100 * (1 - BLEND_MODEL_WEIGHT))}%), which has tested better than either alone.")
    return lines


# ----------------------------------------------------------------------- other markets
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
CORNER_TEAM_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)
CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)
CARD_TEAM_LINES = (0.5, 1.5, 2.5, 3.5)
TRACK_CORNER_LINE = 9.5
TRACK_CARD_LINE = 3.5


SOT_LINES = (6.5, 7.5, 8.5, 9.5, 10.5)
SOT_TEAM_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)
TRACK_SOT_LINE = 8.5


def fit_side_models(allm, ref_date):
    return {
        "corners": CountModel("hc", "ac").fit(allm, ref_date),
        "cards": CountModel("hcards", "acards").fit(allm, ref_date, referee_effect=True),
        "sot": CountModel("hst", "ast").fit(allm, ref_date),
        "ht_share": halftime_share(allm, ref_date),
    }


def all_markets(pr, h, a, side, referee=None, ou25=None, abs_mult=(1.0, 1.0)):
    sot = side["sot"].predict(h, a)
    if sot:  # missing attackers mean fewer shots too: scale like expected goals
        for k, mult in (("mh", abs_mult[0]), ("ma", abs_mult[1])):
            sot[k] *= mult
        sot = side["sot"].predict_from_means(sot["mh"], sot["ma"])
    return {
        "goals": goal_markets(pr["M"], pr["lh"], pr["la"], ou25),
        "halftime": halftime_markets(pr["lh"], pr["la"], side["ht_share"]),
        "corners": count_markets(side["corners"].predict(h, a), CORNER_LINES, CORNER_TEAM_LINES),
        "cards": count_markets(side["cards"].predict(h, a, referee), CARD_LINES, CARD_TEAM_LINES),
        "sot": count_markets(sot, SOT_LINES, SOT_TEAM_LINES),
        "referee": referee,
        "referee_known": bool(referee and referee in side["cards"].refs),
    }


def _line(lines, x):
    return next(l["over"] for l in lines if l["line"] == x)


def quick_picks(f):
    """One headline call per market, with its probability."""
    m, H, A = f["markets"], f["home"], f["away"]
    pH, pD, pA = f["probs"]
    picks = []
    o = int(np.argmax(f["probs"]))
    picks.append({"market": "Result", "pick": [f"{H} win", "Draw", f"{A} win"][o], "p": r3(f["probs"][o])})
    # matches the "safer call": the favourite or a draw
    dc = ((m["goals"]["double_chance"]["1X"], f"{H} or draw") if pH >= pA
          else (m["goals"]["double_chance"]["X2"], f"{A} or draw"))
    picks.append({"market": "Double chance", "pick": dc[1], "p": r3(dc[0])})
    o25 = _line(m["goals"]["lines"], 2.5)
    picks.append({"market": "Goals 2.5", "pick": "Over 2.5" if o25 >= 0.5 else "Under 2.5", "p": r3(max(o25, 1 - o25))})
    b = m["goals"]["btts"]
    picks.append({"market": "Both teams score", "pick": "Yes" if b >= 0.5 else "No", "p": r3(max(b, 1 - b))})
    ht = m["halftime"]["ht_probs"]; hi = int(np.argmax(ht))
    picks.append({"market": "Half-time", "pick": [f"{H} lead", "Level", f"{A} lead"][hi], "p": r3(ht[hi])})
    c = m["corners"]
    if c:
        co = _line(c["lines"], TRACK_CORNER_LINE)
        picks.append({"market": "Corners 9.5", "pick": f"Over {TRACK_CORNER_LINE}" if co >= 0.5 else f"Under {TRACK_CORNER_LINE}",
                      "p": r3(max(co, 1 - co))})
        mi = int(np.argmax([c["more"][0], c["more"][2]]))
        picks.append({"market": "More corners", "pick": [H, A][mi], "p": r3([c["more"][0], c["more"][2]][mi])})
    so = m["sot"]
    if so:
        sv = _line(so["lines"], TRACK_SOT_LINE)
        picks.append({"market": "Shots on target 8.5", "pick": f"Over {TRACK_SOT_LINE}" if sv >= 0.5 else f"Under {TRACK_SOT_LINE}",
                      "p": r3(max(sv, 1 - sv))})
        mi = int(np.argmax([so["more"][0], so["more"][2]]))
        picks.append({"market": "More shots on target", "pick": [H, A][mi], "p": r3([so["more"][0], so["more"][2]][mi])})
    sc = f.get("scorers")
    if sc:
        best = max(sc["home"] + sc["away"], key=lambda p: p["anytime"], default=None)
        if best:
            picks.append({"market": "Likeliest scorer", "pick": best["name"], "p": best["anytime"]})
    k = m["cards"]
    if k:
        ko = _line(k["lines"], TRACK_CARD_LINE)
        picks.append({"market": "Cards 3.5", "pick": f"Over {TRACK_CARD_LINE}" if ko >= 0.5 else f"Under {TRACK_CARD_LINE}",
                      "p": r3(max(ko, 1 - ko))})
    return picks


def strong_picks(f):
    """The calls that clear a STRONG_RULES cut-off for one fixture."""
    H, A, m = f["home"], f["away"], f["markets"]
    pH, pD, pA = f["probs"]
    o = int(np.argmax(f["probs"]))
    cand = {"result": ([f"{H} win", "Draw", f"{A} win"][o], f["probs"][o], None, None)}
    fav_home = pH >= pA
    cand["dc"] = (f"{H} or draw" if fav_home else f"{A} or draw", (pH + pD) if fav_home else (pA + pD), None, None)
    for key in ("sot", "corners"):
        c = m.get(key)
        if c and c["more"][0] + c["more"][2] > 0:
            ph = c["more"][0] / (c["more"][0] + c["more"][2])    # ignoring a tie, which voids the call
            cand[key] = (H if ph >= 0.5 else A, max(ph, 1 - ph), c["more"][1], "home" if ph >= 0.5 else "away")
    out = []
    for rule in STRONG_RULES:
        if rule["key"] in cand:
            pick, p, tie, side = cand[rule["key"]]
            if p >= rule["min_p"]:
                # p_any: chance the pick wins if a tie LOSES the bet (3-way market) instead of voiding it
                p_any = p * (1 - tie) if tie is not None else p
                out.append({"key": rule["key"], "market": rule["market"], "pick": pick, "p": r3(p),
                            "p_any": r3(p_any), "tie": r3(tie) if tie is not None else None, "side": side,
                            "fair": round(1 / p, 2), "fair_any": round(1 / p_any, 2)})
    return out


def strong_status(s, m):
    """How a strong pick ended: "won", "lost", "void" (a tie in corners / shots on target,
    where the stake comes back) or "pending" (match stats not in yet)."""
    o = result_idx(m["hg"], m["ag"])
    if s["key"] == "result":
        return "won" if {f"{m['home']} win": 0, "Draw": 1, f"{m['away']} win": 2}.get(s["pick"]) == o else "lost"
    if s["key"] == "dc":
        ok = o in (0, 1) if s["pick"] == f"{m['home']} or draw" else o in (1, 2)
        return "won" if ok else "lost"
    hk, ak = ("hst", "ast") if s["key"] == "sot" else ("hc", "ac")
    if m.get(hk) is None or m.get(ak) is None:
        return "pending"
    if m[hk] == m[ak]:
        return "void"
    return "won" if (m[hk] > m[ak]) == (s.get("side", "home") == "home") else "lost"


def strong_record(bt):
    """Hit rate of each kind of strong pick: last season (tune_confidence.py) and this season (backtest)."""
    conf = {}
    if os.path.exists(CONF_PATH):
        with open(CONF_PATH, encoding="utf-8") as fh:
            conf = json.load(fh)
    out = []
    for rule in STRONG_RULES:
        k, cut = rule["key"], rule["min_p"]
        rows = []
        for r in bt:
            p = r["probs"]
            if k == "result":
                rows.append((max(p), int(np.argmax(p)) == r["outcome"]))
            elif k == "dc":
                fav_home = p[0] >= p[2]
                rows.append(((p[0] + p[1]) if fav_home else (p[2] + p[1]),
                             r["outcome"] in (0, 1) if fav_home else r["outcome"] in (1, 2)))
            else:
                x = r.get("markets", {}).get("more_" + k)
                if x:
                    rows.append((max(x["p"], 1 - x["p"]), (x["p"] >= 0.5) == x["y"]))
        sel = [h for p, h in rows if p >= cut]
        last = next((b for b in conf.get("at_least", {}).get(k, [])
                     if b["season"] == "2526" and abs(b["min_p"] - cut) < 1e-9), None)
        out.append({**rule, "this_n": len(sel), "this_hits": int(sum(sel)), "this_of": len(rows),
                    "last_n": last["n"] if last else None, "last_hits": last["hits"] if last else None,
                    "last_of": last["of"] if last else None})
    return out


def market_backtest_row(m, pr, mk, h, a):
    """Did each headline market pick come in? m = real match row."""
    out = {}
    tg = m["hg"] + m["ag"]
    o25 = _line(mk["goals"]["lines"], 2.5)
    bk = None
    if m.get("ou25"):
        inv = [1 / m["ou25"][0], 1 / m["ou25"][1]]; bk = inv[0] / sum(inv)
    out["over25"] = {"p": o25, "y": tg > 2.5, "book": bk}
    out["btts"] = {"p": mk["goals"]["btts"], "y": m["hg"] > 0 and m["ag"] > 0}
    if m.get("hthg") is not None:
        ht = mk["halftime"]["ht_probs"]
        out["ht"] = {"probs": ht, "y": result_idx(m["hthg"], m["htag"])}
    if mk["corners"] and m.get("hc") is not None:
        c = mk["corners"]
        out["corners95"] = {"p": _line(c["lines"], TRACK_CORNER_LINE), "y": m["hc"] + m["ac"] > TRACK_CORNER_LINE}
        if m["hc"] != m["ac"]:
            ph = c["more"][0] / (c["more"][0] + c["more"][2])
            out["more_corners"] = {"p": ph, "y": m["hc"] > m["ac"]}
    if mk["sot"] and m.get("hst") is not None:
        so = mk["sot"]
        out["sot85"] = {"p": _line(so["lines"], TRACK_SOT_LINE), "y": m["hst"] + m["ast"] > TRACK_SOT_LINE}
        if m["hst"] != m["ast"]:
            out["more_sot"] = {"p": so["more"][0] / (so["more"][0] + so["more"][2]), "y": m["hst"] > m["ast"]}
    if mk["cards"] and m.get("hcards") is not None:
        out["cards35"] = {"p": _line(mk["cards"]["lines"], TRACK_CARD_LINE), "y": m["hcards"] + m["acards"] > TRACK_CARD_LINE}
    return out


MARKET_LABELS = {
    "over25": "Over/under 2.5 goals", "btts": "Both teams to score", "ht": "Half-time result",
    "corners95": "Over/under 9.5 corners", "more_corners": "Which team wins more corners",
    "cards35": "Over/under 3.5 cards",
    "sot85": "Over/under 8.5 shots on target", "more_sot": "Which team has more shots on target",
}


def summarize_markets(rows):
    out = []
    for key, label in MARKET_LABELS.items():
        rs = [r["markets"][key] for r in rows if key in r.get("markets", {})]
        if not rs:
            continue
        if key == "ht":
            hits = sum(int(np.argmax(x["probs"])) == x["y"] for x in rs)
            base = [np.mean([x["y"] == k for x in rs]) for k in range(3)]
            brier = np.mean([sum((x["probs"][k] - (x["y"] == k)) ** 2 for k in range(3)) for x in rs])
            base_brier = np.mean([sum((base[k] - (x["y"] == k)) ** 2 for k in range(3)) for x in rs])
            out.append({"key": key, "label": label, "n": len(rs), "hits": int(hits), "brier": r3(brier),
                        "base_brier": r3(base_brier), "base_rate": None,
                        "always_majority": int(max(sum(x["y"] == k for x in rs) for k in range(3)))})
            continue
        hits = sum((x["p"] >= 0.5) == x["y"] for x in rs)
        rate = float(np.mean([x["y"] for x in rs]))
        brier = np.mean([(x["p"] - x["y"]) ** 2 for x in rs])
        base_brier = np.mean([(rate - x["y"]) ** 2 for x in rs])
        row = {"key": key, "label": label, "n": len(rs), "hits": int(hits), "brier": r3(brier),
               "base_brier": r3(base_brier), "base_rate": r3(rate),
               "always_majority": int(max(sum(x["y"] for x in rs), len(rs) - sum(x["y"] for x in rs)))}
        bk = [x for x in rs if x.get("book") is not None]
        if bk:
            row["book_n"] = len(bk)
            row["book_hits"] = int(sum((x["book"] >= 0.5) == x["y"] for x in bk))
            row["book_brier"] = r3(np.mean([(x["book"] - x["y"]) ** 2 for x in bk]))
            row["model_brier_same"] = r3(np.mean([(x["p"] - x["y"]) ** 2 for x in bk]))
        out.append(row)
    return out


# ----------------------------------------------------------------------- backtest
def backtest(D, promoted, upto_gw, sc=None):
    scorer_rows = []
    allm = D["history"] + D["current"]
    by_gw = {}
    for fx in D["fixtures"]:
        by_gw.setdefault(fx["gw"], []).append(fx)
    res = {(m["home"], m["away"]): m for m in D["current"]}
    rows = []
    for gw in sorted(g for g in by_gw if g is not None and g <= upto_gw):
        fxs = by_gw[gw]
        cutoff = min(f["kickoff"] for f in fxs if f["kickoff"])
        mod = DixonColes(promoted=promoted, **SETTINGS).fit(allm, cutoff)
        side = fit_side_models(allm, cutoff)
        hist = sc.history(cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")) if sc else None
        for fx in fxs:
            m = res.get((fx["home"], fx["away"]))
            if not m:
                continue
            pr_model = predict(mod, fx["home"], fx["away"])
            pr, used_blend = blended(mod, pr_model, m.get("odds_pre"), fx["home"], fx["away"])
            calls = pick_calls(pr, fx["home"], fx["away"])
            mk_all = all_markets(pr, fx["home"], fx["away"], side, m.get("referee"))
            if sc and fx["gw"] in sc.live and fx["fpl_id"] in sc.fx:
                rows_sc = grade(sc, sc.fx[fx["fpl_id"]], pr["lh"], pr["la"], sc.live[fx["gw"]], hist)
                if rows_sc:
                    scorer_rows.append((f"{fx['home']} v {fx['away']}", rows_sc))
            o = result_idx(m["hg"], m["ag"])
            mk = market_probs(m.get("odds"))
            top = calls["call_score"]["s"]
            actual = f"{m['hg']}-{m['ag']}"
            rows.append({
                "gw": gw, "kickoff": iso(m["kickoff"]), "home": fx["home"], "away": fx["away"],
                "probs": [r3(x) for x in pr["p"]], "pick_score": top, "top3": [s["s"] for s in pr["scores"][:3]],
                "actual": actual, "outcome": o, "correct": int(np.argmax(pr["p"])) == o,
                "exact": top == actual, "most_likely_exact": pr["scores"][0]["s"] == actual,
                "safe": calls["safe"],
                "safe_hit": _safe_hit(calls["safe"], fx["home"], o) if calls["safe"] else None,
                "in_top3": actual in [s["s"] for s in pr["scores"][:3]],
                "rps": r3(rps(pr["p"], o)), "market": [r3(x) for x in mk] if mk else None,
                "market_correct": (int(np.argmax(mk)) == o) if mk else None,
                "market_rps": r3(rps(mk, o)) if mk else None,
                "p_actual_outcome": r3(pr["p"][o]),
                "blend": used_blend, "model_probs": [r3(x) for x in pr_model["p"]],
                "model_correct": int(np.argmax(pr_model["p"])) == o, "model_rps": r3(rps(pr_model["p"], o)),
                "provisional": m["src"].startswith("FPL"),
                "markets": market_backtest_row(m, pr, mk_all, fx["home"], fx["away"]),
            })
    return rows, scorer_rows


def _safe_hit(safe, home, o):
    return o in (0, 1) if safe["label"].startswith(home + " ") else o in (1, 2)


def summarize_track(rows):
    if not rows:
        return None
    mk = [r for r in rows if r["market"]]
    return {
        "n": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "exact": sum(r["exact"] for r in rows),
        "most_likely_exact": sum(bool(r.get("most_likely_exact")) for r in rows),
        "safe_n": sum(1 for r in rows if r.get("safe_hit") is not None),
        "safe_hits": sum(1 for r in rows if r.get("safe_hit")),
        "in_top3": sum(r["in_top3"] for r in rows),
        "rps": r3(np.mean([r["rps"] for r in rows])),
        "market_n": len(mk),
        "market_correct": sum(r["market_correct"] for r in mk),
        "market_rps": r3(np.mean([r["market_rps"] for r in mk])) if mk else None,
        "model_rps_same": r3(np.mean([r["rps"] for r in mk])) if mk else None,
        "always_home_correct": sum(r["outcome"] == 0 for r in rows),
        "model_correct": sum(bool(r.get("model_correct")) for r in rows),
        "model_rps": r3(np.mean([r["model_rps"] for r in rows if r.get("model_rps") is not None])) if any(r.get("model_rps") is not None for r in rows) else None,
        "blend_n": sum(1 for r in rows if r.get("blend")),
    }


# ----------------------------------------------------------------------- log
def update_log(log, fixtures_out, D, now):
    res = {(m["home"], m["away"]): m for m in D["current"]}
    for f in fixtures_out:
        key = str(f["fpl_id"])
        if not f["started"]:
            log[key] = {"fpl_id": f["fpl_id"], "gw": f["gw"], "home": f["home"], "away": f["away"], "kickoff": f["kickoff"],
                        "made_at": iso(now), "model": MODEL_VERSION, "probs": f["probs"], "model_probs": f["model_probs"],
                        "blend": f["blend_used"],
                        "pick_score": f["calls"]["call_score"]["s"], "top3": [s["s"] for s in f["scores"][:3]],
                        "safe": f["calls"]["safe"], "strong": f["strong"], "result": None}
    for key, e in log.items():
        e.setdefault("fpl_id", int(key))   # older entries were saved without it
        m = res.get((e["home"], e["away"]))
        if m:
            o = result_idx(m["hg"], m["ag"])
            actual = f"{m['hg']}-{m['ag']}"
            e["result"] = {"score": actual, "outcome": o, "correct": int(np.argmax(e["probs"])) == o,
                           "exact": actual == e["pick_score"], "in_top3": actual in e["top3"],
                           "rps": r3(rps(e["probs"], o)), "provisional": m["src"].startswith("FPL"),
                           "strong": [{**s, "status": st, "hit": {"won": True, "lost": False}.get(st)}
                                      for s in e.get("strong", []) for st in [strong_status(s, m)]]}
    return log


# ----------------------------------------------------------------------- main
COPIES_PATH = os.path.join(HERE, "local_copies.json")
SITE_CONF = os.path.join(HERE, "site_config.json")


def copy_to_folders():
    """Copy the finished page (data built in) to the folders listed in local_copies.json,
    e.g. a Desktop folder, so a copy opened from there is always current."""
    if not os.path.exists(COPIES_PATH):
        return
    try:
        with open(COPIES_PATH, encoding="utf-8") as fh:
            folders = json.load(fh).get("folders", [])
    except (OSError, ValueError) as e:
        print(f"  ! local_copies.json unreadable, no copies made: {e}")
        return
    for folder in folders:
        try:
            if not os.path.isdir(folder):
                print(f"  ! copy folder missing, skipped: {folder}")
                continue
            for name in ("index.html", "data.json"):
                shutil.copyfile(os.path.join(OUT, name), os.path.join(folder, name))
            print(f"  copied the page to {folder}")
        except OSError as e:   # a locked or offline folder must not fail the build
            print(f"  ! could not copy to {folder}: {e}")


def player_scale(sq, team_ids, cutoff):
    """Our own attack rating (45-99): percentile within position of (xG + 0.7 xA) per 90. Not for goalkeepers."""
    pools = {1: [], 2: [], 3: [], 4: []}
    rts = {t: sq.ratings(t, cutoff) for t in team_ids}
    for rt in rts.values():
        for r in rt.values():
            if r["mins"] >= 450 and r["e"]["status"] != "u" and r["pos"] != 1:
                pools[r["pos"]].append(r["att"])
    for k in pools:
        pools[k].sort()

    def rating(r):
        pool = pools[r["pos"]]
        if not pool:
            return None
        v = r["att"]
        below = sum(1 for x in pool if x < v)
        return int(round(45 + 54 * below / max(1, len(pool) - 1)))
    return rts, rating


def form_tag(r):
    if r["pos"] == 1 or r["last4_min"] < 180:
        return None
    if r["att_recent"] >= 1.25 * r["att"] and r["att_recent"] - r["att"] >= 0.06:
        return "hot"
    if r["att"] >= 0.15 and r["att_recent"] <= 0.75 * r["att"]:
        return "cold"
    return None


def squad_side(rt, rating, chance_fn):
    """A team's players for the page: ratings, form and our guess at the starting XI."""
    rows = []
    for r in rt.values():
        e = r["e"]
        if e["status"] == "u" or r["usual"] <= 0:
            continue
        chance = chance_fn(e)
        rows.append((r, chance))
    rows.sort(key=lambda x: -x[0]["usual"])
    gks = [x for x in rows if x[0]["pos"] == 1]
    outf = [x for x in rows if x[0]["pos"] != 1]
    xi = set()
    if gks:
        xi.add(max(gks, key=lambda x: x[0]["usual"] * x[1])[0]["pid"])
    for r, c in sorted(outf, key=lambda x: -x[0]["usual"] * x[1])[:10]:
        xi.add(r["pid"])
    out = []
    for r, chance in rows:
        out.append({
            "pid": r["pid"], "name": r["e"]["web_name"], "pos": SQ_POS[r["pos"]], "rating": rating(r),
            "att": r3(r["att"]), "att_recent": r3(r["att_recent"]), "form": form_tag(r),
            "usual": r3(r["usual"]), "mws": round(r["min_when_start"]), "mwb": round(r["min_when_bench"], 1), "chance": round(100 * chance),
            "xi": r["pid"] in xi, "starts": r["starts"], "games": r["games"],
            "g": int(round(r["goals"])), "a": int(round(r["assists"])), "xg": round(r["xg"], 1), "xa": round(r["xa"], 1),
            "l4g": int(round(r["last4_g"])), "l4a": int(round(r["last4_a"])), "show": r["usual"] >= 0.05 or r["pid"] in xi,
        })
    return out


def in_form_players(rts, rating, teams):
    rows = []
    for tid, rt in rts.items():
        for r in rt.values():
            if form_tag(r) == "hot" and r["e"]["status"] != "u":
                rows.append({"name": r["e"]["web_name"], "team": teams[tid]["name"], "pos": SQ_POS[r["pos"]],
                             "rating": rating(r), "att": r3(r["att"]), "att_recent": r3(r["att_recent"]),
                             "l4g": int(round(r["last4_g"])), "l4a": int(round(r["last4_a"])), "l4min": int(round(r["last4_min"])),
                             "lift": r3(r["att_recent"] - r["att"])})
    rows.sort(key=lambda x: -x["lift"])
    return rows[:15]


def current_gw_early(boot):
    return next((e["id"] for e in boot["events"] if e["is_current"]), None) or 1


def main():
    offline = "--offline" in sys.argv
    if not offline:
        print("Fetching data...")
        fetch_all()
    with open(META, encoding="utf-8") as fh:
        meta = json.load(fh)
    now = datetime.now(timezone.utc)
    D = load_all()
    hist, cur, boot = D["history"], D["current"], D["boot"]
    allm = hist + cur
    last_season_teams = {m["home"] for m in hist if m["season"] == "2526"}
    cur_teams = sorted({t["name"] for t in D["teams"].values()})
    promoted = set(cur_teams) - last_season_teams
    fpl_id = {t["name"]: tid for tid, t in D["teams"].items()}
    short = {t["name"]: t["short"] for t in D["teams"].values()}

    print("Fitting models...")
    mod = DixonColes(promoted=promoted, **SETTINGS).fit(allm, now)
    long_mod = DixonColes(promoted=promoted, **{**SETTINGS, "half_life": LONG_HL}).fit(allm, now)
    usual = {t: (long_mod.att[t], long_mod.dfn[t]) for t in long_mod.teams}
    short_mod = DixonColes(promoted=promoted, prior_ratings=usual,
                           **{**SETTINGS, "half_life": SHORT_HL, "prior_sd": 0.25}).fit(allm, now)
    mods = {"short": short_mod, "model": mod, "long": long_mod}
    events = boot["events"]
    current_gw = next((e["id"] for e in events if e["is_current"]), None) or 1

    # fixtures to predict: every not-started match in the next two gameweeks that have any
    pending = [f for f in D["fixtures"] if not f["started"] and f["gw"] is not None]
    gws = sorted({f["gw"] for f in pending})[:2]
    upcoming = sorted([f for f in pending if f["gw"] in gws], key=lambda f: f["kickoff"])

    side = fit_side_models(allm, now)
    corners, cards, ht_share = side["corners"], side["cards"], side["ht_share"]
    finished_gws = {e["id"] for e in boot["events"] if e["finished"]}
    live_gws = [e["id"] for e in boot["events"] if e["id"] <= current_gw_early(boot)]
    fetch_live(live_gws, finished_gws, offline=offline)
    sc = Scorers(boot, D["fpl_fixtures"], load_live(live_gws))
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    hist_now = sc.history(now_iso)
    sq = Squad(boot, D["fpl_fixtures"], load_live_full(live_gws))
    sq_rts, sq_rating = player_scale(sq, list(D["teams"]), now_iso)
    forms = {t: team_form(cur, t, mods) for t in cur_teams}
    oddset = load_oddset()
    odds_lm = meta.get("fd_fixtures.csv", {}).get("last_modified")   # when football-data last changed the odds file
    fixtures_out = []
    for fx in upcoming:
        h, a = fx["home"], fx["away"]
        av_h = team_availability(boot, fpl_id[h], fx["gw"], fx["kickoff"], current_gw, now)
        av_a = team_availability(boot, fpl_id[a], fx["gw"], fx["kickoff"], current_gw, now)
        hmult = av_h["attack_mult"] * av_a["concede_mult"]
        amult = av_a["attack_mult"] * av_h["concede_mult"]
        mod_l0 = mod.rates(h, a)
        own_h = mod_l0[0] * (av_h["attack_mult"] - 1)                          # home attackers missing
        opp_h = mod_l0[0] * av_h["attack_mult"] * (av_a["concede_mult"] - 1)   # away defenders missing
        own_a = mod_l0[1] * (av_a["attack_mult"] - 1)
        opp_a = mod_l0[1] * av_a["attack_mult"] * (av_h["concede_mult"] - 1)
        base_l = mods["long"].rates(h, a)
        mod_l = mod.rates(h, a)
        pr_model = predict(mod, h, a, hmult, amult)
        pr, used_blend = blended(mod, pr_model, fx.get("odds"), h, a)
        mk = market_probs(fx.get("odds"))
        f = {
            "fpl_id": fx["fpl_id"], "gw": fx["gw"], "kickoff": iso(fx["kickoff"]), "started": fx["started"],
            "home": h, "away": a, "home_full": FULL_NAME.get(h, h), "away_full": FULL_NAME.get(a, a),
            "home_short": short[h], "away_short": short[a],
            "probs": [r3(x) for x in pr["p"]], "xg": [r3(pr["lh"]), r3(pr["la"])],
            "model_probs": [r3(x) for x in pr_model["p"]], "model_xg": [r3(pr_model["lh"]), r3(pr_model["la"])],
            "blend_used": used_blend, "blend_model_weight": BLEND_MODEL_WEIGHT,
            "scores": pr["scores"], "grid": pr["grid"], "over25": r3(pr["over25"]), "btts": r3(pr["btts"]),
            "waterfall": {
                "home": {"usual": r3(base_l[0]), "form": r3(mod_l[0] - base_l[0]), "own_absences": r3(own_h),
                         "opp_absences": r3(opp_h), "market": r3(pr["lh"] - pr_model["lh"]), "final": r3(pr["lh"])},
                "away": {"usual": r3(base_l[1]), "form": r3(mod_l[1] - base_l[1]), "own_absences": r3(own_a),
                         "opp_absences": r3(opp_a), "market": r3(pr["la"] - pr_model["la"]), "final": r3(pr["la"])},
            },
            "absences": {"home": av_h, "away": av_a},
            "form": {"home": forms[h], "away": forms[a]},
            "h2h": head_to_head(allm + D["champ"], h, a),
            "market": [r3(x) for x in mk] if mk else None,
            "market_odds": fx.get("odds"),
        }
        f["calls"] = pick_calls(pr, h, a)
        f["markets"] = all_markets(pr, h, a, side, fx.get("referee"), fx.get("ou25"),
                                   abs_mult=(av_h["attack_mult"], av_a["attack_mult"]))
        f["markets"]["sot_form"] = {"home": team_count_form(cur, h, "hst", "ast"), "away": team_count_form(cur, a, "hst", "ast")}
        chance_fn = (lambda e, _fx=fx: player_chance(e, _fx["gw"], _fx["kickoff"], current_gw, now)[0] / 100)
        sides, p_goal = sc.predict(fpl_id[h], fpl_id[a], pr["lh"], pr["la"], now_iso, chance_fn=chance_fn, hist=hist_now)
        f["scorers"] = serialize(sides, p_goal, pr["lh"], pr["la"])
        f["markets"]["corner_form"] = {"home": team_count_form(cur, h, "hc", "ac"), "away": team_count_form(cur, a, "hc", "ac")}
        f["markets"]["card_form"] = {"home": team_count_form(cur, h, "hcards", "acards"),
                                     "away": team_count_form(cur, a, "hcards", "acards")}
        f["squad"] = {"home": squad_side(sq_rts[fpl_id[h]], sq_rating, chance_fn),
                      "away": squad_side(sq_rts[fpl_id[a]], sq_rating, chance_fn)}
        f["lineup_base_xg"] = [r3(mod_l0[0]), r3(mod_l0[1])]
        f["picks"] = quick_picks(f)
        f["strong"] = strong_picks(f)
        for sp in f["strong"]:   # result calls were tested with the bookmaker blend; without odds they may move
            sp["early"] = sp["key"] in ("result", "dc") and not used_blend
        f["odds_as_of"] = odds_lm if fx.get("odds") else None
        ev = oddset.get("events", {}).get(f"{h}|{a}")
        f["oddset"] = {**ev, "checked_at": ev.get("checked_at")} if ev else None
        f["verdict"] = verdict(f)
        fixtures_out.append(f)

    print("Backtesting...")
    bt, scorer_bt = backtest(D, promoted, current_gw, sc)
    by_gw = {}
    for r in bt:
        by_gw.setdefault(r["gw"], []).append(r)
    log = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as fh:
            log = json.load(fh)
    log = update_log(log, fixtures_out, D, now)
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)
    live = sorted([e for e in log.values() if e["result"]], key=lambda e: e["kickoff"], reverse=True)

    fd_cur = [m for m in cur if m["src"] == "football-data.co.uk"]
    fpl_extra = [m for m in cur if m["src"] != "football-data.co.uk"]
    ratings = sorted([{"team": t, "short": short[t], "attack": r3(math.exp(mod.mu + mod.att[t])),
                       "concede": r3(math.exp(mod.mu + mod.dfn[t])),
                       "net": forms[t]["level"]["model"], "level_tag": forms[t]["level_tag"],
                       "level_delta": forms[t]["level_delta"]} for t in cur_teams], key=lambda r: -r["net"])
    out = {
        "generated_at": iso(now),
        "data_as_of": {
            "results_csv_fetched": meta.get("E0_2627.csv", {}).get("fetched_at"),
            "results_csv_last_match": iso(max(m["kickoff"] for m in fd_cur)) if fd_cur else None,
            "fpl_fetched": meta.get("fpl_bootstrap.json", {}).get("fetched_at"),
            "odds_fetched": meta.get("fd_fixtures.csv", {}).get("fetched_at"),
            "odds_updated": odds_lm,
            "oddset_url": oddset.get("competition_url"),
            "fpl_results_added": [{"home": m["home"], "away": m["away"], "score": f"{m['hg']}-{m['ag']}",
                                   "kickoff": iso(m["kickoff"]), "provisional": "provisional" in m["src"]} for m in fpl_extra],
        },
        "current_gw": current_gw,
        "gameweeks": gws,
        "short": short,
        "full": {t: FULL_NAME.get(t, t) for t in cur_teams},
        "model": {
            "version": MODEL_VERSION, **SETTINGS, "long_half_life": LONG_HL, "short_half_life": SHORT_HL,
            "home_advantage": r3(math.exp(mod.home_adv)), "rho": r3(mod.rho), "avg_goals": r3(math.exp(mod.mu)),
            "matches_used": mod.n_matches, "promoted": sorted(promoted),
            "attack_loss": ATTACK_LOSS, "defence_loss": DEFENCE_LOSS,
            "blend_model_weight": BLEND_MODEL_WEIGHT, "data_experiments": DATA_EXPERIMENTS,
            "ht_share": r3(ht_share),
            "corners_avg": r3(corners.league_mean), "corners_phi": r3(corners.phi_team), "corners_phi_total": r3(corners.phi_total),
            "cards_avg": r3(cards.league_mean), "cards_phi": r3(cards.phi_team), "cards_phi_total": r3(cards.phi_total),
            "referees_rated": len(cards.refs),
            "sot_avg": r3(side["sot"].league_mean),
            "odds_timing": ODDS_TIMING,
            "lineup_weight": LINEUP_ATTACK_WEIGHT, "squad_tests": SQUAD_TESTS, "squad_tests_games": 87,
        },
        "table": league_table(cur, cur_teams),
        "ratings": ratings,
        "in_form": in_form_players(sq_rts, sq_rating, D["teams"]),
        "fixtures": fixtures_out,
        "track": {
            "backtest": bt[::-1],
            "backtest_summary": summarize_track(bt),
            "markets_summary": summarize_markets(bt),
            "strong_record": strong_record(bt),
            "scorers_summary": summarize_scorers(scorer_bt),
            "backtest_by_gw": [{"gw": g, **summarize_track(rs)} for g, rs in sorted(by_gw.items())],
            "live": live,
            "live_pending": sorted([e for e in log.values() if not e["result"]], key=lambda e: e["kickoff"]),
            "live_summary": summarize_track([{**e, **e["result"], "market": None, "outcome": e["result"]["outcome"]} for e in live]) if live else None,
        },
        "sources": [
            {"name": "football-data.co.uk", "url": "https://www.football-data.co.uk/englandm.php",
             "what": "Premier League results, half-time scores, xG, shots, corners, cards, referees and bookmaker odds (2015-16 to now), plus Championship results 2021-22 to 2025-26. Free CSV downloads.",
             "fetched": meta.get("E0_2627.csv", {}).get("fetched_at")},
            {"name": "Fantasy Premier League (public data)", "url": "https://fantasy.premierleague.com/",
             "what": "Fixture list and kickoff times, latest scores, player injury/suspension news and chance of playing, FPL starting prices, and per-match player minutes, starts, goals, assists, xG and xA (for goalscorers, player ratings and the line-up check).",
             "fetched": meta.get("fpl_bootstrap.json", {}).get("fetched_at")},
        ],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "data.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    if os.path.exists(TEMPLATE):
        with open(TEMPLATE, encoding="utf-8") as fh:
            html = fh.read()
        site = {}
        if os.path.exists(SITE_CONF):   # public Supabase settings for login and the bet log
            with open(SITE_CONF, encoding="utf-8") as fh:
                site = json.load(fh)
        html = html.replace("__SB_URL__", site.get("supabase_url", "")).replace("__SB_KEY__", site.get("supabase_anon_key", ""))
        # artifact.html loads data.json next to it; index.html has the data built in (open it directly,
        # or serve it as the website: the head makes it installable on a phone's home screen)
        with open(os.path.join(OUT, "artifact.html"), "w", encoding="utf-8") as fh:
            fh.write(html)
        payload = json.dumps(out, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
            fh.write('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                     '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                     '<link rel="manifest" href="manifest.webmanifest"><link rel="apple-touch-icon" href="icon-192.png">'
                     '<link rel="icon" href="icon-192.png"><meta name="theme-color" content="#0b6e4f">'
                     '<meta name="apple-mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-title" content="PL Predictor">'
                     '</head><body>\n' + html.replace("/*__DATA__*/null", payload) + "\n</body></html>\n")
        print(f"Wrote {os.path.join(OUT, 'index.html')}")
        copy_to_folders()
    s = out["track"]["backtest_summary"]
    print(f"Predicted {len(fixtures_out)} fixtures (GW {gws}). Backtest: {s['correct']}/{s['n']} outcomes, "
          f"{s['exact']} exact scores, RPS {s['rps']} (bookmakers {s['market_rps']}).")


if __name__ == "__main__":
    main()
