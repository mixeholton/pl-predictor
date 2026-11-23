"""Goalscorer predictions.

For each player:
  * a scoring rate per 90 minutes, mostly from xG (85%) plus a little real goals (15%),
    shrunk towards a typical rate for his position until he has played ~5 full games;
  * expected minutes in this match, from the last 4 team games (60%) and the season (40%),
    times his chance of playing (injury/suspension news).

The team's expected goals from the match model are then shared out between the players
in proportion to rate x minutes (about 3% of goals are own goals). So when a striker is
out, his share moves to the players who replace him.

anytime scorer = 1 - exp(-x), 2+ goals = 1 - exp(-x)(1 + x),
first scorer   = x / (both teams' expected goals) * P(at least one goal).
"""
import json
import math
import os

import numpy as np

from fetch import RAW

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PRIOR90 = {1: 0.0, 2: 0.04, 3: 0.12, 4: 0.30}    # typical goals per 90 by position
PRIOR_MIN = 450                                   # minutes of "typical player" mixed in
OWN_GOAL_SHARE = 0.03
GOALS_WEIGHT = 0.15


def r3(x):
    return round(float(x), 3)


def load_live(gws):
    live = {}
    for gw in gws:
        p = os.path.join(RAW, f"fpl_live_{gw}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        per = {}
        for e in d["elements"]:
            fx = [x["fixture"] for x in e.get("explain", [])]
            s = e["stats"]
            per[e["id"]] = {"minutes": s["minutes"], "goals": s["goals_scored"],
                            "xg": float(s["expected_goals"]), "fixtures": fx}
        live[gw] = per
    return live


class Scorers:
    def __init__(self, boot, fpl_fixtures, live):
        self.boot = boot
        self.players = {e["id"]: e for e in boot["elements"]}
        self.fx = {f["id"]: f for f in fpl_fixtures}
        self.live = live

    def history(self, before_kickoff):
        """Per player: list of (minutes, goals, xg) for each of his team's finished games before the cutoff."""
        team_games = {}
        for f in self.fx.values():
            done = f.get("finished") or f.get("finished_provisional")
            if not done or not f.get("kickoff_time") or f["kickoff_time"] >= before_kickoff:
                continue
            for t in (f["team_h"], f["team_a"]):
                team_games.setdefault(t, []).append(f)
        for t in team_games:
            team_games[t].sort(key=lambda f: f["kickoff_time"])
        hist = {}
        for pid, e in self.players.items():
            games = []
            for f in team_games.get(e["team"], []):
                st = self.live.get(f["event"], {}).get(pid)
                if st and f["id"] in st["fixtures"]:
                    games.append((st["minutes"], st["goals"], st["xg"]))
                else:
                    games.append((0, 0, 0.0))
            hist[pid] = games
        return hist

    @staticmethod
    def rate90(e, games):
        mins = sum(g[0] for g in games)
        xg = sum(g[2] for g in games)
        goals = sum(g[1] for g in games)
        signal = (1 - GOALS_WEIGHT) * xg + GOALS_WEIGHT * goals
        prior = PRIOR90[e["element_type"]]
        return (signal + prior * PRIOR_MIN / 90) / ((mins + PRIOR_MIN) / 90)

    @staticmethod
    def exp_minutes(games):
        if not games:
            return 0.0
        recent = [g[0] for g in games[-4:]]
        return 0.6 * float(np.mean(recent)) + 0.4 * float(np.mean([g[0] for g in games]))

    def predict(self, fpl_team_h, fpl_team_a, lh, la, before_kickoff, chance_fn=None, top=8, hist=None):
        hist = hist if hist is not None else self.history(before_kickoff)
        sides = {}
        tot = lh + la
        p_goal = 1 - math.exp(-tot)
        for side, team, lam in (("home", fpl_team_h, lh), ("away", fpl_team_a, la)):
            rows = []
            for pid, e in self.players.items():
                if e["team"] != team or (e["status"] == "u" and chance_fn):
                    continue
                games = hist.get(pid, [])
                m = self.exp_minutes(games)
                chance = chance_fn(e) if chance_fn else 1.0
                m *= chance
                if m < 1:
                    continue
                r = self.rate90(e, games)
                rows.append({"pid": pid, "e": e, "m": m, "r": r, "w": r * m / 90, "games": games, "chance": chance})
            W = sum(x["w"] for x in rows) or 1.0
            out = []
            for x in rows:
                xg_match = lam * (1 - OWN_GOAL_SHARE) * x["w"] / W
                e, g = x["e"], x["games"]
                out.append({
                    "pid": x["pid"], "name": e["web_name"], "pos": POS[e["element_type"]],
                    "xg": xg_match, "anytime": 1 - math.exp(-xg_match),
                    "brace": 1 - math.exp(-xg_match) * (1 + xg_match),
                    "first": xg_match / tot * p_goal if tot > 0 else 0.0,
                    "exp_minutes": x["m"], "rate90": x["r"], "chance": x["chance"],
                    "season_goals": sum(v[1] for v in g), "season_xg": sum(v[2] for v in g),
                    "season_minutes": sum(v[0] for v in g),
                    "last5_goals": sum(v[1] for v in g[-5:]), "last5_xg": sum(v[2] for v in g[-5:]),
                })
            out.sort(key=lambda p: -p["xg"])
            sides[side] = out if top is None else out[:top]
        return sides, p_goal


def serialize(sides, p_goal, lh, la):
    def row(p):
        return {"name": p["name"], "pos": p["pos"], "xg": r3(p["xg"]), "anytime": r3(p["anytime"]),
                "brace": r3(p["brace"]), "first": r3(p["first"]), "exp_minutes": round(p["exp_minutes"]),
                "chance": round(100 * p["chance"]), "season_goals": p["season_goals"],
                "season_xg": round(p["season_xg"], 2), "season_minutes": p["season_minutes"],
                "last5_goals": p["last5_goals"], "last5_xg": round(p["last5_xg"], 2),
                "rate90": round(p["rate90"], 3)}
    return {"home": [row(p) for p in sides["home"]], "away": [row(p) for p in sides["away"]],
            "no_scorer": r3(1 - p_goal),
            "first_none": r3(1 - p_goal)}


# ---------------------------------------------------------------- backtest grading

BUCKETS = [(0, .1), (.1, .2), (.2, .3), (.3, .45), (.45, 1.01)]


def grade(sc, fx, lh, la, gw_live, hist):
    """Predict all players for one finished match (no team news) and check who scored."""
    sides, _ = sc.predict(fx["team_h"], fx["team_a"], lh, la, fx["kickoff_time"], top=None, hist=hist)
    rows = []
    for side in ("home", "away"):
        for p in sides[side]:
            st = gw_live.get(p["pid"])
            played = bool(st and fx["id"] in st["fixtures"] and st["minutes"] > 0)
            scored = bool(st and fx["id"] in st["fixtures"] and st["goals"] > 0)
            rows.append({"side": side, "name": p["name"], "p": p["anytime"], "scored": scored, "played": played})
    return rows


def summarize_scorers(match_rows):
    """match_rows: list of (match label, rows)."""
    allr = [r for _, rs in match_rows for r in rs]
    if not allr:
        return None
    base = float(np.mean([r["scored"] for r in allr]))
    buckets = []
    for lo, hi in BUCKETS:
        b = [r for r in allr if lo <= r["p"] < hi]
        if b:
            buckets.append({"from": lo, "to": min(hi, 1.0), "n": len(b), "predicted": r3(np.mean([r["p"] for r in b])),
                            "actual": r3(np.mean([r["scored"] for r in b]))})
    top1 = []; top3 = []
    for _, rs in match_rows:
        rs = sorted(rs, key=lambda r: -r["p"])
        if rs:
            top1.append((rs[0]["p"], rs[0]["scored"]))
            top3.append(any(r["scored"] for r in rs[:3]))
    return {
        "matches": len(match_rows), "player_predictions": len(allr),
        "brier": r3(np.mean([(r["p"] - r["scored"]) ** 2 for r in allr])),
        "base_brier": r3(np.mean([(base - r["scored"]) ** 2 for r in allr])),
        "predicted_scorers": r3(sum(r["p"] for r in allr)), "actual_scorers": int(sum(r["scored"] for r in allr)),
        "buckets": buckets,
        "top1_hits": int(sum(s for _, s in top1)), "top1_expected": r3(sum(p for p, _ in top1)),
        "top3_any": int(sum(top3)),
    }
