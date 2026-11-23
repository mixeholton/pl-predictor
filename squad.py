"""Player ratings and squad strength, from the official FPL per-match stats.

Each player gets:
  * an attack value: (xG + 0.7 xA) per 90, shrunk towards what players of his position and
    FPL price usually produce until he has played ~5 full games;
  * a defence value (goalkeepers and defenders): his FPL starting price above the 4.0 floor,
    the game's own view of how good he is;
  * a form number: attack value over his last 4 team games compared with his season.

A team's squad strength for a match is the minutes-weighted sum of those values for the
players expected to play, compared with its usual line-up this season. The goals model's
team ratings already reflect the usual line-up, so only the difference matters.
"""
import json
import math
import os

import numpy as np

from fetch import RAW

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
FALLBACK90 = {1: 0.0, 2: 0.06, 3: 0.18, 4: 0.40}   # xG+0.7xA per 90 by position
PRIOR_MIN = 450
FORM_GAMES = 4


def load_live_full(gws):
    """Per gameweek, per player: minutes, starts, xG, xA, goals, assists, fixtures."""
    live = {}
    for gw in gws:
        p = os.path.join(RAW, f"fpl_live_{gw}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        per = {}
        for e in d["elements"]:
            s = e["stats"]
            per[e["id"]] = {"min": s["minutes"], "start": s.get("starts", 0), "xg": float(s["expected_goals"]),
                            "xa": float(s["expected_assists"]), "g": s["goals_scored"], "a": s["assists"],
                            "fixtures": [x["fixture"] for x in e.get("explain", [])]}
        live[gw] = per
    return live


def start_price(e):
    return (e["now_cost"] - (e.get("cost_change_start") or 0)) / 10


class Squad:
    def __init__(self, boot, fpl_fixtures, live):
        self.players = {e["id"]: e for e in boot["elements"]}
        self.fx = {f["id"]: f for f in fpl_fixtures}
        self.live = live
        self._cache = {}

    # ------------------------------------------------------------------ history
    def history(self, cutoff):
        """Per team: list of finished games before cutoff; per game {pid: stats}."""
        if cutoff in self._cache:
            return self._cache[cutoff]
        games = {}
        for f in sorted(self.fx.values(), key=lambda f: f.get("kickoff_time") or ""):
            done = f.get("finished") or f.get("finished_provisional")
            if not done or not f.get("kickoff_time") or f["kickoff_time"] >= cutoff:
                continue
            gwl = self.live.get(f["event"], {})
            for t in (f["team_h"], f["team_a"]):
                rows = {}
                for pid, st in gwl.items():
                    e = self.players.get(pid)
                    if e and e["team"] == t and f["id"] in st["fixtures"]:
                        n = len(st["fixtures"])
                        rows[pid] = {k: (st[k] / n if k != "start" else st[k]) for k in ("min", "start", "xg", "xa", "g", "a")}
                games.setdefault(t, []).append({"fid": f["id"], "rows": rows})
        self._cache[cutoff] = games
        return games

    def price_prior(self, games):
        """Per position, a straight line from FPL starting price to attack per 90."""
        tot = {}
        for tg in games.values():
            for g in tg:
                for pid, r in g["rows"].items():
                    t = tot.setdefault(pid, [0.0, 0.0])
                    t[0] += r["min"]
                    t[1] += r["xg"] + 0.7 * r["xa"]
        fits = {}
        for pos in (2, 3, 4):
            X, y, w = [], [], []
            for pid, (m, v) in tot.items():
                e = self.players[pid]
                if e["element_type"] == pos and m >= 270:
                    X.append([1.0, start_price(e)])
                    y.append(90 * v / m)
                    w.append(m)
            if len(X) >= 15:
                W = np.sqrt(np.array(w))
                coef = np.linalg.lstsq(np.array(X) * W[:, None], np.array(y) * W, rcond=None)[0]
                fits[pos] = coef
        return fits

    def prior90(self, e, fits):
        pos = e["element_type"]
        if pos == 1:
            return 0.0
        if pos in fits:
            a, b = fits[pos]
            return max(0.01, a + b * start_price(e))
        return FALLBACK90[pos]

    # ------------------------------------------------------------------ ratings
    def ratings(self, team, cutoff):
        """Every player of a team with his values, usual minutes and recent minutes."""
        games = self.history(cutoff)
        fits = self.price_prior(games)
        tg = games.get(team, [])
        n = len(tg)
        out = {}
        for pid, e in self.players.items():
            if e["team"] != team:
                continue
            rows = [g["rows"].get(pid) for g in tg]
            mins = [r["min"] if r else 0 for r in rows]
            val = [(r["xg"] + 0.7 * r["xa"]) if r else 0.0 for r in rows]
            prior = self.prior90(e, fits)
            M, V = sum(mins), sum(val)
            att = (V + prior * PRIOR_MIN / 90) / ((M + PRIOR_MIN) / 90)
            rm, rv = sum(mins[-FORM_GAMES:]), sum(val[-FORM_GAMES:])
            att_recent = (rv + att * PRIOR_MIN / 2 / 90) / ((rm + PRIOR_MIN / 2) / 90)
            starts = [r["start"] if r else 0 for r in rows]
            start_mins = [m for m, s in zip(mins, starts) if s]
            bench_mins = [m for m, s in zip(mins, starts) if not s]
            out[pid] = {
                "pid": pid, "e": e, "pos": e["element_type"], "att": att, "att_recent": att_recent,
                "def": max(0.0, start_price(e) - 4.0) if e["element_type"] in (1, 2) else 0.0,
                "usual": (M / n / 90) if n else 0.0,                       # 90s per game this season
                "recent": (sum(mins[-3:]) / min(3, n) / 90) if n else 0.0,  # last 3 games
                "mins": M, "starts": sum(starts), "games": n,
                "min_when_start": (float(np.mean(start_mins)) if start_mins else 80.0),
                # minutes in games he didn't start (0 when unused), so XI + bench adds up to his usual
                "min_when_bench": (float(np.mean(bench_mins)) if bench_mins else 10.0),
                "goals": sum(r["g"] if r else 0 for r in rows), "assists": sum(r["a"] if r else 0 for r in rows),
                "xg": sum(r["xg"] if r else 0 for r in rows), "xa": sum(r["xa"] if r else 0 for r in rows),
                "last4_g": sum(r["g"] if r else 0 for r in rows[-FORM_GAMES:]),
                "last4_a": sum(r["a"] if r else 0 for r in rows[-FORM_GAMES:]),
                "last4_min": sum(mins[-FORM_GAMES:]),
                "price": start_price(e),
            }
        return out

    @staticmethod
    def strength(rt, weights):
        """Attack, defence and form totals for a set of expected 90s per player."""
        att = sum(weights.get(p, 0) * r["att"] for p, r in rt.items())
        de = sum(weights.get(p, 0) * r["def"] for p, r in rt.items() if r["pos"] in (1, 2))
        form = sum(weights.get(p, 0) * (r["att_recent"] - r["att"]) for p, r in rt.items())
        return att, de, form

    def compare(self, team, cutoff, mode="recent", started=None, chance=None):
        """Squad strength vs usual. mode: 'recent' (last 3 games), 'lineup' (who started), 'news' (chance of playing)."""
        rt = self.ratings(team, cutoff)
        usual = {p: r["usual"] for p, r in rt.items()}
        if mode == "lineup" and started is not None:
            exp = {p: (r["min_when_start"] if p in started else r["min_when_bench"]) / 90 for p, r in rt.items()}
        elif mode == "news" and chance is not None:
            exp = {p: r["usual"] * chance(r["e"]) for p, r in rt.items()}
        else:
            exp = {p: r["recent"] for p, r in rt.items()}
        ua, ud, _ = self.strength(rt, usual)
        ea, ed, ef = self.strength(rt, exp)
        att_dev = (ea / ua - 1) if ua > 0 else 0.0
        def_dev = (ed / ud - 1) if ud > 0 else 0.0
        form_dev = (ef / ua) if ua > 0 else 0.0
        return {"att_dev": att_dev, "def_dev": def_dev, "form_dev": form_dev, "ratings": rt, "exp": exp}

    def started(self, fid, team):
        """Players of a team who started a finished fixture."""
        f = self.fx[fid]
        out = set()
        for pid, st in self.live.get(f["event"], {}).items():
            e = self.players.get(pid)
            if e and e["team"] == team and fid in st["fixtures"] and st["start"]:
                out.add(pid)
        return out


def multipliers(home, away, ka, kd, kf=0.0):
    """Expected-goals multipliers (home, away) from two squad comparisons."""
    mh = math.exp(ka * home["att_dev"] + kf * home["form_dev"] - kd * away["def_dev"])
    ma = math.exp(ka * away["att_dev"] + kf * away["form_dev"] - kd * home["def_dev"])
    return min(1.35, max(0.7, mh)), min(1.35, max(0.7, ma))
