"""Does squad strength help? Walk-forward test on this season (python tune_squad.py).

For each gameweek: refit the goals model on earlier games only, then adjust each side's
expected goals by how its squad compares with its usual line-up:
  * 'recent'  - players used in the last 3 games (known days before kickoff);
  * 'lineup'  - who actually started (known about an hour before kickoff);
  * '+ form'  - also add players' recent attacking form.
Scored with RPS (lower is better) and right results, for the model alone and the 40/60
blend with bookmaker odds.
"""
import json

import numpy as np

from data import load_all
from model import DixonColes, outcome_probs
from squad import Squad, load_live_full, multipliers
from tune import market_probs, rps

BASE = dict(half_life=180, xg_weight=0.7, prior_sd=0.35)
BLEND = 0.4

D = load_all()
hist, cur = D["history"], D["current"]
allm = hist + cur
last = {m["home"] for m in hist if m["season"] == "2526"}
promoted = {m["home"] for m in cur} - last
res = {(m["home"], m["away"]): m for m in cur}
fx_by_gw = {}
for fx in D["fixtures"]:
    if fx["gw"] and (fx["home"], fx["away"]) in res:
        fx_by_gw.setdefault(fx["gw"], []).append(fx)
gws = sorted(fx_by_gw)
sq = Squad(D["boot"], D["fpl_fixtures"], load_live_full(gws))
team_id = {t["name"]: tid for tid, t in D["teams"].items()}

# precompute per game: base rates, squad comparisons, outcome, market
games = []
for gw in gws:
    if gw < 3:
        continue   # need a couple of games to know a team's usual line-up
    fxs = fx_by_gw[gw]
    cutoff = min(f["kickoff"] for f in fxs)
    cut = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    mod = DixonColes(promoted=promoted, **BASE).fit(allm, cutoff)
    for fx in fxs:
        m = res[(fx["home"], fx["away"])]
        th, ta = team_id[fx["home"]], team_id[fx["away"]]
        cmp = {}
        for mode in ("recent", "lineup"):
            cmp[mode] = (sq.compare(th, cut, mode, started=sq.started(fx["fpl_id"], th)),
                         sq.compare(ta, cut, mode, started=sq.started(fx["fpl_id"], ta)))
        o = 0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2)
        mk = market_probs(m["odds_pre"]) if m.get("odds_pre") else None
        games.append({"gw": gw, "mod": mod, "h": fx["home"], "a": fx["away"], "cmp": cmp, "o": o, "mk": mk})
print(f"{len(games)} games, gameweeks 3-{gws[-1]}")


def score(mode, ka, kd, kf):
    r_m, r_b, c_m, c_b = [], [], 0, 0
    for g in games:
        if mode:
            hm, am = multipliers(g["cmp"][mode][0], g["cmp"][mode][1], ka, kd, kf)
        else:
            hm = am = 1.0
        lh, la = g["mod"].rates(g["h"], g["a"], hm, am)
        p = outcome_probs(g["mod"].score_matrix(lh, la))
        pb = [BLEND * x + (1 - BLEND) * y for x, y in zip(p, g["mk"])] if g["mk"] else p
        r_m.append(rps(p, g["o"])); r_b.append(rps(pb, g["o"]))
        c_m += int(np.argmax(p)) == g["o"]; c_b += int(np.argmax(pb)) == g["o"]
    return np.mean(r_m), c_m, np.mean(r_b), c_b


rows = []
base = score(None, 0, 0, 0)
print(f"{'variant':38s} model RPS  right   blend RPS  right")
print(f"{'current model':38s} {base[0]:.4f}   {base[1]:3d}     {base[2]:.4f}   {base[3]:3d}")
rows.append({"name": "current", "rps": base[0], "right": base[1], "blend_rps": base[2], "blend_right": base[3]})
for mode in ("recent", "lineup"):
    for ka in (0.25, 0.5, 0.75, 1.0):
        for kd in (0.0, 0.5, 1.0):
            for kf in (0.0, 0.5):
                s = score(mode, ka, kd, kf)
                name = f"{mode} att {ka} def {kd} form {kf}"
                rows.append({"name": name, "mode": mode, "ka": ka, "kd": kd, "kf": kf,
                             "rps": s[0], "right": s[1], "blend_rps": s[2], "blend_right": s[3]})
for mode in ("recent", "lineup"):
    best = sorted((r for r in rows if r.get("mode") == mode), key=lambda r: r["rps"])[:4]
    print(f"-- {mode}: best 4 by model RPS")
    for r in best:
        print(f"{r['name']:38s} {r['rps']:.4f}   {r['right']:3d}     {r['blend_rps']:.4f}   {r['blend_right']:3d}")
    mid = next(r for r in rows if r.get("mode") == mode and r["ka"] == 0.5 and r["kd"] == 0.5 and r["kf"] == 0.0)
    print(f"{'(middle setting) ' + mid['name']:38s} {mid['rps']:.4f}   {mid['right']:3d}     {mid['blend_rps']:.4f}   {mid['blend_right']:3d}")
with open("tune_squad_results.json", "w", encoding="utf-8") as f:
    json.dump({"games": len(games), "rows": rows}, f, indent=1)
