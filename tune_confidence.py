"""How often do the model's calls come in, by how sure it was? (python tune_confidence.py)

Walk-forward over the 2025-26 season and this season: before each 7-day block of games,
refit the goals model and the corners / shots-on-target models on earlier games only,
blend with the bookmaker odds from a few days before (as on the page), and record each
call with its probability. Then count hits by probability band.

Used to set the "Strongest picks" cut-offs: they are chosen on 2025-26 and checked on
this season, so the page's numbers are not tuned on the games they are judged on.
Writes confidence_results.json.
"""
import json
from datetime import timedelta

import numpy as np

from build import SETTINGS, all_markets, blended, fit_side_models, predict, result_idx
from data import load_all
from model import DixonColes

D = load_all()
hist, cur = D["history"], D["current"]
allm = hist + cur
last = [m for m in hist if m["season"] == "2526"]
prev = {m["home"] for m in hist if m["season"] == "2425"}
PROMOTED = {"2526": {m["home"] for m in last} - prev,
            "2627": {m["home"] for m in cur} - {m["home"] for m in last}}


def walk(matches, season):
    rows = []
    matches = sorted(matches, key=lambda m: m["kickoff"])
    i = 0
    while i < len(matches):
        start = matches[i]["kickoff"]
        block = [m for m in matches[i:] if m["kickoff"] < start + timedelta(days=7)]
        i += len(block)
        if start.month == 8 and season == "2526":
            continue   # opening weeks: last season's ratings only
        mod = DixonColes(promoted=PROMOTED[season], **SETTINGS).fit(allm, start)
        side = fit_side_models(allm, start)
        for m in block:
            h, a = m["home"], m["away"]
            if h not in mod.att or a not in mod.att:
                continue
            pr, _ = blended(mod, predict(mod, h, a), m.get("odds_pre"), h, a)
            mk = all_markets(pr, h, a, side)
            o = result_idx(m["hg"], m["ag"])
            p = pr["p"]
            r = {"season": season, "result_p": max(p), "result_hit": int(np.argmax(p)) == o}
            fav_home = p[0] >= p[2]
            r["dc_p"] = p[0] + p[1] if fav_home else p[2] + p[1]
            r["dc_hit"] = o in (0, 1) if fav_home else o in (1, 2)
            for key, mkey, hk, ak in (("corners", "corners", "hc", "ac"), ("sot", "sot", "hst", "ast")):
                c = mk[mkey]
                if c and m.get(hk) is not None and m[hk] != m[ak]:
                    ph = c["more"][0] / (c["more"][0] + c["more"][2])
                    r[key + "_p"] = max(ph, 1 - ph)
                    r[key + "_hit"] = (ph >= 0.5) == (m[hk] > m[ak])
            rows.append(r)
    return rows


rows = walk(last, "2526") + walk(cur, "2627")
BANDS = {"result": [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.01],
         "dc": [0.0, 0.70, 0.75, 0.80, 0.85, 1.01],
         "corners": [0.5, 0.55, 0.60, 0.65, 0.70, 1.01],
         "sot": [0.5, 0.55, 0.60, 0.65, 0.70, 1.01]}
out = {"games": {s: sum(r["season"] == s for r in rows) for s in ("2526", "2627")}, "bands": {}, "at_least": {}}
for key, edges in BANDS.items():
    out["bands"][key] = []
    out["at_least"][key] = []
    for s in ("2526", "2627"):
        rs = [r for r in rows if r["season"] == s and key + "_p" in r]
        for lo, hi in zip(edges, edges[1:]):
            g = [r for r in rs if lo <= r[key + "_p"] < hi]
            if g:
                out["bands"][key].append({"season": s, "lo": lo, "hi": min(hi, 1.0), "n": len(g),
                                          "hits": sum(r[key + "_hit"] for r in g),
                                          "avg_p": round(float(np.mean([r[key + "_p"] for r in g])), 3)})
        for lo in edges[1:-1]:
            g = [r for r in rs if r[key + "_p"] >= lo]
            out["at_least"][key].append({"season": s, "min_p": lo, "n": len(g), "hits": sum(r[key + "_hit"] for r in g),
                                         "of": len(rs)})
for key in BANDS:
    print(f"== {key}")
    for b in out["at_least"][key]:
        print(f"  {b['season']} p>={b['min_p']:.2f}: {b['hits']:3d}/{b['n']:3d} = {b['hits'] / max(1, b['n']):.1%}"
              f"   ({b['n'] / max(1, b['of']):.0%} of games)")
with open("confidence_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
