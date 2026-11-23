"""Does extra data help? Walk-forward test of model variants (python tune_data.py).

Variants: shot-based xG for seasons without xG, Championship games for promoted sides,
and blending with bookmaker odds. Mean RPS (lower is better) on 2025-26 and 2026-27.
"""
import sys
from datetime import timedelta

import numpy as np

from data import add_shot_xg, fit_shot_xg, load_all
from model import DixonColes, outcome_probs
from tune import market_probs, rps

D = load_all()
hist, cur, champ = D["history"], D["current"], D["champ"]
coef = fit_shot_xg(cur)
print("shot xG: %.3f per shot on target, %.3f per other shot" % coef)
add_shot_xg(hist + cur + champ, coef)

last = [m for m in hist if m["season"] == "2526"]
prom_last = {m["home"] for m in last} - {m["home"] for m in hist if m["season"] == "2425"}
prom_cur = {m["home"] for m in cur} - {m["home"] for m in last}


def walk(test, train, promoted, blend=1.0, **kw):
    out = []
    i = 0
    while i < len(test):
        start = test[i]["kickoff"]
        block = [m for m in test[i:] if m["kickoff"] < start + timedelta(days=7)]
        mod = DixonColes(promoted=promoted, **kw).fit(train, start)
        for m in block:
            p = outcome_probs(mod.score_matrix(*mod.rates(m["home"], m["away"])))
            mk = market_probs(m["odds"]) if m.get("odds") else None
            if mk and blend < 1:
                p = [blend * a + (1 - blend) * b for a, b in zip(p, mk)]
            o = 0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2)
            out.append((rps(p, o), m["home"] in promoted or m["away"] in promoted, rps(mk, o) if mk else None))
        i += len(block)
    return out


def run(name, kw, use_e1, blend=1.0):
    res = []
    for test, train_e0, prom in ((last[60:], hist, prom_last), (cur, hist + cur, prom_cur)):
        train = train_e0 + (champ if use_e1 else [])
        r = walk(test, train, prom, blend=blend, **{**BASE, **kw})
        res.append((np.mean([x[0] for x in r]), np.mean([x[0] for x in r if x[1]])))
    print(f"{name:40s} 25-26 {res[0][0]:.4f} (promoted {res[0][1]:.4f})   "
          f"26-27 {res[1][0]:.4f} (promoted {res[1][1]:.4f})   avg {(res[0][0] + res[1][0]) / 2:.4f}")


BASE = dict(half_life=180, xg_weight=0.7, prior_sd=0.35)
run("current model", {}, False)
if "--blend" not in sys.argv:
    for pr in ((-0.25, 0.25), (-0.35, 0.35), (-0.5, 0.5)):
        for w in (1.0, 0.5, 0.25):
            run(f"+ Championship prior {pr[0]:+.2f} weight {w}", {"e1_prior": pr, "e1_weight": w}, True)
print("\nblending with bookmaker odds (current model):")
for b in (1.0, 0.6, 0.5, 0.4, 0.3, 0.0):
    run(f"  model weight {b:.1f}", {}, False, blend=b)
