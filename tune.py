"""Pick the model settings by walk-forward testing (never peeking at future results).

Usage: python tune.py
Prints mean RPS (ranked probability score, lower is better) for a grid of settings
on (a) the 2025-26 season, refitting weekly, and (b) this season so far.
"""
from datetime import timedelta

import numpy as np

from data import load_all
from model import DixonColes, outcome_probs


def rps(p, outcome):
    o = [0, 0, 0]; o[outcome] = 1
    c1 = p[0] - o[0]; c2 = p[0] + p[1] - o[0] - o[1]
    return 0.5 * (c1 * c1 + c2 * c2)


def market_probs(odds):
    inv = [1 / x for x in odds]; s = sum(inv)
    return [x / s for x in inv]


def walk(matches, all_matches, promoted, hl, xw, sd):
    """Refit before each 7-day block of matches."""
    out = []
    i = 0
    while i < len(matches):
        start = matches[i]["kickoff"]
        block = [m for m in matches[i:] if m["kickoff"] < start + timedelta(days=7)]
        mod = DixonColes(half_life=hl, xg_weight=xw, prior_sd=sd, promoted=promoted).fit(all_matches, start)
        for m in block:
            if m["home"] not in mod.att or m["away"] not in mod.att:
                continue
            lh, la = mod.rates(m["home"], m["away"])
            p = outcome_probs(mod.score_matrix(lh, la))
            res = 0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2)
            mk = rps(market_probs(m["odds"]), res) if m.get("odds") else None
            out.append((rps(p, res), mk, int(np.argmax(p)) == res))
        i += len(block)
    return out


def main():
    D = load_all()
    hist, cur = D["history"], D["current"]
    allm = hist + cur
    last = [m for m in hist if m["season"] == "2526"]
    prev = {m["home"] for m in hist if m["season"] == "2425"}
    prom_last = {m["home"] for m in last} - prev
    prom_cur = {m["home"] for m in cur} - {m["home"] for m in last}
    print("promoted 25-26:", prom_last, " promoted 26-27:", prom_cur)
    test_last = [m for m in last if m["kickoff"].month not in (8,)]  # skip opening weeks
    for hl in (60, 90, 120, 180, 270):
        for sd in (0.25, 0.35):
            r1 = walk(test_last, hist, prom_last, hl, 0.5, sd)
            print(f"25-26 hl={hl:>3} sd={sd}: RPS {np.mean([x[0] for x in r1]):.4f}  "
                  f"acc {np.mean([x[2] for x in r1]):.3f}  market RPS {np.mean([x[1] for x in r1 if x[1] is not None]):.4f}")
    for hl in (90, 120, 180):
        for xw in (0.0, 0.5, 0.7, 1.0):
            r2 = walk(cur, allm, prom_cur, hl, xw, 0.35)
            print(f"26-27 hl={hl:>3} xg={xw}: RPS {np.mean([x[0] for x in r2]):.4f}  "
                  f"acc {np.mean([x[2] for x in r2]):.3f}  market RPS {np.mean([x[1] for x in r2 if x[1] is not None]):.4f}  n={len(r2)}")


if __name__ == "__main__":
    main()
