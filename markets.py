"""Extra predictions beyond the result: goals lines, half-time, corners and cards.

Goals and half-time come straight from the goals model's expected goals.
Corners and cards get their own team ratings, fitted the same way as goals
(weighted, recent games count more, mild prior), because a team that scores a lot
is not necessarily one that wins lots of corners or picks up lots of bookings.
Counts are spread with a negative binomial, which allows for more variation than a
plain Poisson (one match has 4 corners, the next 16).
"""
import math

import numpy as np

from model import DixonColes, score_matrix

# ---------------------------------------------------------------- distributions


def poisson_pmf(lam, n=40):
    return np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(n)])


def negbin_pmf(mean, phi, n=40):
    """Negative binomial with the given mean and variance = phi * mean (phi > 1)."""
    if phi <= 1.01 or mean <= 0:
        return poisson_pmf(max(mean, 1e-6), n)
    r = mean / (phi - 1)
    p = r / (r + mean)
    out = np.zeros(n)
    out[0] = p ** r
    for k in range(1, n):
        out[k] = out[k - 1] * (k - 1 + r) / k * (1 - p)
    return out / out.sum()


def over(pmf, line):
    return float(pmf[int(math.floor(line)) + 1:].sum())


def r3(x):
    return round(float(x), 3)


# ---------------------------------------------------------------- goals


def goal_markets(M, lh, la, ou_market=None):
    n = M.shape[0]
    tot = np.zeros(2 * n)
    for i in range(n):
        for j in range(n):
            tot[i + j] += M[i, j]
    home_goals, away_goals = M.sum(axis=1), M.sum(axis=0)
    pH = float(np.tril(M, -1).sum()); pD = float(np.trace(M)); pA = float(np.triu(M, 1).sum())
    margin = {}
    for k, lab in ((1, "1"), (2, "2"), (3, "3+")):
        if lab == "3+":
            h = sum(M[i, j] for i in range(n) for j in range(n) if i - j >= 3)
            a = sum(M[i, j] for i in range(n) for j in range(n) if j - i >= 3)
        else:
            h = sum(M[j + k, j] for j in range(n - k)); a = sum(M[j, j + k] for j in range(n - k))
        margin[lab] = [r3(h), r3(a)]
    p_any = 1 - float(M[0, 0])
    # first goal: each goal is the home side's with share lh/(lh+la)
    first_home = p_any * lh / (lh + la)
    out = {
        "expected_total": r3(lh + la),
        "total_dist": [r3(x) for x in list(tot[:6]) + [tot[6:].sum()]],   # 0..5, 6+
        "lines": [{"line": l, "over": r3(over(tot, l))} for l in (0.5, 1.5, 2.5, 3.5, 4.5)],
        "home_lines": [{"line": l, "over": r3(over(home_goals, l))} for l in (0.5, 1.5, 2.5)],
        "away_lines": [{"line": l, "over": r3(over(away_goals, l))} for l in (0.5, 1.5, 2.5)],
        "btts": r3(M[1:, 1:].sum()),
        "clean_sheet": [r3(away_goals[0]), r3(home_goals[0])],      # home keeps one, away keeps one
        "win_to_nil": [r3(M[1:, 0].sum()), r3(M[0, 1:].sum())],
        "double_chance": {"1X": r3(pH + pD), "X2": r3(pD + pA), "12": r3(pH + pA)},
        "draw_no_bet": [r3(pH / (pH + pA)), r3(pA / (pH + pA))],
        "handicap_15": [r3(sum(M[i, j] for i in range(n) for j in range(n) if i - j >= 2)),
                        r3(sum(M[i, j] for i in range(n) for j in range(n) if j - i >= 2))],
        "margin": margin,
        "first_goal": [r3(first_home), r3(M[0, 0]), r3(p_any - first_home)],   # home, no goal, away
        "odd": r3(sum(tot[k] for k in range(1, len(tot), 2))),
    }
    if ou_market:
        inv = [1 / ou_market[0], 1 / ou_market[1]]
        out["market_over25"] = r3(inv[0] / sum(inv))
    return out


def halftime_markets(lh, la, share):
    """First half ~ `share` of each side's expected goals, second half the rest (independent)."""
    H1 = score_matrix(lh * share, la * share, 0.0)
    H2 = score_matrix(lh * (1 - share), la * (1 - share), 0.0)
    n = H1.shape[0]

    def res(i, j):
        return 0 if i > j else (1 if i == j else 2)
    ht = [0.0, 0.0, 0.0]
    htft = np.zeros((3, 3))
    for i in range(n):
        for j in range(n):
            p1 = H1[i, j]
            if p1 < 1e-7:
                continue
            ht[res(i, j)] += p1
            for k in range(n - i):
                for l in range(n - j):
                    htft[res(i, j), res(i + k, j + l)] += p1 * H2[k, l]
    labels = ["Home", "Draw", "Away"]
    combos = sorted(({"ht": labels[a], "ft": labels[b], "p": r3(htft[a, b])} for a in range(3) for b in range(3)),
                    key=lambda c: -c["p"])
    t1 = float(sum(H1[i, j] for i in range(n) for j in range(n) if i + j >= 1))
    goals_2h_more = 1 - share > share
    return {
        "ht_probs": [r3(x) for x in ht],
        "htft": combos,
        "ht_goal": r3(t1),                      # at least one first-half goal
        "ht_over15": r3(float(sum(H1[i, j] for i in range(n) for j in range(n) if i + j >= 2))),
        "ht_xg": [r3(lh * share), r3(la * share)],
        "share": r3(share),
        "second_half_busier": goals_2h_more,
    }


def halftime_share(matches, ref_date, seasons=3):
    ms = [m for m in matches if m.get("hthg") is not None and m["kickoff"] < ref_date]
    ms = sorted(ms, key=lambda m: m["kickoff"])[-380 * seasons:]
    ft = sum(m["hg"] + m["ag"] for m in ms)
    return sum(m["hthg"] + m["htag"] for m in ms) / ft if ft else 0.43


# ---------------------------------------------------------------- counts (corners, cards)


class CountModel:
    """Team ratings for a match count (corners won, cards shown) + negative binomial spread."""

    def __init__(self, home_key, away_key, half_life=180, prior_sd=0.25):
        self.hk, self.ak = home_key, away_key
        self.half_life, self.prior_sd = half_life, prior_sd

    def fit(self, matches, ref_date, referee_effect=False):
        rows = [dict(m, hg=m[self.hk], ag=m[self.ak], hxg=None, axg=None) for m in matches
                if m.get(self.hk) is not None and m.get(self.ak) is not None and m["kickoff"] < ref_date]
        self.dc = DixonColes(half_life=self.half_life, xg_weight=0, prior_sd=self.prior_sd).fit(rows, ref_date)
        self.teams = set(self.dc.teams)
        # dispersion from recent residuals (variance / mean)
        recent = [r for r in rows if (ref_date - r["kickoff"]).days <= 400]
        num_t = num_h = 0.0; n = 0
        for r in recent:
            mh, ma = self.dc.rates(r["home"], r["away"])
            num_h += ((r["hg"] - mh) ** 2 / mh + (r["ag"] - ma) ** 2 / ma) / 2
            num_t += (r["hg"] + r["ag"] - mh - ma) ** 2 / (mh + ma)
            n += 1
        self.phi_team = max(1.0, num_h / n) if n else 1.2
        self.phi_total = max(1.0, num_t / n) if n else 1.1
        self.league_mean = float(np.mean([r["hg"] + r["ag"] for r in recent])) if recent else None
        self.refs = {}
        if referee_effect:
            w_by = {}
            for r in rows:
                ref = r.get("referee")
                if not ref:
                    continue
                age = (ref_date - r["kickoff"]).days
                if age > 800:
                    continue
                mh, ma = self.dc.rates(r["home"], r["away"])
                w = 0.5 ** (age / 365)
                a = w_by.setdefault(ref, [0.0, 0.0, 0])
                a[0] += w * (r["hg"] + r["ag"]); a[1] += w * (mh + ma); a[2] += 1
            for ref, (obs, exp, cnt) in w_by.items():
                k = 6.0  # shrink: a referee needs a decent sample before he moves the numbers much
                self.refs[ref] = {"mult": (obs + k) / (exp + k) if exp else 1.0, "games": cnt}
        return self

    def predict(self, home, away, referee=None):
        if home not in self.teams or away not in self.teams:
            return None
        mh, ma = self.dc.rates(home, away)
        ref = self.refs.get(referee) if referee else None
        mult = ref["mult"] if ref else 1.0
        out = self.predict_from_means(mh * mult, ma * mult)
        out["ref_mult"] = mult
        out["ref_games"] = ref["games"] if ref else 0
        return out

    def predict_from_means(self, mh, ma):
        ph, pa = negbin_pmf(mh, self.phi_team), negbin_pmf(ma, self.phi_team)
        tot = negbin_pmf(mh + ma, self.phi_total, 60)
        more_h = float(sum(ph[i] * pa[:i].sum() for i in range(len(ph))))
        more_a = float(sum(pa[i] * ph[:i].sum() for i in range(len(pa))))
        return {"mh": mh, "ma": ma, "tot": tot, "ph": ph, "pa": pa,
                "more": [more_h, 1 - more_h - more_a, more_a],
                "ref_mult": 1.0, "ref_games": 0}


def count_markets(pred, lines, team_lines):
    if pred is None:
        return None
    tot = pred["tot"]
    dist_lo = int(max(0, math.floor(pred["mh"] + pred["ma"]) - 6))
    return {
        "expected": [r3(pred["mh"]), r3(pred["ma"])],
        "expected_total": r3(pred["mh"] + pred["ma"]),
        "lines": [{"line": l, "over": r3(over(tot, l))} for l in lines],
        "home_lines": [{"line": l, "over": r3(over(pred["ph"], l))} for l in team_lines],
        "away_lines": [{"line": l, "over": r3(over(pred["pa"], l))} for l in team_lines],
        "more": [r3(x) for x in pred["more"]],
        "dist": {"from": dist_lo, "p": [r3(x) for x in tot[dist_lo:dist_lo + 13]]},
        "most_likely_total": int(np.argmax(tot)),
        "ref_mult": r3(pred["ref_mult"]), "ref_games": pred["ref_games"],
    }


def team_count_form(cur, team, hk, ak, n=6):
    gs = []
    for m in cur:
        if team not in (m["home"], m["away"]) or m.get(hk) is None:
            continue
        home = m["home"] == team
        gs.append((m[hk], m[ak]) if home else (m[ak], m[hk]))
    gs = gs[-n:]
    if not gs:
        return None
    return {"n": len(gs), "for_pg": r3(np.mean([g[0] for g in gs])), "against_pg": r3(np.mean([g[1] for g in gs]))}
