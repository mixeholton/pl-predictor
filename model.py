"""Dixon-Coles goals model.

Each team gets an attack rating and a defence rating. Expected goals in a match:

    home goals ~ Poisson( exp(mu + home_adv + attack[home] + defence[away]) )
    away goals ~ Poisson( exp(mu            + attack[away] + defence[home]) )

(defence is "goals let in" - higher means leakier.)

* Ratings are fitted by weighted maximum likelihood. Recent matches weigh more:
  a match `half_life` days old counts half as much as one played today. That is how
  form enters the model.
* For this season the target is a blend of real goals and xG (xG is a steadier
  signal of how well a team actually played).
* A mild prior pulls every rating towards 0 (an average PL side) so teams with
  few matches don't get extreme ratings. Promoted teams start from a weaker prior.
* Dixon & Coles' rho corrects the plain Poisson for low scores (0-0, 1-0, 0-1, 1-1),
  which happen a bit more/less often than independent Poisson says.
"""
import math

import numpy as np

MAX_GOALS = 10


class DixonColes:
    def __init__(self, half_life=120.0, xg_weight=0.5, prior_sd=0.35, window_days=800,
                 promoted=(), promoted_att=-0.20, promoted_def=0.20, prior_ratings=None,
                 shot_weight=0.0, e1_prior=(-0.35, 0.35), e1_weight=1.0):
        self.half_life = half_life
        self.xg_weight = xg_weight  # share of the target that is xG (when available)
        self.prior_sd = prior_sd
        self.window_days = window_days
        self.promoted = set(promoted)
        self.promoted_att = promoted_att
        self.promoted_def = promoted_def
        # optional {team: (attack, defence)} to shrink towards instead of 0 (used for the
        # short-term "current level" model, which starts from each team's usual level)
        self.prior_ratings = prior_ratings or {}
        # share of the target taken from shot-based xG in games that have no real xG
        self.shot_weight = shot_weight
        # Championship teams start from a lower baseline (attack, defence) and their games
        # can count less than Premier League games
        self.e1_prior = e1_prior
        self.e1_weight = e1_weight

    # ------------------------------------------------------------------ fitting
    def fit(self, matches, ref_date):
        ms = [m for m in matches if m["kickoff"] < ref_date
              and (ref_date - m["kickoff"]).days <= self.window_days]
        teams = sorted({m["home"] for m in ms} | {m["away"] for m in ms} | self.promoted)
        T = len(teams)
        ix = {t: i for i, t in enumerate(teams)}
        # Championship games share the team ratings but get their own scoring level,
        # so a promoted side's Championship form carries over on the same scale.
        has_e1 = any(m.get("div") == "E1" for m in ms)
        P = 2 + 2 * T + (1 if has_e1 else 0)
        e1c = 2 + 2 * T
        rows, ys, ws = [], [], []
        for m in ms:
            age = (ref_date - m["kickoff"]).total_seconds() / 86400
            w = 0.5 ** (age / self.half_life) * (self.e1_weight if m.get("div") == "E1" else 1.0)
            hy, ay = float(m["hg"]), float(m["ag"])
            if m.get("hxg") is not None and m.get("axg") is not None and self.xg_weight > 0:
                hy = (1 - self.xg_weight) * hy + self.xg_weight * m["hxg"]
                ay = (1 - self.xg_weight) * ay + self.xg_weight * m["axg"]
            elif m.get("hsxg") is not None and m.get("asxg") is not None and self.shot_weight > 0:
                hy = (1 - self.shot_weight) * hy + self.shot_weight * m["hsxg"]
                ay = (1 - self.shot_weight) * ay + self.shot_weight * m["asxg"]
            h, a = ix[m["home"]], ix[m["away"]]
            e1 = e1c if m.get("div") == "E1" else None
            rows.append((0, 1, 2 + h, 2 + T + a, e1)); ys.append(hy); ws.append(w)
            rows.append((0, None, 2 + a, 2 + T + h, e1)); ys.append(ay); ws.append(w)
        n = len(rows)
        X = np.zeros((n, P))
        for i, r in enumerate(rows):
            for c in r:
                if c is not None:
                    X[i, c] = 1.0
        y = np.array(ys); w = np.array(ws)

        prior = np.zeros(P)
        for t in self.promoted:
            if t in ix:
                prior[2 + ix[t]] = self.promoted_att
                prior[2 + T + ix[t]] = self.promoted_def
        if has_e1 and self.e1_prior:
            for t in {m["home"] for m in ms if m.get("div") == "E1"}:
                prior[2 + ix[t]] = self.e1_prior[0]
                prior[2 + T + ix[t]] = self.e1_prior[1]
        for t, (pa, pd) in self.prior_ratings.items():
            if t in ix:
                prior[2 + ix[t]] = pa
                prior[2 + T + ix[t]] = pd
        pen = np.full(P, 1.0 / (self.prior_sd ** 2)); pen[0] = 0.0; pen[1] = 1.0 / (0.5 ** 2)
        if has_e1:
            pen[e1c] = 1.0

        beta = prior.copy()
        beta[0] = math.log(max(y.mean(), 0.5)); beta[1] = 0.2
        for _ in range(50):
            eta = X @ beta
            lam = np.exp(eta)
            g = X.T @ (w * (y - lam)) - pen * (beta - prior)
            H = -(X.T * (w * lam)) @ X - np.diag(pen)
            step = np.linalg.solve(H, g)
            beta = beta - step
            if np.max(np.abs(step)) < 1e-8:
                break

        self.teams = teams
        self.ix = ix
        self.mu = beta[0]
        self.home_adv = beta[1]
        self.att = {t: beta[2 + i] for t, i in ix.items()}
        self.dfn = {t: beta[2 + T + i] for t, i in ix.items()}
        self.e1_offset = float(beta[e1c]) if has_e1 else None
        self.n_matches = len(ms)
        self.eff_matches = float(sum(ws[::2]))
        self.ref_date = ref_date
        self.rho = self._fit_rho(ms, ref_date)
        return self

    def _fit_rho(self, ms, ref_date):
        best, best_ll = 0.0, -1e18
        data = []
        for m in ms:
            age = (ref_date - m["kickoff"]).total_seconds() / 86400
            w = 0.5 ** (age / self.half_life)
            if m["hg"] <= 1 and m["ag"] <= 1 and m.get("div", "E0") == "E0":
                lh, la = self.rates(m["home"], m["away"])
                data.append((w, m["hg"], m["ag"], lh, la))
        for rho in np.arange(-0.25, 0.151, 0.005):
            ll = 0.0
            ok = True
            for w, hg, ag, lh, la in data:
                t = _tau(hg, ag, lh, la, rho)
                if t <= 0:
                    ok = False
                    break
                ll += w * math.log(t)
            # tau also changes normalisation slightly; ignored (standard DC practice)
            if ok and ll > best_ll:
                best, best_ll = float(rho), ll
        return best

    # --------------------------------------------------------------- predicting
    def rates(self, home, away, home_mult=1.0, away_mult=1.0):
        lh = math.exp(self.mu + self.home_adv + self.att[home] + self.dfn[away]) * home_mult
        la = math.exp(self.mu + self.att[away] + self.dfn[home]) * away_mult
        return lh, la

    def score_matrix(self, lh, la):
        return score_matrix(lh, la, self.rho)


def _tau(x, y, lh, la, rho):
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(lh, la, rho):
    ph = np.array([math.exp(-lh) * lh ** k / math.factorial(k) for k in range(MAX_GOALS + 1)])
    pa = np.array([math.exp(-la) * la ** k / math.factorial(k) for k in range(MAX_GOALS + 1)])
    M = np.outer(ph, pa)
    for x in (0, 1):
        for y_ in (0, 1):
            M[x, y_] *= max(_tau(x, y_, lh, la, rho), 0.0)
    return M / M.sum()


def outcome_probs(M):
    h = float(np.tril(M, -1).sum())
    d = float(np.trace(M))
    a = float(np.triu(M, 1).sum())
    return h, d, a


def expected_points_from_xg(xg_for, xg_against):
    """Points a team 'deserved' from its chances (independent Poisson on the xG values)."""
    M = score_matrix(max(xg_for, 0.01), max(xg_against, 0.01), 0.0)
    h, d, a = outcome_probs(M)
    return 3 * h + d
