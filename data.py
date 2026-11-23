"""Load and merge the raw files into one clean list of matches."""
import csv
import json
import os
from datetime import datetime, timezone

from fetch import RAW, PAST_SEASONS, CURRENT_SEASON, CHAMP_SEASONS

# FPL team name -> football-data.co.uk team name (our canonical name)
FPL_TO_FD = {
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}

FULL_NAME = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Leeds": "Leeds United",
    "Brighton": "Brighton & Hove Albion",
    "Hull": "Hull City",
    "Coventry": "Coventry City",
    "Ipswich": "Ipswich Town",
}


def _f(x):
    try:
        return float(x) if x not in ("", None) else None
    except ValueError:
        return None


def _date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(s)


def read_fd_csv(name, season, div="E0"):
    path = os.path.join(RAW, name)
    out = []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("HomeTeam") or r.get("FTHG") in ("", None):
                continue
            d = _date(r["Date"])
            t = r.get("Time") or "15:00"
            try:
                hh, mm = [int(x) for x in t.split(":")[:2]]
            except ValueError:
                hh, mm = 15, 0
            # football-data times are UK local; treat as UTC (+/-1h does not matter here)
            ko = d.replace(hour=hh, minute=mm, tzinfo=timezone.utc)
            odds = None
            for pre in ("AvgC", "Avg", "B365C", "B365"):
                h, dr, a = _f(r.get(pre + "H")), _f(r.get(pre + "D")), _f(r.get(pre + "A"))
                if h and dr and a:
                    odds = (h, dr, a)
                    break
            odds_pre = None  # odds a few days before kickoff (what we'd have had when predicting)
            for pre in ("Avg", "B365", "AvgC", "B365C"):
                h, dr, a = _f(r.get(pre + "H")), _f(r.get(pre + "D")), _f(r.get(pre + "A"))
                if h and dr and a:
                    odds_pre = (h, dr, a)
                    break
            ou25 = None
            for pre in ("AvgC", "Avg", "B365C", "B365"):
                o, u = _f(r.get(pre + ">2.5")), _f(r.get(pre + "<2.5"))
                if o and u:
                    ou25 = (o, u)
                    break
            hy, ay, hr_, ar_ = (_f(r.get(k)) for k in ("HY", "AY", "HR", "AR"))
            cards_ok = None not in (hy, ay, hr_, ar_)
            out.append({
                "season": season,
                "div": div,
                "hs": _f(r.get("HS")), "as": _f(r.get("AS")),
                "kickoff": ko,
                "home": r["HomeTeam"].strip(),
                "away": r["AwayTeam"].strip(),
                "hg": int(r["FTHG"]),
                "ag": int(r["FTAG"]),
                "hxg": _f(r.get("HxG")),
                "axg": _f(r.get("AxG")),
                "hst": _f(r.get("HST")),
                "ast": _f(r.get("AST")),
                "odds": odds,
                "ou25": ou25,
                "odds_pre": odds_pre,
                "hthg": int(_f(r.get("HTHG"))) if _f(r.get("HTHG")) is not None else None,
                "htag": int(_f(r.get("HTAG"))) if _f(r.get("HTAG")) is not None else None,
                "hc": _f(r.get("HC")), "ac": _f(r.get("AC")),
                "hcards": hy + hr_ if cards_ok else None, "acards": ay + ar_ if cards_ok else None,
                "referee": (r.get("Referee") or "").strip() or None,
                "src": "football-data.co.uk",
            })
    return out


def load_fpl():
    with open(os.path.join(RAW, "fpl_bootstrap.json"), encoding="utf-8") as f:
        boot = json.load(f)
    with open(os.path.join(RAW, "fpl_fixtures.json"), encoding="utf-8") as f:
        fixtures = json.load(f)
    teams = {}
    for t in boot["teams"]:
        name = FPL_TO_FD.get(t["name"], t["name"])
        teams[t["id"]] = {"name": name, "short": t["short_name"], "fpl_id": t["id"]}
    return boot, fixtures, teams


def fpl_kickoff(fx):
    return datetime.fromisoformat(fx["kickoff_time"].replace("Z", "+00:00")) if fx.get("kickoff_time") else None


def load_fd_upcoming_odds():
    """Pre-match average odds for fixtures in the next few days (football-data fixtures.csv)."""
    path = os.path.join(RAW, "fd_fixtures.csv")
    odds = {}
    if not os.path.exists(path):
        return odds
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Div") != "E0":
                continue
            h, d, a = _f(r.get("AvgH")), _f(r.get("AvgD")), _f(r.get("AvgA"))
            o, u = _f(r.get("Avg>2.5")), _f(r.get("Avg<2.5"))
            odds[(r["HomeTeam"].strip(), r["AwayTeam"].strip())] = {
                "odds": (h, d, a) if h and d and a else None,
                "ou25": (o, u) if o and u else None,
                "referee": (r.get("Referee") or "").strip() or None,
            }
    return odds


def load_all():
    """Returns dict with history (all seasons), current-season matches, FPL objects."""
    history = []
    for s in PAST_SEASONS:
        history += read_fd_csv(f"E0_{s}.csv", s)
    current = read_fd_csv(f"E0_{CURRENT_SEASON}.csv", CURRENT_SEASON)
    champ = []
    for s in CHAMP_SEASONS:
        if os.path.exists(os.path.join(RAW, f"E1_{s}.csv")):
            champ += read_fd_csv(f"E1_{s}.csv", s, div="E1")

    boot, fpl_fx, teams = load_fpl()
    upcoming_odds = load_fd_upcoming_odds()
    seen = {(m["home"], m["away"]): m for m in current}

    fixtures = []
    for fx in fpl_fx:
        home = teams[fx["team_h"]]["name"]
        away = teams[fx["team_a"]]["name"]
        ko = fpl_kickoff(fx)
        done = bool(fx.get("finished") or fx.get("finished_provisional"))
        row = {
            "fpl_id": fx["id"], "gw": fx["event"], "kickoff": ko, "home": home, "away": away,
            "finished": done, "provisional": done and not fx.get("finished"),
            "started": bool(fx.get("started")),
            "hg": fx.get("team_h_score"), "ag": fx.get("team_a_score"),
        }
        m = seen.get((home, away))
        if m is not None:
            m["gw"] = fx["event"]
            m["fpl_id"] = fx["id"]
            m["kickoff"] = ko or m["kickoff"]
        elif done and row["hg"] is not None:
            # result newer than the football-data CSV: take the score from FPL (no xG yet)
            up = upcoming_odds.get((home, away)) or {}
            current.append({
                "season": CURRENT_SEASON, "div": "E0", "hs": None, "as": None,
                "kickoff": ko, "home": home, "away": away,
                "hg": int(row["hg"]), "ag": int(row["ag"]), "hxg": None, "axg": None,
                "hst": None, "ast": None, "odds": up.get("odds"), "odds_pre": up.get("odds"), "ou25": up.get("ou25"),
                "hthg": None, "htag": None, "hc": None, "ac": None, "hcards": None, "acards": None,
                "referee": up.get("referee"),
                "src": "FPL (provisional)" if row["provisional"] else "FPL",
                "gw": fx["event"], "fpl_id": fx["id"],
            })
        up = upcoming_odds.get((home, away)) or {}
        row["odds"] = up.get("odds") or (m["odds"] if m else None)
        row["ou25"] = up.get("ou25") or (m["ou25"] if m else None)
        row["referee"] = up.get("referee") or (m["referee"] if m else None)
        fixtures.append(row)
    current.sort(key=lambda m: m["kickoff"])
    return {
        "history": history,
        "champ": champ,
        "current": current,
        "fixtures": fixtures,
        "boot": boot,
        "fpl_fixtures": fpl_fx,
        "teams": teams,
    }


def fit_shot_xg(matches):
    """Shot-based stand-in for xG, for seasons without xG: xG ~ a*on-target + b*off-target.

    Fitted (least squares, no intercept) on this season's games, which have both."""
    X, y = [], []
    for m in matches:
        for xg, st, sh in ((m.get("hxg"), m.get("hst"), m.get("hs")), (m.get("axg"), m.get("ast"), m.get("as"))):
            if xg is not None and st is not None and sh is not None:
                X.append((st, sh - st)); y.append(xg)
    if len(y) < 40:
        return None
    import numpy as np
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return float(coef[0]), float(coef[1])


def add_shot_xg(matches, coef):
    if not coef:
        return
    a, b = coef
    for m in matches:
        for side in ("h", "a"):
            st, sh = m.get(side + "st"), m.get(side + "s")
            m[side + "sxg"] = a * st + b * (sh - st) if st is not None and sh is not None else None
