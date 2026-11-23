"""Download the raw data the predictor uses.

Sources (all free, no account or key needed):
  * football-data.co.uk  - results, match stats (xG, shots) and bookmaker odds as CSV.
                           Free to download; credit football-data.co.uk.
  * fantasy.premierleague.com/api - the Premier League's public Fantasy JSON:
                           fixture list with kickoff times, live/provisional scores,
                           and player availability news (injuries, suspensions).
                           Public, no login or key. Used lightly (2 requests per refresh)
                           for personal, non-commercial use.

Every file's download time is recorded in data/raw/meta.json so the page can show
"data as of" dates.
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
META = os.path.join(RAW, "meta.json")

CURRENT_SEASON = "2627"
PAST_SEASONS = ["1516", "1617", "1718", "1819",
                "1920", "2021", "2122", "2223", "2324", "2425", "2526"]  # head-to-head, referees, ratings
CHAMP_SEASONS = ["2122", "2223", "2324", "2425", "2526"]  # Championship: rates promoted sides

UA = "pl-predictor/1.0 (personal use)"

SOURCES = {
    # name: (url, refresh_every_run)
    "E0_2627.csv": (f"https://www.football-data.co.uk/mmz4281/{CURRENT_SEASON}/E0.csv", True),
    "fd_fixtures.csv": ("https://www.football-data.co.uk/fixtures.csv", True),
    "fpl_bootstrap.json": ("https://fantasy.premierleague.com/api/bootstrap-static/", True),
    "fpl_fixtures.json": ("https://fantasy.premierleague.com/api/fixtures/", True),
}
for s in PAST_SEASONS:
    SOURCES[f"E0_{s}.csv"] = (f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv", False)
for s in CHAMP_SEASONS:
    SOURCES[f"E1_{s}.csv"] = (f"https://www.football-data.co.uk/mmz4281/{s}/E1.csv", False)


def load_meta():
    if os.path.exists(META):
        with open(META, encoding="utf-8") as f:
            return json.load(f)
    return {}


def fetch_all(force=False, offline=False):
    os.makedirs(RAW, exist_ok=True)
    meta = load_meta()
    for name, (url, refresh) in SOURCES.items():
        path = os.path.join(RAW, name)
        if offline or (os.path.exists(path) and not refresh and not force):
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        last_modified = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                    last_modified = r.headers.get("Last-Modified")
                break
            except Exception as e:  # network hiccup: retry, then keep the old copy
                if attempt == 2:
                    print(f"  ! could not fetch {name}: {e} (keeping previous copy)")
                    body = None
                time.sleep(2)
        if body is None:
            continue
        if len(body) < 200:
            print(f"  ! {name} looks empty ({len(body)} bytes), keeping previous copy")
            continue
        with open(path, "wb") as f:
            f.write(body)
        meta[name] = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if last_modified:  # when the source last changed the file (e.g. when the odds were taken)
            try:
                meta[name]["last_modified"] = parsedate_to_datetime(last_modified).astimezone(timezone.utc).isoformat(timespec="seconds")
            except (TypeError, ValueError):
                pass
        print(f"  fetched {name} ({len(body)//1024} KB)" + (f", last changed {meta[name].get('last_modified')}" if last_modified else ""))
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def fetch_live(gws, finished, offline=False):
    """Per-gameweek player stats (minutes, goals, xG per match) from the FPL 'live' feed.

    Finished gameweeks are downloaded once and kept; unfinished ones are refreshed."""
    meta = load_meta()
    for gw in gws:
        name = f"fpl_live_{gw}.json"
        path = os.path.join(RAW, name)
        if offline or (os.path.exists(path) and gw in finished and meta.get(name, {}).get("final")):
            continue
        url = f"https://fantasy.premierleague.com/api/event/{gw}/live/"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
        except Exception as e:
            print(f"  ! could not fetch {name}: {e}")
            continue
        with open(path, "wb") as f:
            f.write(body)
        meta[name] = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "final": gw in finished}
        print(f"  fetched {name}")
        time.sleep(0.5)  # be gentle
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    fetch_all(force="--force" in sys.argv, offline="--offline" in sys.argv)
