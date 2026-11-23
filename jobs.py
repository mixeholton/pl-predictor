"""Cloud job helper (GitHub Actions): answers the page's buttons and sends match-day notifications.

Reads the owner's saved data from Supabase through the token-protected `feed` function
(no service key needed). Environment: SUPABASE_URL, SUPABASE_ANON_KEY, FEED_TOKEN, NTFY_TOPIC.

  python jobs.py check      look for a "Refresh predictions" / "Send test notification" request;
                            sends the test at once; prints build=true when a refresh is wanted
  python jobs.py done       after a successful refresh: mark the request done and notify
  python jobs.py failed     after a failed refresh: mark the request failed
  python jobs.py matchday   on a day with a Premier League match, notify today's strong picks
                            (unless on a break or past a limit)

Never logs in to, reads or bets on any bookmaker.
"""
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import notify

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "out", "data.json")
LOCAL = ZoneInfo("Europe/Copenhagen")
STALE = timedelta(minutes=20)
MARGIN = 1.05
DEF_LIMITS = {"weekly_budget": 200, "monthly_loss_limit": 300}


def rpc(name, body):
    url = os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/rpc/" + name
    key = os.environ["SUPABASE_ANON_KEY"]
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"apikey": key, "Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def feed():
    return rpc("feed", {"token": os.environ["FEED_TOKEN"]}) or {}


def update(path, patch):
    rpc("feed_update", {"token": os.environ["FEED_TOKEN"], "p_path": path, "p_patch": patch})


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def output(**kv):
    lines = "".join(f"{k}={v}\n" for k, v in kv.items())
    print(lines, end="")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(lines)


def pending_job(docs):
    job = docs.get("job")
    if not job:
        return None
    st, t = job.get("status"), parse(job.get("checking_at") or job.get("requested_at"))
    if st == "waiting" or (st == "checking" and t and datetime.now(timezone.utc) - t > STALE):
        return job
    return None


def cmd_check():
    job = pending_job(feed())
    if not job:
        output(build="false")
        return
    action = job.get("action")
    update("job", {"status": "checking", "checking_at": now_iso()})
    if action == "test_notify":
        try:
            notify.send("Test from the page: phone notifications work.")
            update("job", {"status": "done", "done_at": now_iso(), "summary": "Test notification sent."})
        except Exception as e:   # report on the page, don't fail the run
            update("job", {"status": "failed", "done_at": now_iso(), "summary": f"Couldn't send: {e}"[:200]})
        output(build="false")
    elif action == "refresh":
        output(build="true")
    else:
        update("job", {"status": "failed", "done_at": now_iso(), "summary": "Unknown request"})
        output(build="false")


def strong_soon(data, days=4):
    now = datetime.now(timezone.utc)
    return [(f, s) for f in data["fixtures"] for s in (f.get("strong") or [])
            if now < parse(f["kickoff"]) <= now + timedelta(days=days)]


def cmd_done():
    with open(DATA, encoding="utf-8") as fh:
        n = len(strong_soon(json.load(fh)))
    update("job", {"status": "done", "done_at": now_iso(),
                   "summary": f"Predictions refreshed and page republished. {n} strong picks in the next 4 days."})
    notify.send(f"Predictions refreshed: {n} strong picks in the next 4 days. Open the page to see them.")


def cmd_failed():
    update("job", {"status": "failed", "done_at": now_iso(), "summary": "The refresh failed. See the GitHub Actions log."})


# ----------------------------------------------------------------------- match day
def effective_limits(docs):
    lim = docs.get("limits") or {}
    e = {"weekly_budget": lim.get("weekly_budget", DEF_LIMITS["weekly_budget"]),
         "monthly_loss_limit": lim.get("monthly_loss_limit", DEF_LIMITS["monthly_loss_limit"]),
         "break_until": lim.get("break_until")}
    p = lim.get("pending") or {}
    if p and parse(p.get("effective_at")) and datetime.now(timezone.utc) >= parse(p["effective_at"]):
        for k in ("weekly_budget", "monthly_loss_limit"):
            if p.get(k) is not None:
                e[k] = p[k]
        if p.get("end_break"):
            e["break_until"] = None
    return e


def bet_status(b, live):
    if b.get("status") != "placed":
        return "skipped"
    if b.get("manual_result"):
        return b["manual_result"]
    e = next((x for x in live if x.get("fpl_id") == b.get("fixture_id")), None)
    res = e and e.get("result")
    key = b.get("key")
    if key in ("result", "dc", "score"):
        if not res:
            return "open"
        o = res["outcome"]
        if key == "score":
            return "won" if res["score"] == b.get("score") else "lost"
        if key == "result":
            want = {f"{b['home']} win": 0, "Draw": 1, f"{b['away']} win": 2}.get(b.get("pick"))
            return "won" if want == o else "lost"
        ok = o <= 1 if b.get("pick") == f"{b['home']} or draw" else o >= 1
        return "won" if ok else "lost"
    if key == "other":
        return "open"
    s = res and next((x for x in res.get("strong", []) if x.get("key") == key), None)
    st = (s.get("status") or ("won" if s.get("hit") else "lost" if s.get("hit") is False else "pending")) if s else "open"
    if st == "void" and b.get("tie_rule") == "lose":
        st = "lost"
    return st


def room_left(docs, live):
    """(week left, loss left) in kr, the same way the page counts them."""
    e = effective_limits(docs)
    now = datetime.now(LOCAL)
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    week = month_net = open_stakes = 0.0
    for path, b in docs.items():
        if not path.startswith("limits/bets/") or b.get("status") != "placed":
            continue
        st = bet_status(b, live)
        if parse(b.get("answered_at")) and parse(b["answered_at"]) >= week_start:
            week += b.get("stake", 0)
        if st in ("open", "pending"):
            open_stakes += b.get("stake", 0)
            continue
        profit = b["stake"] * (b["odds"] - 1) if st == "won" and b.get("odds") else (-b["stake"] if st == "lost" else 0)
        if parse(b.get("kickoff")) and parse(b["kickoff"]) >= month_start:
            month_net += profit
    return e["weekly_budget"] - week, e["monthly_loss_limit"] - (max(0, -month_net) + open_stakes), e


def cmd_matchday():
    with open(DATA, encoding="utf-8") as fh:
        data = json.load(fh)
    today = datetime.now(LOCAL).date()
    now = datetime.now(timezone.utc)
    todays = [f for f in data["fixtures"] if parse(f["kickoff"]).astimezone(LOCAL).date() == today and parse(f["kickoff"]) > now]
    if not todays:
        print("No Premier League match today.")
        return
    docs = feed()
    week_left, loss_left, e = room_left(docs, data["track"].get("live") or [])
    if e.get("break_until") and parse(e["break_until"]) > now:
        print("On a break: no notification.")
        return
    if week_left < 20 or loss_left < 20:
        print("A limit is reached: no notification.")
        return
    picks = []
    for f in todays:
        for s in f.get("strong") or []:
            if s["key"] == "sot" or f"limits/bets/{f['fpl_id']}_{s['key']}" in docs:
                continue
            fair = s["fair_any"] if s["key"] == "corners" else s["fair"]
            need = math.ceil(fair * MARGIN * 100 - 1e-9) / 100
            label = f"{s['pick']} more corners" if s["key"] == "corners" else s["pick"]
            when = parse(f["kickoff"]).astimezone(LOCAL).strftime("%H:%M")
            picks.append(f"{label} {when}, worth it at {need:.2f}+")
    if not picks:
        print("No strong picks left to notify today.")
        return
    msg = ". ".join(picks[:3]) + (f". +{len(picks) - 3} more" if len(picks) > 3 else "") + ". Check Oddset, then yes/no on the bet slip."
    notify.send(msg[:240])
    print("Sent:", msg)


if __name__ == "__main__":
    {"check": cmd_check, "done": cmd_done, "failed": cmd_failed, "matchday": cmd_matchday}[sys.argv[1]]()
