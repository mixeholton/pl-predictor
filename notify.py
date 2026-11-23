"""Send a push notification to your phone through ntfy (https://ntfy.sh).

ntfy is a free, open-source push service: install the ntfy app (iPhone or Android),
subscribe to the topic in notify_config.json, and every message posted to that topic
pops up on the phone. No account or key is needed; the topic name is the only "secret",
so keep it private (anyone who knows it can read or send to it).

Usage:
  python notify.py --test                 send a test notification
  python notify.py "message text"         send a message (opens the bet slip when tapped)
"""
import json
import os
import secrets
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(HERE, "notify_config.json")
SITE_CONF = os.path.join(HERE, "site_config.json")


def page_url():
    if os.path.exists(SITE_CONF):
        with open(SITE_CONF, encoding="utf-8") as fh:
            return json.load(fh).get("page_url", "")
    return ""


PAGE = os.environ.get("PAGE_URL") or page_url()


def load_conf():
    if os.environ.get("NTFY_TOPIC"):   # in the cloud job the topic comes from a hidden secret
        return {"server": os.environ.get("NTFY_SERVER", "https://ntfy.sh"), "topic": os.environ["NTFY_TOPIC"]}
    if not os.path.exists(CONF):   # first run: make a long random topic name
        conf = {"server": "https://ntfy.sh", "topic": "plp-" + secrets.token_urlsafe(18)}
        with open(CONF, "w", encoding="utf-8") as fh:
            json.dump(conf, fh, indent=1)
    with open(CONF, encoding="utf-8") as fh:
        return json.load(fh)


def send(message, title="PL Predictor", tags="soccer", priority="high"):   # high = pop-up banner + sound on the phone
    conf = load_conf()
    url = f"{conf['server'].rstrip('/')}/{conf['topic']}"
    headers = {"Title": title, "Tags": tags, "Priority": priority, "Click": PAGE,
               "Actions": f"view, Open bet slip, {PAGE}"}
    try:
        req = urllib.request.Request(url, data=message.encode("utf-8"), method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.URLError as e:
        if "CERTIFICATE" not in str(e).upper():
            raise
    # Python's certificate store on some Windows PCs is out of date; Windows' own curl
    # uses the system store instead. The message goes in on stdin, never through a shell.
    cmd = ["curl", "-sS", "-o", os.devnull, "-w", "%{http_code}", "--data-binary", "@-"]
    for k, v in headers.items():
        cmd += ["-H", f"{k}: {v}"]
    out = subprocess.run(cmd + [url], input=message.encode("utf-8"), capture_output=True, timeout=60)
    code = out.stdout.decode().strip()
    if out.returncode != 0 or code != "200":
        raise RuntimeError(f"curl failed ({code}): {out.stderr.decode().strip()}")
    return int(code)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    msg = ("Test from PL Predictor: notifications work. Tap to open the bet slip."
           if sys.argv[1] == "--test" else " ".join(sys.argv[1:]))
    try:
        status = send(msg)
    except Exception as e:   # report, don't crash the scheduled run
        print(f"notification failed: {e}")
        sys.exit(2)
    print(f"sent ({status})")
