#!/usr/bin/env python3
"""nudge-watch: Bridge Hermes Periodic Nudge → msg-bridge → Rovv aggregation."""
import hashlib, json, os, time, urllib.request

MEMORY_FILE = os.path.expanduser("~/.hermes/memories/MEMORY.md")
BRIDGE_URL = "http://localhost:8083/msg"
STATE_FILE = os.path.expanduser("~/.hermes/.nudge-watch-state")

def main():
    try:
        md5 = hashlib.md5(open(MEMORY_FILE, "rb").read()).hexdigest()
        prev = open(STATE_FILE).read().strip() if os.path.exists(STATE_FILE) else ""
        if md5 == prev:
            return  # no changes

        lines = open(MEMORY_FILE).readlines()
        recent = "".join(lines[-8:]).strip()[-1000:]

        payload = json.dumps({
            "from": "hermes-nudge",
            "to": "rovv",
            "text": recent,
        }).encode("utf-8")

        req = urllib.request.Request(
            BRIDGE_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        urllib.request.urlopen(req, timeout=5)

        with open(STATE_FILE, "w") as f:
            f.write(md5)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"[nudge-watch {ts}] pushed {len(recent)} chars")
    except Exception as e:
        print(f"[nudge-watch] error: {e}")

if __name__ == "__main__":
    main()
