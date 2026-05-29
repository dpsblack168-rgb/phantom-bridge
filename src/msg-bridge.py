#!/usr/bin/env python3
"""Rovv-Rovi 同机消息桥 — HTTP API + 文件邮箱"""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

MSG_DIR = os.path.expanduser("~/messages")
TO_ROVI = os.path.join(MSG_DIR, "to-rovi.jsonl")
TO_ROVV = os.path.join(MSG_DIR, "to-rovv.jsonl")
PORT = 8083

os.makedirs(MSG_DIR, exist_ok=True)

class MsgHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except Exception:
            return self._json({"error": "bad json"}, 400)

        sender = data.get("from", "unknown")
        text = data.get("text", "")
        target = data.get("to", "rovi")  # rovi or rovv

        if not text:
            return self._json({"error": "text required"}, 400)

        entry = {
            "from": sender,
            "text": text,
            "ts": datetime.now().isoformat()
        }

        if target == "rovi":
            out = TO_ROVI
        elif target == "rovv":
            out = TO_ROVV
        else:
            return self._json({"error": "to must be rovi or rovv"}, 400)

        with open(out, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Also print to terminal for debugging
        print(f"\n📩 [{entry['ts']}] {entry['from']} → {target}: {entry['text']}", flush=True)

        self._json({"ok": True, "entry": entry})

    def do_GET(self):
        if self.path == "/health":
            return self._json({"ok": True, "service": "rovv-rovi-bridge"})

        if self.path == "/msgs/rovi":
            path = TO_ROVI
        elif self.path == "/msgs/rovv":
            path = TO_ROVV
        else:
            return self._json({"error": "use /msgs/rovi or /msgs/rovv"}, 404)

        msgs = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        msgs.append(json.loads(line))
        self._json({"count": len(msgs), "messages": msgs[-20:]})  # last 20

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), MsgHandler)
    print(f"🏯🏯 Rovv-Rovi 消息桥已启动 → http://0.0.0.0:{PORT}")
    print(f"   POST /msg  {{from, to, text}}    发消息")
    print(f"   GET  /msgs/rovi  /msgs/rovv       看消息")
    print(f"   文件邮箱: ~/messages/to-rovi.jsonl  ~/messages/to-rovv.jsonl")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 消息桥关闭")
