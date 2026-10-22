#!/usr/bin/env python3
"""SpriteKiln web UI server — standard library only.

  python3 webui/serve.py [--port 8189] [--host 127.0.0.1] [--comfy http://127.0.0.1:8188]

Serves the static app in this folder, reverse-proxies the ComfyUI API (so the browser stays
same-origin) and exposes the SpriteKiln workflows, read from disk on every request:

  GET /sk/workflows          index.json entries + any extra api/*.api.json, with an `id` per workflow
  GET /sk/workflow/<id>      the API-format graph workflows/api/<id>.api.json
  GET /sk/config             the ComfyUI address this server talks to
(/pf/... still works as an alias from the PixelForge days.)

Proxied to ComfyUI: /prompt /queue /history /interrupt /upload/image /view /object_info
/system_stats, and the /ws websocket (live progress and previews).
"""
import argparse
import json
import os
import re
import select
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY_PREFIXES = ("/prompt", "/queue", "/history", "/interrupt", "/upload/image", "/view", "/object_info",
                  "/system_stats")
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "host", "origin", "referer", "content-length"}
UI_PREFIXES = ("SpriteKiln_", "PixelForge_")
ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

COMFY = "http://127.0.0.1:8188"
WORKFLOWS = os.path.join(os.path.dirname(HERE), "workflows")


def api_id(ui_file):
    """'SpriteKiln_01_text_to_pixel_art.json' (or the older 'PixelForge_…') -> '01_text_to_pixel_art'"""
    name = os.path.basename(ui_file)
    name = name[:-5] if name.endswith(".json") else name
    for prefix in UI_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def list_workflows():
    api_dir = os.path.join(WORKFLOWS, "api")
    on_disk = sorted(f[:-len(".api.json")] for f in os.listdir(api_dir) if f.endswith(".api.json"))
    try:
        with open(os.path.join(WORKFLOWS, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        index = []
    out, seen = [], set()
    for entry in index:
        wid = api_id(entry.get("file", ""))
        if wid in on_disk and wid not in seen:
            seen.add(wid)
            out.append(dict(entry, id=wid))
    for wid in on_disk:
        if wid not in seen:
            out.append({"id": wid, "title": wid.replace("_", " "), "description": "", "file": None})
    return out


class Handler(SimpleHTTPRequestHandler):
    server_version = "SpriteKilnUI/1.0"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)

    def end_headers(self):
        if not getattr(self, "_proxying", False):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ---- routing -------------------------------------------------------------------------------
    def _is_proxy(self):
        path = urllib.parse.urlsplit(self.path).path
        return any(path == p or path.startswith(p + "/") for p in PROXY_PREFIXES)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            return self.proxy_websocket()
        if path.startswith(("/sk/", "/pf/")):
            return self.sk_api("/sk/" + path[4:])
        if self._is_proxy():
            return self.proxy()
        return super().do_GET()

    def do_HEAD(self):
        if self._is_proxy():
            return self.proxy()
        return super().do_HEAD()

    def do_POST(self):
        if self._is_proxy():
            return self.proxy()
        self.send_error(404)

    # ---- SpriteKiln workflow API ---------------------------------------------------------------
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def sk_api(self, path):
        try:
            if path == "/sk/workflows":
                return self.send_json(list_workflows())
            if path == "/sk/config":
                return self.send_json({"comfy": COMFY, "workflows": WORKFLOWS})
            if path.startswith("/sk/workflow/"):
                wid = urllib.parse.unquote(path[len("/sk/workflow/"):])
                fp = os.path.join(WORKFLOWS, "api", wid + ".api.json")
                if not ID_RE.match(wid) or not os.path.isfile(fp):
                    return self.send_json({"error": "no workflow %r" % wid}, 404)
                with open(fp, encoding="utf-8") as f:
                    return self.send_json(json.load(f))
        except Exception as e:  # noqa: BLE001 - report anything to the browser
            return self.send_json({"error": str(e)}, 500)
        self.send_json({"error": "not found"}, 404)

    # ---- HTTP reverse proxy --------------------------------------------------------------------
    def proxy(self):
        self._proxying = True
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        req = urllib.request.Request(COMFY + self.path, data=body, headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            resp = e
        except (urllib.error.URLError, OSError) as e:
            self._proxying = False
            return self.send_json({"error": {"type": "comfy_unreachable", "message":
                                             "ComfyUI is not reachable at %s (%s)" % (COMFY, e)}}, 502)
        with resp:
            data = resp.read()
            self.send_response(resp.status if hasattr(resp, "status") else resp.code)
            for k, v in resp.headers.items():
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

    # ---- websocket tunnel (raw bytes after the handshake) ---------------------------------------
    def proxy_websocket(self):
        target = urllib.parse.urlsplit(COMFY)
        try:
            up = socket.create_connection((target.hostname, target.port or 80), timeout=10)
        except OSError:
            return self.send_error(502, "ComfyUI websocket not reachable")
        up.settimeout(None)
        lines = ["GET %s HTTP/1.1" % self.path, "Host: %s" % target.netloc]
        lines += ["%s: %s" % (k, v) for k, v in self.headers.items() if k.lower() not in ("host", "origin")]
        up.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        client = self.connection
        self.close_connection = True
        socks = [client, up]
        try:
            while True:
                ready, _, err = select.select(socks, [], socks, 60)
                if err:
                    break
                for s in ready:
                    data = s.recv(65536)
                    if not data:
                        return
                    (up if s is client else client).sendall(data)
        except OSError:
            pass
        finally:
            up.close()


def main():
    global COMFY, WORKFLOWS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other devices")
    ap.add_argument("--comfy", default=COMFY, help="ComfyUI address (default %(default)s)")
    ap.add_argument("--workflows", default=WORKFLOWS, help="folder holding index.json and api/")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every request")
    a = ap.parse_args()
    COMFY = a.comfy.rstrip("/")
    WORKFLOWS = os.path.abspath(a.workflows)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    httpd.daemon_threads = True
    httpd.verbose = a.verbose
    print("SpriteKiln UI on http://%s:%d  (ComfyUI: %s, workflows: %s)" % (
        "127.0.0.1" if a.host == "0.0.0.0" else a.host, a.port, COMFY, WORKFLOWS))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
