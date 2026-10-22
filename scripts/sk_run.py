#!/usr/bin/env python3
"""Run a SpriteKiln API workflow against a local ComfyUI, with input overrides — a scriptable
"Retro Diffusion API" for your own machine. Standard library only.

Examples
  python3 sk_run.py 01_text_to_pixel_art --set "SK_StylePreset.subject=a goblin archer" \
      --set "SK_StylePreset.style=character (side view)"
  python3 sk_run.py 10_animate_sprite --image sprite=./hero.png --set "SK_AnimationPreset.action=run"
  python3 sk_run.py 06_rotate_8_directions --image "Front-facing sprite=./hero.png" --seed 7

  --set NODE.INPUT=VALUE   NODE is a node id or its title (as shown in the graph); VALUE is parsed as JSON
                            when possible (numbers, true/false), else used as a string.
  --image NODE=PATH        upload PATH and point that LoadImage node at it ("sprite" = first LoadImage).
  --seed N                 set every seed / noise_seed in the graph.
  --list                   print the workflow's nodes and editable inputs and exit.
  --out DIR                copy the result files there (default ./spritekiln_out)
  --server URL / --timeout seconds
"""
import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(os.path.dirname(HERE), "workflows", "api")


def http(server, path, data=None, headers=None):
    req = urllib.request.Request(server + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req) as r:
        return r.read()


def upload(server, path):
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    body = (("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n"
             % (boundary, name, mimetypes.guess_type(name)[0] or "image/png")).encode()
            + open(path, "rb").read()
            + ("\r\n--%s\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--%s--\r\n"
               % (boundary, boundary)).encode())
    res = json.loads(http(server, "/upload/image", body, {"Content-Type": "multipart/form-data; boundary=" + boundary}))
    return (res.get("subfolder") + "/" if res.get("subfolder") else "") + res["name"]


def find_node(api, key, cls=None):
    if key in api:
        return key
    hits = [k for k, v in api.items() if v.get("_meta", {}).get("title", "").lower() == key.lower()
            or v["class_type"].lower() == key.lower()]
    if cls:
        hits = [k for k in hits if api[k]["class_type"] == cls] or hits
    if not hits:
        sys.exit("no node %r (use --list)" % key)
    return hits[0]


def parse(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workflow", help="name (e.g. 01_text_to_pixel_art) or path to an .api.json")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="spritekiln_out", help="where to copy the outputs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()

    path = a.workflow if os.path.exists(a.workflow) else os.path.join(API_DIR, a.workflow + ".api.json")
    api = json.load(open(path))
    if a.list:
        for k, v in api.items():
            vals = {n: x for n, x in v["inputs"].items() if not isinstance(x, list)}
            print("%3s %-28s %-32s %s" % (k, v["class_type"], v["_meta"]["title"], json.dumps(vals)[:150]))
        return
    for item in a.image:
        key, p = item.split("=", 1)
        nid = find_node(api, "LoadImage" if key == "sprite" else key, "LoadImage")
        api[nid]["inputs"]["image"] = upload(a.server, p)
    for item in a.set:
        lhs, v = item.split("=", 1)
        key, inp = lhs.rsplit(".", 1)
        nid = find_node(api, key)
        cur = api[nid]["inputs"].get(inp)
        # keep strings as strings (combo options like "4" must not become numbers)
        api[nid]["inputs"][inp] = v if isinstance(cur, str) else parse(v)
    if a.seed is not None:
        for v in api.values():
            for n in ("seed", "noise_seed"):
                if n in v["inputs"] and not isinstance(v["inputs"][n], list):
                    v["inputs"][n] = a.seed

    cid = uuid.uuid4().hex
    try:
        res = json.loads(http(a.server, "/prompt", json.dumps({"prompt": api, "client_id": cid}).encode(),
                              {"Content-Type": "application/json"}))
    except urllib.error.HTTPError as e:
        sys.exit(e.read().decode())
    pid = res["prompt_id"]
    print("queued", pid)
    t0 = time.time()
    while True:
        hist = json.loads(http(a.server, "/history/" + pid)).get(pid)
        if hist and hist.get("status", {}).get("completed") or (hist and hist["status"].get("status_str") == "error"):
            break
        if time.time() - t0 > a.timeout:
            sys.exit("timeout")
        time.sleep(2)
    st = hist["status"]
    if st.get("status_str") == "error":
        for m in st.get("messages", []):
            if m[0] == "execution_error":
                sys.exit("error in node %s (%s): %s" % (m[1].get("node_id"), m[1].get("node_type"),
                                                         m[1].get("exception_message")))
        sys.exit("error")
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for node_out in hist["outputs"].values():
        for img in node_out.get("images", []):
            if img.get("type") != "output":
                continue
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": "output"})
            open(os.path.join(a.out, img["filename"]), "wb").write(http(a.server, "/view?" + q))
            n += 1
    print("done in %.1fs, %d files -> %s" % (time.time() - t0, n, a.out))


if __name__ == "__main__":
    main()
