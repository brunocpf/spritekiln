"""Tiny graph builder that emits ComfyUI workflows in both UI (litegraph) and API format.

Widget order, slot types and defaults come from a live `object_info` snapshot so the
UI files load cleanly in the ComfyUI frontend.
"""
import json

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
FRONTEND_ONLY = {"MarkdownNote", "Note"}
PREVIEW_NODES = {"PreviewImage", "SaveImage", "SK_SavePixelArt", "LoadImage", "SaveAnimatedWEBP", "PreviewAny"}


class Out:
    def __init__(self, node, idx):
        self.node, self.idx = node, idx


class Node:
    def __init__(self, graph, nid, cls, group, title, inputs, mode):
        self.graph, self.id, self.cls, self.group, self.title = graph, nid, cls, group, title
        self.inputs, self.mode = inputs, mode

    def __getitem__(self, idx):
        return Out(self, idx)


def _spec_kind(spec):
    """-> (is_widget, type_name, options)"""
    t = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(t, list):
        return True, "COMBO", opts
    if t in WIDGET_TYPES and not opts.get("forceInput"):
        return True, t, opts
    return False, t, opts


class Graph:
    def __init__(self, object_info, name, title, description=""):
        self.oi, self.name, self.title, self.description = object_info, name, title, description
        self.nodes, self.groups = [], []

    # ------------------------------------------------------------------ building
    def add(self, cls, group, title=None, node_mode=0, **inputs):
        """node_mode: litegraph state (0 normal, 4 bypass). Everything else in **inputs is a node input
        (including inputs literally named `mode`)."""
        if cls not in self.oi and cls not in FRONTEND_ONLY:
            raise KeyError("unknown node class %s" % cls)
        if group not in self.groups:
            self.groups.append(group)
        n = Node(self, len(self.nodes) + 1, cls, group, title, inputs, node_mode)
        if cls not in FRONTEND_ONLY:
            spec = self._inputs(cls)
            for k in inputs:
                if k not in spec:
                    raise KeyError("%s has no input %r (has %s)" % (cls, k, list(spec)))
        self.nodes.append(n)
        return n

    def note(self, group, text):
        return self.add("MarkdownNote", group, title="Note", text=text)

    def _inputs(self, cls):
        info = self.oi[cls]["input"]
        out = {}
        for sec in ("required", "optional"):
            for k, v in (info.get(sec) or {}).items():
                out[k] = (sec, v)
        return out

    def _default(self, cls, name, spec):
        is_w, t, opts = _spec_kind(spec)
        if "default" in opts:
            return opts["default"]
        if t == "COMBO":
            options = spec[0] if isinstance(spec[0], list) else opts.get("options", [])
            return options[0] if options else ""
        return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(t)

    # ------------------------------------------------------------------ bypass handling for API export
    def _resolve(self, value):
        """Follow bypassed nodes back to a real output (same type pass-through)."""
        while isinstance(value, Out) and value.node.mode == 4:
            n = value.node
            out_type = self.oi[n.cls]["output"][value.idx]
            nxt = None
            for k, v in n.inputs.items():
                if isinstance(v, Out) and self.oi[v.node.cls]["output"][v.idx] == out_type:
                    nxt = v
                    break
            if nxt is None:
                return None
            value = nxt
        return value

    # ------------------------------------------------------------------ API format
    def to_api(self):
        api = {}
        for n in self.nodes:
            if n.cls in FRONTEND_ONLY or n.mode in (2, 4):
                continue
            ins = {}
            for name, (sec, spec) in self._inputs(n.cls).items():
                if name in n.inputs:
                    v = self._resolve(n.inputs[name])
                    if v is None:
                        continue
                    ins[name] = [str(v.node.id), v.idx] if isinstance(v, Out) else v
                else:
                    is_w, _, _ = _spec_kind(spec)
                    if is_w and sec == "required":
                        ins[name] = self._default(n.cls, name, spec)
            api[str(n.id)] = {"class_type": n.cls, "inputs": ins, "_meta": {"title": n.title or n.cls}}
        return api

    # ------------------------------------------------------------------ UI format
    def _size(self, n):
        if n.cls in FRONTEND_ONLY:
            lines = n.inputs["text"].count("\n") + len(n.inputs["text"]) // 60 + 2
            return [420, 40 + 20 * lines]
        slots = widgets = 0
        extra = 0
        for name, (sec, spec) in self._inputs(n.cls).items():
            is_w, t, opts = _spec_kind(spec)
            if is_w:
                widgets += 1
                if t == "STRING" and opts.get("multiline"):
                    extra += 90
            else:
                slots += 1
        h = 40 + 22 * max(slots, len(self.oi[n.cls]["output"])) + 26 * widgets + extra
        if n.cls in PREVIEW_NODES:
            h += 300
        w = 360 if extra else 320
        return [w, h]

    def _layout(self):
        """Groups left->right (wrapping), nodes inside a group in dependency columns."""
        pos, sizes = {}, {n.id: self._size(n) for n in self.nodes}
        group_boxes = []
        gx, gy, row_h, max_row_w = 0, 0, 0, 3400
        for g in self.groups:
            members = [n for n in self.nodes if n.group == g]
            ids = {n.id for n in members}
            depth = {}
            for n in members:  # nodes are added in dependency order
                d = 0
                for v in n.inputs.values():
                    if isinstance(v, Out) and v.node.id in ids:
                        d = max(d, depth[v.node.id] + 1)
                depth[n.id] = d if n.cls not in FRONTEND_ONLY else 0
            cols = {}
            for n in members:
                cols.setdefault(depth[n.id], []).append(n)
            col_x, block_h = 0, 0
            col_pos = {}
            for c in sorted(cols):
                width = max(sizes[n.id][0] for n in cols[c])
                y = 0
                for n in cols[c]:
                    col_pos[n.id] = (col_x, y)
                    y += sizes[n.id][1] + 40
                block_h = max(block_h, y)
                col_x += width + 50
            block_w = col_x - 50
            if gx > 0 and gx + block_w > max_row_w:
                gx, gy, row_h = 0, gy + row_h + 120, 0
            for nid, (x, y) in col_pos.items():
                pos[nid] = [gx + x + 20, gy + y + 70]
            group_boxes.append((g, [gx, gy, block_w + 40, block_h + 60]))
            gx += block_w + 120
            row_h = max(row_h, block_h + 60)
        return pos, sizes, group_boxes

    def to_ui(self):
        pos, sizes, group_boxes = self._layout()
        links, nodes_ui = [], []
        out_links = {}  # (node id, slot) -> [link ids]
        link_id = 0
        pending = []
        for n in self.nodes:
            if n.cls in FRONTEND_ONLY:
                continue
            for name, v in n.inputs.items():
                if isinstance(v, Out):
                    link_id += 1
                    pending.append((link_id, v, n, name))
                    out_links.setdefault((v.node.id, v.idx), []).append(link_id)
        link_for_input = {(n.id, name): (lid, v) for lid, v, n, name in pending}

        for order, n in enumerate(self.nodes):
            base = {"id": n.id, "type": n.cls, "pos": pos[n.id], "size": sizes[n.id], "flags": {},
                    "order": order, "mode": n.mode, "properties": {"Node name for S&R": n.cls}}
            if n.title:
                base["title"] = n.title
            if n.cls in FRONTEND_ONLY:
                base.update({"inputs": [], "outputs": [], "widgets_values": [n.inputs["text"]],
                             "color": "#432", "bgcolor": "#653"})
                nodes_ui.append(base)
                continue
            ins, widgets = [], []
            in_slot = 0
            for name, (sec, spec) in self._inputs(n.cls).items():
                is_w, t, opts = _spec_kind(spec)
                lk = link_for_input.get((n.id, name))
                entry = {"name": name, "type": t, "link": lk[0] if lk else None}
                if is_w:
                    entry["widget"] = {"name": name}
                    v = n.inputs.get(name)
                    val = self._default(n.cls, name, spec) if (v is None or isinstance(v, Out)) else v
                    widgets.append(val)
                    ctrl = opts.get("control_after_generate")
                    if ctrl:
                        widgets.append(ctrl if isinstance(ctrl, str) else
                                       ("randomize" if "seed" in name else "fixed"))
                    if opts.get("image_upload"):
                        widgets.append("image")
                    if sec == "optional" and not lk and name not in n.inputs:
                        pass
                ins.append(entry)
                if lk:
                    links.append([lk[0], lk[1].node.id, lk[1].idx, n.id, in_slot, t if not is_w else t])
                in_slot += 1
            outs = []
            info = self.oi[n.cls]
            for i, (t, nm) in enumerate(zip(info["output"], info.get("output_name") or info["output"])):
                o = {"name": nm, "type": t if isinstance(t, str) else "COMBO",
                     "links": out_links.get((n.id, i), []), "slot_index": i}
                if (info.get("output_is_list") or [False] * 99)[i]:
                    o["shape"] = 6
                outs.append(o)
            base.update({"inputs": ins, "outputs": outs, "widgets_values": widgets})
            nodes_ui.append(base)

        # link types must be the producer's output type
        for l in links:
            prod = next(x for x in self.nodes if x.id == l[1])
            l[5] = self.oi[prod.cls]["output"][l[2]]
        colors = ["#3f789e", "#8A8", "#b06634", "#a1309b", "#88A", "#b58b2a", "#444", "#335"]
        groups = [{"id": i + 1, "title": g, "bounding": box, "color": colors[i % len(colors)],
                   "font_size": 24, "flags": {}} for i, (g, box) in enumerate(group_boxes)]
        return {"id": "spritekiln-" + self.name, "revision": 0, "last_node_id": len(self.nodes),
                "last_link_id": link_id, "nodes": nodes_ui, "links": links, "groups": groups,
                "config": {}, "extra": {"ds": {"scale": 0.55, "offset": [40, 40]},
                                        "spritekiln": {"title": self.title, "description": self.description}},
                "version": 0.4}

    def save(self, ui_path, api_path):
        with open(ui_path, "w") as f:
            json.dump(self.to_ui(), f, indent=1, ensure_ascii=False)
        with open(api_path, "w") as f:
            json.dump(self.to_api(), f, indent=1, ensure_ascii=False)
