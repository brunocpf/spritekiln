"""Spritesheets, animation frames and export (PNG / GIF / APNG / Aseprite-style JSON)."""
import json
import os

import numpy as np
import torch
from PIL import Image

import folder_paths

from . import sk_core as core

CAT = "SpriteKiln/sprite"


def _flatten_images(images):
    """list of IMAGE batches (or a single batch) -> list of (rgb, alpha)."""
    if isinstance(images, torch.Tensor):
        images = [images]
    frames = []
    for batch in images:
        frames.extend(core.split_tensor(batch))
    return frames


class SK_ListToBatch:
    """Image list (e.g. one image per direction) -> one batch, padded to the largest frame."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, images):
        return (core.join_tensor(_flatten_images(images)),)


class SK_SpriteSheet:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "columns": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "0 = all frames in one row."}),
            "padding": ("INT", {"default": 0, "min": 0, "max": 64}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sheet",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, images, columns, padding):
        cols_n, pad = columns[0], padding[0]
        frames = _flatten_images(images)
        return (core.join_tensor([make_sheet(frames, cols_n, pad)]),)


def make_sheet(frames, columns, padding):
    n = len(frames)
    cols = n if columns <= 0 else min(columns, n)
    rows = -(-n // cols)
    fh = max(f[0].shape[0] for f in frames)
    fw = max(f[0].shape[1] for f in frames)
    H, W = rows * fh + (rows - 1) * padding, cols * fw + (cols - 1) * padding
    rgb = np.zeros((H, W, 3), np.uint8)
    alpha = np.zeros((H, W), np.float32)
    for i, (f, a) in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = r * (fh + padding), c * (fw + padding)
        rgb[y:y + f.shape[0], x:x + f.shape[1]] = f
        alpha[y:y + f.shape[0], x:x + f.shape[1]] = 1.0 if a is None else a
    return rgb, alpha


class SK_SliceSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sheet": ("IMAGE",),
            "rows": ("INT", {"default": 4, "min": 1, "max": 64}),
            "columns": ("INT", {"default": 4, "min": 1, "max": 64}),
            "inset": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Pixels trimmed from every cell edge."}),
            "order": (["row_major", "column_major"],),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, sheet, rows, columns, inset, order):
        out = []
        for rgb, alpha in core.split_tensor(sheet):
            H, W = rgb.shape[:2]
            cells = [(r, c) for r in range(rows) for c in range(columns)]
            if order == "column_major":
                cells = [(r, c) for c in range(columns) for r in range(rows)]
            for r, c in cells:
                y0, y1 = round(r * H / rows) + inset, round((r + 1) * H / rows) - inset
                x0, x1 = round(c * W / columns) + inset, round((c + 1) * W / columns) - inset
                out.append((rgb[y0:y1, x0:x1], None if alpha is None else alpha[y0:y1, x0:x1]))
        return (core.join_tensor(out),)


class SK_FrameSelect:
    """Pick N animation frames from a video: evenly, or find the best loop point first."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "count": ("INT", {"default": 8, "min": 1, "max": 128}),
            "mode": (["auto", "loop_best", "even", "even_inclusive", "ping_pong"],
                     {"tooltip": "loop_best: end where the clip looks most like its first frame. even: spread over the "
                                 "clip without its last frame (loops). even_inclusive: first and last frame included "
                                 "(one-shot actions, keyframes). auto: loop_best if `loop` is true, else even_inclusive."}),
            "skip_start": ("INT", {"default": 0, "min": 0, "max": 1000,
                                   "tooltip": "Drop frames at the start (video models often settle in)."}),
            "min_loop_fraction": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05}),
        }, "optional": {"loop": ("BOOLEAN", {"default": True, "forceInput": True})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "info")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, frames, count, mode, skip_start, min_loop_fraction, loop=True):
        if mode == "auto":
            mode = "loop_best" if loop else "even_inclusive"
        f = frames[min(skip_start, frames.shape[0] - 1):]
        n = f.shape[0]
        info = "%d source frames" % n
        end = n
        if mode == "loop_best" and n > 4:
            small = torch.nn.functional.interpolate(f[..., :3].permute(0, 3, 1, 2), size=(48, 48), mode="area")
            ref = small[0:1]
            d = ((small - ref) ** 2).mean((1, 2, 3))
            lo = max(2, int(n * min_loop_fraction))
            cand = torch.arange(lo, n)
            best = int(cand[d[lo:].argmin()]) if len(cand) else n - 1
            end = best  # frame `best` ~= frame 0, so the loop is [0, best)
            info += ", loop closes at frame %d (diff %.4f)" % (best, float(d[best]))
        if mode == "ping_pong":
            half = max(1, (count + 2) // 2)
            idx = np.linspace(0, n - 1, half).round().astype(int).tolist()
            idx = idx + idx[-2:0:-1]
            idx = idx[:count]
        elif mode == "even_inclusive":
            idx = np.linspace(0, n - 1, count).round().astype(int).tolist()
        else:
            idx = np.floor(np.arange(count) * end / count).astype(int).tolist()
        info += ", picked %s" % idx
        return f[idx], info


class SK_Stabilize:
    """Align animation frames on the sprite (removes drift from video models). Needs alpha."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "anchor": (["bottom_center", "center", "center_x", "none"],
                       {"tooltip": "bottom_center aligns the feet (best for characters); center for flying things."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, frames, anchor):
        fr = core.split_tensor(frames)
        if anchor == "none" or fr[0][1] is None:
            return (frames,)

        def point(a):
            ys, xs = np.where(a >= 0.5)
            if len(xs) == 0:
                return None
            if anchor == "bottom_center":
                # anchor on the feet: horizontal centre of the lowest ~12% of the silhouette, so a sword
                # or arm swinging out does not drag the body sideways
                bottom = ys.max()
                band = max(2, int(round((bottom - ys.min() + 1) * 0.12)))
                feet = ys > bottom - band
                return xs[feet].mean(), bottom
            return xs.mean(), ys.mean()

        ref = point(fr[0][1])
        out = []
        for rgb, a in fr:
            p = point(a)
            if p is None or ref is None:
                out.append((rgb, a))
                continue
            dx = int(round(ref[0] - p[0]))
            dy = 0 if anchor == "center_x" else int(round(ref[1] - p[1]))
            rgb2 = np.zeros_like(rgb)
            a2 = np.zeros_like(a)
            h, w = a.shape
            ys0, ys1 = max(0, dy), min(h, h + dy)
            xs0, xs1 = max(0, dx), min(w, w + dx)
            rgb2[ys0:ys1, xs0:xs1] = rgb[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
            a2[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
            out.append((rgb2, a2))
        return (core.join_tensor(out, force_alpha=True),)


def _gif_frames(frames):
    """RGBA frames -> P-mode PIL images sharing one palette, index 255 = transparent."""
    cols, counts = core.unique_colors([f[0] for f in frames], [f[1] for f in frames])
    pal = cols if len(cols) <= 255 else core.kmeans_palette(cols, counts, 255)
    lab = core.srgb_to_oklab(pal)
    flat_pal = np.zeros((256, 3), np.uint8)
    flat_pal[:len(pal)] = pal
    out = []
    for rgb, a in frames:
        idx = core.nearest_index(core.srgb_to_oklab(rgb), lab).astype(np.uint8)
        if a is not None:
            idx[a < 0.5] = 255
        im = Image.fromarray(idx, "P")
        im.putpalette(flat_pal.reshape(-1).tolist())
        out.append(im)
    return out


class SK_SavePixelArt:
    """Save sprites at native resolution (+ upscaled copies). Batches become an animation:
    individual frames, a spritesheet, GIF, APNG and an Aseprite-compatible JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "spritekiln/sprite"}),
            "upscale": ("INT", {"default": 8, "min": 1, "max": 32, "tooltip": "Extra nearest-neighbour upscaled copy."}),
            "as_animation": ("BOOLEAN", {"default": True, "tooltip": "Treat a batch as frames of one animation."}),
            "fps": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 60.0, "step": 0.5}),
            "sheet_columns": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "0 = one row."}),
        }, "optional": {"row_names": ("STRING", {"forceInput": True,
                                                 "tooltip": "Space-separated tag names per sheet row (e.g. directions)."})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, images, filename_prefix, upscale, as_animation, fps, sheet_columns, row_names=""):
        frames = core.split_tensor(images)
        h, w = frames[0][0].shape[:2]
        outdir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h)
        ui = []

        def save_png(rgb, a, name, scale=1):
            if scale > 1:
                rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
                a = None if a is None else np.repeat(np.repeat(a, scale, 0), scale, 1)
            arr = rgb if a is None else np.dstack([rgb, np.round(a * 255).astype(np.uint8)])
            Image.fromarray(arr, "RGB" if a is None else "RGBA").save(os.path.join(outdir, name), compress_level=4)
            return {"filename": name, "subfolder": subfolder, "type": "output"}

        base = "%s_%05d" % (fname, counter)
        if not as_animation or len(frames) == 1:
            for i, (rgb, a) in enumerate(frames):
                tag = base if len(frames) == 1 else "%s_%02d" % (base, i)
                save_png(rgb, a, tag + ".png")
                ui.append(save_png(rgb, a, tag + "_x%d.png" % upscale, upscale) if upscale > 1
                          else {"filename": tag + ".png", "subfolder": subfolder, "type": "output"})
            return {"ui": {"images": ui}}

        # animation
        dur = int(round(1000 / fps))
        for i, (rgb, a) in enumerate(frames):
            save_png(rgb, a, "%s_f%02d.png" % (base, i))
        sheet_rgb, sheet_a = make_sheet(frames, sheet_columns, 0)
        ui.append(save_png(sheet_rgb, sheet_a, base + "_sheet.png"))
        if upscale > 1:
            ui[-1] = save_png(sheet_rgb, sheet_a, base + "_sheet_x%d.png" % upscale, upscale)
        cols = len(frames) if sheet_columns <= 0 else min(sheet_columns, len(frames))
        meta = {"frames": {}, "meta": {
            "app": "SpriteKiln", "image": base + "_sheet.png", "format": "RGBA8888",
            "size": {"w": int(sheet_rgb.shape[1]), "h": int(sheet_rgb.shape[0])}, "scale": "1",
            "frameTags": [{"name": "anim", "from": 0, "to": len(frames) - 1, "direction": "forward"}]}}
        rows = -(-len(frames) // cols)
        if rows > 1:  # one Aseprite tag per sheet row (e.g. one row per direction)
            names = (row_names or "").split()
            meta["meta"]["frameTags"] = [{"name": names[r] if r < len(names) else "row%d" % r, "from": r * cols,
                                          "to": min(len(frames), (r + 1) * cols) - 1, "direction": "forward"}
                                         for r in range(rows)]
        for i in range(len(frames)):
            r, c = divmod(i, cols)
            meta["frames"]["%s %d.png" % (fname, i)] = {
                "frame": {"x": c * w, "y": r * h, "w": w, "h": h}, "rotated": False, "trimmed": False,
                "spriteSourceSize": {"x": 0, "y": 0, "w": w, "h": h}, "sourceSize": {"w": w, "h": h}, "duration": dur}
        with open(os.path.join(outdir, base + "_sheet.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if rows > 1:  # an upscaled GIF per row as well
            for r in range(rows):
                part = frames[r * cols:(r + 1) * cols]
                big = [(np.repeat(np.repeat(rgb, upscale, 0), upscale, 1),
                        None if a is None else np.repeat(np.repeat(a, upscale, 0), upscale, 1)) for rgb, a in part]
                gif = _gif_frames(big)
                tag = (row_names or "").split()[r] if r < len((row_names or "").split()) else "row%d" % r
                gname = "%s_%s_x%d.gif" % (base, tag, upscale)
                gif[0].save(os.path.join(outdir, gname), save_all=True, append_images=gif[1:], duration=dur,
                            loop=0, transparency=255, disposal=2, optimize=False)
                ui.append({"filename": gname, "subfolder": subfolder, "type": "output"})
        for scale, suffix in ((1, ""), (upscale, "_x%d" % upscale)) if upscale > 1 else ((1, ""),):
            big = [(np.repeat(np.repeat(rgb, scale, 0), scale, 1),
                    None if a is None else np.repeat(np.repeat(a, scale, 0), scale, 1)) for rgb, a in frames]
            gif = _gif_frames(big)
            gname = base + suffix + ".gif"
            gif[0].save(os.path.join(outdir, gname), save_all=True, append_images=gif[1:], duration=dur,
                        loop=0, transparency=255, disposal=2, optimize=False)
            if scale > 1 or upscale == 1:
                ui.append({"filename": gname, "subfolder": subfolder, "type": "output"})
            apng = [Image.fromarray(rgb if a is None else np.dstack([rgb, np.round(a * 255).astype(np.uint8)]))
                    for rgb, a in big]
            apng[0].save(os.path.join(outdir, base + suffix + ".apng.png"), save_all=True,
                         append_images=apng[1:], duration=dur, loop=0)
        return {"ui": {"images": ui}}


class SK_BatchToList:
    """Batch -> list, so downstream nodes (e.g. an edit model) run once per frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, images):
        return ([images[i:i + 1] for i in range(images.shape[0])],)


class SK_LayerExtract:
    """Equipment layer = pixels of `equipped` that differ from `base` (same grid, same pose).
    Result is a transparent overlay that the engine draws on top of the base animation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": ("IMAGE",), "equipped": ("IMAGE",),
            "tolerance": ("FLOAT", {"default": 0.07, "min": 0.0, "max": 0.5, "step": 0.005,
                                    "tooltip": "OKLab distance under which a pixel counts as unchanged body."}),
            "min_area": ("INT", {"default": 3, "min": 0, "max": 256, "tooltip": "Drop specks smaller than this."}),
            "grow": ("INT", {"default": 0, "min": 0, "max": 4, "tooltip": "Also take N neighbouring changed-ish pixels."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("layer", "composite")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, base, equipped, tolerance, min_area, grow):
        from scipy import ndimage
        layers, comps = [], []
        bf = core.split_tensor(base)
        for i, (e_rgb, e_a) in enumerate(core.split_tensor(equipped)):
            b_rgb, b_a = bf[min(i, len(bf) - 1)]
            if b_rgb.shape != e_rgb.shape:
                raise ValueError("base and equipped frames differ in size: %s vs %s" % (b_rgb.shape, e_rgb.shape))
            ba = np.ones(b_rgb.shape[:2], np.float32) if b_a is None else (b_a >= 0.5).astype(np.float32)
            ea = np.ones(e_rgb.shape[:2], np.float32) if e_a is None else (e_a >= 0.5).astype(np.float32)
            d = np.sqrt(((core.srgb_to_oklab(e_rgb) - core.srgb_to_oklab(b_rgb)) ** 2).sum(-1))
            changed = (ea > 0) & ((ba == 0) | (d > tolerance))
            if grow:
                near = ndimage.binary_dilation(changed, iterations=grow) & (ea > 0) & (d > tolerance * 0.5)
                changed |= near
            if min_area:
                lbl, n = ndimage.label(changed)
                if n:
                    sizes = ndimage.sum(changed, lbl, range(1, n + 1))
                    changed &= np.isin(lbl, np.where(sizes >= min_area)[0] + 1)
            la = changed.astype(np.float32)
            layers.append((e_rgb, la))
            comp = np.where(changed[..., None], e_rgb, b_rgb)
            comps.append((comp, np.maximum(ba, la)))
        return core.join_tensor(layers, force_alpha=True), core.join_tensor(comps, force_alpha=True)


class SK_LayerComposite:
    """Stack layers over a base (all same frame layout) — preview any armour/weapon combination."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base": ("IMAGE",)},
                "optional": {"layer_1": ("IMAGE",), "layer_2": ("IMAGE",), "layer_3": ("IMAGE",), "layer_4": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, base, layer_1=None, layer_2=None, layer_3=None, layer_4=None):
        out = []
        layers = [core.split_tensor(l) for l in (layer_1, layer_2, layer_3, layer_4) if l is not None]
        for i, (rgb, a) in enumerate(core.split_tensor(base)):
            rgb = rgb.copy()
            a = np.ones(rgb.shape[:2], np.float32) if a is None else a.copy()
            for lf in layers:
                l_rgb, l_a = lf[min(i, len(lf) - 1)]
                m = np.ones(rgb.shape[:2], bool) if l_a is None else l_a >= 0.5
                if l_rgb.shape != rgb.shape:
                    raise ValueError("layer frame size %s != base %s" % (l_rgb.shape, rgb.shape))
                rgb[m] = l_rgb[m]
                a[m] = 1.0
            out.append((rgb, a))
        return (core.join_tensor(out, force_alpha=True),)


class SK_CropToContent:
    """Crop every frame to the union of all opaque pixels (+ margin): uniform, tight frame size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "margin": ("INT", {"default": 2, "min": 0, "max": 64}),
            "multiple": ("INT", {"default": 2, "min": 1, "max": 64, "tooltip": "Round the frame size up to a multiple."}),
            "square": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "box")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, frames, margin, multiple, square):
        fr = core.split_tensor(frames)
        if fr[0][1] is None:
            return frames, "no alpha"
        union = np.zeros(fr[0][1].shape, bool)
        for _, a in fr:
            union |= a >= 0.5
        H, W = union.shape
        x0, y0, x1, y1 = core.trim_box(union.astype(np.float32), margin)
        w, h = x1 - x0, y1 - y0
        if square:
            w = h = max(w, h)
        w, h = -(-w // multiple) * multiple, -(-h // multiple) * multiple
        if w > W or h > H:  # cannot grow past the frame: shrink to the largest valid size inside it
            w, h = min(w, W // multiple * multiple or W), min(h, H // multiple * multiple or H)
            if square:
                w = h = min(w, h)
        cx, by = (x0 + x1) // 2, y1  # keep the feet line
        x0 = int(np.clip(cx - w // 2, 0, max(0, W - w)))
        y0 = int(np.clip(by - h, 0, max(0, H - h)))
        out = [(rgb[y0:y0 + h, x0:x0 + w], a[y0:y0 + h, x0:x0 + w]) for rgb, a in fr]
        return core.join_tensor(out, force_alpha=True), "x=%d y=%d w=%d h=%d" % (x0, y0, w, h)


NODES = {
    "SK_ListToBatch": (SK_ListToBatch, "Image List → Batch"),
    "SK_SpriteSheet": (SK_SpriteSheet, "Frames → Spritesheet"),
    "SK_SliceSheet": (SK_SliceSheet, "Spritesheet → Frames"),
    "SK_FrameSelect": (SK_FrameSelect, "Select Animation Frames (loop finder)"),
    "SK_Stabilize": (SK_Stabilize, "Stabilize Sprite Frames"),
    "SK_SavePixelArt": (SK_SavePixelArt, "Save Pixel Art (PNG/GIF/APNG/sheet)"),
    "SK_BatchToList": (SK_BatchToList, "Batch → Image List"),
    "SK_CropToContent": (SK_CropToContent, "Crop Frames to Content"),
    "SK_LayerExtract": (SK_LayerExtract, "Extract Equipment Layer"),
    "SK_LayerComposite": (SK_LayerComposite, "Composite Layers (paper doll)"),
}
