"""HD (non-pixel) 2D art mode: painted / cartoon style presets, soft-edged cutouts, HD canvases,
and character -> parts export for cutout rigs (Spine, DragonBones, Godot Skeleton2D)."""
import json
import os

import numpy as np
import torch
from PIL import Image

import folder_paths

from . import sk_core as core

CAT = "SpriteKiln/hd"

# name: (template, gen_w, gen_h, out_long_side, transparent)
HD_STYLES = {
    "painted cartoon platformer (Rayman-like)": (
        "vibrant hand-painted 2D cartoon platformer game character art of {subject}, bold readable shapes, "
        "soft painterly shading, rich saturated colors, smooth clean edges, full body, dynamic appealing pose",
        832, 1216, 768, True),
    "hand-painted dark fantasy (metroidvania)": (
        "hand-painted 2D game character art of {subject}, dark fantasy metroidvania style, moody muted palette, "
        "painterly brush textures, strong silhouette, full body", 832, 1216, 768, True),
    "cel-shaded anime 2D": (
        "cel-shaded anime style 2D game character art of {subject}, clean line art, two-tone shading, vivid colors, "
        "full body", 832, 1216, 768, True),
    "flat vector cartoon": (
        "flat vector 2D cartoon game art of {subject}, clean bold outlines, flat colors, simple shading, mobile game "
        "style, full body", 1024, 1024, 768, True),
    "comic ink (bold outlines)": (
        "2D comic book style game character art of {subject}, bold black ink outlines, cel shading, halftone accents, "
        "full body", 832, 1216, 768, True),
    "storybook watercolor": (
        "storybook watercolor 2D game art of {subject}, soft washes, gentle paper texture, whimsical, full body",
        832, 1216, 768, True),
    "chibi mobile game": (
        "cute chibi 2D mobile game character art of {subject}, big head small body, glossy cartoon shading, "
        "full body", 1024, 1024, 768, True),
    "creature / monster (painted)": (
        "hand-painted 2D cartoon game monster art of {subject}, expressive, bold shapes, painterly shading, full body",
        1024, 1024, 768, True),
    "environment prop (painted)": (
        "hand-painted 2D game environment prop of {subject}, single object, painterly shading, clean edges",
        1024, 1024, 512, True),
    "item / icon (HD)": (
        "polished 2D game item icon of {subject}, single object, centered, painterly shading, clean outline",
        1024, 1024, 256, True),
    "VFX / effect (HD)": (
        "2D game visual effect of {subject}, bright glowing hand-painted energy, on a plain solid black background",
        1024, 1024, 512, False),
    "painted background / parallax layer": (
        "lush hand-painted 2D side-scrolling game background of {subject}, layered depth, atmospheric lighting, "
        "no characters, wide panorama", 1344, 768, 1920, False),
    "texture (painted, tileable)": (
        "seamless tileable hand-painted 2D game texture of {subject}, top-down, even lighting, no objects",
        1024, 1024, 512, False),
    "raw prompt": ("{subject}", 1024, 1024, 1024, False),
}
HD_NEG = ("pixel art, pixelated, low resolution, jpeg artifacts, blurry, watermark, text, signature, frame, photo, "
          "realistic photograph, 3d render, cropped")
BG = {"white": "plain solid white background, no ground shadow", "gray": "plain solid flat gray background, no ground shadow",
      "green": "plain solid bright green background, no ground shadow", "black": "plain solid black background"}


class SK_HDStylePreset:
    """Painted / cartoon 2D presets (no pixel art) -> prompt, model resolution, output size, transparency."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "subject": ("STRING", {"multiline": True, "default": "a cheerful lanky hero with a big red scarf and a wooden sword"}),
            "style": (list(HD_STYLES.keys()),),
            "background": (["auto", "white", "gray", "green", "black", "scene"],),
            "out_long_side": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "0 = style default"}),
            "extra": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("prompt", "negative", "gen_width", "gen_height", "out_long_side", "transparent")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, subject, style, background, out_long_side, extra):
        tpl, gw, gh, ols, transparent = HD_STYLES[style]
        prompt = tpl.format(subject=subject.strip())
        if background == "auto":
            background = "white" if transparent else "scene"
        if background != "scene":
            prompt += ", " + BG[background]
            transparent = True
        if extra.strip():
            prompt += ", " + extra.strip()
        return prompt, HD_NEG, gw, gh, out_long_side or ols, transparent


def _resize_rgba(rgb, a, w, h):
    """Premultiplied Lanczos resize (no dark/light halos around transparent edges)."""
    af = (np.ones(rgb.shape[:2], np.float32) if a is None else a)[..., None]
    pre = np.dstack([rgb.astype(np.float32) * af, af[..., 0] * 255]).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(pre, "RGBA").resize((w, h), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32)
    a2 = arr[..., 3] / 255.0
    rgb2 = np.where(a2[..., None] > 1e-3, arr[..., :3] / np.maximum(a2[..., None], 1e-3), 0)
    return np.clip(np.round(rgb2), 0, 255).astype(np.uint8), (None if a is None else a2.astype(np.float32))


class SK_HDCutout:
    """Full-resolution cutout with soft, anti-aliased edges: segmentation mask + backdrop key (enclosed gaps, ground
    shadow), colour decontamination of edge pixels, optional shared trim and resize for whole animations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mode": (["mask_and_key", "mask", "key_refine", "none"],),
            "tolerance": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.005}),
            "remove_shadow": ("BOOLEAN", {"default": True}),
            "soft_edges": ("BOOLEAN", {"default": True, "tooltip": "Keep the segmentation's soft alpha at the silhouette."}),
            "decontaminate": ("BOOLEAN", {"default": True, "tooltip": "Remove backdrop colour bleeding from edge pixels."}),
            "trim": ("BOOLEAN", {"default": True, "tooltip": "Crop to content (one shared box for the whole batch)."}),
            "margin": ("INT", {"default": 8, "min": 0, "max": 512}),
            "out_long_side": ("INT", {"default": 0, "min": 0, "max": 8192, "tooltip": "Resize (Lanczos); 0 = keep."}),
            "enabled": ("BOOLEAN", {"default": True}),
        }, "optional": {"foreground_mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, mode, tolerance, remove_shadow, soft_edges, decontaminate, trim, margin, out_long_side,
            enabled=True, foreground_mask=None):
        from scipy import ndimage
        frames = []
        for i, (rgb, alpha) in enumerate(core.split_tensor(image)):
            a = np.ones(rgb.shape[:2], np.float32) if alpha is None else alpha
            if enabled and mode != "none":
                soft = None
                if mode in ("mask_and_key", "mask"):
                    if foreground_mask is None:
                        raise ValueError("mode %s needs foreground_mask (e.g. from Remove Background)" % mode)
                    soft = core.mask_to_np(foreground_mask, i, rgb.shape[:2]).clip(0, 1)
                if mode == "mask":
                    hard = soft >= 0.5
                else:
                    hard = core.refine_background(rgb, None if soft is None else (soft >= 0.5).astype(np.float32),
                                                  tolerance, shadow=remove_shadow) > 0
                if soft_edges:
                    band = ndimage.binary_dilation(hard, iterations=2)
                    if soft is not None:
                        a = np.where(ndimage.binary_erosion(hard, iterations=2), 1.0, soft * band)
                    else:
                        a = ndimage.gaussian_filter(hard.astype(np.float32), 0.8) * band
                else:
                    a = hard.astype(np.float32)
                a = a.astype(np.float32) * (1.0 if alpha is None else alpha)
                if decontaminate:
                    _, bg = core.key_background(rgb, tolerance, "border")
                    edge = (a > 0.02) & (a < 0.98)
                    f = rgb.astype(np.float32)
                    un = (f - (1 - a[..., None]) * bg.astype(np.float32)) / np.maximum(a[..., None], 0.02)
                    rgb = np.where(edge[..., None], np.clip(un, 0, 255), f).astype(np.uint8)
            frames.append((rgb, a))
        if trim:
            union = np.zeros(frames[0][1].shape, np.float32)
            for _, a in frames:
                union = np.maximum(union, a)
            x0, y0, x1, y1 = core.trim_box((union > 0.02).astype(np.float32) if union.max() > 0 else
                                           np.ones_like(union), margin)
            frames = [(r[y0:y1, x0:x1], a[y0:y1, x0:x1]) for r, a in frames]
        if out_long_side:
            h, w = frames[0][0].shape[:2]
            s = out_long_side / max(h, w)
            nw, nh = max(1, round(w * s)), max(1, round(h * s))
            frames = [_resize_rgba(r, a, nw, nh) for r, a in frames]
        img = core.join_tensor(frames, force_alpha=True)
        return img, img[..., 3]


class SK_HDCanvas:
    """Place HD art (RGBA ok) on a flat backdrop canvas for edit / video models, Lanczos-scaled to fit."""

    COLORS = {"gray": (128, 128, 128), "white": (255, 255, 255), "green": (0, 177, 64), "black": (0, 0, 0)}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            "padding_percent": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 45.0, "step": 1.0,
                                          "tooltip": "Free space around the subject (room for motion)."}),
            "background": (list(cls.COLORS.keys()),),
            "trim_first": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("canvas", "width", "height")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, width, height, padding_percent, background, trim_first):
        bg = np.array(self.COLORS[background], np.float32)
        out = []
        for rgb, a in core.split_tensor(image):
            a = np.ones(rgb.shape[:2], np.float32) if a is None else a
            if trim_first and (a > 0.02).any() and (a < 0.98).any():
                x0, y0, x1, y1 = core.trim_box((a > 0.02).astype(np.float32), 0)
                rgb, a = rgb[y0:y1, x0:x1], a[y0:y1, x0:x1]
            h, w = rgb.shape[:2]
            avail_w, avail_h = width * (1 - 2 * padding_percent / 100), height * (1 - 2 * padding_percent / 100)
            s = min(avail_w / w, avail_h / h)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            r2, a2 = _resize_rgba(rgb, a, nw, nh)
            canvas = np.empty((height, width, 3), np.float32)
            canvas[:] = bg
            y, x = (height - nh) // 2, (width - nw) // 2
            canvas[y:y + nh, x:x + nw] = r2 * a2[..., None] + bg * (1 - a2[..., None])
            out.append((canvas.clip(0, 255).astype(np.uint8), None))
        return core.join_tensor(out), width, height


DEFAULT_PARTS = "cape, left arm, left leg, right leg, torso, head, right arm, weapon"


class SK_PartPrompts:
    """Body-part list (draw order: back to front) -> segmentation prompts + completion prompts (list outputs)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "parts": ("STRING", {"multiline": True, "default": DEFAULT_PARTS,
                                 "tooltip": "Comma-separated, back-to-front draw order. Use what your rig needs."}),
            "character": ("STRING", {"default": "the character"}),
            "completion_template": ("STRING", {"multiline": True, "default":
                "Extend the {part} of {character} into the empty area: the {part} continues underneath the parts that "
                "covered it, so draw its hidden section (same colours, shading and art style) to make it whole."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("segment_prompts", "completion_prompts", "names")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, parts, character, completion_template):
        names = [p.strip() for p in parts.replace("\n", ",").split(",") if p.strip()]
        comp = [completion_template.format(part=n, character=character) for n in names]
        return list(names), comp, names


class SK_PartLayout:
    """Turn per-part segmentation masks into an exclusive, inpaint-ready layout.

    Every visible character pixel goes to the front-most part that claims it (draw order is back -> front).
    For each part: `isolated` = the part alone on white, `inpaint_mask` = where the part is probably hidden
    (covered by parts in front of it, near the part), `visible_mask` = its own pixels. Leftover character
    pixels become a final "rest" layer so nothing is lost."""

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, True, True)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "part_masks": ("MASK",),
            "names": ("STRING", {"forceInput": True}),
            "prompts": ("STRING", {"forceInput": True, "tooltip": "Completion prompt per part (same order as names)."}),
            "hidden_reach_percent": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0, "step": 5.0,
                                               "tooltip": "How far (relative to the part's size) a hidden section may extend."}),
            "min_part_px": ("INT", {"default": 64, "min": 0, "max": 100000}),
            "rest_layer": ("BOOLEAN", {"default": True}),
        }, "optional": {"character_mask": ("MASK", {"tooltip": "Whole-character foreground (e.g. BiRefNet)."})}}

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("isolated", "inpaint_mask", "visible_mask", "names", "prompts")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, part_masks, names, prompts, hidden_reach_percent, min_part_px, rest_layer, character_mask=None):
        from scipy import ndimage
        rgb, a0 = core.split_tensor(image[0])[0]
        H, W = rgb.shape[:2]
        reach, min_px, rest = hidden_reach_percent[0], min_part_px[0], rest_layer[0]
        fg = np.ones((H, W), bool) if a0 is None else a0 >= 0.5
        if character_mask is not None:
            fg &= core.mask_to_np(character_mask[0], 0, (H, W)) >= 0.5
        masks = [(core.mask_to_np(m, 0, (H, W)) >= 0.5) & fg for m in part_masks]
        names = list(names)[:len(masks)]
        prompts = (list(prompts) + [""] * len(masks))[:len(masks)]
        # exclusive assignment, front-most part wins
        claimed = np.zeros((H, W), bool)
        vis = [None] * len(masks)
        for i in range(len(masks) - 1, -1, -1):
            vis[i] = masks[i] & ~claimed
            claimed |= vis[i]
        if rest and (fg & ~claimed).sum() >= min_px:
            vis.insert(0, fg & ~claimed)   # leftovers drawn at the very back
            names.insert(0, "rest")
            prompts.insert(0, "Fill in the missing area of the character piece so it is whole, same art style. "
                              "Plain white background.")
        iso, inp, vm, out_names, out_prompts = [], [], [], [], []
        for i, (v, name, prompt) in enumerate(zip(vis, names, prompts)):
            if v.sum() < min_px:
                continue
            front = np.zeros((H, W), bool)
            for v2 in vis[i + 1:]:
                front |= v2
            ys, xs = np.where(v)
            r = int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * reach / 100) + 1
            hidden = front & ndimage.binary_dilation(v, iterations=max(1, r)) if reach > 0 else np.zeros_like(v)
            hidden = ndimage.binary_dilation(hidden, iterations=3) & ~v
            if name == "rest" and i == 0 and rest:
                hidden[:] = False  # leftovers are kept as they are, never invented
            canvas = np.full((H, W, 3), 255, np.uint8)
            canvas[v] = rgb[v]
            iso.append(core.join_tensor([(canvas, None)]))
            inp.append(torch.from_numpy(hidden.astype(np.float32))[None])
            vm.append(torch.from_numpy(v.astype(np.float32))[None])
            out_names.append(name)
            out_prompts.append(prompt)
        return iso, inp, vm, out_names, out_prompts


class SK_PartCutout:
    """One aligned part layer: its visible pixels (identical to the original art) plus the inpainted hidden
    section from `completed`, limited to `inpaint_mask` and the completed image's foreground."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Original art (same canvas)."}),
            "visible_mask": ("MASK",),
        }, "optional": {"completed": ("IMAGE",), "inpaint_mask": ("MASK",), "completed_mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("layer",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, visible_mask, completed=None, inpaint_mask=None, completed_mask=None):
        rgb, _ = core.split_tensor(image)[0]
        H, W = rgb.shape[:2]
        v = core.mask_to_np(visible_mask, 0, (H, W)) >= 0.5
        alpha = v.astype(np.float32)
        out = rgb.copy()
        if completed is not None and inpaint_mask is not None:
            crgb, _ = core.split_tensor(completed)[0]
            if crgb.shape[:2] != (H, W):
                crgb, _ = _resize_rgba(crgb, None, W, H)
            hid = core.mask_to_np(inpaint_mask, 0, (H, W)) >= 0.5
            cm = core.mask_to_np(completed_mask, 0, (H, W)) >= 0.5 if completed_mask is not None else \
                np.ones((H, W), bool)
            # inpainted pixels that are just backdrop (the model "erased" instead of extending) are rejected
            _, bgc = core.key_background(crgb, 0.08, "border")
            not_bg = np.sqrt(((core.srgb_to_oklab(crgb) - core.srgb_to_oklab(bgc)) ** 2).sum(-1)) > 0.12
            add = hid & cm & ~v & not_bg
            out[add] = crgb[add]
            alpha = np.maximum(alpha, add.astype(np.float32))
        return (core.join_tensor([(out, alpha)]),)


class SK_SaveParts:
    """Write rig parts: full-canvas aligned PNGs, tightly cropped PNGs, parts.json (position, size, draw order,
    pivot guess) and a composite preview. Import the aligned PNGs as layers into Spine / DragonBones / Godot."""

    INPUT_IS_LIST = True
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "layers": ("IMAGE",),
            "names": ("STRING", {"forceInput": True}),
            "folder": ("STRING", {"default": "spritekiln/parts/character"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, layers, names, folder):
        folder = folder[0]
        frames = []
        for l in layers:
            frames.extend(core.split_tensor(l))
        if len(names) < len(frames):
            names = list(names) + ["layer%d" % i for i in range(len(names), len(frames))]
        base = os.path.join(folder_paths.get_output_directory(), folder)
        n = 1
        while os.path.exists("%s_%03d" % (base, n)):
            n += 1
        outdir = "%s_%03d" % (base, n)
        os.makedirs(os.path.join(outdir, "cropped"), exist_ok=True)
        H, W = frames[0][0].shape[:2]
        meta = {"canvas": {"w": W, "h": H}, "draw_order": "back_to_front", "parts": []}
        comp = np.zeros((H, W, 3), np.float32)
        comp_a = np.zeros((H, W), np.float32)
        for z, ((rgb, a), name) in enumerate(zip(frames, names)):
            a = np.zeros((H, W), np.float32) if a is None else a
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
            Image.fromarray(np.dstack([rgb, (a * 255).astype(np.uint8)]), "RGBA").save(os.path.join(outdir, "%02d_%s.png" % (z, safe)))
            if (a > 0.02).any():
                x0, y0, x1, y1 = core.trim_box((a > 0.02).astype(np.float32), 0)
                ys, xs = np.where(a > 0.5) if (a > 0.5).any() else np.where(a > 0.02)
                Image.fromarray(np.dstack([rgb[y0:y1, x0:x1], (a[y0:y1, x0:x1] * 255).astype(np.uint8)]), "RGBA") \
                    .save(os.path.join(outdir, "cropped", "%02d_%s.png" % (z, safe)))
                meta["parts"].append({"name": name, "file": "%02d_%s.png" % (z, safe), "z": z,
                                      "x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0),
                                      "center": [float(xs.mean()), float(ys.mean())]})
            comp = comp * (1 - a[..., None]) + rgb.astype(np.float32) * a[..., None]
            comp_a = comp_a + a * (1 - comp_a)
        cx = W / 2
        cy = H / 2
        for p in meta["parts"]:
            # pivot guess: the point of the part's box closest to the canvas centre (shoulder / hip / neck)
            p["pivot_guess"] = [float(np.clip(cx, p["x"], p["x"] + p["w"])), float(np.clip(cy, p["y"], p["y"] + p["h"]))]
        with open(os.path.join(outdir, "parts.json"), "w") as f:
            json.dump(meta, f, indent=1)
        comp_rgb = comp.clip(0, 255).astype(np.uint8)
        Image.fromarray(np.dstack([comp_rgb, (comp_a * 255).astype(np.uint8)]), "RGBA").save(os.path.join(outdir, "composite.png"))
        sub = os.path.relpath(outdir, folder_paths.get_output_directory())
        ui = [{"filename": "composite.png", "subfolder": sub, "type": "output"}]
        return {"ui": {"images": ui}, "result": (core.join_tensor([(comp_rgb, comp_a)]),)}



class SK_SpriteNormalMap:
    """Normal maps for 2D lighting (Godot CanvasTexture, Unity 2D lights / Sprite secondary textures, Unreal Paper2D).
    Model-free: silhouette inflation + shading relief, pixel-exact at native resolution, works on whole sheets.
    Optionally blends AI normals (MoGe) for HD art."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Sprite / sheet with transparency."}),
            "inflate": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                  "tooltip": "How round the silhouette is (pillow / bevel)."}),
            "radius": ("INT", {"default": 0, "min": 0, "max": 512,
                               "tooltip": "Bevel radius in px; 0 = automatic per sprite island."}),
            "detail": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 3.0, "step": 0.05,
                                 "tooltip": "Relief taken from the art's shading (lighter = higher)."}),
            "detail_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.1,
                                      "tooltip": "0 for pixel art; 1-2 for HD art."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
            "convention": (["opengl", "directx"], {"tooltip": "OpenGL (+Y up): Godot, Unity, Blender. DirectX: Unreal."}),
            "invert_detail": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "ai_normals": ("IMAGE", {"tooltip": "Normal map from MoGe (OpenGL), any size."}),
            "ai_weight": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("normal_map", "height")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, inflate, radius, detail, detail_blur, strength, convention, invert_detail,
            ai_normals=None, ai_weight=0.6):
        outs, heights = [], []
        ai = None if ai_normals is None else core.split_tensor(ai_normals)
        for i, (rgb, alpha) in enumerate(core.split_tensor(image)):
            h, m = core.sprite_height(rgb, alpha, inflate, radius or None, detail, detail_blur, invert_detail)
            n = core.height_to_normals(h, m, strength, "opengl")
            if ai is not None and ai_weight > 0:
                arg, _ = ai[min(i, len(ai) - 1)]
                if arg.shape[:2] != rgb.shape[:2]:
                    arg = np.asarray(Image.fromarray(arg).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
                na = core.decode_normals(arg)
                n = n * (1 - ai_weight) + na * ai_weight
                n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
                n[~m] = (0, 0, 1)
            if convention == "directx":
                n[..., 1] *= -1
            a = np.ones(rgb.shape[:2], np.float32) if alpha is None else alpha
            outs.append((core.encode_normals(n), a))
            hn = h - h[m].min() if m.any() else h
            hn = (hn / max(hn.max(), 1e-6) * 255).astype(np.uint8)
            heights.append((np.dstack([hn] * 3), a))
        return core.join_tensor(outs, force_alpha=True), core.join_tensor(heights, force_alpha=True)


class SK_NormalLightPreview:
    """Preview a sprite lit by a light circling around it (uses the normal map) — like an engine's 2D point light."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "normal_map": ("IMAGE",),
            "frames": ("INT", {"default": 12, "min": 1, "max": 64}),
            "elevation_deg": ("FLOAT", {"default": 35.0, "min": 5.0, "max": 89.0, "step": 1.0}),
            "ambient": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            "convention": (["opengl", "directx"],),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("lit_frames",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, normal_map, frames, elevation_deg, ambient, convention):
        rgb, a = core.split_tensor(image)[0]
        nrgb, _ = core.split_tensor(normal_map)[0]
        n = core.decode_normals(nrgb)
        el = np.deg2rad(elevation_deg)
        out = []
        for k in range(frames):
            az = 2 * np.pi * k / frames
            l = (np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el))
            out.append((core.shade(rgb, a, n, l, ambient, convention=convention), a))
        return (core.join_tensor(out, force_alpha=a is not None),)


class SK_Flatten:
    """RGBA -> RGB over a flat colour (for models that need an opaque image, e.g. MoGe)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "background": ("STRING", {"default": "#808080"})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, background):
        c = core.parse_hex_palette(background)
        bg = np.array(c[0] if c is not None else (128, 128, 128), np.float32)
        out = []
        for rgb, a in core.split_tensor(image):
            a = np.ones(rgb.shape[:2], np.float32) if a is None else a
            out.append(((rgb * a[..., None] + bg * (1 - a[..., None])).astype(np.uint8), None))
        return (core.join_tensor(out),)

NODES = {
    "SK_HDStylePreset": (SK_HDStylePreset, "HD Style Preset (painted / cartoon 2D)"),
    "SK_HDCutout": (SK_HDCutout, "HD Cutout (soft edges, trim, resize)"),
    "SK_HDCanvas": (SK_HDCanvas, "HD Art → Model Canvas"),
    "SK_PartPrompts": (SK_PartPrompts, "Rig Part Prompts"),
    "SK_PartLayout": (SK_PartLayout, "Rig Part Layout (exclusive + hidden areas)"),
    "SK_PartCutout": (SK_PartCutout, "Rig Part Cutout (aligned layer)"),
    "SK_SaveParts": (SK_SaveParts, "Save Rig Parts (PNG layers + JSON)"),
    "SK_SpriteNormalMap": (SK_SpriteNormalMap, "Sprite Normal Map (2D lighting)"),
    "SK_NormalLightPreview": (SK_NormalLightPreview, "Normal Map Light Preview"),
    "SK_Flatten": (SK_Flatten, "Flatten RGBA → RGB"),
}
