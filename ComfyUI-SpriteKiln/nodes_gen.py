"""Prompt presets (styles, directions, animations), seamless tiling and tileset builder."""
import json

import numpy as np
import torch

from . import sk_core as core

CAT = "SpriteKiln/generate"

# name: (template, art_w, art_h, colors, transparent, pixel_lora_strength)
# Lower LoRA strength = finer native detail from the model (needed for large / HD sprites);
# higher = chunkier pixels (classic low-res looks).
STYLES = {
    "default": ("pixel art of {subject}, crisp clean pixels, limited color palette", 128, 128, 32, False, 0.8),
    "8-bit (NES)": ("8-bit NES era pixel art of {subject}, very limited palette, chunky pixels, retro game", 64, 64, 12, False, 1.0),
    "16-bit (SNES / Mega Drive)": ("16-bit SNES era pixel art of {subject}, retro game graphics, limited palette, crisp pixels", 128, 128, 32, False, 0.85),
    "GBA (32-bit handheld)": ("Game Boy Advance era pixel art of {subject}, vibrant saturated colors, clean dark outlines, handheld JRPG sprite art", 96, 96, 32, False, 0.7),
    "32-bit (PS1 / Saturn / arcade)": ("32-bit era 2D pixel art of {subject}, late-90s arcade and console 2D game art, detailed shading, rich colors", 160, 160, 64, False, 0.5),
    "HD-2D character (Octopath-like)": ("HD-2D JRPG pixel art character sprite of {subject}, small detailed sprite with chibi proportions, soft shading, full body, centered", 48, 64, 32, True, 0.8),
    "HD-2D portrait": ("HD-2D JRPG pixel art character portrait of {subject}, detailed face, soft painterly pixel shading", 96, 96, 48, False, 0.6),
    "high-detail modern (indie)": ("high-resolution detailed modern pixel art of {subject}, rich lighting and color, hand-crafted indie game art", 256, 256, 64, False, 0.35),
    "arcade fighter (large sprite)": ("large detailed 2D fighting game pixel art sprite of {subject}, full body, dynamic pose, 90s arcade fighting game", 128, 160, 48, True, 0.45),
    "simple / flat": ("simple pixel art of {subject}, flat colors, minimal shading, bold dark outline", 64, 64, 12, False, 0.9),
    "detailed": ("highly detailed pixel art of {subject}, intricate shading, subtle dithering, rich palette", 256, 256, 48, False, 0.45),
    "game asset": ("pixel art game asset of {subject}, single object, centered, clean dark outline", 64, 64, 24, True, 0.8),
    "character (side view)": ("pixel art character sprite of {subject}, full body, side view facing right, standing pose, centered", 64, 64, 24, True, 0.8),
    "character (front, RPG)": ("pixel art RPG character sprite of {subject}, full body, front view facing the viewer, chibi proportions, centered", 48, 48, 20, True, 0.9),
    "character (top-down)": ("top-down RPG pixel art character sprite of {subject}, seen from above at 3/4 angle, facing down, full body", 32, 32, 16, True, 1.0),
    "character turnaround": ("pixel art character turnaround sheet of {subject}, the same character shown in front view, side view and back view, side by side, evenly spaced", 192, 64, 24, True, 0.8),
    "monster / creature": ("pixel art monster sprite of {subject}, full body, menacing, game enemy, centered", 64, 64, 24, True, 0.8),
    "boss (large)": ("large detailed pixel art boss monster sprite of {subject}, full body, imposing, game boss", 160, 160, 48, True, 0.5),
    "portrait": ("pixel art portrait of {subject}, head and shoulders, RPG dialogue portrait, expressive face", 96, 96, 32, False, 0.7),
    "item / icon": ("pixel art item icon of {subject}, single object, centered, dark outline, inventory icon", 32, 32, 16, True, 1.0),
    "item sheet": ("pixel art item sheet, a neat grid of separate game items: {subject}, each item isolated", 128, 128, 32, True, 0.8),
    "skill icon": ("pixel art skill ability icon of {subject}, square icon, glowing magical effect, game UI", 32, 32, 24, False, 0.9),
    "VFX / effect": ("pixel art game visual effect of {subject}, bright glowing energy, on a plain solid black background", 64, 64, 16, False, 0.8),
    "UI element": ("pixel art game UI element, {subject}, clean flat design, crisp edges", 64, 64, 16, True, 0.9),
    "texture (tileable)": ("seamless tileable pixel art texture of {subject}, top-down, flat even lighting, repeating pattern, no objects", 32, 32, 16, False, 0.9),
    "environment / background": ("pixel art side-scrolling game background of {subject}, wide landscape, atmospheric depth", 320, 180, 48, False, 0.6),
    "isometric": ("isometric pixel art of {subject}, 2:1 isometric projection, clean edges", 128, 128, 32, False, 0.75),
    "isometric asset": ("isometric pixel art game asset of {subject}, single object, 2:1 isometric projection, centered", 64, 64, 24, True, 0.8),
    "top-down map": ("top-down pixel art RPG map of {subject}, overworld seen from directly above, game level", 256, 256, 32, False, 0.6),
    "top-down asset": ("top-down pixel art game asset of {subject}, seen from directly above, centered", 32, 32, 16, True, 1.0),
    "1-bit": ("1-bit pixel art of {subject}, pure black and white only, high contrast, no gray", 64, 64, 2, False, 0.9),
    "low-res / tiny": ("tiny low resolution pixel art of {subject}, very large chunky pixels, minimal detail, iconic shapes", 32, 32, 8, False, 1.0),
    "block texture (MC-style)": ("16x16 voxel game block texture of {subject}, seamless, flat face, pixel art", 16, 16, 16, False, 1.0),
    "inventory item (MC-style)": ("16x16 voxel game inventory item of {subject}, pixel art, centered", 16, 16, 16, True, 1.0),
    "painterly": ("painterly pixel art of {subject}, soft textured shading, muted harmonious palette", 128, 128, 48, False, 0.4),
    "anime / cel": ("anime style pixel art of {subject}, cel shading, vibrant colors, expressive", 128, 128, 32, False, 0.7),
    "raw prompt": ("{subject}", 64, 64, 32, False, 0.8),
}

TRIGGERS = {
    "z_image (pixel LoRA)": "Pixel art style. ",
    "flux2 klein (pixel LoRA)": "pixel art sprite, ",
    "none": "",
}

BG_TEXT = {"white": "plain solid white background, no ground shadow", "gray": "plain solid flat gray background, no ground shadow",
           "green": "plain solid bright green background, no ground shadow", "black": "plain solid black background"}


def _gen_size(w, h, long_side, multiple=16):
    f = max(1, long_side // max(w, h))
    gw = max(multiple, int(round(w * f / multiple)) * multiple)
    gh = max(multiple, int(round(h * f / multiple)) * multiple)
    return gw, gh


class SK_StylePreset:
    """Retro-Diffusion-like style presets -> prompt, art resolution, colour count, model resolution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "subject": ("STRING", {"multiline": True, "default": "a brave knight with a red cape and a silver sword"}),
            "style": (list(STYLES.keys()),),
            "model": (list(TRIGGERS.keys()),),
            "art_width": ("INT", {"default": 0, "min": 0, "max": 1024, "tooltip": "0 = style default"}),
            "art_height": ("INT", {"default": 0, "min": 0, "max": 1024, "tooltip": "0 = style default"}),
            "colors": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip": "0 = style default"}),
            "background": (["auto", "white", "gray", "green", "black", "scene"],
                           {"tooltip": "auto: plain background for sprite styles (easy to key out)."}),
            "long_side": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64,
                                  "tooltip": "Model resolution (long side) before pixelising."}),
            "extra": ("STRING", {"multiline": False, "default": ""}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT", "INT", "BOOLEAN", "FLOAT")
    RETURN_NAMES = ("prompt", "negative", "art_width", "art_height", "gen_width", "gen_height", "colors", "transparent",
                    "lora_strength")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, subject, style, model, art_width, art_height, colors, background, long_side, extra):
        tpl, w, h, c, transparent, lora = STYLES[style]
        w = art_width or w
        h = art_height or h
        c = colors or c
        prompt = TRIGGERS[model] + tpl.format(subject=subject.strip())
        if background == "auto":
            background = "white" if transparent else "scene"
            if style == "VFX / effect":
                background = "scene"  # the template already asks for black
        if background != "scene":
            prompt += ", " + BG_TEXT[background]
            transparent = True
        if extra.strip():
            prompt += ", " + extra.strip()
        negative = ("blurry, smooth gradients, anti-aliasing, jpeg artifacts, photo, 3d render, realistic, "
                    "noise, text, watermark, signature, frame, cropped")
        gw, gh = _gen_size(w, h, long_side)
        return prompt, negative, w, h, gw, gh, c, transparent, float(lora)


# canonical order: clockwise from the front.  (label, Klein phrase, Qwen multi-angle azimuth)
DIRS8 = [
    ("S", "front view, facing the viewer", "front view"),
    # Qwen azimuths name the side of the subject the camera sees: seeing its right side = it faces screen-right.
    ("SE", "three-quarter front view, body turned toward the lower right of the image", "front-right quarter view"),
    ("E", "side profile view, facing right", "right side view"),
    ("NE", "three-quarter back view, facing away toward the upper right of the image", "back-right quarter view"),
    ("N", "back view, facing away from the viewer", "back view"),
    ("NW", "three-quarter back view, facing away toward the upper left of the image", "back-left quarter view"),
    ("W", "side profile view, facing left", "left side view"),
    ("SW", "three-quarter front view, body turned toward the lower left of the image", "front-left quarter view"),
]
MIRROR = {"W": "E", "NW": "NE", "SW": "SE"}


def _dir_set(directions):
    if directions == "4":
        return [d for d in DIRS8 if d[0] in ("S", "E", "N", "W")]
    return DIRS8


class SK_DirectionPrompts:
    """One edit prompt per facing direction (list output -> the sampler runs once per direction)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "directions": (["8", "4"],),
            "mirror_west": ("BOOLEAN", {"default": True,
                                        "tooltip": "Only generate S/SE/E/NE/N and mirror the west-facing ones "
                                                   "(faster, perfectly symmetric like hand-made sprites)."}),
            "prompt_style": (["flux2 klein", "qwen multi-angle LoRA"],),
            "template": ("STRING", {"multiline": True, "default":
                "Turn the character in the image to a {direction}. Keep exactly the same character design, outfit, colors, "
                "proportions, size and pixel art style. Full body, centered, {background}."}),
            "background": (["white", "gray", "green", "black"],),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompts", "labels")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, directions, mirror_west, prompt_style, template, background):
        prompts, labels = [], []
        for label, klein, qwen in _dir_set(directions):
            if mirror_west and label in MIRROR:
                continue
            if prompt_style.startswith("qwen"):
                p = "<sks> %s eye-level shot medium shot" % qwen
            else:
                p = template.format(direction=klein, background=BG_TEXT[background])
            prompts.append(p)
            labels.append(label)
        return prompts, labels


class SK_DirectionAssemble:
    """Generated direction images (list, same order as SK_DirectionPrompts) -> full ordered batch.
    Each list item may be a single image or an animation (batch of frames): output is then
    direction-major (all frames of dir 1, then dir 2 ...) -> save with sheet_columns = frames
    for one row per direction. West-facing directions that were skipped are mirrored."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "labels": ("STRING", {"forceInput": True}),
            "order": (["clockwise from S", "RPG rows (S, W, E, N)", "S, SW, W, NW, N, NE, E, SE"],),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("frames", "labels", "frames_per_direction")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, images, labels, order):
        order = order[0]
        directions = "8" if any(l in ("SE", "NE", "SW", "NW") for l in labels) else "4"
        got = {}
        for lab, img in zip(labels, images):
            got[lab] = core.split_tensor(img)
        for west, east in MIRROR.items():
            if west not in got and east in got:
                got[west] = [(rgb[:, ::-1].copy(), None if a is None else a[:, ::-1].copy()) for rgb, a in got[east]]
        want = [d[0] for d in _dir_set(directions)]
        if order.startswith("RPG"):
            want = ["S", "W", "E", "N"] if directions == "4" else ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
        elif order.startswith("S, SW"):
            want = [d for d in ["S", "SW", "W", "NW", "N", "NE", "E", "SE"] if d in want]
        missing = [d for d in want if d not in got]
        if missing:
            raise ValueError("missing directions: %s" % missing)
        n = min(len(got[d]) for d in want)
        frames = [f for d in want for f in got[d][:n]]
        return core.join_tensor(frames), " ".join(want), n


ACTIONS = {
    # action: (motion text, sprite frames, gif fps, loops)
    "idle": ("stands in place with a subtle idle breathing animation, gentle bobbing, blinking", 6, 6, True),
    "walk": ("walks in place, a smooth walk cycle with legs and arms swinging, like on a treadmill; the character stays in the center of the frame", 8, 10, True),
    "run": ("runs in place, a fast run cycle with big strides and pumping arms; the character stays in the center of the frame", 8, 12, True),
    "sword slash": ("performs a fast sword slash: winds up, swings the sword in a wide arc in front of it with a motion trail, then returns to the ready stance", 6, 12, False),
    "attack combo (3 hits)": ("performs a three-hit melee combo: a quick horizontal slash, a second backhand slash, then a powerful overhead strike, then returns to the ready stance", 12, 12, False),
    "attack (thrust / stab)": ("lunges forward and thrusts the weapon straight ahead, then pulls back to the ready stance", 6, 12, False),
    "attack (ranged)": ("raises the weapon, aims and fires a shot forward, recoils, then returns to the ready stance", 6, 10, False),
    "kick": ("throws a quick high kick forward, then returns to the fighting stance", 6, 12, False),
    "cast spell": ("raises its hands and casts a glowing magic spell with sparkling energy, then lowers its hands", 8, 10, False),
    "charge up": ("crouches slightly and charges up power, glowing energy aura swirling around the body, pulsing", 8, 10, True),
    "crouch": ("crouches down low, bending the knees, and holds the crouched pose", 4, 10, False),
    "crouch idle": ("stays crouched down low, breathing gently while crouching", 4, 6, True),
    "block / guard": ("raises its weapon or shield to block, braces for impact, holds the guard", 4, 10, False),
    "dodge roll": ("does a quick forward dodge roll, tucking into a ball and rolling, then stands back up in place", 8, 14, False),
    "dash": ("dashes forward in a quick burst with motion lines, leaning forward", 6, 14, False),
    "slide": ("drops into a low slide along the ground, one leg extended forward", 6, 12, False),
    "jump": ("crouches, jumps straight up, then lands back in the same spot", 8, 10, False),
    "climb": ("climbs upward on a ladder, alternating arms and legs, climbing cycle", 6, 10, True),
    "hurt": ("is hit, flinches backward in pain, then recovers", 4, 8, False),
    "death": ("is defeated, staggers and collapses to the ground, lying still", 8, 8, False),
    "victory / emote": ("celebrates a victory, raising its weapon triumphantly in the air and cheering", 8, 8, False),
    # creatures & monsters
    "creature idle (breathe)": ("breathes slowly, its body gently expanding and contracting, small idle movements", 6, 6, True),
    "slime bounce": ("squashes down and bounces up in place with a jelly-like squash and stretch", 6, 10, True),
    "wing flap / fly": ("hovers in place flapping its wings in a steady flying cycle", 6, 10, True),
    "slither / crawl": ("slithers and crawls in place with a wavy body motion, staying in the center of the frame", 8, 10, True),
    "monster attack (bite / claw)": ("lunges forward and bites or claws at the air, then returns to its stance", 6, 12, False),
    "roar": ("rears back and roars aggressively, mouth wide open, then settles", 8, 10, False),
    # props & environment
    "fire / torch flicker": ("the flames flicker and dance continuously, with small sparks; the object itself does not move", 6, 10, True),
    "sway (tree / grass / banner)": ("gently sways back and forth in the wind; the base stays fixed in place", 8, 8, True),
    "water flow / waterfall": ("the water flows and ripples continuously in a seamless loop", 8, 10, True),
    "chest / door opening": ("slowly opens, the lid or door swinging open, revealing a soft glow", 6, 10, False),
    "coin / item spin": ("spins around its vertical axis in place, a full smooth rotation", 8, 12, True),
    "hover / float": ("floats up and down gently in place with a soft magical glow", 6, 8, True),
    "glow pulse": ("pulses with light, glowing brighter and dimmer in a steady rhythm", 6, 8, True),
    "explode / break": ("cracks and bursts apart into fragments and dust, then disappears", 8, 12, False),
    "custom": ("{custom}", 8, 10, True),
}

WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
           "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, "
           "camera movement, camera pan, zoom, background change, blurry, morphing, extra limbs, text, watermark")


class SK_AnimationPreset:
    """Motion prompt + frame counts for sprite animations (Wan 2.2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "subject": ("STRING", {"multiline": True, "default": "the pixel art knight"}),
            "action": (list(ACTIONS.keys()),),
            "view": (["side view", "front view", "three-quarter view", "top-down view", "isometric view",
                      "keep facing direction"],),
            "custom_motion": ("STRING", {"multiline": True, "default": ""}),
            "background": (["gray", "white", "green", "black"],),
            "art": (["pixel art", "HD 2D (painted / cartoon)"],),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("positive", "negative", "sprite_frames", "fps", "loops")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, subject, action, view, custom_motion, background, art="pixel art"):
        motion, frames, fps, loops = ACTIONS[action]
        motion = motion.format(custom=custom_motion.strip())
        if view == "keep facing direction":
            view = "the character keeps facing exactly the same direction as in the first frame"
        pos = ("2D pixel art sprite animation, %s, %s %s. Static locked camera, no zoom, no panning, %s stays unchanged, "
               "crisp pixel art, consistent character design, smooth looping game animation."
               % (view, subject.strip(), motion, BG_TEXT[background]))
        if not loops:
            pos = pos.replace(", smooth looping game animation.", ", one-shot game animation.")
        if art.startswith("HD"):
            pos = pos.replace("2D pixel art sprite animation", "2D hand-painted cartoon game character animation") \
                     .replace("crisp pixel art", "clean smooth hand-painted 2D cartoon style")
        return pos, WAN_NEG, frames, float(fps), loops


# ---------------------------------------------------------------- seamless tiling

def _wrap_pad(x, px, py):
    if px:
        x = torch.cat([x[..., -px:], x, x[..., :px]], dim=-1)
    if py:
        x = torch.cat([x[..., -py:, :], x, x[..., :py, :]], dim=-2)
    return x


class SK_SeamlessTiling:
    """Make any diffusion model (incl. DiTs: Z-Image, Flux) produce tileable images.

    Every model call sees the latent circularly rolled by a different offset, so the wrap-around
    seam keeps moving into the middle of the picture where the model paints across it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "axes": (["both", "x", "y"],),
            "align": ("INT", {"default": 2, "min": 1, "max": 16,
                              "tooltip": "Roll in multiples of the model's patch size (2 for Flux/Z-Image/Wan)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, model, axes, align):
        m = model.clone()

        def wrapper(apply_model, args):
            x = args["input"]
            H, W = x.shape[-2], x.shape[-1]
            # offset index from the timestep: same seed -> same result, every step a different seam position
            k = int(round(float(args["timestep"].flatten()[0]) * 1000))
            # golden-ratio sequence: well spread offsets, half a frame on the first call
            fx, fy = (0.5 + k * 0.618034) % 1.0, (0.5 + k * 0.754878) % 1.0
            sx = int(W * fx) // align * align if axes in ("both", "x") else 0
            sy = int(H * fy) // align * align if axes in ("both", "y") else 0
            c = dict(args["c"])
            cc = c.get("c_concat")
            if isinstance(cc, torch.Tensor) and cc.shape[-2:] == x.shape[-2:]:
                c["c_concat"] = torch.roll(cc, (sy, sx), dims=(-2, -1))
            out = apply_model(torch.roll(x, (sy, sx), dims=(-2, -1)), args["timestep"], **c)
            return torch.roll(out, (-sy, -sx), dims=(-2, -1))

        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


class SK_SeamlessVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "vae": ("VAE",),
            "axes": (["both", "x", "y"],),
            "pad_latent": ("INT", {"default": 8, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, samples, vae, axes, pad_latent):
        z = samples["samples"]
        px = pad_latent if axes in ("both", "x") else 0
        py = pad_latent if axes in ("both", "y") else 0
        img = vae.decode(_wrap_pad(z, px, py))
        if img.dim() == 5:
            img = img.reshape(-1, *img.shape[-3:])
        r = img.shape[1] / (z.shape[-2] + 2 * py)
        oy, ox = int(round(py * r)), int(round(px * r))
        h, w = int(round(z.shape[-2] * r)), int(round(z.shape[-1] * r))
        return (img[:, oy:oy + h, ox:ox + w, :],)


class SK_TilePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "repeat_x": ("INT", {"default": 3, "min": 1, "max": 16}),
                             "repeat_y": ("INT", {"default": 3, "min": 1, "max": 16})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, repeat_x, repeat_y):
        return (image.repeat(1, repeat_y, repeat_x, 1),)


# ---------------------------------------------------------------- tilesets

def _periodic_noise(size, cells, rng):
    g = rng.random((cells, cells)).astype(np.float32)
    t = (np.arange(size) + 0.5) / size * cells
    i0 = np.floor(t).astype(int) % cells
    i1 = (i0 + 1) % cells
    f = t - np.floor(t)
    f = f * f * (3 - 2 * f)
    rows = g[i0][:, :] * (1 - f)[:, None] + g[i1] * f[:, None]
    return rows[:, i0] * (1 - f)[None, :] + rows[:, i1] * f[None, :]


def _blob_reduce(mask):
    # bits: 1 N, 2 NE, 4 E, 8 SE, 16 S, 32 SW, 64 W, 128 NW. Corners count only with both sides.
    n, ne, e, se, s, sw, w, nw = [(mask >> i) & 1 for i in range(8)]
    ne &= n & e
    se &= s & e
    sw &= s & w
    nw &= n & w
    return n | ne << 1 | e << 2 | se << 3 | s << 4 | sw << 5 | w << 6 | nw << 7


class SK_Tileset:
    """Build an autotile set from two seamless pixel textures (A drawn on top of B).
    wang16: 2-corner / dual-grid set (index = NW + 2*NE + 4*SW + 8*SE corner is A).
    blob47: 8-neighbour blob set (mapping of neighbour bitmask -> tile index is returned)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "texture_a": ("IMAGE", {"tooltip": "Upper terrain (e.g. grass). Seamless, tile-sized."}),
            "texture_b": ("IMAGE", {"tooltip": "Lower terrain (e.g. dirt). Seamless, tile-sized."}),
            "layout": (["wang16", "blob47"],),
            "edge_noise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
            "noise_cells": ("INT", {"default": 4, "min": 1, "max": 16}),
            "edge_shade": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "Darken A pixels along the border (1 = none)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tileset", "mapping")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, texture_a, texture_b, layout, edge_noise, noise_cells, edge_shade, seed):
        A, _ = core.split_tensor(texture_a)[0]
        B, _ = core.split_tensor(texture_b)[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("autotile textures must be square tiles (got %dx%d)" % (A.shape[1], A.shape[0]))
        ts = A.shape[0]
        if B.shape[:2] != A.shape[:2]:
            ys = (np.arange(ts) * B.shape[0] / ts).astype(int)
            xs = (np.arange(A.shape[1]) * B.shape[1] / A.shape[1]).astype(int)
            B = B[ys][:, xs]
        rng = np.random.default_rng(seed)
        noise = _periodic_noise(ts, noise_cells, rng) - 0.5
        # pixel coordinates in [0,1], extended by one pixel each side for the edge-shade test
        t = (np.arange(-1, ts + 1) + 0.5) / ts
        t = np.clip(t, 0, 1)

        def field(ctrl):
            """ctrl: 3x3 control values (corners, edge mids, centre) -> (ts+2)^2 field."""
            u = t * 2
            i = np.minimum(np.floor(u).astype(int), 1)
            f = u - i
            yi, yf = i[:, None], f[:, None]
            xi, xf = i[None, :], f[None, :]
            c00, c01 = ctrl[yi, xi], ctrl[yi, xi + 1]
            c10, c11 = ctrl[yi + 1, xi], ctrl[yi + 1, xi + 1]
            return (c00 * (1 - xf) + c01 * xf) * (1 - yf) + (c10 * (1 - xf) + c11 * xf) * yf

        npad = np.pad(noise, 1, mode="wrap")

        def render(ctrl):
            v = field(ctrl) + edge_noise * npad
            isA = v > 0.5
            core_a = isA[1:-1, 1:-1]
            out = np.where(core_a[..., None], A, B).astype(np.float32)
            if edge_shade < 1:
                border = core_a & ~(isA[:-2, 1:-1] & isA[2:, 1:-1] & isA[1:-1, :-2] & isA[1:-1, 2:])
                out[border] *= edge_shade
            return out.astype(np.uint8)

        tiles, mapping = [], {}
        if layout == "wang16":
            for idx in range(16):
                nw, ne, sw, se = idx & 1, (idx >> 1) & 1, (idx >> 2) & 1, (idx >> 3) & 1
                top, bot = (nw + ne) / 2, (sw + se) / 2
                left, right = (nw + sw) / 2, (ne + se) / 2
                ctr = (nw + ne + sw + se) / 4
                ctrl = np.array([[nw, top, ne], [left, ctr, right], [sw, bot, se]], np.float32)
                tiles.append(render(ctrl))
                mapping[str(idx)] = idx
            cols = 4
        else:
            reduced = sorted({_blob_reduce(m) for m in range(256)})
            for i, m in enumerate(reduced):
                n, ne, e, se, s, sw, w, nw = [(m >> b) & 1 for b in range(8)]
                ctrl = np.array([[nw, n, ne], [w, 1, e], [sw, s, se]], np.float32)
                tiles.append(render(ctrl))
                mapping[str(m)] = i
            for m in range(256):
                mapping[str(m)] = mapping[str(_blob_reduce(m))]
            cols = 8
        rows = -(-len(tiles) // cols)
        sheet = np.zeros((rows * ts, cols * ts, 3), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * ts:(r + 1) * ts, c * ts:(c + 1) * ts] = tile
        info = {"layout": layout, "tile_size": ts, "columns": cols,
                "bits": "wang16: 1=NW 2=NE 4=SW 8=SE corner is A" if layout == "wang16" else
                        "blob47: 1=N 2=NE 4=E 8=SE 16=S 32=SW 64=W 128=NW neighbour is A",
                "mask_to_tile": mapping}
        return core.join_tensor([(sheet, None)]), json.dumps(info)


NODES = {
    "SK_StylePreset": (SK_StylePreset, "Pixel Style Preset"),
    "SK_DirectionPrompts": (SK_DirectionPrompts, "Direction Prompts (4/8-dir)"),
    "SK_DirectionAssemble": (SK_DirectionAssemble, "Assemble Directions"),
    "SK_AnimationPreset": (SK_AnimationPreset, "Animation Preset"),
    "SK_SeamlessTiling": (SK_SeamlessTiling, "Seamless Tiling (model patch)"),
    "SK_SeamlessVAEDecode": (SK_SeamlessVAEDecode, "Seamless VAE Decode"),
    "SK_TilePreview": (SK_TilePreview, "Tile Preview"),
    "SK_Tileset": (SK_Tileset, "Autotile Tileset (wang16 / blob47)"),
}
