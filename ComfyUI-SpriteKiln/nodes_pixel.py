"""Pixel-perfect conversion, palettes, cleanup and background nodes."""
import numpy as np
import torch

from . import sk_core as core
from .palettes import palette_names, get_palette

CAT = "SpriteKiln/pixel"
DITHERS = ["none", "bayer-2", "bayer-4", "bayer-8", "floyd-steinberg"]


class SK_PixelPerfect:
    """Turn a high-res 'fake' pixel-art image (model output) into a true pixel grid."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["fixed_size", "auto_detect", "scale_factor", "match_reference"],
                         {"tooltip": "fixed_size: exact art resolution. auto_detect: find the pixel grid "
                                     "(for existing fake pixel art). scale_factor: source px per art px. "
                                     "match_reference: same size as `reference` (edit round-trips)."}),
                "width": ("INT", {"default": 64, "min": 4, "max": 1024}),
                "height": ("INT", {"default": 64, "min": 4, "max": 1024}),
                "scale": ("INT", {"default": 16, "min": 1, "max": 128,
                                  "tooltip": "Source pixels per art pixel (scale_factor / match_reference)."}),
                "method": (["dominant", "median", "mean", "nearest"], {"default": "dominant"}),
                "pre_colors": ("INT", {"default": 48, "min": 2, "max": 256,
                                       "tooltip": "Colour clusters used by the 'dominant' vote."}),
            },
            "optional": {
                "foreground_mask": ("MASK", {"tooltip": "1 = keep (e.g. from Remove Background)."}),
                "reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("pixel_art", "preview", "info")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, mode, width, height, scale, method, pre_colors, foreground_mask=None, reference=None):
        frames, previews, infos = [], [], []
        for i, (rgb, alpha) in enumerate(core.split_tensor(image)):
            H, W = rgb.shape[:2]
            if foreground_mask is not None:
                m = core.mask_to_np(foreground_mask, i, (H, W))
                alpha = m if alpha is None else alpha * m
            box = None
            if mode == "fixed_size":
                ow, oh = width, height
            elif mode == "scale_factor":
                ow, oh = max(1, round(W / scale)), max(1, round(H / scale))
            elif mode == "match_reference":
                if reference is None:
                    raise ValueError("match_reference needs the `reference` image input")
                oh, ow = reference.shape[1], reference.shape[2]
                box = (0, 0, min(W, ow * scale), min(H, oh * scale))
            else:
                if i == 0:  # one grid for the whole batch keeps frames aligned
                    grid = core.detect_pixel_scale(rgb)
                sx, ox, sy, oy = grid
                # first grid line -> include a partial leading cell when it is more than half a pixel wide
                ox = ox % sx
                oy = oy % sy
                if ox > sx / 2:
                    ox -= sx
                if oy > sy / 2:
                    oy -= sy
                ow, oh = max(1, int(round((W - ox) / sx))), max(1, int(round((H - oy) / sy)))
                box = (ox, oy, ox + ow * sx, oy + oh * sy)
                infos.append("detected pixel size %.2f x %.2f, offset (%.1f,%.1f) -> %dx%d" % (sx, sy, ox, oy, ow, oh))
            out, out_a = core.downscale(rgb, alpha, ow, oh, method=method, box=box, pre_colors=pre_colors, seed=i)
            frames.append((out, out_a))
            f = max(1, min(W // ow, H // oh))
            prev = np.repeat(np.repeat(out, f, 0), f, 1)
            pa = None if out_a is None else np.repeat(np.repeat(out_a, f, 0), f, 1)
            previews.append((prev, pa))
            if mode != "auto_detect":
                infos.append("%dx%d -> %dx%d" % (W, H, ow, oh))
        return core.join_tensor(frames), core.join_tensor(previews), "\n".join(infos)


class SK_Palette:
    """Reduce / remap colours. Shared mode uses one palette for the whole batch (animations)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source": (["auto", "preset", "palette_image", "hex"],
                           {"tooltip": "auto: k-means N colours from the image. palette_image: use every colour "
                                       "of the image wired to `palette_image` (e.g. the original sprite)."}),
                "colors": ("INT", {"default": 16, "min": 2, "max": 256}),
                "preset": (palette_names(),),
                "dither": (DITHERS, {"default": "none", "tooltip": "Bayer is stable across animation frames; "
                                                                   "Floyd-Steinberg shimmers in animations."}),
                "dither_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "shared_palette": ("BOOLEAN", {"default": True, "tooltip": "One palette for all frames."}),
            },
            "optional": {
                "palette_image": ("IMAGE",),
                "hex_palette": ("STRING", {"default": "", "multiline": False,
                                           "tooltip": "e.g. #000000 #1d2b53 #7e2553 ..."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "palette", "hex")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, source, colors, preset, dither, dither_strength, shared_palette,
            palette_image=None, hex_palette=""):
        frames = core.split_tensor(image)

        def fixed_palette():
            if source == "preset":
                pal = get_palette(preset)
                if pal is None:
                    raise ValueError("palette %r has no readable colours (expected #rrggbb or AARRGGBB lines)" % preset)
                return pal
            if source == "hex":
                pal = core.parse_hex_palette(hex_palette)
                if pal is None:
                    raise ValueError("hex_palette has no #rrggbb colours")
                return pal
            if source == "palette_image":
                if palette_image is None:
                    raise ValueError("source=palette_image needs the palette_image input")
                pf = core.split_tensor(palette_image)
                cols, counts = core.unique_colors([f[0] for f in pf], [f[1] for f in pf])
                return cols if len(cols) <= colors else core.kmeans_palette(cols, counts, colors)
            return None

        pal = fixed_palette()
        if pal is not None and len(pal) > colors and source != "palette_image":
            # limit a big preset to the N colours the image actually needs
            cols, counts = core.unique_colors([f[0] for f in frames], [f[1] for f in frames])
            idx = core.nearest_index(core.srgb_to_oklab(cols), core.srgb_to_oklab(pal))
            use = np.bincount(idx, weights=counts, minlength=len(pal))
            pal = pal[np.argsort(-use)[:colors]]
        if pal is None and shared_palette:
            cols, counts = core.unique_colors([f[0] for f in frames], [f[1] for f in frames])
            pal = core.kmeans_palette(cols, counts, colors)

        out, used = [], []
        for i, (rgb, alpha) in enumerate(frames):
            p = pal
            if p is None:
                cols, counts = core.unique_colors([rgb], [alpha])
                p = core.kmeans_palette(cols, counts, colors, seed=i)
            q = core.quantize(rgb, p, dither, dither_strength, alpha)
            out.append((q, alpha))
            used.append(p)
        final = core.sort_palette(np.unique(np.concatenate(used), axis=0))
        strip = core.palette_strip(final)
        return core.join_tensor(out), core.join_tensor([(strip, None)]), core.palette_to_hex(final)


class SK_Cleanup:
    """Remove near-shade specks left by downscaling, harden alpha, add outlines.
    Keeps high-contrast single pixels (eyes, highlights) and dither patterns."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "speck_size": ("INT", {"default": 2, "min": 0, "max": 16,
                                       "tooltip": "Merge same-colour clusters up to this many pixels (0 = off)."}),
                "max_contrast": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Only merge specks this close (OKLab) to their surroundings. "
                                                      "Raise to also remove contrasting noise; lower to protect detail."}),
                "outline": (["none", "outer_auto", "outer_color", "inner_auto", "inner_color"],),
                "outline_color": ("STRING", {"default": "#140c1c"}),
                "outline_diagonal": ("BOOLEAN", {"default": False}),
                "darken": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, speck_size, max_contrast, outline, outline_color, outline_diagonal, darken):
        col = core.parse_hex_palette(outline_color)
        col = None if col is None else col[0]
        out = []
        for rgb, alpha in core.split_tensor(image):
            if alpha is not None:
                alpha = (alpha >= 0.5).astype(np.float32)
            if speck_size:
                rgb, alpha = core.remove_specks(rgb, alpha, speck_size, max_contrast)
            if outline != "none":
                mode = "outer" if outline.startswith("outer") else "inner"
                c = col if outline.endswith("color") else None
                rgb, alpha = core.add_outline(rgb, alpha, c, mode, outline_diagonal, darken)
            out.append((rgb, alpha))
        return (core.join_tensor(out),)


class SK_BackgroundRemove:
    """Make the background transparent: border flood-fill colour key, or an external mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["mask_and_key", "key_refine", "border_flood", "global_key", "mask", "mask_and_flood"],
                         {"tooltip": "mask_and_key (best for generated sprites): segmentation mask + backdrop-colour key "
                                     "incl. enclosed gaps, edge defringe and ground-shadow removal. key_refine: the same "
                                     "without a mask. border_flood: plain colour key from the edges."}),
                "remove_shadow": ("BOOLEAN", {"default": True, "tooltip": "mask_and_key / key_refine: drop grey ground shadows."}),
                "tolerance": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.005,
                                        "tooltip": "OKLab distance to the detected background colour."}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = keep the background (fully opaque)."}),
            },
            "optional": {"foreground_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "foreground_mask")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, mode, tolerance, mask_threshold, enabled=True, remove_shadow=True, foreground_mask=None):
        out, masks = [], []
        for i, (rgb, alpha) in enumerate(core.split_tensor(image)):
            a = np.ones(rgb.shape[:2], np.float32)
            if not enabled:
                pass
            elif mode in ("mask_and_key", "key_refine"):
                fm = None
                if mode == "mask_and_key":
                    if foreground_mask is None:
                        raise ValueError("mode mask_and_key needs foreground_mask (e.g. from Remove Background)")
                    fm = (core.mask_to_np(foreground_mask, i, rgb.shape[:2]) >= mask_threshold).astype(np.float32)
                a = core.refine_background(rgb, fm, tolerance, shadow=remove_shadow)
            elif mode in ("border_flood", "global_key", "mask_and_flood"):
                a, _ = core.key_background(rgb, tolerance, "global" if mode == "global_key" else "border")
            if enabled and mode in ("mask", "mask_and_flood"):
                if foreground_mask is None:
                    raise ValueError("mode %s needs foreground_mask" % mode)
                m = (core.mask_to_np(foreground_mask, i, rgb.shape[:2]) >= mask_threshold).astype(np.float32)
                a = m if mode == "mask" else a * m
            if alpha is not None:
                a = a * (alpha >= 0.5)
            out.append((rgb, a))
            masks.append(torch.from_numpy(a))
        return core.join_tensor(out, force_alpha=True), torch.stack(masks)


class SK_ScaleNearest:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "scale": ("INT", {"default": 8, "min": 1, "max": 64})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, scale):
        return (image.repeat_interleave(scale, 1).repeat_interleave(scale, 2),)


class SK_ToCanvas:
    """Prepare small pixel art as model input: integer nearest upscale onto a flat background."""

    BG = {"gray": (128, 128, 128), "white": (255, 255, 255), "black": (0, 0, 0),
          "green": (0, 177, 64), "magenta": (255, 0, 255), "blue": (0, 71, 187)}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "long_side": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "multiple": ("INT", {"default": 16, "min": 1, "max": 64,
                                     "tooltip": "Canvas is padded to a multiple of this (16 Flux/Z-Image, 32 Wan)."}),
                "background": (list(cls.BG.keys()) + ["custom"],),
                "custom_color": ("STRING", {"default": "#808080"}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 256,
                                    "tooltip": "Extra art-pixels of empty space around the sprite (room to move)."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Edit region at sprite resolution (1 = change)."}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                                         "tooltip": "Exact model canvas (e.g. 1280x704 for Wan 5B, 832x480 for Wan 14B). "
                                                    "The sprite is centred with integer scaling. 0 = use long_side."}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "IMAGE", "MASK")
    RETURN_NAMES = ("canvas", "scale", "width", "height", "padded_sprite", "canvas_mask")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, long_side, multiple, background, custom_color, padding, mask=None, target_width=0,
            target_height=0):
        if background == "custom":
            c = core.parse_hex_palette(custom_color)
            bg = tuple(int(v) for v in (c[0] if c is not None else (128, 128, 128)))
        else:
            bg = self.BG[background]
        frames, sprites, masks = [], [], []
        scale = W = H = 1
        for i, (rgb, alpha) in enumerate(core.split_tensor(image)):
            m = None if mask is None else core.mask_to_np(mask, i, rgb.shape[:2])
            if padding:
                a0 = alpha if alpha is not None else np.ones(rgb.shape[:2], np.float32)
                rgb = np.pad(rgb, ((padding, padding), (padding, padding), (0, 0)))
                alpha = np.pad(a0, padding)
                m = None if m is None else np.pad(m, padding)
            if target_width and target_height:
                # centre the sprite on an art-pixel canvas whose integer upscale fills the model canvas
                h0, w0 = rgb.shape[:2]
                s = max(1, min(target_width // w0, target_height // h0))
                # padding 0 = tiles / exact content: no centring band (it would break wrap-around)
                AW, AH = (max(w0, target_width // s), max(h0, target_height // s)) if padding else (w0, h0)
                px, py = (AW - w0) // 2, (AH - h0) // 2
                a0 = alpha if alpha is not None else np.ones((h0, w0), np.float32)
                rgb = np.pad(rgb, ((py, AH - h0 - py), (px, AW - w0 - px), (0, 0)))
                alpha = np.pad(a0, ((py, AH - h0 - py), (px, AW - w0 - px)))
                m = None if m is None else np.pad(m, ((py, AH - h0 - py), (px, AW - w0 - px)))
            h, w = rgb.shape[:2]
            a = alpha if alpha is not None else np.ones((h, w), np.float32)
            comp = (rgb.astype(np.float32) * a[..., None] + np.array(bg, np.float32) * (1 - a[..., None]))
            comp = comp.astype(np.uint8)
            sprites.append((comp, a))
            if target_width and target_height:
                scale = max(1, min(target_width // w, target_height // h))
            else:
                scale = max(1, long_side // max(h, w))
            big = np.repeat(np.repeat(comp, scale, 0), scale, 1)
            fill_h = target_height if (target_width and target_height and padding) else 0
            fill_w = target_width if (target_width and target_height and padding) else 0
            H = -(-max(big.shape[0], fill_h) // multiple) * multiple
            W = -(-max(big.shape[1], fill_w) // multiple) * multiple
            canvas = np.empty((H, W, 3), np.uint8)
            canvas[:] = bg
            canvas[:big.shape[0], :big.shape[1]] = big
            frames.append((canvas, None))
            cm = np.zeros((H, W), np.float32)
            if m is not None:
                cm[:big.shape[0], :big.shape[1]] = np.repeat(np.repeat(m, scale, 0), scale, 1)
            masks.append(torch.from_numpy(cm))
        return core.join_tensor(frames), scale, W, H, core.join_tensor(sprites), torch.stack(masks)


class SK_Outpaint:
    """Grow the canvas of a small sprite (art pixels) — pair with an edit model to extend scenes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "left": ("INT", {"default": 0, "min": 0, "max": 512}),
            "top": ("INT", {"default": 0, "min": 0, "max": 512}),
            "right": ("INT", {"default": 16, "min": 0, "max": 512}),
            "bottom": ("INT", {"default": 0, "min": 0, "max": 512}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "new_area_mask")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, left, top, right, bottom):
        B, H, W, C = image.shape
        out = torch.zeros((B, H + top + bottom, W + left + right, C))
        out[:, top:top + H, left:left + W] = image
        if C == 3:
            # fill the new area with the edge colours so edit models see a neutral extension
            out = torch.nn.functional.pad(image.permute(0, 3, 1, 2), (left, right, top, bottom), mode="replicate").permute(0, 2, 3, 1)
        mask = torch.ones((B, H + top + bottom, W + left + right))
        mask[:, top:top + H, left:left + W] = 0
        return out, mask


class SK_RegionMask:
    """Rectangle mask in art pixels (precise edit regions for small sprites), optionally merged with a painted mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Sprite the mask is for (sets the size)."}),
            "x": ("INT", {"default": 0, "min": 0, "max": 4096}),
            "y": ("INT", {"default": 0, "min": 0, "max": 4096}),
            "width": ("INT", {"default": 16, "min": 0, "max": 4096, "tooltip": "0 = no rectangle"}),
            "height": ("INT", {"default": 16, "min": 0, "max": 4096}),
            "grow": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Dilate by N pixels."}),
            "invert": ("BOOLEAN", {"default": False}),
        }, "optional": {"painted_mask": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, x, y, width, height, grow, invert, painted_mask=None):
        from scipy import ndimage
        B, H, W = image.shape[:3]
        m = np.zeros((H, W), np.float32)
        if width and height:
            m[y:y + height, x:x + width] = 1.0
        if painted_mask is not None:
            m = np.maximum(m, (core.mask_to_np(painted_mask, 0, (H, W)) >= 0.5).astype(np.float32))
        if grow:
            m = ndimage.binary_dilation(m > 0, iterations=grow).astype(np.float32)
        if invert:
            m = 1.0 - m
        return (torch.from_numpy(m)[None].repeat(B, 1, 1),)


class SK_MaskedMerge:
    """Keep `original` pixels outside the mask, take `edited` pixels inside (alpha aware)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "edited": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, original, edited, mask):
        out = []
        ed = core.split_tensor(edited)
        for i, (rgb, a) in enumerate(core.split_tensor(original)):
            e_rgb, e_a = ed[min(i, len(ed) - 1)]
            m = core.mask_to_np(mask, i, rgb.shape[:2]) >= 0.5
            if e_rgb.shape != rgb.shape:
                raise ValueError("original and edited must have the same size (%s vs %s)" % (rgb.shape, e_rgb.shape))
            r = np.where(m[..., None], e_rgb, rgb)
            aa = np.ones(rgb.shape[:2], np.float32) if a is None else a
            ea = np.ones(rgb.shape[:2], np.float32) if e_a is None else e_a
            out.append((r, np.where(m, ea, aa) if (a is not None or e_a is not None) else None))
        return (core.join_tensor(out),)


class SK_PixelUpscale:
    """Pixel-art aware upscaling: Scale2x / Scale3x (EPX family) round staircases without adding colours."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "method": (["scale2x", "scale3x", "scale4x (2x twice)", "nearest 2x", "nearest 3x", "nearest 4x"],)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, method):
        out = []
        for rgb, a in core.split_tensor(image):
            if method == "scale2x":
                rgb, a = core.scale2x(rgb, a)
            elif method == "scale3x":
                rgb, a = core.scale3x(rgb, a)
            elif method.startswith("scale4x"):
                rgb, a = core.scale2x(*core.scale2x(rgb, a))
            else:
                f = int(method.split()[1][0])
                rgb = np.repeat(np.repeat(rgb, f, 0), f, 1)
                a = None if a is None else np.repeat(np.repeat(a, f, 0), f, 1)
            out.append((rgb, a))
        return (core.join_tensor(out),)


class SK_ColorRamp:
    """Hue-shifted colour ramps (shadows toward blue/violet, highlights toward yellow) from base colours."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_colors": ("STRING", {"default": "#c0392b #2e86c1 #27ae60", "tooltip": "One ramp per colour."}),
            "steps": ("INT", {"default": 6, "min": 2, "max": 16}),
            "hue_shift": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0}),
            "light_range": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05}),
            "saturation_curve": ("FLOAT", {"default": 0.3, "min": -1.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("palette", "hex")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, base_colors, steps, hue_shift, light_range, saturation_curve):
        bases = core.parse_hex_palette(base_colors)
        if bases is None:
            raise ValueError("base_colors has no #rrggbb colours")
        rows = [core.color_ramp(b, steps, hue_shift, light_range, saturation_curve) for b in bases]
        sw = 16
        img = np.zeros((len(rows) * sw, steps * sw, 3), np.uint8)
        for r, ramp in enumerate(rows):
            for c, col in enumerate(ramp):
                img[r * sw:(r + 1) * sw, c * sw:(c + 1) * sw] = col
        return core.join_tensor([(img, None)]), core.palette_to_hex(np.concatenate(rows))


class SK_PaletteSwap:
    """Exact colour replacement: colour i of `from_colors` -> colour i of `to_colors` (team colours,
    armour tiers, enemy variants). Optional tolerance catches near shades."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "from_colors": ("STRING", {"default": "#a22633 #e43b44"}),
            "to_colors": ("STRING", {"default": "#124e89 #0099db"}),
            "tolerance": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.3, "step": 0.005}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, from_colors, to_colors, tolerance):
        src, dst = core.parse_hex_palette(from_colors), core.parse_hex_palette(to_colors)
        if src is None or dst is None or len(src) != len(dst):
            raise ValueError("from_colors and to_colors need the same number of #rrggbb colours")
        ls = core.srgb_to_oklab(src)
        out = []
        for rgb, a in core.split_tensor(image):
            lab = core.srgb_to_oklab(rgb)
            d = np.sqrt(((lab[..., None, :] - ls[None, None]) ** 2).sum(-1))
            k = d.argmin(-1)
            hit = d.min(-1) <= tolerance + 1e-6
            out.append((np.where(hit[..., None], dst[k], rgb), a))
        return (core.join_tensor(out),)


class SK_HueShift:
    """Recolour in OKLab: rotate hue, scale chroma, shift lightness (whole sprite or a mask)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "hue_degrees": ("FLOAT", {"default": 120.0, "min": -180.0, "max": 180.0, "step": 1.0}),
            "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            "lightness": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
        }, "optional": {"mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, image, hue_degrees, saturation, lightness, mask=None):
        out = []
        for i, (rgb, a) in enumerate(core.split_tensor(image)):
            m = None if mask is None else core.mask_to_np(mask, i, rgb.shape[:2]) >= 0.5
            out.append((core.hue_shift_image(rgb, hue_degrees, saturation, lightness, m), a))
        return (core.join_tensor(out),)


NODES = {
    "SK_PixelPerfect": (SK_PixelPerfect, "Pixel Perfect (unfake / downscale)"),
    "SK_Palette": (SK_Palette, "Palette (quantize / remap / dither)"),
    "SK_Cleanup": (SK_Cleanup, "Pixel Cleanup + Outline"),
    "SK_BackgroundRemove": (SK_BackgroundRemove, "Background → Transparent"),
    "SK_ScaleNearest": (SK_ScaleNearest, "Scale Nearest (integer)"),
    "SK_ToCanvas": (SK_ToCanvas, "Sprite → Model Canvas"),
    "SK_Outpaint": (SK_Outpaint, "Expand Canvas (outpaint)"),
    "SK_RegionMask": (SK_RegionMask, "Region Mask (art pixels)"),
    "SK_MaskedMerge": (SK_MaskedMerge, "Masked Merge"),
    "SK_PixelUpscale": (SK_PixelUpscale, "Pixel Upscale (Scale2x/3x)"),
    "SK_ColorRamp": (SK_ColorRamp, "Colour Ramp (hue-shifted)"),
    "SK_PaletteSwap": (SK_PaletteSwap, "Palette Swap"),
    "SK_HueShift": (SK_HueShift, "Hue Shift / Recolour"),
}
