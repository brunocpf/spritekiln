"""Tests for the SpriteKiln pixel algorithms (no models needed).

Run with ComfyUI's Python (needs numpy, scipy, torch):
    ~/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python tests/test_core.py      (or: -m pytest tests)
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "ComfyUI-SpriteKiln")
spec = importlib.util.spec_from_file_location("sk_core", os.path.join(PKG, "sk_core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

PAL = np.array([[20, 12, 28], [200, 60, 60], [240, 200, 120], [60, 120, 200], [255, 255, 255]], np.uint8)


def blocky(n=32, seed=0):
    return PAL[np.random.default_rng(seed).integers(0, len(PAL), (n, n))]


def upscale(img, f):
    idx = (np.arange(int(img.shape[0] * f)) / f).astype(int)
    return img[idx][:, idx]


def test_oklab_roundtrip():
    rgb = np.random.default_rng(1).integers(0, 256, (64, 3)).astype(np.uint8)
    back = core.oklab_to_srgb(core.srgb_to_oklab(rgb))
    assert np.abs(back.astype(int) - rgb).max() <= 1


def test_detect_fractional_grid():
    for f in (8.0, 15.625, 21.3):
        sx, ox, sy, oy = core.detect_pixel_scale(upscale(blocky(), f))
        assert abs(sx - f) < 0.15 and abs(sy - f) < 0.15, (f, sx, sy)


def test_downscale_recovers_art():
    art = blocky(32)
    big = upscale(art, 16).astype(np.float32) + np.random.default_rng(2).normal(0, 5, (512, 512, 3))
    big = np.clip(big, 0, 255).astype(np.uint8)
    out, _ = core.downscale(big, None, 32, 32, "dominant")
    assert (np.abs(out.astype(int) - art).max(-1) < 10).mean() > 0.99


def test_quantize_exact_palette_colours():
    art = blocky(16)
    assert (core.quantize(art, PAL) == art).all()


def test_floyd_steinberg_respects_alpha():
    rgb = np.full((8, 8, 3), 128, np.uint8)
    rgb[:, :4] = PAL[1]                       # opaque left half, exact palette colour
    alpha = np.zeros((8, 8), np.float32)
    alpha[:, :4] = 1
    q = core.quantize(rgb, PAL, "floyd-steinberg", 1.0, alpha)
    assert (q[:, :4] == PAL[1]).all(), "error from transparent pixels leaked into the sprite"


def test_specks_keep_contrast_detail_and_dither():
    img = np.zeros((9, 9, 3), np.uint8)
    img[:] = (90, 60, 40)
    img[4, 4] = (255, 255, 255)               # 1-px highlight / eye: high contrast -> keep
    img[1, 1] = (94, 63, 42)                  # near-shade noise -> merge
    out, _ = core.remove_specks(img, None, 2, 0.12)
    assert tuple(out[4, 4]) == (255, 255, 255)
    assert tuple(out[1, 1]) == (90, 60, 40)
    checker = np.where(((np.indices((8, 8)).sum(0)) % 2)[..., None] == 1, PAL[0], PAL[4]).astype(np.uint8)
    out2, _ = core.remove_specks(checker, None, 2, 0.12)
    assert (out2 == checker).all(), "dither pattern was destroyed"


def test_specks_cluster_size():
    img = np.full((10, 10, 3), 100, np.uint8)
    img[2, 2:4] = 104                          # 2-px near-shade cluster
    out, _ = core.remove_specks(img, None, 2, 0.12)
    assert (out[2, 2:4] == 100).all()
    out1, _ = core.remove_specks(img, None, 1, 0.12)
    assert (out1[2, 2:4] == 104).all()


def test_parse_palettes():
    assert core.parse_hex_palette("#ff0000 00ff00, #0000FF").tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    paintnet = ";paint.net Palette File\n;comment ffffffff\nFF1A1C2C\nFF5D275D\n"
    assert core.parse_hex_palette(paintnet).tolist() == [[26, 28, 44], [93, 39, 93]]
    assert core.parse_hex_palette("no colours here") is None


def test_scale2x_no_new_colours():
    art = blocky(16)
    up, _ = core.scale2x(art)
    assert up.shape == (32, 32, 3)
    assert {tuple(c) for c in up.reshape(-1, 3)} <= {tuple(c) for c in art.reshape(-1, 3)}
    diag = np.zeros((4, 4, 3), np.uint8)
    diag[np.arange(4), np.arange(4)] = 255   # staircase gets rounded
    up, _ = core.scale2x(diag)
    assert up[1, 2].tolist() == [255, 255, 255] or up[2, 1].tolist() == [255, 255, 255]
    up3, _ = core.scale3x(art)
    assert up3.shape == (48, 48, 3)


def test_color_ramp_is_ordered():
    ramp = core.color_ramp((180, 60, 60), steps=6)
    L = core.srgb_to_oklab(ramp)[:, 0]
    assert (np.diff(L) > 0).all()


def test_key_background_ignores_grid_lines():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:] = (200, 40, 40)                     # red backdrop
    img[:, :3] = 0                             # black grid line on two edges
    img[:3, :] = 0
    img[24:40, 24:40] = (60, 120, 200)         # subject
    alpha, _ = core.key_background(img, 0.1)
    assert alpha[30, 30] == 1 and alpha[50, 50] == 0 and alpha[1, 30] == 0


def test_refine_background_holes_halo_shadow():
    img = np.full((200, 200, 3), 255, np.uint8)            # white backdrop
    img[40:160, 60:140] = (40, 60, 160)                     # body
    img[100:160, 90:110] = 255                              # enclosed gap between the legs (big) -> remove
    img[60:63, 80:83] = 255                                 # 3x3 eye highlight (small) -> keep
    img[160:166, 50:150] = (170, 170, 170)                  # grey ground shadow under the feet -> remove
    img[39, 60:140] = (235, 235, 240)                       # anti-aliased halo on the top edge -> remove
    a = core.refine_background(img, None, 0.08)
    assert a[120, 100] == 0, "enclosed gap kept"
    assert a[61, 81] == 1, "small highlight removed"
    assert a[163, 100] == 0, "ground shadow kept"
    assert a[39, 100] == 0, "halo kept"
    assert a[80, 70] == 1, "body removed"


def test_refine_background_keeps_teeth_when_mask_says_foreground():
    img = np.full((400, 400, 3), 255, np.uint8)
    img[40:380, 100:300] = (230, 170, 120)                 # character
    img[100:106, 175:225] = 254                             # teeth: enclosed white, 300 px (0.44% of the character)
    img[300:380, 195:205] = 255                             # gap between the legs, 800 px (1.2%)
    soft = np.zeros((400, 400), np.float32)
    soft[40:380, 100:300] = 1.0                             # segmentation: whole body incl. teeth...
    soft[300:380, 195:205] = 0.0                            # ...but not the leg gap
    a = core.refine_background(img, soft, 0.08)
    assert a[103, 200] == 1, "teeth removed"
    assert a[340, 200] == 0, "leg gap kept"


def test_normals_disc_points_outward():
    yy, xx = np.indices((64, 64))
    alpha = (((yy - 32) ** 2 + (xx - 32) ** 2) <= 28 ** 2).astype(np.float32)
    rgb = np.full((64, 64, 3), 128, np.uint8)
    h, m = core.sprite_height(rgb, alpha, inflate=1.0, detail=0.0)
    n = core.height_to_normals(h, m, 1.0, "opengl")
    assert n[32, 6, 0] < -0.3 and n[32, 58, 0] > 0.3, "left/right rims should face left/right"
    assert n[6, 32, 1] > 0.3 and n[58, 32, 1] < -0.3, "OpenGL: top rim faces up (+Y)"
    assert n[32, 32, 2] > 0.95, "centre faces the viewer"
    assert (n[~m] == (0, 0, 1)).all()
    nd = core.height_to_normals(h, m, 1.0, "directx")
    assert nd[6, 32, 1] < -0.3, "DirectX flips green"
    enc = core.encode_normals(n)
    assert np.abs(core.decode_normals(enc) - n).max() < 0.02


def run_all():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except AssertionError as e:
            failed += 1
            print("FAIL", name, e)
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
