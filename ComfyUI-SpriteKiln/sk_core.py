"""Pure numpy pixel-art algorithms used by the SpriteKiln nodes.

Images here are numpy arrays: RGB uint8 (H, W, 3) and alpha float32 (H, W) in 0..1.
"""
import numpy as np
import torch

# ---------------------------------------------------------------- tensors <-> numpy

def split_tensor(img):
    """ComfyUI IMAGE [B,H,W,C] -> list of (rgb uint8, alpha float32 or None)."""
    out = []
    arr = img.detach().cpu().float().clamp(0, 1).numpy()
    for frame in arr:
        rgb = np.round(frame[..., :3] * 255).astype(np.uint8)
        alpha = frame[..., 3].astype(np.float32) if frame.shape[-1] == 4 else None
        out.append((rgb, alpha))
    return out


def join_tensor(frames, force_alpha=False):
    """list of (rgb, alpha) -> IMAGE tensor. RGBA if any frame has alpha (or force_alpha)."""
    has_alpha = force_alpha or any(a is not None for _, a in frames)
    h = max(f[0].shape[0] for f in frames)
    w = max(f[0].shape[1] for f in frames)
    out = np.zeros((len(frames), h, w, 4 if has_alpha else 3), np.float32)
    for i, (rgb, a) in enumerate(frames):
        fh, fw = rgb.shape[:2]
        out[i, :fh, :fw, :3] = rgb.astype(np.float32) / 255.0
        if has_alpha:
            out[i, :fh, :fw, 3] = 1.0 if a is None else a
    return torch.from_numpy(out)


def mask_to_np(mask, index, shape):
    """ComfyUI MASK [B,H,W] (or [H,W]) -> float32 (H,W) resized (nearest) to shape."""
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    m = m[min(index, m.shape[0] - 1)].detach().cpu().float().numpy()
    if m.shape != tuple(shape):
        ys = (np.arange(shape[0]) * m.shape[0] / shape[0]).astype(int)
        xs = (np.arange(shape[1]) * m.shape[1] / shape[1]).astype(int)
        m = m[ys][:, xs]
    return m.astype(np.float32)

# ---------------------------------------------------------------- color science

def srgb_to_oklab(rgb):
    """rgb uint8/float (...,3) in 0..255 -> oklab float32 (...,3)."""
    c = np.asarray(rgb, np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    l = 0.4122214708 * c[..., 0] + 0.5363325363 * c[..., 1] + 0.0514459929 * c[..., 2]
    m = 0.2119034982 * c[..., 0] + 0.6806995451 * c[..., 1] + 0.1073969566 * c[..., 2]
    s = 0.0883024619 * c[..., 0] + 0.2817188376 * c[..., 1] + 0.6299787005 * c[..., 2]
    l, m, s = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ], -1).astype(np.float32)


def oklab_to_srgb(lab):
    lab = np.asarray(lab, np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    c = np.clip(np.stack([r, g, bb], -1), 0, 1)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def nearest_index(lab_pixels, lab_palette, chunk=262144):
    """Index of the nearest palette entry for each pixel (both in OKLab)."""
    flat = lab_pixels.reshape(-1, 3)
    out = np.empty(flat.shape[0], np.int32)
    for i in range(0, flat.shape[0], chunk):
        d = ((flat[i:i + chunk, None, :] - lab_palette[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out.reshape(lab_pixels.shape[:-1])

# ---------------------------------------------------------------- palettes

def unique_colors(rgb_list, alpha_list=None):
    """Unique colors + counts over several frames (only opaque pixels when alpha given)."""
    packs = []
    for i, rgb in enumerate(rgb_list):
        p = rgb.reshape(-1, 3).astype(np.uint32)
        if alpha_list is not None and alpha_list[i] is not None:
            p = p[alpha_list[i].reshape(-1) >= 0.5]
        packs.append((p[:, 0] << 16) | (p[:, 1] << 8) | p[:, 2])
    packed = np.concatenate(packs) if packs else np.zeros(0, np.uint32)
    if packed.size == 0:
        return np.zeros((1, 3), np.uint8), np.ones(1, np.float32)
    u, counts = np.unique(packed, return_counts=True)
    cols = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1).astype(np.uint8)
    return cols, counts.astype(np.float32)


def kmeans_palette(colors, weights, k, iters=16, seed=0, snap=True, max_samples=20000):
    """Weighted k-means++ in OKLab over unique colors. Returns (k',3) uint8 palette.

    snap=True returns the most common real colour of each cluster (keeps hand-picked colours exact);
    snap=False returns cluster centroids (better for noisy model output).
    """
    k = int(max(1, min(k, len(colors))))
    if len(colors) <= k:
        return colors.copy()
    rng = np.random.default_rng(seed)
    if len(colors) > max_samples:
        keep = rng.choice(len(colors), max_samples, replace=False, p=weights / weights.sum())
        colors, weights = colors[keep], weights[keep]
    lab = srgb_to_oklab(colors)
    w = weights / weights.sum()
    centers = [lab[rng.choice(len(lab), p=w)]]
    d2 = ((lab - centers[0]) ** 2).sum(-1)
    for _ in range(1, k):
        p = d2 * w
        if p.sum() <= 0:
            break
        c = lab[rng.choice(len(lab), p=p / p.sum())]
        centers.append(c)
        d2 = np.minimum(d2, ((lab - c) ** 2).sum(-1))
    centers = np.array(centers, np.float32)
    for _ in range(iters):
        idx = nearest_index(lab, centers)
        new = np.zeros_like(centers)
        tot = np.zeros(len(centers), np.float32)
        np.add.at(new, idx, lab * weights[:, None])
        np.add.at(tot, idx, weights)
        empty = tot == 0
        new[~empty] /= tot[~empty, None]
        new[empty] = centers[empty]
        if np.abs(new - centers).max() < 1e-4:
            centers = new
            break
        centers = new
    if not snap:
        return np.unique(oklab_to_srgb(centers), axis=0)
    # snap each center to the most common real color of its cluster: keeps hues crisp
    idx = nearest_index(lab, centers)
    pal = []
    for c in range(len(centers)):
        members = np.where(idx == c)[0]
        if len(members):
            pal.append(colors[members[np.argmax(weights[members])]])
    pal = np.unique(np.array(pal, np.uint8), axis=0)
    return pal


def sort_palette(pal):
    lab = srgb_to_oklab(pal)
    return pal[np.argsort(lab[:, 0])]


def palette_to_hex(pal):
    return " ".join("#%02x%02x%02x" % tuple(int(v) for v in c) for c in pal)


def parse_hex_palette(text):
    """'#rrggbb' / 'rrggbb' tokens (Lospec .hex, CSS lists) and Paint.NET 'AARRGGBB' lines.
    Lines starting with ';' are comments. Returns None when no colour is found."""
    import re
    cols = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        for tok in re.findall(r"#?\b([0-9a-fA-F]{8}|[0-9a-fA-F]{6})\b", line):
            h = tok[2:] if len(tok) == 8 else tok
            cols.append([int(h[i:i + 2], 16) for i in (0, 2, 4)])
    return np.array(cols, np.uint8) if cols else None


def palette_strip(pal, swatch=16):
    """Palette -> swatch image (rgb uint8), max 16 per row."""
    n = len(pal)
    cols = min(16, n)
    rows = (n + cols - 1) // cols
    img = np.zeros((rows * swatch, cols * swatch, 3), np.uint8)
    for i, c in enumerate(pal):
        r, q = divmod(i, cols)
        img[r * swatch:(r + 1) * swatch, q * swatch:(q + 1) * swatch] = c
    return img

# ---------------------------------------------------------------- quantize / dither

BAYER2 = np.array([[0, 2], [3, 1]], np.float32) / 4.0


def bayer(n):
    m = BAYER2 * 4
    size = 2
    while size < n:
        m = np.block([[4 * m + 0, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
        size *= 2
    return (m + 0.5) / (size * size) - 0.5


def quantize(rgb, pal, dither="none", strength=1.0, alpha=None):
    """Map rgb (H,W,3) uint8 onto palette (K,3) uint8.

    Floyd-Steinberg skips transparent pixels (alpha < 0.5) and never pushes error into them.
    Prefer Bayer for animations: its pattern is fixed in screen space, so it does not shimmer."""
    lab_pal = srgb_to_oklab(pal)
    if dither == "none" or strength <= 0:
        return pal[nearest_index(srgb_to_oklab(rgb), lab_pal)]
    if dither.startswith("bayer"):
        n = int(dither.split("-")[1])
        m = bayer(n)
        h, w = rgb.shape[:2]
        tile = np.tile(m, (h // n + 1, w // n + 1))[:h, :w]
        # spread ~ typical distance between palette colors
        spread = 255.0 / max(2.0, np.cbrt(len(pal))) * strength
        shifted = np.clip(rgb.astype(np.float32) + tile[..., None] * spread, 0, 255)
        return pal[nearest_index(srgb_to_oklab(shifted), lab_pal)]
    # floyd-steinberg (serpentine) in sRGB space
    h, w = rgb.shape[:2]
    buf = rgb.astype(np.float32).copy()
    out = np.zeros_like(rgb)
    palf = pal.astype(np.float32)
    opaque = np.ones((h, w), bool) if alpha is None else alpha >= 0.5
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        d = 1 if y % 2 == 0 else -1
        for x in xs:
            if not opaque[y, x]:
                continue
            old = np.clip(buf[y, x], 0, 255)
            i = int(((srgb_to_oklab(old)[None] - lab_pal) ** 2).sum(-1).argmin())
            out[y, x] = pal[i]
            err = (old - palf[i]) * strength
            for yy, xx, wgt in ((y, x + d, 7), (y + 1, x - d, 3), (y + 1, x, 5), (y + 1, x + d, 1)):
                if 0 <= xx < w and yy < h and opaque[yy, xx]:
                    buf[yy, xx] += err * wgt / 16
    return out

# ---------------------------------------------------------------- grid detection / downscale

def _edge_profile(rgb, axis):
    f = rgb.astype(np.float32)
    d = np.abs(np.diff(f, axis=axis)).sum(-1)
    # axis=1: edges between columns, averaged over rows; axis=0: between rows
    return d.mean(0) if axis == 1 else d.mean(1)


def detect_pixel_scale(rgb, max_scale=64, min_scale=2.0):
    """Estimate the size of one 'art pixel' in a fake-pixel-art image (fractional periods allowed).

    Returns (scale_x, offset_x, scale_y, offset_y): grid lines sit at offset + k * scale.
    """
    res = []
    for axis in (1, 0):  # x then y
        prof = _edge_profile(rgb, axis)  # prof[i] = edge strength at boundary i+1
        n = len(prof)
        pos = np.arange(n, dtype=np.float64) + 1.0
        w = prof.astype(np.float64)
        base = w.mean() + 1e-9
        periods = np.arange(min_scale, min(max_scale, n / 4) + 1e-9, 0.02)
        scores, offsets = [], []
        for p in periods:
            ang = 2 * np.pi * pos / p
            phase = np.angle((w * np.exp(1j * ang)).sum())
            off = (phase / (2 * np.pi) * p) % p
            pred = np.arange(off, n + 1, p)
            i0 = np.clip(np.floor(pred).astype(int) - 1, 0, n - 1)
            i1 = np.clip(i0 + 1, 0, n - 1)
            scores.append(np.maximum(w[i0], w[i1]).mean() / base)
            offsets.append(off)
        scores = np.array(scores)
        if len(scores) == 0 or scores.max() < 1.5:
            res += [1.0, 0.0]
            continue
        top = scores.max()
        # multiples of the true period score as high as the period itself: take the smallest
        good = np.where(scores >= 0.85 * top)[0]
        first = good[0]
        # within that first cluster of good periods take the local best
        j = first
        while j + 1 < len(scores) and j + 1 in good and scores[j + 1] >= scores[j]:
            j += 1
        res += [float(periods[j]), float(offsets[j])]
    return res[0], res[1], res[2], res[3]


def downscale(rgb, alpha, out_w, out_h, method="dominant", box=None, pre_colors=64, seed=0):
    """Downscale a hi-res (fake) pixel-art image to exactly out_w x out_h art pixels.

    box = (x0, y0, x1, y1) region of the source covering the art grid (default: full image).
    Methods: dominant (mode of cell after k-means), median, mean, nearest (cell center).
    """
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box if box else (0, 0, W, H)
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    fx = (xs + 0.5 - x0) / (x1 - x0) * out_w
    fy = (ys + 0.5 - y0) / (y1 - y0) * out_h
    cx = np.floor(fx).astype(np.int64)
    cy = np.floor(fy).astype(np.int64)
    validx = (cx >= 0) & (cx < out_w)
    validy = (cy >= 0) & (cy < out_h)
    # weight pixels near the centre of their cell higher (avoids anti-aliased borders)
    wx = np.where(np.abs(fx - cx - 0.5) < 0.3, 1.0, 0.2).astype(np.float32)
    wy = np.where(np.abs(fy - cy - 0.5) < 0.3, 1.0, 0.2).astype(np.float32)
    cell = (cy[:, None] * out_w + cx[None, :])
    valid = validy[:, None] & validx[None, :]
    weight = (wy[:, None] * wx[None, :]) * valid
    ncell = out_w * out_h
    cell_f = np.where(valid, cell, 0).reshape(-1)
    weight_f = weight.reshape(-1)

    out_alpha = None
    opaque = np.ones(H * W, bool)
    if alpha is not None:
        a = alpha.reshape(-1)
        asum = np.bincount(cell_f, weights=weight_f * a, minlength=ncell)
        wsum = np.bincount(cell_f, weights=weight_f, minlength=ncell) + 1e-6
        out_alpha = (asum / wsum >= 0.5).astype(np.float32).reshape(out_h, out_w)
        opaque = a >= 0.5

    flat = rgb.reshape(-1, 3)
    if method == "nearest":
        sx = np.clip(((np.arange(out_w) + 0.5) / out_w * (x1 - x0) + x0).astype(int), 0, W - 1)
        sy = np.clip(((np.arange(out_h) + 0.5) / out_h * (y1 - y0) + y0).astype(int), 0, H - 1)
        out = rgb[sy][:, sx]
    elif method in ("mean", "median"):
        out = np.zeros((ncell, 3), np.float32)
        wv = weight_f * opaque
        if method == "mean":
            tot = np.bincount(cell_f, weights=wv, minlength=ncell) + 1e-6
            for c in range(3):
                out[:, c] = np.bincount(cell_f, weights=wv * flat[:, c], minlength=ncell) / tot
        else:
            centre = wv > 0.5
            has_centre = np.bincount(cell_f[centre], minlength=ncell) > 0
            sel = (wv > 0) & (centre | ~has_centre[cell_f])
            ids = cell_f[sel]
            for c in range(3):
                vals = flat[sel, c].astype(np.float32)
                o = np.lexsort((vals, ids))
                ids_s, vals_s = ids[o], vals[o]
                starts = np.searchsorted(ids_s, np.arange(ncell))
                ends = np.searchsorted(ids_s, np.arange(ncell), side="right")
                mid = np.clip((starts + ends) // 2, 0, max(0, len(vals_s) - 1))
                v = vals_s[mid] if len(vals_s) else np.zeros(ncell)
                out[:, c] = np.where(ends > starts, v, 0)
        out = np.clip(np.round(out), 0, 255).astype(np.uint8).reshape(out_h, out_w, 3)
    else:  # dominant
        sel = opaque & (weight_f > 0)
        cols, counts = unique_colors([flat[sel].reshape(-1, 1, 3)])
        pal = kmeans_palette(cols, counts, pre_colors, seed=seed, snap=False)
        idx = nearest_index(srgb_to_oklab(flat), srgb_to_oklab(pal)).reshape(-1)
        k = len(pal)
        votes = np.bincount(cell_f * k + idx, weights=weight_f * sel, minlength=ncell * k).reshape(ncell, k)
        win = votes.argmax(1)
        out = pal[win].reshape(out_h, out_w, 3)
    return out, out_alpha

# ---------------------------------------------------------------- cleanup / outline

def _neighbors(arr):
    p = np.pad(arr, 1, mode="edge")
    return [p[1 + dy:1 + dy + arr.shape[0], 1 + dx:1 + dx + arr.shape[1]]
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]


def remove_specks(rgb, alpha, max_size=2, max_contrast=0.12):
    """Merge small same-colour clusters (<= max_size px) into the surrounding majority colour,
    but only when they are close to it (OKLab distance <= max_contrast).

    High-contrast single pixels (eyes, highlights, sparkles) and dither patterns between
    clearly different colours are kept; near-shade noise left by downscaling is removed.
    Isolated opaque specks floating in transparency are removed when they are <= max_size."""
    from scipy import ndimage
    rgb = rgb.copy()
    a = None if alpha is None else alpha.copy()
    opaque = np.ones(rgb.shape[:2], bool) if a is None else a >= 0.5
    packed = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
    lab = srgb_to_oklab(rgb)
    H, W = packed.shape
    st = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for col in np.unique(packed[opaque]):
        lbl, n = ndimage.label((packed == col) & opaque, structure=st)
        if n == 0:
            continue
        sizes = np.bincount(lbl.ravel())
        for k in np.where(sizes[1:] <= max_size)[0] + 1:
            ys, xs = np.where(lbl == k)
            ring = ndimage.binary_dilation(lbl == k, structure=st) & (lbl != k)
            ring_op = ring & opaque
            if not ring_op.any():
                if a is not None:
                    a[ys, xs] = 0.0  # floating speck in transparency
                continue
            vals, cnt = np.unique(packed[ring_op], return_counts=True)
            maj = vals[cnt.argmax()]
            if cnt.max() < 0.6 * ring_op.sum():
                continue  # no clear surrounding colour: probably intentional detail
            my, mx = np.argwhere(ring_op & (packed == maj))[0]
            if np.sqrt(((lab[ys[0], xs[0]] - lab[my, mx]) ** 2).sum()) <= max_contrast:
                packed[ys, xs] = maj
                rgb[ys, xs] = rgb[my, mx]
    return rgb, a


def remove_orphans(rgb, alpha, passes=1):
    """Backward-compatible wrapper (single-pixel, contrast-aware)."""
    for _ in range(passes):
        rgb, alpha = remove_specks(rgb, alpha, 1, 0.12)
    return rgb, alpha


def add_outline(rgb, alpha, color=None, mode="outer", diagonal=False, darken=0.45):
    """1px outline around opaque area. color None = auto (darkened neighbour color)."""
    if alpha is None:
        return rgb, alpha
    a = alpha >= 0.5
    p = np.pad(a, 1)
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    nb_any = np.zeros_like(a)
    for dy, dx in offs:
        nb_any |= p[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]]
    out = rgb.copy()
    new_a = alpha.copy()
    if mode == "outer":
        edge = (~a) & nb_any
        if color is None:
            # average of opaque neighbour colors, darkened
            acc = np.zeros(rgb.shape, np.float32)
            cnt = np.zeros(a.shape, np.float32)
            prgb = np.pad(rgb.astype(np.float32), ((1, 1), (1, 1), (0, 0)))
            for dy, dx in offs:
                m = p[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]]
                acc += prgb[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]] * m[..., None]
                cnt += m
            col = (acc / np.maximum(cnt, 1)[..., None]) * darken
            out[edge] = np.clip(col[edge], 0, 255).astype(np.uint8)
        else:
            out[edge] = color
        new_a[edge] = 1.0
    else:  # inner: recolor the opaque border pixels
        inner = np.ones_like(a)
        for dy, dx in offs:
            inner &= p[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]]
        edge = a & ~inner
        if color is None:
            out[edge] = (rgb[edge].astype(np.float32) * darken).astype(np.uint8)
        else:
            out[edge] = color
    return out, new_a


def key_background(rgb, tolerance=0.08, mode="border", min_share=0.15):
    """Alpha from flood-filling background colours connected to the image border.

    Every colour that covers at least `min_share` of the border (after merging similar shades)
    counts as background, so grid lines or a frame around the sprite do not hide the real backdrop."""
    from scipy import ndimage
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).reshape(-1, 1, 3)
    cols, counts = unique_colors([border])
    pal = kmeans_palette(cols, counts, 6, snap=False)
    bl = srgb_to_oklab(border.reshape(-1, 3))
    idx = nearest_index(bl, srgb_to_oklab(pal))
    share = np.bincount(idx, minlength=len(pal)) / len(idx)
    bgs = [pal[i] for i in np.argsort(-share) if share[i] >= min_share] or [pal[int(share.argmax())]]
    lab = srgb_to_oklab(rgb)
    bgmask = np.zeros(rgb.shape[:2], bool)
    for bg in bgs:
        similar = np.sqrt(((lab - srgb_to_oklab(bg)) ** 2).sum(-1)) <= tolerance
        if mode == "global":
            bgmask |= similar
        else:
            lbl, _ = ndimage.label(similar)
            edge_labels = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
            bgmask |= np.isin(lbl, edge_labels[edge_labels > 0])
    return (~bgmask).astype(np.float32), bgs[0]


def trim_box(alpha, pad=0):
    ys, xs = np.where(alpha >= 0.5)
    if len(ys) == 0:
        return 0, 0, alpha.shape[1], alpha.shape[0]
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(alpha.shape[1], xs.max() + 1 + pad), min(alpha.shape[0], ys.max() + 1 + pad))


# ---------------------------------------------------------------- pixel-art upscaling (EPX / Scale2x)

def scale2x(rgb, alpha=None):
    """Scale2x/EPX: doubles resolution, rounding diagonal staircases without new colours."""
    packed = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
    if alpha is not None:
        packed = np.where(alpha >= 0.5, packed, -1)
    p = np.pad(packed, 1, mode="edge")
    E = p[1:-1, 1:-1]
    B, D, F, Hh = p[:-2, 1:-1], p[1:-1, :-2], p[1:-1, 2:], p[2:, 1:-1]
    ok = (B != Hh) & (D != F)
    e0 = np.where(ok & (D == B), D, E)
    e1 = np.where(ok & (B == F), F, E)
    e2 = np.where(ok & (D == Hh), D, E)
    e3 = np.where(ok & (Hh == F), F, E)
    h, w = E.shape
    out = np.empty((h * 2, w * 2), np.int64)
    out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2] = e0, e1, e2, e3
    a2 = None if alpha is None else (out >= 0).astype(np.float32)
    out = np.maximum(out, 0)
    rgb2 = np.stack([(out >> 16) & 255, (out >> 8) & 255, out & 255], -1).astype(np.uint8)
    return rgb2, a2


def scale3x(rgb, alpha=None):
    """Scale3x (AdvMAME3x)."""
    packed = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
    if alpha is not None:
        packed = np.where(alpha >= 0.5, packed, -1)
    p = np.pad(packed, 1, mode="edge")
    A, B, C = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    D, E, F = p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:]
    G, Hh, I = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    ok = (B != Hh) & (D != F)
    o = [None] * 9
    o[0] = np.where(ok & (D == B), D, E)
    o[1] = np.where(ok & (((D == B) & (E != C)) | ((B == F) & (E != A))), B, E)
    o[2] = np.where(ok & (B == F), F, E)
    o[3] = np.where(ok & (((D == B) & (E != G)) | ((D == Hh) & (E != A))), D, E)
    o[4] = E
    o[5] = np.where(ok & (((B == F) & (E != I)) | ((Hh == F) & (E != C))), F, E)
    o[6] = np.where(ok & (D == Hh), D, E)
    o[7] = np.where(ok & (((D == Hh) & (E != I)) | ((Hh == F) & (E != G))), Hh, E)
    o[8] = np.where(ok & (Hh == F), F, E)
    h, w = E.shape
    out = np.empty((h * 3, w * 3), np.int64)
    for k in range(9):
        out[k // 3::3, k % 3::3] = o[k]
    a3 = None if alpha is None else (out >= 0).astype(np.float32)
    out = np.maximum(out, 0)
    return np.stack([(out >> 16) & 255, (out >> 8) & 255, out & 255], -1).astype(np.uint8), a3


# ---------------------------------------------------------------- colour ramps / hue shift

def color_ramp(base, steps=7, hue_shift=20.0, light_range=0.55, sat_curve=0.25):
    """Pixel-art ramp around `base` (rgb): darker shades shift hue toward blue/purple, lighter
    shades toward yellow (classic hue-shifted ramps). hue_shift = degrees at the ends."""
    lab = srgb_to_oklab(np.array(base, np.float32))
    L, a, b = lab
    C, h = np.hypot(a, b), np.arctan2(b, a)
    out = []
    for i in range(steps):
        t = i / max(1, steps - 1) * 2 - 1           # -1 (dark) .. +1 (light)
        Li = np.clip(L + t * light_range / 2, 0.05, 0.98)
        target = np.deg2rad(265.0 if t < 0 else 95.0)  # OKLab hue of blue-violet / yellow
        d = (target - h + np.pi) % (2 * np.pi) - np.pi
        hi = h + np.sign(d) * min(abs(d), np.deg2rad(abs(hue_shift) * abs(t)))
        Ci = C * (1 - sat_curve * t * t) + (0.02 * (1 - abs(t)))
        out.append(oklab_to_srgb(np.array([Li, Ci * np.cos(hi), Ci * np.sin(hi)], np.float32)))
    return np.array(out, np.uint8)


def hue_shift_image(rgb, degrees=0.0, sat=1.0, light=0.0, mask=None):
    lab = srgb_to_oklab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C, h = np.hypot(a, b) * sat, np.arctan2(b, a) + np.deg2rad(degrees)
    new = oklab_to_srgb(np.stack([np.clip(L + light, 0, 1), C * np.cos(h), C * np.sin(h)], -1))
    if mask is not None:
        new = np.where(mask[..., None], new, rgb)
    return new


def refine_background(rgb, fg_mask=None, tolerance=0.08, hole_frac=0.01, fringe=3, shadow=True):
    """Clean a foreground mask (e.g. BiRefNet) with the known backdrop colour:
    - background-coloured regions touching the border are removed (classic key),
    - enclosed background-coloured regions are removed when the segmentation mask says "background" there,
      or when they are large (>= hole_frac of the character's bounding box: gaps between legs/arms).
      Small enclosed regions the mask calls foreground (teeth, eye whites, highlights) are kept,
    - background-tinted pixels within `fringe` px of the silhouette edge are removed (anti-alias halo)."""
    from scipy import ndimage
    H, W = rgb.shape[:2]
    keyed, bg = key_background(rgb, tolerance, "border")
    lab = srgb_to_oklab(rgb)
    dist = np.sqrt(((lab - srgb_to_oklab(bg)) ** 2).sum(-1))
    near = dist <= tolerance
    fg = keyed > 0
    if fg.any():
        ys, xs = np.where(fg)
        char_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        lbl, n = ndimage.label(near & fg)
        if n:
            sizes = ndimage.sum(near & fg, lbl, range(1, n + 1))
            fg &= ~np.isin(lbl, np.where(sizes >= hole_frac * char_area)[0] + 1)
    if fg_mask is not None:
        fg &= fg_mask >= 0.5
    if fringe:
        edge = fg & ~ndimage.binary_erosion(fg, iterations=fringe)
        fg &= ~(edge & (dist <= tolerance * 2.5))
    if shadow and fg.any():
        # ground shadow: grey (low chroma), darker than the backdrop, in the bottom band of the silhouette,
        # touching the background
        L = lab[..., 0]
        C = np.hypot(lab[..., 1], lab[..., 2])
        bgl = srgb_to_oklab(bg)
        ys = np.where(fg.any(1))[0]
        band = np.zeros_like(fg)
        band[max(ys[-1] - int(0.15 * (ys[-1] - ys[0] + 1)), 0):ys[-1] + 1] = True
        cand = fg & band & (C < 0.04 + np.hypot(bgl[1], bgl[2])) & (L < bgl[0] - 0.02) & (L > 0.25)
        lbl, n = ndimage.label(cand)
        if n:
            touching = np.unique(lbl[ndimage.binary_dilation(~fg) & cand])
            fg &= ~np.isin(lbl, touching[touching > 0])
    # drop tiny floating islands left behind
    lbl, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(fg, lbl, range(1, n + 1))
        fg &= np.isin(lbl, np.where(sizes >= max(16, 0.0002 * H * W))[0] + 1)
    return fg.astype(np.float32)


# ---------------------------------------------------------------- sprite normal maps

def sprite_height(rgb, alpha, inflate=1.0, radius=None, detail=0.35, detail_blur=0.0, invert_detail=False):
    """Height field for a sprite: a rounded 'pillow' from the silhouette (distance to the alpha edge, circular
    profile) plus fine relief from luminance. Works per island, so whole spritesheets can be processed at once."""
    from scipy import ndimage
    a = np.ones(rgb.shape[:2], bool) if alpha is None else alpha >= 0.5
    d = ndimage.distance_transform_edt(a)
    if radius is None or radius <= 0:
        lbl, n = ndimage.label(a)
        # per-island radius: a third of the island's thickness, so thin limbs and big bodies both round off
        radius_map = np.zeros(a.shape, np.float32)
        if n:
            mx = ndimage.maximum(d, lbl, range(1, n + 1))
            radius_map = np.where(lbl > 0, np.asarray(mx, np.float32)[np.maximum(lbl - 1, 0)], 1.0)
        r = np.maximum(radius_map * 0.9, 1.0)
    else:
        r = float(radius)
    t = np.clip(d / r, 0, 1)
    dome = np.sqrt(1 - (1 - t) ** 2)          # quarter-circle profile: steep at the rim, flat on top
    lum = srgb_to_oklab(rgb)[..., 0]
    if detail_blur > 0:
        lum = ndimage.gaussian_filter(lum, detail_blur)
    if invert_detail:
        lum = 1 - lum
    lum = (lum - lum[a].mean()) if a.any() else lum
    h = inflate * dome * (r if np.ndim(r) == 0 else r) * 0.5 + detail * lum * 8.0
    if detail_blur > 0:
        # HD art: smooth the whole field inside the silhouette (normalised convolution) so the distance
        # transform's medial-axis ridges don't show up as facets
        w = ndimage.gaussian_filter(a.astype(np.float32), detail_blur * 2)
        h = ndimage.gaussian_filter(np.where(a, h, 0), detail_blur * 2) / np.maximum(w, 1e-3)
    return np.where(a, h, 0).astype(np.float32), a


def height_to_normals(h, mask, strength=1.0, convention="opengl"):
    """Tangent-space normals from a height field. Returns float32 (H,W,3) in -1..1 (x right, y up, z out)."""
    from scipy import ndimage
    gx = ndimage.sobel(h, axis=1) / 8.0
    gy = ndimage.sobel(h, axis=0) / 8.0       # +y = down in image space
    n = np.dstack([-gx * strength, gy * strength, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n[~mask] = (0, 0, 1)
    if convention == "directx":
        n[..., 1] *= -1
    return n.astype(np.float32)


def encode_normals(n):
    return np.clip(np.round((n + 1) * 0.5 * 255), 0, 255).astype(np.uint8)


def decode_normals(rgb):
    n = rgb.astype(np.float32) / 255.0 * 2 - 1
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


def shade(rgb, alpha, normals, light_dir, ambient=0.35, light_color=(1.0, 0.95, 0.85), convention="opengl"):
    """Lambert preview of albedo lit by a directional light (light_dir in x right, y up, z out)."""
    n = normals.copy()
    if convention == "directx":
        n[..., 1] *= -1
    l = np.asarray(light_dir, np.float32)
    l /= np.linalg.norm(l)
    ndl = np.clip((n * l).sum(-1), 0, 1)[..., None]
    lit = rgb.astype(np.float32) / 255 * (ambient + (1 - ambient) * ndl * np.asarray(light_color, np.float32))
    return np.clip(lit * 255, 0, 255).astype(np.uint8)
