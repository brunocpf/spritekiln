"""Generate the SpriteKiln workflow suite (UI + API JSON) from a live ComfyUI object_info.

    python3 build_workflows.py [--server http://127.0.0.1:8188] [--object-info file.json]
"""
import argparse
import json
import os
import urllib.request

from graph import Graph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ----------------------------------------------------------------------------- model files
Z_UNET, Z_CLIP, Z_VAE = "z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors"
Z_LORA = "spritekiln_zimage_pixel_art.safetensors"
K_UNET, K_BASE, K_CLIP, K_VAE = ("flux-2-klein-4b.safetensors", "flux-2-klein-base-4b.safetensors",
                                 "qwen_3_4b.safetensors", "flux2-vae.safetensors")
K_LORA = "spritekiln_klein_pixel_art.safetensors"
K_SHEET_LORA = "spritekiln_klein_base_4walk_sheet.safetensors"
K_2X2_LORA = "spritekiln_klein_2x2_multiview.safetensors"
W_CLIP = "umt5_xxl_fp16.safetensors"
W14_HIGH, W14_LOW, W14_VAE = ("wan2.2_i2v_high_noise_14B_Q4_K_M.gguf", "wan2.2_i2v_low_noise_14B_Q4_K_M.gguf",
                              "wan_2.1_vae.safetensors")
W14_LIGHT_H, W14_LIGHT_L = "wan2.2_i2v_lightning_4step_high.safetensors", "wan2.2_i2v_lightning_4step_low.safetensors"
W14_PIXEL = "spritekiln_wan22_pixel_animate.safetensors"
Q_UNET, Q_CLIP, Q_VAE = "qwen-image-edit-2511-Q4_K_M.gguf", "qwen_2.5_vl_7b.safetensors", "qwen_image_vae.safetensors"
Q_ANGLES, Q_LIGHT = ("qwen-image-edit-2511-multiple-angles-lora.safetensors",
                     "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
BIREFNET = "birefnet.safetensors"
SAM3 = "sam3.1_multiplex_fp16.safetensors"
EX_HD = "spritekiln_example_hd_character.png"   # HD painted character on a plain background

EX_SPRITE = "spritekiln_example_sprite.png"      # 64x64 RGBA knight
EX_REF = "spritekiln_example_reference.png"      # any image (photo / concept art)
EX_STYLE = "spritekiln_example_style.png"        # pixel art whose style should be copied
EX_SHEET = "spritekiln_example_sheet.png"        # 4x4 sprite sheet
EX_FAKE = "spritekiln_example_fake_pixel.png"    # blurry upscaled "AI pixel art"

SEED = 1234
WAN_NEG_TEXT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                "camera movement, zoom, background change, blurry, morphing, extra limbs, text, watermark")


# ----------------------------------------------------------------------------- shared blocks
def zimage_models(g, grp, lora=0.8):
    """lora: a float or a linked output (e.g. SK_StylePreset.lora_strength); None = no pixel LoRA (HD mode)."""
    unet = g.add("UNETLoader", grp, unet_name=Z_UNET, weight_dtype="default")
    m = unet
    if lora is not None:
        m = g.add("LoraLoaderModelOnly", grp, title="Pixel-art LoRA (Z-Image)", model=unet[0], lora_name=Z_LORA,
                  strength_model=lora)
    m = g.add("ModelSamplingAuraFlow", grp, model=m[0], shift=3.0)
    clip = g.add("CLIPLoader", grp, clip_name=Z_CLIP, type="lumina2", device="default")
    vae = g.add("VAELoader", grp, vae_name=Z_VAE)
    return m, clip, vae


def zimage_sample(g, grp, model, clip, text, width=1024, height=1024, latent=None, denoise=1.0, batch=1, seed=SEED):
    pos = g.add("CLIPTextEncode", grp, title="Prompt", text=text, clip=clip[0])
    neg = g.add("ConditioningZeroOut", grp, conditioning=pos[0])
    if latent is None:
        latent = g.add("EmptySD3LatentImage", grp, width=width, height=height, batch_size=batch)[0]
    return g.add("KSampler", grp, model=model[0], seed=seed, steps=8, cfg=1.0, sampler_name="res_multistep",
                 scheduler="simple", positive=pos[0], negative=neg[0], latent_image=latent, denoise=denoise)


def klein_models(g, grp, base=False, loras=()):
    unet = g.add("UNETLoader", grp, title="FLUX.2 klein 4B " + ("base" if base else "distilled"),
                 unet_name=K_BASE if base else K_UNET, weight_dtype="default")
    m = unet
    for name, strength, title in loras:
        m = g.add("LoraLoaderModelOnly", grp, title=title, model=m[0], lora_name=name, strength_model=strength)
    clip = g.add("CLIPLoader", grp, clip_name=K_CLIP, type="flux2", device="default")
    vae = g.add("VAELoader", grp, vae_name=K_VAE)
    return m, clip, vae


def klein_sample(g, grp, model, clip, vae, text, width, height, refs=(), base=False, negative="", seed=SEED,
                 steps=None, cfg=None, latent=None):
    pos = g.add("CLIPTextEncode", grp, title="Prompt", text=text, clip=clip[0])
    if base:
        neg = g.add("CLIPTextEncode", grp, title="Negative", text=negative, clip=clip[0])
    else:
        neg = g.add("ConditioningZeroOut", grp, conditioning=pos[0])
    for ref in refs:
        pos = g.add("ReferenceLatent", grp, conditioning=pos[0], latent=ref)
        neg = g.add("ReferenceLatent", grp, conditioning=neg[0], latent=ref)
    noise = g.add("RandomNoise", grp, noise_seed=seed)
    guider = g.add("CFGGuider", grp, model=model[0], positive=pos[0], negative=neg[0],
                   cfg=cfg if cfg is not None else (5.0 if base else 1.0))
    sampler = g.add("KSamplerSelect", grp, sampler_name="euler")
    sigmas = g.add("Flux2Scheduler", grp, steps=steps or (20 if base else 4), width=width, height=height)
    if latent is None:
        latent = g.add("EmptyFlux2LatentImage", grp, width=width, height=height, batch_size=1)[0]
    s = g.add("SamplerCustomAdvanced", grp, noise=noise[0], guider=guider[0], sampler=sampler[0],
              sigmas=sigmas[0], latent_image=latent)
    return g.add("VAEDecode", grp, samples=s[0], vae=vae[0])


def encode_ref(g, grp, image, vae, scale=True):
    if scale:
        image = g.add("ImageScaleToTotalPixels", grp, image=image, upscale_method="lanczos", megapixels=1.0,
                      resolution_steps=16)[0]
    return g.add("VAEEncode", grp, pixels=image, vae=vae[0])[0]


def birefnet_mask(g, grp, image):
    m = g.add("LoadBackgroundRemovalModel", grp, bg_removal_name=BIREFNET)
    return g.add("RemoveBackground", grp, bg_removal_model=m[0], image=image)[0]


def load_sprite(g, grp, title, image):
    """LoadImage drops transparency; JoinImageWithAlpha restores it (its alpha = 1 - LoadImage's mask)."""
    li = g.add("LoadImage", grp, title=title, image=image)
    return g.add("JoinImageWithAlpha", grp, title=title + " (RGBA)", image=li[0], alpha=li[1])


def save(g, grp, image, prefix, animation=False, fps=8.0, columns=0, upscale=8):
    return g.add("SK_SavePixelArt", grp, images=image, filename_prefix="spritekiln/" + prefix, upscale=upscale,
                 as_animation=animation, fps=fps, sheet_columns=columns)


# ----------------------------------------------------------------------------- workflows
def wf01(oi):
    g = Graph(oi, "01_text_to_pixel_art", "Text → Pixel Art (Z-Image Turbo)",
              "RD Fast equivalent: style presets, exact art size, palette size, transparent background.")
    P, M, S, O = "1 · Prompt & style", "2 · Generate (Z-Image Turbo + pixel LoRA, 8 steps)", "3 · Pixel-perfect", "4 · Save"
    g.note(P, "## Text → Pixel Art\nPick a **style** (sets art size, colour count, background). Override art size / colours "
              "with non-zero values. Sprite styles get a plain background that is removed automatically.\n\n"
              "Raise `batch_size` in *Empty latent* for variations. Outputs land in `output/spritekiln/`.")
    st = g.add("SK_StylePreset", P, subject="a brave knight with a red cape, silver armor and a sword",
               style="character (side view)", model="z_image (pixel LoRA)", art_width=0, art_height=0, colors=0,
               background="auto", long_side=1024, extra="")
    m, clip, vae = zimage_models(g, M, lora=st[8])
    ks = zimage_sample(g, M, m, clip, st[0], st[4], st[5])
    dec = g.add("VAEDecode", M, samples=ks[0], vae=vae[0])
    g.add("PreviewImage", M, title="Raw model output", images=dec[0])
    mask = birefnet_mask(g, S, dec[0])
    bg = g.add("SK_BackgroundRemove", S, image=dec[0], mode="mask_and_key", tolerance=0.08, mask_threshold=0.5,
               enabled=st[7], foreground_mask=mask)
    pp = g.add("SK_PixelPerfect", S, image=bg[0], mode="fixed_size", width=st[2], height=st[3], scale=16,
               method="dominant", pre_colors=48)
    pal = g.add("SK_Palette", S, image=pp[0], source="auto", colors=st[6], preset="PICO-8", dither="none",
                dither_strength=0.5, shared_palette=True)
    cl = g.add("SK_Cleanup", S, image=pal[0], speck_size=2, max_contrast=0.12, outline="none", outline_color="#140c1c",
               outline_diagonal=False, darken=0.45)
    save(g, O, cl[0], "text2pixel")
    g.add("PreviewImage", O, title="Palette", images=pal[1])
    return g


def wf02(oi):
    g = Graph(oi, "02_image_to_pixel_art", "Image → Pixel Art (convert + AI redraw)",
              "Any picture to pixel art: a direct algorithmic conversion and a Z-Image img2img redraw (strength).")
    I, D, R, O = "1 · Input & size", "2a · Direct conversion (no AI)", "2b · AI redraw (img2img)", "3 · Save"
    g.note(I, "## Image → Pixel Art\n**2a** converts the image directly (fast, faithful).\n**2b** redraws it as real pixel "
              "art with Z-Image: `denoise` 0.4 = close to the source, 0.75 = reinterpretation (RD 'strength').\n"
              "Describe the image in the prompt for best results.")
    img = g.add("LoadImage", I, image=EX_REF)
    aw = g.add("PrimitiveInt", I, title="Art width", value=96)
    ah = g.add("PrimitiveInt", I, title="Art height", value=96)
    cols = g.add("PrimitiveInt", I, title="Colours", value=24)
    pp = g.add("SK_PixelPerfect", D, image=img[0], mode="fixed_size", width=aw[0], height=ah[0], scale=16,
               method="mean", pre_colors=48)
    pal = g.add("SK_Palette", D, image=pp[0], source="auto", colors=cols[0], preset="PICO-8", dither="none",
                dither_strength=0.3, shared_palette=True)
    save(g, O, pal[0], "img2pixel_direct")
    m, clip, vae = zimage_models(g, R)
    scaled = g.add("ImageScaleToTotalPixels", R, image=img[0], upscale_method="lanczos", megapixels=1.0,
                   resolution_steps=16)
    lat = g.add("VAEEncode", R, pixels=scaled[0], vae=vae[0])
    ks = zimage_sample(g, R, m, clip, "Pixel art style. pixel art of the scene, crisp pixels, limited palette, retro game graphics",
                       latent=lat[0], denoise=0.6)
    dec = g.add("VAEDecode", R, samples=ks[0], vae=vae[0])
    g.add("PreviewImage", R, title="Redraw (model res)", images=dec[0])
    pp2 = g.add("SK_PixelPerfect", R, image=dec[0], mode="fixed_size", width=aw[0], height=ah[0], scale=16,
                method="dominant", pre_colors=48)
    pal2 = g.add("SK_Palette", R, image=pp2[0], source="auto", colors=cols[0], preset="PICO-8", dither="none",
                 dither_strength=0.5, shared_palette=True)
    save(g, O, pal2[0], "img2pixel_redraw")
    return g


def wf03(oi):
    g = Graph(oi, "03_reference_to_pixel_art", "Reference Images → Pixel Art (FLUX.2 klein)",
              "RD Plus / PixelLab style-reference equivalent: subject + style reference images and a text prompt.")
    P, R, M, S, O = "1 · Prompt", "2 · Reference images", "3 · Generate (klein 4B + pixel LoRA)", "4 · Pixel-perfect", "5 · Save"
    g.note(P, "## Reference → Pixel Art\nImage 1 = **subject** (character, object, photo). Image 2 = **style** reference "
              "(a pixel art piece whose look you want). Refer to them as *image 1 / image 2* in the prompt.\n"
              "Only one reference? Select the *Style reference* nodes and bypass them (Ctrl+B).")
    st = g.add("SK_StylePreset", P, subject="the character from image 1, redrawn as a pixel art sprite in the exact pixel art "
                                            "style, palette and shading of image 2",
               style="game asset", model="flux2 klein (pixel LoRA)", art_width=64, art_height=64, colors=0,
               background="auto", long_side=1024, extra="")
    m, clip, vae = klein_models(g, M, loras=[(K_LORA, st[8], "Pixel-art LoRA (klein)")])
    subj = g.add("LoadImage", R, title="Subject reference (image 1)", image=EX_REF)
    sty = g.add("LoadImage", R, title="Style reference (image 2)", image=EX_STYLE)
    r1 = encode_ref(g, R, subj[0], vae)
    sty_scaled = g.add("ImageScaleToTotalPixels", R, title="Style reference scale", image=sty[0],
                       upscale_method="nearest-exact", megapixels=1.0, resolution_steps=16)
    r2 = g.add("VAEEncode", R, title="Style reference encode", pixels=sty_scaled[0], vae=vae[0])[0]
    dec = klein_sample(g, M, m, clip, vae, st[0], st[4], st[5], refs=[r1, r2])
    g.add("PreviewImage", M, title="Raw model output", images=dec[0])
    mask = birefnet_mask(g, S, dec[0])
    bg = g.add("SK_BackgroundRemove", S, image=dec[0], mode="mask_and_key", tolerance=0.08, mask_threshold=0.5,
               enabled=st[7], foreground_mask=mask)
    pp = g.add("SK_PixelPerfect", S, image=bg[0], mode="fixed_size", width=st[2], height=st[3], scale=16,
               method="dominant", pre_colors=48)
    pal = g.add("SK_Palette", S, image=pp[0], source="auto", colors=st[6], preset="PICO-8", dither="none",
                dither_strength=0.5, shared_palette=True)
    cl = g.add("SK_Cleanup", S, image=pal[0], speck_size=2, max_contrast=0.12, outline="none", outline_color="#140c1c",
               outline_diagonal=False, darken=0.45)
    save(g, O, cl[0], "ref2pixel")
    return g


def _sprite_in(g, grp, title="Sprite to edit", padding=0, long_side=1024, multiple=16, bg="gray", mask=None,
               target=None):
    sp = load_sprite(g, grp, title, EX_SPRITE)
    kw = {"mask": mask} if mask is not None else {}
    if target:
        kw.update(target_width=target[0], target_height=target[1])
    cv = g.add("SK_ToCanvas", grp, image=sp[0], long_side=long_side, multiple=multiple, background=bg,
               custom_color="#808080", padding=padding, **kw)
    return sp, cv


def wf04(oi):
    g = Graph(oi, "04_edit_pixel_art", "Edit Pixel Art (text + image references, FLUX.2 klein)",
              "Instruction edits on an existing sprite; optional second image as an extra reference.")
    I, M, S, O = "1 · Sprite + references", "2 · Edit (klein 4B distilled)", "3 · Back to the pixel grid", "4 · Save"
    g.note(I, "## Edit pixel art\nLoad a sprite at its **native** size (e.g. 64×64 PNG). It is upscaled onto a 1024 canvas, "
              "edited by FLUX.2 klein, then snapped back to the exact same grid.\nTwo outputs: free colours, and "
              "locked to the sprite's original palette.\nThe *Extra reference* (image 2) is optional — bypass it with Ctrl+B.")
    sp, cv = _sprite_in(g, I)
    ref = g.add("LoadImage", I, title="Extra reference (image 2)", image=EX_REF)
    m, clip, vae = klein_models(g, M, loras=[(K_LORA, 0.5, "Pixel-art LoRA (klein)")])
    r1 = g.add("VAEEncode", M, pixels=cv[0], vae=vae[0])[0]
    r2 = encode_ref(g, M, ref[0], vae)
    dec = klein_sample(g, M, m, clip, vae,
                       "Edit the pixel art sprite in image 1: give the knight a golden crown and a blue cape. Keep everything "
                       "else identical: same pose, same size, same pixel art style, same outline and colors, same gray background.",
                       cv[2], cv[3], refs=[r1, r2])
    g.add("PreviewImage", M, title="Raw edit", images=dec[0])
    pp = g.add("SK_PixelPerfect", S, image=dec[0], mode="match_reference", width=64, height=64, scale=cv[1],
               method="dominant", pre_colors=48, reference=cv[4])
    bg = g.add("SK_BackgroundRemove", S, image=pp[0], mode="border_flood", tolerance=0.06, mask_threshold=0.5, enabled=True)
    free = g.add("SK_Palette", S, title="Palette: free (new colours allowed)", image=bg[0], source="auto", colors=32,
                 preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True)
    lock = g.add("SK_Palette", S, title="Palette: locked to the original sprite", image=bg[0], source="palette_image",
                 colors=256, preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True, palette_image=sp[0])
    save(g, O, free[0], "edit_free")
    save(g, O, lock[0], "edit_locked")
    return g


def wf05(oi):
    g = Graph(oi, "05_inpaint_pixel_art", "Inpaint Pixel Art (region edit, FLUX.2 klein)",
              "Change only a rectangle (or painted area) of a sprite; everything else stays pixel-identical.")
    I, M, S, O = "1 · Sprite + region", "2 · Inpaint (klein 4B)", "3 · Merge on the pixel grid", "4 · Save"
    g.note(I, "## Inpaint\nSet the region in **art pixels** (x, y, width, height on the original sprite) or paint a mask "
              "in the image's mask editor and connect it to `painted_mask`. Pixels outside the region are copied back "
              "untouched, so the sprite never drifts.")
    sp = load_sprite(g, I, "Sprite", EX_SPRITE)
    rm = g.add("SK_RegionMask", I, image=sp[0], x=16, y=0, width=32, height=20, grow=1, invert=False)
    cv = g.add("SK_ToCanvas", I, image=sp[0], long_side=1024, multiple=16, background="gray", custom_color="#808080",
               padding=0, mask=rm[0])
    m, clip, vae = klein_models(g, M, loras=[(K_LORA, 0.5, "Pixel-art LoRA (klein)")])
    enc = g.add("VAEEncode", M, pixels=cv[0], vae=vae[0])
    lat = g.add("SetLatentNoiseMask", M, samples=enc[0], mask=cv[5])
    dec = klein_sample(g, M, m, clip, vae,
                       "Pixel art sprite: the knight now wears a horned viking helmet. Same pixel art style, same palette, "
                       "same outline, gray background.", cv[2], cv[3], refs=[enc[0]], latent=lat[0])
    g.add("PreviewImage", M, title="Raw inpaint", images=dec[0])
    pp = g.add("SK_PixelPerfect", S, image=dec[0], mode="match_reference", width=64, height=64, scale=cv[1],
               method="dominant", pre_colors=48, reference=cv[4])
    bg = g.add("SK_BackgroundRemove", S, image=pp[0], mode="border_flood", tolerance=0.06, mask_threshold=0.5, enabled=True)
    pal = g.add("SK_Palette", S, image=bg[0], source="auto", colors=32, preset="PICO-8", dither="none",
                dither_strength=0.5, shared_palette=True)
    merged = g.add("SK_MaskedMerge", S, original=sp[0], edited=pal[0], mask=rm[0])
    save(g, O, merged[0], "inpaint")
    return g


def _rotation_post(g, S, O, dec, cv, sp, prompts, prefix, method):
    pp = g.add("SK_PixelPerfect", S, image=dec[0], mode="match_reference", width=64, height=64, scale=cv[1],
               method="dominant", pre_colors=48, reference=cv[4])
    bg = g.add("SK_BackgroundRemove", S, image=pp[0], mode="border_flood", tolerance=0.08, mask_threshold=0.5, enabled=True)
    pal = g.add("SK_Palette", S, title="Lock to the sprite's palette", image=bg[0], source="palette_image", colors=256,
                preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True, palette_image=sp[0])
    asm = g.add("SK_DirectionAssemble", O, images=pal[0], labels=prompts[1], order="clockwise from S")
    save(g, O, asm[0], prefix, animation=True, fps=4.0, columns=0)
    g.add("PreviewAny", O, title="Direction order", source=asm[1])


def wf06(oi):
    g = Graph(oi, "06_rotate_8_directions", "Rotate Sprite → 4/8 Directions (FLUX.2 klein)",
              "PixelLab-style rotations: one edit per facing; west-facing views are mirrored for perfect symmetry.")
    I, M, S, O = "1 · Sprite + directions", "2 · One edit per direction (klein 4B)", "3 · Snap to grid + palette", "4 · Assemble & save"
    g.note(I, "## Rotations\nLoad a **front-facing** sprite. The graph runs once per direction (S, SE, E, NE, N; W/SW/NW "
              "are mirrored). Set `directions` to 4 for S/E/N/W.\nOutput: a sheet in clockwise order from S, a turnaround "
              "GIF and individual frames. For higher consistency use workflow 07 (Qwen multi-angle).")
    sp, cv = _sprite_in(g, I, title="Front-facing sprite")
    dp = g.add("SK_DirectionPrompts", I, directions="8", mirror_west=True, prompt_style="flux2 klein",
               template="Turn the character in the image to a {direction}. Keep exactly the same character design, outfit, "
                        "colors, proportions, size and pixel art style. Full body, centered, {background}.",
               background="gray")
    m, clip, vae = klein_models(g, M, loras=[(K_LORA, 0.6, "Pixel-art LoRA (klein)")])
    r1 = g.add("VAEEncode", M, pixels=cv[0], vae=vae[0])[0]
    dec = klein_sample(g, M, m, clip, vae, dp[0], cv[2], cv[3], refs=[r1])
    _rotation_post(g, S, O, dec, cv, sp, dp, "rotate8", "klein")
    return g


def wf07(oi):
    g = Graph(oi, "07_rotate_hq_qwen_multiangle", "Rotate Sprite HQ (Qwen-Image-Edit-2511 + Multiple-Angles LoRA)",
              "96-camera-pose LoRA for the most consistent turnarounds. Slower (20B, GGUF Q4).")
    I, M, S, O = "1 · Sprite + directions", "2 · Qwen-Image-Edit-2511 (4-step Lightning)", "3 · Snap to grid + palette", "4 · Assemble & save"
    g.note(I, "## HQ rotations\nSame as workflow 06 but with Qwen-Image-Edit-2511 + fal's Multiple-Angles LoRA "
              "(`<sks> right side view eye-level shot medium shot` prompts). Needs the *hq-rotate* models and ComfyUI-GGUF.\n"
              "If left/right come out swapped for your subject, change `prompt_style` back to klein or swap the outputs.")
    sp, cv = _sprite_in(g, I, title="Front-facing sprite")
    dp = g.add("SK_DirectionPrompts", I, directions="8", mirror_west=True, prompt_style="qwen multi-angle LoRA",
               template="{direction}", background="gray")
    unet = g.add("UnetLoaderGGUF", M, unet_name=Q_UNET)
    m = g.add("LoraLoaderModelOnly", M, title="Multiple-Angles LoRA", model=unet[0], lora_name=Q_ANGLES, strength_model=0.9)
    m = g.add("ModelSamplingAuraFlow", M, model=m[0], shift=3.1)
    m = g.add("CFGNorm", M, model=m[0], strength=1.0)
    m = g.add("LoraLoaderModelOnly", M, title="Lightning 4-step LoRA", model=m[0], lora_name=Q_LIGHT, strength_model=1.0)
    clip = g.add("CLIPLoader", M, clip_name=Q_CLIP, type="qwen_image", device="default")
    vae = g.add("VAELoader", M, vae_name=Q_VAE)
    # the canvas is already ~1 MP and a multiple of 16; resizing it would break the pixel-grid round trip
    pos = g.add("TextEncodeQwenImageEditPlus", M, clip=clip[0], prompt=dp[0], vae=vae[0], image1=cv[0])
    neg = g.add("TextEncodeQwenImageEditPlus", M, clip=clip[0], prompt="", vae=vae[0], image1=cv[0])
    pos = g.add("FluxKontextMultiReferenceLatentMethod", M, conditioning=pos[0], reference_latents_method="index_timestep_zero")
    neg = g.add("FluxKontextMultiReferenceLatentMethod", M, conditioning=neg[0], reference_latents_method="index_timestep_zero")
    lat = g.add("VAEEncode", M, pixels=cv[0], vae=vae[0])
    ks = g.add("KSampler", M, model=m[0], seed=SEED, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
               positive=pos[0], negative=neg[0], latent_image=lat[0], denoise=1.0)
    dec = g.add("VAEDecode", M, samples=ks[0], vae=vae[0])
    _rotation_post(g, S, O, dec, cv, sp, dp, "rotate8_hq", "qwen")
    return g


def wf08(oi):
    g = Graph(oi, "08_object_multiview_2x2", "Object → 4 Views 2×2 Sheet (klein base + sprite-sheet LoRA)",
              "fal's 2x2 sprite-sheet LoRA: isometric left/right, side and top-down views of an object in one pass.")
    I, M, S, O = "1 · Object", "2 · Generate 2×2 sheet (klein base, 20 steps)", "3 · Pixel-perfect + slice", "4 · Save"
    g.note(I, "## Object multi-view\nBest with a single object (prop, vehicle, item, building) on a plain background.\n"
              "Layout: TL isometric facing bottom-right, TR isometric facing bottom-left, BL side profile, BR top-down.\n"
              "Set *Art size per view* for the final sprite size.")
    sp, cv = _sprite_in(g, I, title="Object image", bg="white")
    view = g.add("PrimitiveInt", I, title="Art size per view", value=64)
    m, clip, vae = klein_models(g, M, base=True, loras=[(K_2X2_LORA, 1.1, "2x2 sprite-sheet LoRA (fal)")])
    r1 = g.add("VAEEncode", M, pixels=cv[0], vae=vae[0])[0]
    dec = klein_sample(g, M, m, clip, vae, "2x2 sprite sheet, pixel art", 1024, 1024, refs=[r1], base=True,
                       negative="blurry, photo, text")
    g.add("PreviewImage", M, title="Raw 2x2 sheet", images=dec[0])
    sl = g.add("SK_SliceSheet", S, sheet=dec[0], rows=2, columns=2, inset=6, order="row_major")
    mask = birefnet_mask(g, S, sl[0])
    bg = g.add("SK_BackgroundRemove", S, title="Remove the LoRA's red background", image=sl[0], mode="mask_and_key",
               tolerance=0.1, mask_threshold=0.5, enabled=True, foreground_mask=mask)
    pp = g.add("SK_PixelPerfect", S, image=bg[0], mode="fixed_size", width=view[0], height=view[0], scale=8,
               method="dominant", pre_colors=48)
    pal = g.add("SK_Palette", S, image=pp[0], source="auto", colors=24, preset="PICO-8", dither="none",
                dither_strength=0.5, shared_palette=True)
    save(g, O, pal[0], "multiview", animation=True, fps=2.0, columns=2)
    return g


WALK_PROMPT = ("The spritesheet is a 4 by 4 grid of four rows of frames - first row is 3 walking frames facing down and 1 frame "
               "both arms raised, second row is 3 walking frames facing left and 1 frame jumping left, third row is 3 walking "
               "frames facing right and 1 frame jumping right, fourth row is 3 walking frames back view facing up and 1 frame "
               "lying on floor.")


def _walk_post(g, S, O, dec):
    mask = birefnet_mask(g, S, dec[0])
    bg = g.add("SK_BackgroundRemove", S, image=dec[0], mode="mask_and_key", tolerance=0.08, mask_threshold=0.5, enabled=True,
               foreground_mask=mask)
    pp = g.add("SK_PixelPerfect", S, title="512 → 128 (x4, as trained)", image=bg[0], mode="fixed_size", width=128,
               height=128, scale=4, method="dominant", pre_colors=48)
    pal = g.add("SK_Palette", S, image=pp[0], source="auto", colors=24, preset="PICO-8", dither="none",
                dither_strength=0.5, shared_palette=True)
    cl = g.add("SK_Cleanup", S, image=pal[0], speck_size=2, max_contrast=0.12, outline="none", outline_color="#140c1c",
               outline_diagonal=False, darken=0.45)
    save(g, O, cl[0], "walksheet")
    fr = g.add("SK_SliceSheet", O, sheet=cl[0], rows=4, columns=4, inset=0, order="row_major")
    for row, name in enumerate(["down", "left", "right", "up"]):
        walk = g.add("ImageFromBatch", O, title="Walk " + name, image=fr[0], batch_index=row * 4, length=3)
        loop = g.add("SK_FrameSelect", O, title="Ping-pong 0-1-2-1", frames=walk[0], count=4, mode="ping_pong",
                     skip_start=0, min_loop_fraction=0.4)
        save(g, O, loop[0], "walk_" + name, animation=True, fps=6.0, upscale=8)


def wf09(oi):
    g = Graph(oi, "09_walk_spritesheet_from_text", "4-Direction Walk Spritesheet from Text (klein base + LoRA)",
              "32×32 character sheet: walk down/left/right/up, arms raised, jumps, lying down — plus a GIF per direction.")
    P, M, S, O = "1 · Character", "2 · Generate 512² sheet (klein base, 4-walk LoRA)", "3 · Pixel-perfect 128²", "4 · Sheet + GIFs"
    g.note(P, "## Walk spritesheet\nDescribe the character only; the layout text is appended automatically. "
              "Generates at 512×512 (training size) and snaps to a true 128×128 sheet of 32×32 frames.")
    subj = g.add("PrimitiveStringMultiline", P, title="Character", value="A pixel art spritesheet of a medieval knight wearing "
                                                                         "metal armor and a helmet with a red plume.")
    txt = g.add("StringConcatenate", P, string_a=subj[0], string_b=WALK_PROMPT, delimiter=" ")
    m, clip, vae = klein_models(g, M, base=True, loras=[(K_SHEET_LORA, 1.0, "4-walk spritesheet LoRA")])
    dec = klein_sample(g, M, m, clip, vae, txt[0], 512, 512, base=True, negative="blurry, smooth, photo, text, watermark",
                       steps=24)
    g.add("PreviewImage", M, title="Raw sheet", images=dec[0])
    _walk_post(g, S, O, dec)
    return g


def wf09b(oi):
    g = Graph(oi, "09b_walk_spritesheet_from_reference", "4-Direction Walk Spritesheet from a Reference (klein base + LoRA)",
              "Same as 09 but from an image of your character (concept art, a sprite, a photo).")
    P, M, S, O = "1 · Character reference", "2 · Generate 512² sheet (klein base edit)", "3 · Pixel-perfect 128²", "4 · Sheet + GIFs"
    g.note(P, "## Walk spritesheet from a reference\nUse a clear, full-body image of the character.")
    ref = load_sprite(g, P, "Character reference", EX_SPRITE)
    cv = g.add("SK_ToCanvas", P, image=ref[0], long_side=512, multiple=16, background="white", custom_color="#ffffff", padding=4)
    m, clip, vae = klein_models(g, M, base=True, loras=[(K_SHEET_LORA, 1.0, "4-walk spritesheet LoRA")])
    r1 = g.add("VAEEncode", M, pixels=cv[0], vae=vae[0])[0]
    dec = klein_sample(g, M, m, clip, vae, "Create a pixel art spritesheet of the character in the image. " + WALK_PROMPT,
                       512, 512, refs=[r1], base=True, negative="blurry, smooth, photo, text, watermark", steps=24)
    g.add("PreviewImage", M, title="Raw sheet", images=dec[0])
    _walk_post(g, S, O, dec)
    return g


def wan14(g, grp, positive_text, negative_text, start, width, height, length, end=None, seamless=False, pixel=True):
    """Wan 2.2 I2V 14B (GGUF Q4) + Lightning 4-step + pixel-animate LoRA, high→low noise experts.
    end=None: free image-to-video. end=image: first/last-frame (loops when end == start)."""
    clip = g.add("CLIPLoader", grp, clip_name=W_CLIP, type="wan", device="default")
    vae = g.add("VAELoader", grp, vae_name=W14_VAE)
    models = []
    for unet_name, light, tag in ((W14_HIGH, W14_LIGHT_H, "high"), (W14_LOW, W14_LIGHT_L, "low")):
        u = g.add("UnetLoaderGGUF", grp, title="Wan 2.2 I2V 14B " + tag + " noise", unet_name=unet_name)
        x = g.add("LoraLoaderModelOnly", grp, title="Lightning 4-step (" + tag + ")", model=u[0], lora_name=light,
                  strength_model=1.0)
        if pixel:
            x = g.add("LoraLoaderModelOnly", grp, title="Pixel-animate LoRA (" + tag + ")", model=x[0],
                      lora_name=W14_PIXEL, strength_model=1.0)
        x = g.add("ModelSamplingSD3", grp, model=x[0], shift=5.0)
        if seamless:
            x = g.add("SK_SeamlessTiling", grp, title="Seamless (" + tag + ")", model=x[0], axes="both", align=2)
        models.append(x)
    pos = g.add("CLIPTextEncode", grp, title="Motion prompt", text=positive_text, clip=clip[0])
    neg = g.add("CLIPTextEncode", grp, title="Negative", text=negative_text, clip=clip[0])
    if end is None:
        cond = g.add("WanImageToVideo", grp, positive=pos[0], negative=neg[0], vae=vae[0], width=width, height=height,
                     length=length, batch_size=1, start_image=start)
    else:
        cond = g.add("WanFirstLastFrameToVideo", grp, positive=pos[0], negative=neg[0], vae=vae[0], width=width,
                     height=height, length=length, batch_size=1, start_image=start, end_image=end)
    k1 = g.add("KSamplerAdvanced", grp, title="High noise (steps 0-2)", model=models[0][0], add_noise="enable",
               noise_seed=SEED, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple", positive=cond[0],
               negative=cond[1], latent_image=cond[2], start_at_step=0, end_at_step=2, return_with_leftover_noise="enable")
    k2 = g.add("KSamplerAdvanced", grp, title="Low noise (steps 2-4)", model=models[1][0], add_noise="disable",
               noise_seed=SEED, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple", positive=cond[0],
               negative=cond[1], latent_image=k1[0], start_at_step=2, end_at_step=10000, return_with_leftover_noise="disable")
    if seamless:
        return g.add("SK_SeamlessVAEDecode", grp, samples=k2[0], vae=vae[0], axes="both", pad_latent=4)
    return g.add("VAEDecode", grp, samples=k2[0], vae=vae[0])


def _anim_post(g, S, O, frames, cv, sp, ap, prefix):
    sel = g.add("SK_FrameSelect", S, frames=frames, count=ap[2], mode="auto", skip_start=1, min_loop_fraction=0.4,
                loop=ap[4])
    g.add("PreviewAny", S, title="Loop info", source=sel[1])
    pp = g.add("SK_PixelPerfect", S, image=sel[0], mode="match_reference", width=64, height=64, scale=cv[1],
               method="dominant", pre_colors=48, reference=cv[4])
    bg = g.add("SK_BackgroundRemove", S, image=pp[0], mode="border_flood", tolerance=0.08, mask_threshold=0.5, enabled=True)
    pal = g.add("SK_Palette", S, title="Lock to the sprite's palette", image=bg[0], source="palette_image", colors=256,
                preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True, palette_image=sp[0])
    stab = g.add("SK_Stabilize", S, frames=pal[0], anchor="bottom_center")
    cl = g.add("SK_Cleanup", S, image=stab[0], speck_size=2, max_contrast=0.12, outline="none", outline_color="#140c1c",
               outline_diagonal=False, darken=0.45)
    crop = g.add("SK_CropToContent", O, frames=cl[0], margin=2, multiple=2, square=False)
    save(g, O, crop[0], prefix, animation=True, fps=ap[3], columns=0)


def wf10(oi):
    g = Graph(oi, "10_animate_sprite", "Animate Sprite – any action or free text (Wan 2.2 14B I2V)",
              "RD Animation / PixelLab text animation: 36 presets (combat, movement, creatures, props, FX) or free text.")
    I, M, S, O = "1 · Sprite + action", "2 · Video (Wan 2.2 14B, 4 steps)", "3 · Frames → pixel grid (shared palette)", "4 · Save GIF + sheet"
    g.note(I, "## Animate a sprite\nPick an **action** (or `custom` + your own sentence). The preset sets the motion "
              "prompt, number of sprite frames, GIF fps and whether it loops. Loops get an automatic loop point; one-shot "
              "actions are sampled start→end. Frames are snapped to the sprite's grid and palette, stabilised on the feet "
              "and cropped to a uniform size.\nFor guaranteed return-to-start loops use workflow 11.")
    sp, cv = _sprite_in(g, I, title="Sprite", padding=8, long_side=640, multiple=16, target=(640, 640))
    ap = g.add("SK_AnimationPreset", I, subject="the pixel art knight", action="sword slash", view="side view",
               custom_motion="", background="gray")
    vf = g.add("PrimitiveInt", I, title="Video frames (4n+1)", value=33)
    dec = wan14(g, M, ap[0], ap[1], cv[0], cv[2], cv[3], vf[0])
    g.add("SaveAnimatedWEBP", M, title="Raw video preview", images=dec[0], filename_prefix="spritekiln/raw_anim",
          fps=16.0, lossless=False, quality=85, method="default")
    _anim_post(g, S, O, dec[0], cv, sp, ap, "anim")
    return g


def wf11(oi):
    g = Graph(oi, "11_animate_sprite_hq_loop", "Animate Sprite – seamless loop (Wan 2.2 14B first = last frame)",
              "First and last video frame are your sprite: perfect loops and 'returns to the starting pose' motions.")
    I, M, S, O = "1 · Sprite + action", "2 · Video (Wan 2.2 14B, first=last frame, 4 steps)", "3 · Frames → pixel grid", "4 · Save GIF + sheet"
    g.note(I, "## Looping animation\nThe clip is pinned to your sprite at both ends, so it always loops and always "
              "returns to the start pose — ideal for idles, walks, and text like *'raises his arm, points to the sky, "
              "then lowers it again'* (action = custom). For one-shot moves that end elsewhere use 10 or 17.")
    sp, cv = _sprite_in(g, I, title="Sprite", padding=8, long_side=640, multiple=16, target=(640, 640))
    ap = g.add("SK_AnimationPreset", I, subject="the pixel art knight", action="walk", view="side view", custom_motion="",
               background="gray")
    vf = g.add("PrimitiveInt", I, title="Video frames (4n+1)", value=33)
    dec = wan14(g, M, ap[0], ap[1], cv[0], cv[2], cv[3], vf[0], end=cv[0])
    g.add("SaveAnimatedWEBP", M, title="Raw video preview", images=dec[0], filename_prefix="spritekiln/raw_anim_hq",
          fps=16.0, lossless=False, quality=85, method="default")
    _anim_post(g, S, O, dec[0], cv, sp, ap, "anim_loop")
    return g


def _seamless_texture(g, grp, m, clip, vae, prompt, tile, colors, title):
    st = m  # patched model
    ks = zimage_sample(g, grp, st, clip, prompt, 512, 512)
    dec = g.add("SK_SeamlessVAEDecode", grp, samples=ks[0], vae=vae[0], axes="both", pad_latent=8)
    pp = g.add("SK_PixelPerfect", grp, title=title, image=dec[0], mode="fixed_size", width=tile, height=tile, scale=16,
               method="dominant", pre_colors=48)
    pal = g.add("SK_Palette", grp, image=pp[0], source="auto", colors=colors, preset="PICO-8", dither="none",
                dither_strength=0.5, shared_palette=True)
    return pal


def wf12(oi):
    g = Graph(oi, "12_seamless_tile_texture", "Seamless Tile / Texture (Z-Image + circular tiling)",
              "RD Tile 'single tile' equivalent: perfectly wrapping pixel textures at any tile size.")
    P, M, O = "1 · Prompt", "2 · Generate tileable (Z-Image + seamless patch)", "3 · Preview 3×3 + save"
    g.note(P, "## Seamless tile\nThe model is patched to wrap around its edges, so the result tiles in both directions. "
              "Use the *texture (tileable)* or *block texture* style. `batch_size` > 1 gives tile variations.")
    st = g.add("SK_StylePreset", P, subject="green grass with a few small flowers, large chunky pixels, simple 16-bit RPG "
                                            "game tile", style="texture (tileable)",
               model="z_image (pixel LoRA)", art_width=32, art_height=32, colors=12, background="scene",
               long_side=512, extra="")
    m, clip, vae = zimage_models(g, M)
    sm = g.add("SK_SeamlessTiling", M, model=m[0], axes="both", align=2)
    ks = zimage_sample(g, M, sm, clip, st[0], st[4], st[5])
    dec = g.add("SK_SeamlessVAEDecode", M, samples=ks[0], vae=vae[0], axes="both", pad_latent=8)
    pp = g.add("SK_PixelPerfect", M, image=dec[0], mode="fixed_size", width=st[2], height=st[3], scale=16,
               method="dominant", pre_colors=48)
    pal = g.add("SK_Palette", M, image=pp[0], source="auto", colors=st[6], preset="PICO-8", dither="none",
                dither_strength=0.5, shared_palette=True)
    save(g, O, pal[0], "tile")
    tp = g.add("SK_TilePreview", O, image=pal[0], repeat_x=3, repeat_y=3)
    big = g.add("SK_ScaleNearest", O, image=tp[0], scale=6)
    g.add("PreviewImage", O, title="3×3 tiling check", images=big[0])
    return g


def wf13(oi):
    g = Graph(oi, "13_autotile_tileset", "Autotile Tileset – wang16 + blob47 (Z-Image + seamless + SpriteKiln)",
              "RD Tile 'tileset' equivalent: two seamless terrains combined into dual-grid and 47-blob autotile sets.")
    P, A, B, T, O = "1 · Terrains", "2a · Terrain A (upper)", "2b · Terrain B (lower)", "3 · Build tilesets", "4 · Save"
    g.note(P, "## Autotile tileset\nTerrain A is drawn over terrain B. **wang16** = 2-corner / dual-grid set (Godot 4 "
              "terrain 'corners' mode, Tiled 'corner' wang sets). **blob47** = classic 8-neighbour autotile; the `mapping` "
              "output gives neighbour-bitmask → tile index for your engine.")
    ts = g.add("PrimitiveInt", P, title="Tile size (px)", value=16)
    pa = g.add("PrimitiveStringMultiline", P, title="Terrain A",
               value="Pixel art style. seamless tileable pixel art texture of green grass, large chunky pixels, simple 16-bit RPG game tile, top-down, flat even lighting")
    pb = g.add("PrimitiveStringMultiline", P, title="Terrain B",
               value="Pixel art style. seamless tileable pixel art texture of brown dirt with a few small stones, large chunky pixels, simple 16-bit RPG game tile, top-down, flat even lighting")
    m, clip, vae = zimage_models(g, P)
    sm = g.add("SK_SeamlessTiling", P, model=m[0], axes="both", align=2)
    ta = _seamless_texture(g, A, sm, clip, vae, pa[0], ts[0], 8, "Terrain A → tile")
    tb = _seamless_texture(g, B, sm, clip, vae, pb[0], ts[0], 8, "Terrain B → tile")
    w16 = g.add("SK_Tileset", T, texture_a=ta[0], texture_b=tb[0], layout="wang16", edge_noise=0.35, noise_cells=3,
                edge_shade=0.65, seed=7)
    b47 = g.add("SK_Tileset", T, texture_a=ta[0], texture_b=tb[0], layout="blob47", edge_noise=0.35, noise_cells=3,
                edge_shade=0.65, seed=7)
    save(g, O, w16[0], "tileset_wang16")
    save(g, O, b47[0], "tileset_blob47")
    g.add("PreviewAny", O, title="blob47 mapping (bitmask → tile)", source=b47[1])
    return g


def wf14(oi):
    g = Graph(oi, "14_pixelize_and_recolor", "Pixelize / Unfake + Palette Swap (no AI)",
              "Fix 'fake' AI pixel art (auto grid detection), reduce colours, recolour to retro palettes, add outlines.")
    I, F, R, O = "1 · Input", "2 · Unfake (auto-detect grid)", "3 · Palette swap", "4 · Save"
    g.note(I, "## Pixelize / unfake\n`auto_detect` finds the real pixel size of blurry or upscaled pixel art (even "
              "fractional sizes like 15.6 px) and rebuilds the exact grid. Switch to `fixed_size` to pixelize photos.\n"
              "Palette presets include PICO-8, Game Boy, C64, CGA, Endesga, Sweetie 16… Drop Lospec `.hex` files into "
              "`custom_nodes/ComfyUI-SpriteKiln/palettes/` to add more.")
    img = g.add("LoadImage", I, image=EX_FAKE)
    pp = g.add("SK_PixelPerfect", F, image=img[0], mode="auto_detect", width=64, height=64, scale=16, method="dominant",
               pre_colors=48)
    g.add("PreviewAny", F, title="Detected grid", source=pp[2])
    pal = g.add("SK_Palette", F, title="Clean colours", image=pp[0], source="auto", colors=32, preset="PICO-8",
                dither="none", dither_strength=0.5, shared_palette=True)
    cl = g.add("SK_Cleanup", F, image=pal[0], speck_size=2, max_contrast=0.12, outline="none", outline_color="#140c1c",
               outline_diagonal=False, darken=0.45)
    save(g, O, cl[0], "unfaked")
    gb = g.add("SK_Palette", R, title="Game Boy + Bayer dither", image=cl[0], source="preset", colors=4,
               preset="Game Boy (DMG)", dither="bayer-4", dither_strength=0.4, shared_palette=True)
    p8 = g.add("SK_Palette", R, title="PICO-8", image=cl[0], source="preset", colors=16, preset="PICO-8", dither="none",
               dither_strength=0.5, shared_palette=True)
    save(g, O, gb[0], "recolor_gameboy")
    save(g, O, p8[0], "recolor_pico8")
    return g


def wf15(oi):
    g = Graph(oi, "15_spritesheet_to_animation", "Spritesheet → GIF / APNG / Aseprite JSON (no AI)",
              "Slice any sheet, stabilise, export animation files for engines.")
    I, O = "1 · Sheet", "2 · Export"
    g.note(I, "## Sheet → animation\nSet rows × columns of your sheet. Exports frames, a re-packed sheet, an "
              "Aseprite-compatible JSON, GIF and APNG (native and upscaled).")
    img = load_sprite(g, I, "Sheet", EX_SHEET)
    sl = g.add("SK_SliceSheet", I, sheet=img[0], rows=4, columns=4, inset=0, order="row_major")
    row = g.add("ImageFromBatch", O, title="Pick frames (row 1)", image=sl[0], batch_index=0, length=4)
    stab = g.add("SK_Stabilize", O, frames=row[0], anchor="bottom_center")
    save(g, O, stab[0], "sheet_export", animation=True, fps=8.0)
    return g


def wf16(oi):
    g = Graph(oi, "16_8dir_animated_spritesheet", "8-Direction Animated Spritesheet (klein rotations + Wan 2.2 14B)",
              "PixelLab-style: one sprite -> 8 facings -> the same action animated in every facing -> one row per direction.")
    I, R, RS, A, S, O = ("1 · Sprite + action", "2 · Rotate (klein 4B, 5 facings)", "3 · Facings → pixel grid",
                         "4 · Animate every facing (Wan 2.2 14B, 4 steps)", "5 · Frames → pixel grid", "6 · Assemble 8-row sheet")
    g.note(I, "## 8-direction animated sheet\nLoad a **front-facing** sprite and pick an action. The sprite is rotated to "
              "S, SE, E, NE, N (W/SW/NW are mirrored), each facing is animated with the same action, frames are snapped to "
              "the sprite's grid and palette, and everything is packed into one sheet: **one row per direction** "
              "(clockwise from S, or RPG order), one column per frame, plus an Aseprite JSON.\n"
              "Runtime ≈ 5 × an HQ animation run (needs the anim models). Set `directions` to 4 for S/E/N/W only.")
    sp, cv = _sprite_in(g, I, title="Front-facing sprite")
    dp = g.add("SK_DirectionPrompts", I, directions="8", mirror_west=True, prompt_style="flux2 klein",
               template="Turn the character in the image to a {direction}. Keep exactly the same character design, outfit, "
                        "colors, proportions, size and pixel art style. Full body, centered, {background}.",
               background="gray")
    ap = g.add("SK_AnimationPreset", I, subject="the pixel art knight", action="walk", view="keep facing direction",
               custom_motion="", background="gray")
    vf = g.add("PrimitiveInt", I, title="Video frames (4n+1)", value=33)
    m, clip, vae = klein_models(g, R, loras=[(K_LORA, 0.6, "Pixel-art LoRA (klein)")])
    r1 = g.add("VAEEncode", R, pixels=cv[0], vae=vae[0])[0]
    dec = klein_sample(g, R, m, clip, vae, dp[0], cv[2], cv[3], refs=[r1])
    pp = g.add("SK_PixelPerfect", RS, image=dec[0], mode="match_reference", width=64, height=64, scale=cv[1],
               method="dominant", pre_colors=48, reference=cv[4])
    bg = g.add("SK_BackgroundRemove", RS, image=pp[0], mode="border_flood", tolerance=0.08, mask_threshold=0.5, enabled=True)
    pal = g.add("SK_Palette", RS, title="Lock to the sprite's palette", image=bg[0], source="palette_image", colors=256,
                preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True, palette_image=sp[0])
    cv2 = g.add("SK_ToCanvas", A, title="Facing → video canvas", image=pal[0], long_side=640, multiple=16,
                background="gray", custom_color="#808080", padding=8, target_width=640, target_height=640)
    vdec = wan14(g, A, ap[0], ap[1], cv2[0], cv2[2], cv2[3], vf[0])
    sel = g.add("SK_FrameSelect", S, frames=vdec[0], count=ap[2], mode="auto", skip_start=1, min_loop_fraction=0.4,
                loop=ap[4])
    fpp = g.add("SK_PixelPerfect", S, image=sel[0], mode="match_reference", width=64, height=64, scale=cv2[1],
                method="dominant", pre_colors=48, reference=cv2[4])
    fbg = g.add("SK_BackgroundRemove", S, image=fpp[0], mode="border_flood", tolerance=0.08, mask_threshold=0.5, enabled=True)
    fpal = g.add("SK_Palette", S, title="Lock to the sprite's palette", image=fbg[0], source="palette_image", colors=256,
                 preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True, palette_image=sp[0])
    stab = g.add("SK_Stabilize", S, frames=fpal[0], anchor="bottom_center")
    asm = g.add("SK_DirectionAssemble", O, images=stab[0], labels=dp[1], order="clockwise from S")
    crop = g.add("SK_CropToContent", O, frames=asm[0], margin=2, multiple=2, square=False)
    g.add("SK_SavePixelArt", O, images=crop[0], filename_prefix="spritekiln/anim8dir", upscale=6, as_animation=True,
          fps=ap[3], sheet_columns=asm[2], row_names=asm[1])
    g.add("PreviewAny", O, title="Row order", source=asm[1])
    return g


def wf17(oi):
    g = Graph(oi, "17_keyframe_inbetween", "Keyframe In-betweening (Wan 2.2 14B first → last frame)",
              "Animate between two poses you control (e.g. stand → crouch, wind-up → slash end). Chain clips for combos.")
    I, M, S, O = "1 · Start + end keyframes", "2 · In-between video (Wan 2.2 14B, 4 steps)", "3 · Frames → pixel grid", "4 · Save"
    g.note(I, "## Keyframe in-betweening\nLoad two sprites of the **same size**: the first and the last pose of the move "
              "(make the end pose with workflow 04 *Edit*, e.g. 'make the knight crouch'). Describe the motion between them.\n"
              "**Combos:** run once per move and use the previous move's end pose as the next start pose; the frames line "
              "up exactly, so the GIFs/sheets can be concatenated (workflow 15 or any engine).")
    sp = load_sprite(g, I, "Start pose", EX_SPRITE)
    ep = load_sprite(g, I, "End pose", EX_SPRITE)
    cs = g.add("SK_ToCanvas", I, title="Start canvas", image=sp[0], long_side=640, multiple=16, background="gray",
               custom_color="#808080", padding=8, target_width=640, target_height=640)
    ce = g.add("SK_ToCanvas", I, title="End canvas", image=ep[0], long_side=640, multiple=16, background="gray",
               custom_color="#808080", padding=8, target_width=640, target_height=640)
    motion = g.add("PrimitiveStringMultiline", I, title="Motion between the poses", value=(
        "2D pixel art sprite animation, side view. The knight swings his sword in a fast wide slash from the wind-up pose "
        "to the follow-through pose. Static locked camera, no zoom, plain solid flat gray background stays unchanged, "
        "crisp pixel art, consistent character design."))
    nf = g.add("PrimitiveInt", I, title="Sprite frames", value=6)
    vf = g.add("PrimitiveInt", I, title="Video frames (4n+1)", value=17)
    dec = wan14(g, M, motion[0], WAN_NEG_TEXT, cs[0], cs[2], cs[3], vf[0], end=ce[0])
    sel = g.add("SK_FrameSelect", S, frames=dec[0], count=nf[0], mode="even_inclusive", skip_start=0,
                min_loop_fraction=0.4)
    pp = g.add("SK_PixelPerfect", S, image=sel[0], mode="match_reference", width=64, height=64, scale=cs[1],
               method="dominant", pre_colors=48, reference=cs[4])
    bg = g.add("SK_BackgroundRemove", S, image=pp[0], mode="border_flood", tolerance=0.08, mask_threshold=0.5, enabled=True)
    pal = g.add("SK_Palette", S, title="Lock to the start pose's palette", image=bg[0], source="palette_image", colors=256,
                preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True, palette_image=sp[0])
    stab = g.add("SK_Stabilize", S, frames=pal[0], anchor="bottom_center")
    crop = g.add("SK_CropToContent", O, frames=stab[0], margin=2, multiple=2, square=False)
    save(g, O, crop[0], "keyframes", animation=True, fps=10.0)
    return g


def wf18(oi):
    g = Graph(oi, "18_equipment_layer", "Equipment Layer for an Animation (paper-doll / swappable gear)",
              "Re-dress every frame of a base animation with a weapon/armour reference and extract it as a "
              "transparent layer that lines up with the base sheet frame-for-frame.")
    I, M, S, L, O = ("1 · Base animation + gear", "2 · Dress every frame (klein 4B, same seed)", "3 · Back to the grid",
                     "4 · Extract layer", "5 · Save layer + preview")
    g.note(I, "## Swappable equipment\nLoad the **base** animation sheet (a body without the gear, e.g. from 10/16/17) "
              "and set its rows × columns. Load a picture of the **gear** (image 2) and describe it.\n"
              "Output: `*_layer_sheet.png` has the same layout as the base sheet and only contains the gear pixels. "
              "Your engine draws base + armour layer + weapon layer per frame. Run once per item; combine them with "
              "*PF Composite Layers*.\nTip: generate the base as a plain 'underwear' version of the character so armour "
              "only ever adds pixels. Raise `tolerance` if body pixels leak into the layer.")
    sheet = load_sprite(g, I, "Base animation sheet", EX_SHEET)
    allf = g.add("SK_SliceSheet", I, sheet=sheet[0], rows=4, columns=4, inset=0, order="row_major")
    frames = g.add("ImageFromBatch", I, title="Frames to dress (start, count)", image=allf[0], batch_index=0, length=4)
    gear = g.add("LoadImage", I, title="Gear reference (image 2)", image=EX_REF)
    item = g.add("PrimitiveStringMultiline", I, title="Gear description", value="a golden crown and a blue cape")
    txt = g.add("StringConcatenate", I, string_a="Dress the pixel art character in image 1 with", string_b=item[0],
                delimiter=" ")
    txt = g.add("StringConcatenate", I, string_a=txt[0], string_b=(
        "as shown in image 2. Keep exactly the same pose, body, proportions, position, size and pixel art style; only "
        "add the equipment. Plain solid flat gray background."), delimiter=" ")
    lst = g.add("SK_BatchToList", I, images=frames[0])
    cv = g.add("SK_ToCanvas", M, image=lst[0], long_side=1024, multiple=16, background="gray", custom_color="#808080",
               padding=0)
    m, clip, vae = klein_models(g, M, loras=[(K_LORA, 0.5, "Pixel-art LoRA (klein)")])
    r1 = g.add("VAEEncode", M, pixels=cv[0], vae=vae[0])[0]
    r2 = encode_ref(g, M, gear[0], vae)
    dec = klein_sample(g, M, m, clip, vae, txt[0], cv[2], cv[3], refs=[r1, r2])
    pp = g.add("SK_PixelPerfect", S, image=dec[0], mode="match_reference", width=64, height=64, scale=cv[1],
               method="dominant", pre_colors=48, reference=cv[4])
    bg = g.add("SK_BackgroundRemove", S, image=pp[0], mode="border_flood", tolerance=0.06, mask_threshold=0.5, enabled=True)
    ex = g.add("SK_LayerExtract", L, base=lst[0], equipped=bg[0], tolerance=0.07, min_area=3, grow=0)
    layer = g.add("SK_ListToBatch", L, title="Layer frames", images=ex[0])
    comp = g.add("SK_ListToBatch", L, title="Composite frames", images=ex[1])
    pal = g.add("SK_Palette", L, title="One palette for the layer", image=layer[0], source="auto", colors=16,
                preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True)
    g.add("SK_SavePixelArt", O, images=pal[0], filename_prefix="spritekiln/gear_layer", upscale=8, as_animation=True,
          fps=8.0, sheet_columns=0)
    g.add("SK_SavePixelArt", O, images=comp[0], filename_prefix="spritekiln/gear_preview", upscale=8, as_animation=True,
          fps=8.0, sheet_columns=0)
    return g


def wf19(oi):
    g = Graph(oi, "19_text_to_animation", "Text → Animation / VFX (no sprite needed)",
              "Describe a character, creature, prop or effect and its motion: Z-Image draws frame 1, Wan 2.2 animates it.")
    P, G, M, S, O = ("1 · What and how it moves", "2 · First frame (Z-Image + style)", "3 · Animate (Wan 2.2 14B)",
                     "4 · Frames → pixel grid", "5 · Save")
    g.note(P, "## Text → animation\nOne prompt for **what** (style preset + subject) and one for **how it moves** "
              "(animation preset or custom). Works for effects with no base sprite: pick the *VFX / effect* style and "
              "e.g. 'fire / torch flicker', 'explode / break' or a custom motion. Effects are drawn on black and keyed out.")
    st = g.add("SK_StylePreset", P, subject="a swirling purple magic fireball with sparks", style="VFX / effect",
               model="z_image (pixel LoRA)", art_width=64, art_height=64, colors=16, background="auto", long_side=640,
               extra="")
    ap = g.add("SK_AnimationPreset", P, subject="the magic fireball", action="custom",
               view="keep facing direction", custom_motion="swirls and pulses, flames licking outward, sparks flying, in a "
                                                           "seamless loop", background="black")
    vf = g.add("PrimitiveInt", P, title="Video frames (4n+1)", value=33)
    m, clip, vae = zimage_models(g, G, lora=st[8])
    ks = zimage_sample(g, G, m, clip, st[0], st[4], st[5])
    first = g.add("VAEDecode", G, samples=ks[0], vae=vae[0])
    g.add("PreviewImage", G, title="Frame 1", images=first[0])
    pp0 = g.add("SK_PixelPerfect", G, title="Frame 1 → pixel grid", image=first[0], mode="fixed_size", width=st[2],
                height=st[3], scale=16, method="dominant", pre_colors=48)
    pal0 = g.add("SK_Palette", G, image=pp0[0], source="auto", colors=st[6], preset="PICO-8", dither="none",
                 dither_strength=0.5, shared_palette=True)
    cv = g.add("SK_ToCanvas", G, title="Sprite → video canvas", image=pal0[0], long_side=640, multiple=16,
               background="black", custom_color="#000000", padding=8, target_width=640, target_height=640)
    dec = wan14(g, M, ap[0], ap[1], cv[0], cv[2], cv[3], vf[0])
    g.add("SaveAnimatedWEBP", M, title="Raw video preview", images=dec[0], filename_prefix="spritekiln/raw_text2anim",
          fps=16.0, lossless=False, quality=85, method="default")
    sel = g.add("SK_FrameSelect", S, frames=dec[0], count=ap[2], mode="auto", skip_start=1, min_loop_fraction=0.4,
                loop=ap[4])
    pp = g.add("SK_PixelPerfect", S, image=sel[0], mode="match_reference", width=64, height=64, scale=cv[1],
               method="dominant", pre_colors=48, reference=cv[4])
    bg = g.add("SK_BackgroundRemove", S, title="Key out the backdrop (on: effects / sprites)", image=pp[0],
               mode="border_flood", tolerance=0.1, mask_threshold=0.5, enabled=True)
    pal = g.add("SK_Palette", S, title="Lock to frame 1's palette", image=bg[0], source="palette_image", colors=256,
                preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True, palette_image=pal0[0])
    crop = g.add("SK_CropToContent", O, frames=pal[0], margin=2, multiple=2, square=False)
    save(g, O, crop[0], "text2anim", animation=True, fps=ap[3])
    return g


def wf20(oi):
    g = Graph(oi, "20_animated_tile", "Animated Seamless Tile (water, lava, grass…)",
              "RD animated tiles: a seamless pixel tile animated with Wan 2.2 while staying tileable in every frame.")
    P, T, M, S, O = ("1 · Tile + motion", "2 · Seamless tile (Z-Image)", "3 · Animate seamlessly (Wan 2.2 14B)",
                     "4 · Frames → tile grid", "5 · Save + 3×3 check")
    g.note(P, "## Animated tile\nGenerates a seamless tile, then animates it with the video model patched to wrap "
              "around its edges, with first frame = last frame so it loops. Every frame stays tileable.")
    ts = g.add("PrimitiveInt", P, title="Tile size (px)", value=32)
    tp = g.add("PrimitiveStringMultiline", P, title="Tile", value=(
        "Pixel art style. seamless tileable pixel art texture of deep blue ocean water with small white wave crests, "
        "large chunky pixels, simple 16-bit RPG game tile, top-down, flat even lighting"))
    mp = g.add("PrimitiveStringMultiline", P, title="Motion", value=(
        "2D pixel art game tile animation, top-down view of water, the waves ripple and flow gently, shimmering "
        "highlights move across the surface, static locked camera, seamless looping texture animation, crisp pixel art."))
    vf = g.add("PrimitiveInt", P, title="Video frames (4n+1)", value=33)
    nf = g.add("PrimitiveInt", P, title="Tile frames", value=8)
    m, clip, vae = zimage_models(g, T)
    sm = g.add("SK_SeamlessTiling", T, model=m[0], axes="both", align=2)
    tile = _seamless_texture(g, T, sm, clip, vae, tp[0], ts[0], 12, "Tile → pixel grid")
    cv = g.add("SK_ToCanvas", T, title="Tile → video canvas", image=tile[0], long_side=640, multiple=16,
               background="gray", custom_color="#808080", padding=0, target_width=640, target_height=640)
    dec = wan14(g, M, mp[0], WAN_NEG_TEXT, cv[0], cv[2], cv[3], vf[0], end=cv[0], seamless=True)
    sel = g.add("SK_FrameSelect", S, frames=dec[0], count=nf[0], mode="even", skip_start=0, min_loop_fraction=0.4)
    pp = g.add("SK_PixelPerfect", S, image=sel[0], mode="match_reference", width=32, height=32, scale=cv[1],
               method="dominant", pre_colors=48, reference=cv[4])
    pal = g.add("SK_Palette", S, title="Lock to the tile's palette", image=pp[0], source="palette_image", colors=256,
                preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True, palette_image=tile[0])
    save(g, O, pal[0], "anim_tile", animation=True, fps=8.0)
    prev = g.add("SK_TilePreview", O, image=pal[0], repeat_x=3, repeat_y=3)
    big = g.add("SK_ScaleNearest", O, image=prev[0], scale=4)
    g.add("SaveAnimatedWEBP", O, title="3×3 tiling check (animated)", images=big[0],
          filename_prefix="spritekiln/anim_tile_3x3", fps=8.0, lossless=True, quality=100, method="default")
    return g


def wf21(oi):
    g = Graph(oi, "21_color_and_upscale_tools", "Colour Ramps, Palette Swap, Hue Shift, Scale2x (no AI)",
              "Hue-shifted ramps, enemy/team colour variants, recolours and pixel-art upscaling.")
    I, C, U, O = "1 · Sprite", "2 · Colour tools", "3 · Pixel-art upscale", "4 · Save"
    g.note(I, "## Colour & upscale tools\n**Colour Ramp** builds hue-shifted ramps (shadows lean blue-violet, highlights "
              "lean yellow) from base colours — feed its `hex` into *PF Palette* (source = hex) to recolour anything with "
              "it. **Palette Swap** maps exact colours (team colours, armour tiers). **Hue Shift** recolours in OKLab. "
              "**Scale2x/3x** upscale pixel art smoothly without inventing colours.")
    sp = load_sprite(g, I, "Sprite", EX_SPRITE)
    ramp = g.add("SK_ColorRamp", C, base_colors="#b13e53 #3b5dc9 #38b764 #ef7d57", steps=6, hue_shift=25.0,
                 light_range=0.6, saturation_curve=0.3)
    g.add("PreviewImage", C, title="Ramps", images=ramp[0])
    remap = g.add("SK_Palette", C, title="Recolour with the ramps", image=sp[0], source="hex", colors=64,
                  preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True, hex_palette=ramp[1])
    pal = g.add("SK_Palette", C, title="Sprite palette (copy colours from here)", image=sp[0], source="auto", colors=32,
                preset="PICO-8", dither="none", dither_strength=0.5, shared_palette=True)
    g.add("PreviewAny", C, title="Sprite colours", source=pal[2])
    swap = g.add("SK_PaletteSwap", C, image=sp[0], from_colors="#a22633 #e43b44", to_colors="#124e89 #0099db",
                 tolerance=0.12)
    hue = g.add("SK_HueShift", C, title="Hue shift (enemy variant)", image=sp[0], hue_degrees=140.0, saturation=1.1,
                lightness=0.0)
    up2 = g.add("SK_PixelUpscale", U, image=sp[0], method="scale2x")
    up3 = g.add("SK_PixelUpscale", U, image=sp[0], method="scale3x")
    save(g, O, remap[0], "ramp_recolor")
    save(g, O, swap[0], "palette_swap")
    save(g, O, hue[0], "hue_variant")
    save(g, O, up2[0], "scale2x", upscale=4)
    save(g, O, up3[0], "scale3x", upscale=3)
    return g


# ----------------------------------------------------------------------------- HD painted / cartoon mode
HD_EDIT_TEMPLATE = ("Turn the character in the image to a {direction}. Keep exactly the same character design, outfit, "
                    "colors, proportions, size and hand-painted art style. Full body, centered, {background}.")


def hd_cutout(g, grp, image, out_long_side, trim=True, title=None, mode="mask_and_key", enabled=True):
    mask = birefnet_mask(g, grp, image)
    kw = {"title": title} if title else {}
    return g.add("SK_HDCutout", grp, image=image, mode=mode, tolerance=0.08, remove_shadow=True, soft_edges=True,
                 decontaminate=True, trim=trim, margin=8, out_long_side=out_long_side, enabled=enabled,
                 foreground_mask=mask, **kw)


def wf22(oi):
    g = Graph(oi, "22_hd_text_to_art", "HD 2D Art from Text (painted / cartoon, no pixels)",
              "Rayman-style painted cartoon characters, creatures, props, icons and VFX as clean transparent PNGs.")
    P, M, S, O = "1 · Prompt & style", "2 · Generate (Z-Image Turbo, no pixel LoRA)", "3 · Soft-edged cutout", "4 · Save"
    g.note(P, "## HD 2D art\nNo pixel grid, no palette: full-resolution painted / cartoon art. The style sets the "
              "prompt, the model resolution (portrait for characters) and the output size. Sprites get a plain "
              "background that is removed with soft anti-aliased edges and edge-colour cleanup.")
    st = g.add("SK_HDStylePreset", P, subject="a cheerful lanky hero with a big red scarf, a blue tunic and a wooden sword",
               style="painted cartoon platformer (Rayman-like)", background="auto", out_long_side=0, extra="")
    m, clip, vae = zimage_models(g, M, lora=None)
    ks = zimage_sample(g, M, m, clip, st[0], st[2], st[3])
    dec = g.add("VAEDecode", M, samples=ks[0], vae=vae[0])
    g.add("PreviewImage", M, title="Raw model output", images=dec[0])
    cut = hd_cutout(g, S, dec[0], st[4], enabled=st[5])
    save(g, O, cut[0], "hd_art", upscale=1)
    return g


def wf23(oi):
    g = Graph(oi, "23_hd_edit", "HD 2D Edit with Text + Image References (FLUX.2 klein)",
              "Change outfits, props, colours or poses of HD 2D art; optional second reference image.")
    I, M, S, O = "1 · Art + reference", "2 · Edit (klein 4B, no pixel LoRA)", "3 · Cutout", "4 · Save"
    g.note(I, "## HD edit\nLoad HD art (transparent PNG or on a plain background). It is placed on a 1024 canvas, "
              "edited by FLUX.2 klein, and cut out again at full resolution. The extra reference (image 2) is optional "
              "— bypass it with Ctrl+B.")
    art = load_sprite(g, I, "Art to edit", EX_HD)
    cv = g.add("SK_HDCanvas", I, image=art[0], width=1024, height=1024, padding_percent=8.0, background="white",
               trim_first=True)
    ref = g.add("LoadImage", I, title="Extra reference (image 2)", image=EX_REF)
    m, clip, vae = klein_models(g, M)
    r1 = g.add("VAEEncode", M, pixels=cv[0], vae=vae[0])[0]
    r2 = encode_ref(g, M, ref[0], vae)
    dec = klein_sample(g, M, m, clip, vae,
                       "Edit the character in image 1: give them a golden crown and a purple cape. Keep everything else "
                       "identical: same pose, size, proportions and hand-painted art style, plain white background.",
                       cv[1], cv[2], refs=[r1, r2])
    g.add("PreviewImage", M, title="Raw edit", images=dec[0])
    out_size = g.add("PrimitiveInt", O, title="Output long side (px)", value=768)
    cut = hd_cutout(g, S, dec[0], out_size[0])
    save(g, O, cut[0], "hd_edit", upscale=1)
    return g


def wf24(oi):
    g = Graph(oi, "24_hd_rotate_8_directions", "HD 2D Rotate → 4/8 Directions (FLUX.2 klein)",
              "Turnarounds of painted / cartoon characters with mirrored west views and a shared-size sheet.")
    I, M, S, O = "1 · Art + directions", "2 · One edit per direction (klein 4B)", "3 · Cutouts", "4 · Assemble & save"
    g.note(I, "## HD rotations\nSame as workflow 06 without pixel snapping. Each facing is cut out with soft edges; "
              "all facings share one crop box and size so they line up in the sheet.")
    art = load_sprite(g, I, "Front-facing art", EX_HD)
    cv = g.add("SK_HDCanvas", I, image=art[0], width=832, height=1216, padding_percent=6.0, background="white",
               trim_first=True)
    dp = g.add("SK_DirectionPrompts", I, directions="8", mirror_west=True, prompt_style="flux2 klein",
               template=HD_EDIT_TEMPLATE, background="white")
    m, clip, vae = klein_models(g, M)
    r1 = g.add("VAEEncode", M, pixels=cv[0], vae=vae[0])[0]
    dec = klein_sample(g, M, m, clip, vae, dp[0], cv[1], cv[2], refs=[r1])
    cut = hd_cutout(g, S, dec[0], 0, trim=False)
    asm = g.add("SK_DirectionAssemble", O, images=cut[0], labels=dp[1], order="clockwise from S")
    fit = g.add("SK_HDCutout", O, title="Shared crop + size", image=asm[0], mode="none", tolerance=0.08,
                remove_shadow=False, soft_edges=True, decontaminate=False, trim=True, margin=8, out_long_side=512,
                enabled=True)
    g.add("SK_SavePixelArt", O, images=fit[0], filename_prefix="spritekiln/hd_rotate", upscale=1, as_animation=True,
          fps=4.0, sheet_columns=0, row_names=asm[1])
    return g


def wf25(oi):
    g = Graph(oi, "25_hd_animate", "HD 2D Animate – presets or free text (Wan 2.2 14B)",
              "Painted / cartoon character animation from one image: 35 actions or your own sentence, HD sheet + GIF/APNG.")
    I, M, S, O = "1 · Art + action", "2 · Video (Wan 2.2 14B, 4 steps, no pixel LoRA)", "3 · Frames → cutouts", "4 · Save"
    g.note(I, "## HD animation\nThe video model animates the art directly (the pixel-animate LoRA is off). Frames are "
              "picked (loop point or start→end), cut out with soft edges, stabilised on the feet and saved with one "
              "shared crop. APNG keeps full colour and soft alpha; the GIF is a preview.")
    art = load_sprite(g, I, "Art", EX_HD)
    cv = g.add("SK_HDCanvas", I, image=art[0], width=640, height=640, padding_percent=12.0, background="gray",
               trim_first=True)
    ap = g.add("SK_AnimationPreset", I, subject="the cartoon hero", action="sword slash", view="side view",
               custom_motion="", background="gray", art="HD 2D (painted / cartoon)")
    vf = g.add("PrimitiveInt", I, title="Video frames (4n+1)", value=33)
    dec = wan14(g, M, ap[0], ap[1], cv[0], cv[1], cv[2], vf[0], pixel=False)
    g.add("SaveAnimatedWEBP", M, title="Raw video preview", images=dec[0], filename_prefix="spritekiln/raw_hd_anim",
          fps=16.0, lossless=False, quality=85, method="default")
    sel = g.add("SK_FrameSelect", S, frames=dec[0], count=ap[2], mode="auto", skip_start=1, min_loop_fraction=0.4,
                loop=ap[4])
    cut = hd_cutout(g, S, sel[0], 0, trim=False)
    stab = g.add("SK_Stabilize", S, frames=cut[0], anchor="bottom_center")
    size = g.add("PrimitiveInt", O, title="Output long side (px)", value=384)
    fit = g.add("SK_HDCutout", O, title="Shared crop + size", image=stab[0], mode="none", tolerance=0.08,
                remove_shadow=False, soft_edges=True, decontaminate=False, trim=True, margin=8, out_long_side=size[0],
                enabled=True)
    save(g, O, fit[0], "hd_anim", animation=True, fps=ap[3], upscale=1)
    return g


def wf26(oi):
    g = Graph(oi, "26_hd_textures_and_parallax", "HD Seamless Texture + Parallax Background Layer",
              "Painted tileable textures (both axes) and horizontally looping parallax backgrounds.")
    T, B, O = "1 · Seamless texture (tiles both ways)", "2 · Parallax layer (loops horizontally)", "3 · Save"
    g.note(T, "## Painted textures & parallax\nThe texture wraps in x and y. The parallax layer wraps only "
              "horizontally, so it scrolls forever in a side-scroller.")
    m, clip, vae = zimage_models(g, T, lora=None)
    st = g.add("SK_HDStylePreset", T, subject="mossy cobblestone path with small flowers", style="texture (painted, tileable)",
               background="scene", out_long_side=0, extra="")
    sm = g.add("SK_SeamlessTiling", T, model=m[0], axes="both", align=2)
    ks = zimage_sample(g, T, sm, clip, st[0], st[2], st[3])
    tex = g.add("SK_SeamlessVAEDecode", T, samples=ks[0], vae=vae[0], axes="both", pad_latent=8)
    tsz = g.add("ImageScale", T, title="Texture size", image=tex[0], upscale_method="lanczos", width=512, height=512,
                crop="disabled")
    save(g, O, tsz[0], "hd_texture", upscale=1)
    prev = g.add("SK_TilePreview", O, image=tsz[0], repeat_x=3, repeat_y=2)
    g.add("PreviewImage", O, title="Tiling check", images=prev[0])
    bst = g.add("SK_HDStylePreset", B, subject="enchanted forest with giant mushrooms and distant hills",
                style="painted background / parallax layer", background="scene", out_long_side=0, extra="")
    bm = g.add("SK_SeamlessTiling", B, model=m[0], axes="x", align=2)
    bks = zimage_sample(g, B, bm, clip, bst[0], bst[2], bst[3])
    bg = g.add("SK_SeamlessVAEDecode", B, samples=bks[0], vae=vae[0], axes="x", pad_latent=8)
    save(g, O, bg[0], "hd_parallax", upscale=1)
    bprev = g.add("SK_TilePreview", O, image=bg[0], repeat_x=2, repeat_y=1)
    g.add("PreviewImage", O, title="Horizontal loop check", images=bprev[0])
    return g


def wf27(oi):
    g = Graph(oi, "27_character_to_rig_parts", "Character → Rig Parts for Cutout Animation (SAM3 + klein inpainting)",
              "Split a character into named parts (head, torso, arms, legs, weapon…), inpaint the hidden sections, export "
              "aligned layers + JSON for Spine / DragonBones / Godot Skeleton2D.")
    I, D, C, O = ("1 · Character + part list", "2 · Find each part (SAM3 text prompts)",
                  "3 · Inpaint hidden sections (klein 4B)", "4 · Aligned layers + JSON")
    g.note(I, "## Character → rig parts\nWorks best on a **neutral pose** (T- or A-pose, limbs apart) on a plain "
              "background. List the parts in **back-to-front draw order**; names are also the SAM3 prompts ('left arm', "
              "'sword', 'tail', 'wing'…). Every visible pixel goes to the front-most part that claims it; leftovers become "
              "a 'rest' layer. For each part, klein inpaints only where the part is covered by parts in front of it "
              "(e.g. the shoulder under the head), on the original canvas — visible pixels never move or change.\n"
              "Output (output/spritekiln/parts/…): full-canvas aligned PNGs (drop them into Spine/DragonBones/Godot as "
              "layers, in order), cropped PNGs, parts.json (box, draw order, pivot guess) and composite.png.")
    art = load_sprite(g, I, "Character (neutral pose)", EX_HD)
    cv = g.add("SK_HDCanvas", I, image=art[0], width=1024, height=1024, padding_percent=5.0, background="white",
               trim_first=True)
    pp = g.add("SK_PartPrompts", I, parts="cape, left arm, left leg, right leg, torso, head, right arm, sword",
               character="the character",
               completion_template="Extend the {part} of {character} into the empty area: the {part} continues "
                                   "underneath the parts that covered it, so draw its hidden section (same colours, "
                                   "shading and art style) to make it whole.")
    sam = g.add("CheckpointLoaderSimple", D, title="SAM3.1", ckpt_name=SAM3)
    seg_c = g.add("CLIPTextEncode", D, title="Part prompt", text=pp[0], clip=sam[1])
    det = g.add("SAM3_Detect", D, model=sam[0], image=cv[0], threshold=0.35, refine_iterations=2,
                individual_masks=False, conditioning=seg_c[0])
    body = birefnet_mask(g, D, cv[0])
    lay = g.add("SK_PartLayout", D, image=cv[0], part_masks=det[0], names=pp[2], prompts=pp[1], hidden_reach_percent=25.0,
                min_part_px=64, rest_layer=True, character_mask=body)
    m, clip, vae = klein_models(g, C)
    enc = g.add("VAEEncode", C, pixels=lay[0], vae=vae[0])
    lat = g.add("SetLatentNoiseMask", C, samples=enc[0], mask=lay[1])
    dec = klein_sample(g, C, m, clip, vae, lay[4], cv[1], cv[2], refs=[enc[0]], latent=lat[0])
    cmask = birefnet_mask(g, C, dec[0])
    part = g.add("SK_PartCutout", O, image=cv[0], visible_mask=lay[2], completed=dec[0], inpaint_mask=lay[1],
                 completed_mask=cmask)
    g.add("SK_SaveParts", O, layers=part[0], names=lay[3], folder="spritekiln/parts/character")
    return g


MOGE = "moge_2_vitl_normal_fp16.safetensors"


def _normal_outputs(g, O, art, nm, prefix, upscale):
    g.add("SK_SavePixelArt", O, title="Normal map", images=nm[0], filename_prefix="spritekiln/" + prefix + "_normal",
          upscale=upscale, as_animation=False, fps=8.0, sheet_columns=0)
    g.add("PreviewImage", O, title="Height", images=nm[1])
    lit = g.add("SK_NormalLightPreview", O, image=art[0], normal_map=nm[0], frames=12, elevation_deg=35.0, ambient=0.3,
                convention="opengl")
    g.add("SK_SavePixelArt", O, title="Lit preview (light circling)", images=lit[0],
          filename_prefix="spritekiln/" + prefix + "_lit", upscale=upscale, as_animation=True, fps=10.0, sheet_columns=0)


def wf28(oi):
    g = Graph(oi, "28_sprite_normal_maps", "Sprite Normal Maps for 2D Lighting (pixel or HD, no AI)",
              "Normal + height maps so engine lights shade sprites like HD-2D games; works on single sprites and whole sheets.")
    I, N, O = "1 · Sprite or sheet", "2 · Normals", "3 · Save + lit preview"
    g.note(I, "## Sprite normal maps\nModel-free and pixel-exact: a rounded 'pillow' from each sprite's silhouette "
              "(`inflate`, `radius`) plus relief from its shading (`detail`). Feed a whole spritesheet — every frame gets "
              "its normals in the same layout, ready as the engine's normal texture.\n**Engines:** Godot 4 → "
              "CanvasTexture `normal_texture` (OpenGL). Unity URP 2D → Sprite *Secondary Textures* `_NormalMap` "
              "(OpenGL). Unreal Paper2D → DirectX convention.\nPixel art: `detail_blur` 0. HD art: 1–2, or use 28b.")
    art = load_sprite(g, I, "Sprite or sheet (transparent PNG)", EX_SPRITE)
    nm = g.add("SK_SpriteNormalMap", N, image=art[0], inflate=1.0, radius=0, detail=0.35, detail_blur=0.0, strength=1.0,
               convention="opengl", invert_detail=False)
    _normal_outputs(g, O, art, nm, "sprite", 8)
    return g


def wf28b(oi):
    g = Graph(oi, "28b_hd_normal_maps_ai", "HD Normal Maps with AI Shape (MoGe + sprite inflation)",
              "For painted / HD art: MoGe reads real 3D shape (faces, folds, armour) and is blended with silhouette rounding.")
    I, A, N, O = "1 · HD art", "2 · AI normals (MoGe-2)", "3 · Blend with sprite normals", "4 · Save + lit preview"
    g.note(I, "## HD normal maps (AI)\nMoGe-2 estimates surface normals from the art; they are blended "
              "(`ai_weight`) with the model-free silhouette normals so edges still round off cleanly. Use 28 for pixel "
              "art (MoGe needs more than a few dozen pixels to see shape).")
    art = load_sprite(g, I, "HD art (transparent PNG)", EX_HD)
    flat = g.add("SK_Flatten", A, image=art[0], background="#808080")
    mm = g.add("LoadMoGeModel", A, model_name=MOGE)
    geo = g.add("MoGeInference", A, moge_model=mm[0], image=flat[0], resolution_level=9, fov_x_degrees=0.0,
                batch_size=1, force_projection=True, apply_mask=False, refine_steps=0)
    ain = g.add("MoGeRender", A, moge_geometry=geo[0], output="normal_opengl")
    g.add("PreviewImage", A, title="MoGe normals", images=ain[0])
    nm = g.add("SK_SpriteNormalMap", N, image=art[0], inflate=1.0, radius=0, detail=0.2, detail_blur=1.5, strength=1.0,
               convention="opengl", invert_detail=False, ai_normals=ain[0], ai_weight=0.6)
    _normal_outputs(g, O, art, nm, "hd", 1)
    return g


WORKFLOWS = [wf01, wf02, wf03, wf04, wf05, wf06, wf07, wf08, wf09, wf09b, wf10, wf11, wf12, wf13, wf14, wf15, wf16, wf17, wf18, wf19, wf20, wf21, wf22, wf23, wf24, wf25, wf26, wf27, wf28, wf28b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--object-info", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "workflows"))
    a = ap.parse_args()
    if a.object_info:
        oi = json.load(open(a.object_info))
    else:
        oi = json.load(urllib.request.urlopen(a.server + "/object_info"))
    os.makedirs(os.path.join(a.out, "api"), exist_ok=True)
    index = []
    for fn in WORKFLOWS:
        g = fn(oi)
        g.save(os.path.join(a.out, "SpriteKiln_%s.json" % g.name), os.path.join(a.out, "api", "%s.api.json" % g.name))
        index.append({"file": "SpriteKiln_%s.json" % g.name, "title": g.title, "description": g.description,
                      "nodes": len(g.nodes)})
        print("built %-45s %3d nodes" % (g.name, len(g.nodes)))
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
