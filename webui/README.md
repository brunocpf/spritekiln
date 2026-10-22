# SpriteKiln web UI

A Retro Diffusion / PixelLab-style front end for the SpriteKiln ComfyUI workflows. It runs in
your browser, talks to your local ComfyUI, and needs nothing beyond the Python that ships with macOS:
no npm, no build step, no extra packages.

## Start it

ComfyUI must already be running (by default at `http://127.0.0.1:8188`). Then, from the project folder (the one holding `webui/` and `workflows/`):

```bash
python3 webui/serve.py
```

and open <http://127.0.0.1:8189>.

Options:

| flag | default | |
|---|---|---|
| `--port` | `8189` | port of the web UI |
| `--host` | `127.0.0.1` | `0.0.0.0` makes it reachable from other devices on your network (no auth, so only on a trusted network) |
| `--comfy` | `http://127.0.0.1:8188` | the ComfyUI server to drive |
| `--workflows` | `../workflows` | folder holding `index.json` and `api/*.api.json` |
| `-v` | off | log every request |

## How it works

`serve.py` (standard library only) does three things:

1. Serves `index.html`, `app.js` and `style.css` from this folder.
2. Reverse-proxies the ComfyUI API, so the browser stays on one origin: `/prompt`, `/queue`,
   `/history/<id>`, `/interrupt`, `/upload/image`, `/view`, `/object_info`, `/system_stats`, plus the
   `/ws` websocket (live step progress and latent previews). It drops the browser's `Origin` header
   so ComfyUI's cross-origin protection does not reject the proxied calls.
3. Lists the workflows from disk on every request, so regenerated workflows show up after a page reload,
   without restarting anything:
   - `GET /sk/workflows`: the entries of `workflows/index.json` with an `id` (`SpriteKiln_01_text_to_pixel_art.json`
     → `01_text_to_pixel_art`), plus any `api/*.api.json` that the index does not list.
   - `GET /sk/workflow/<id>`: the API graph `workflows/api/<id>.api.json`.

The page then does what `scripts/sk_run.py` does: upload images, patch the graph, `POST /prompt`, poll
`/queue` + `/history/<prompt_id>`, and show the files from `/view`.

## Sidebar → workflows

| Group | Workflows |
|---|---|
| Generate | 01 text → pixel art, 02 image → pixel art, 03 reference images → pixel art, 19 text → animation / VFX |
| Edit | 04 edit, 05 inpaint, 21 colour ramps / palette swap / hue shift / Scale2x |
| Rotate & views | 06 rotate 4/8 directions, 07 rotate HQ (Qwen), 08 object 2×2 views |
| Sheets & animation | 09 walk sheet from text, 09b walk sheet from reference, 10 animate sprite, 11 seamless loop, 16 8-direction animated sheet, 17 keyframe in-betweening, 18 equipment layer |
| Tiles | 12 seamless tile, 13 autotile tileset, 20 animated tile |
| Tools | 14 pixelize / unfake + recolour, 15 spritesheet → GIF / APNG / Aseprite JSON |
| HD 2D art | 22 HD art from text, 23 HD edit, 24 HD rotate 4/8 directions, 25 HD animate, 26 HD seamless texture + parallax layer |
| Rigging | 27 character → rig parts (cutout animation) |
| Lighting | 28 sprite normal maps (pixel or HD, no AI), 28b HD normal maps with AI shape |

A workflow whose number is not in this table (a new one) appears under **Other**. The groups are the
`GROUPS` constant at the top of `app.js`.

## How the form is built

Nothing is hand-written per workflow. The form is read from the API graph and from ComfyUI's `/object_info`,
which supplies the dropdown options, min/max/step and tooltips:

- **Image**: every `LoadImage`, labelled by its title. Click, drag and drop, or paste (⌘V). The file is
  uploaded to ComfyUI's input folder as `sk_<hash>_<name>` (`overwrite=true`), so a re-upload of the same picture
  reuses the file. You can also pick a file already in the input folder.
- **Prompt**: `CLIPTextEncode` nodes whose text is typed in rather than wired, and every
  `PrimitiveString` / `PrimitiveStringMultiline` node (labelled by title). Prompts titled "Negative" go
  under Advanced.
- **Main sections**: all typed-in inputs of `SK_StylePreset`, `SK_HDStylePreset`, `SK_AnimationPreset` (incl. its
  `art` pixel / HD switch), `SK_DirectionPrompts`, `SK_PartPrompts` (parts list, character), `SK_RegionMask`,
  `SK_ColorRamp`, `SK_PaletteSwap`, `SK_HueShift`, an `ImageScale` that has its own title (e.g. "Texture size"), and
  `SK_SliceSheet` when it slices your own uploaded image. A few rarely touched preset inputs (model, long side,
  extra, prompt template, background, completion template) are under Advanced as "… — more". Two nodes with the same
  name are told apart by their style ("Style · texture (painted, tileable)") or a number ("Palette (2)").
- **Settings**: every `PrimitiveInt` (labelled by title; "(4n+1)" frame counts are rounded for you), one
  **Seed** with 🎲 (new seed) and *Random each run* / *🔒 Locked* (applies to every sampler seed:
  `KSampler`, `KSamplerAdvanced`, `RandomNoise`), and **Animation FPS** for animation saves when the FPS
  is a typed-in value.
- **Normal map** (28, 28b): `SK_SpriteNormalMap`'s inflate, radius, detail, detail blur, strength, convention
  and AI weight. The lit preview (light circling the sprite) is shown as a GIF.
- **Advanced** (collapsed): `SK_Palette`, `SK_Cleanup`, `SK_BackgroundRemove`, `SK_PixelPerfect`, `SK_ToCanvas`,
  `SK_Tileset`, `SK_Stabilize`, and for the HD / rigging tools `SK_HDCanvas` (width, height, padding %, background),
  `SK_HDCutout` (output long side, margin, soft edges, remove shadow), `SK_PartLayout` (hidden reach %, minimum part
  size, rest layer) and `SK_PartCutout` (expand %, when the workflow sets it). In the no-AI workflows (14, 15, 21) these are the point of the tool, so they are
  shown on the main form instead.

Only inputs that are typed in, not wired from another node, appear; a wired input is driven by something
else in the graph. Your last values per workflow are remembered in this browser. **Reset** restores the
workflow's defaults.

## Presets and looks (reuse settings across tools)

- **Presets ▾** on every tool section (Style, Palette, Pixel grid, Cleanup & outline, Background removal, Model canvas…)
  saves that section's values under a name. Presets are stored per section *type*, so a Palette preset saved in
  14 appears in the Palette sections of 01, 02, 04 and every other tool that has a Palette section. **Apply** sets the
  values; **✕** deletes your own presets.
- **Looks ▾** (under the tool's title) bundles several sections: *Save current* stores every section of the tool, and
  *Apply* sets each matching section of the current tool and says which sections it changed or skipped.
- A preset never changes the **subject**, the **character** or the **model** (that is tied to the AI model a tool loads).
- Reference locks are protected. A palette wired to the original sprite ("Lock to the sprite's palette") keeps its
  source and colour count, and a pixel grid wired to a reference keeps its `match_reference` mode. The status line
  says when something was kept.
- **Built-in presets**:
  - Style: *FFBE-style chibi 96×120 (tested)*, *GBA chibi 32×32*, *FFT-style unit 32×48*, *HD anime JRPG 80×112*
  - Pixel grid: *Detailed (64 colour clusters)*, *Smooth shading (median)*
  - Palette: *Clean, no dither*, *Rich 48 colours*, *PICO-8*, *Game Boy + Bayer*
  - Cleanup: *Keep detail*, *Soft auto outline*, *Dark outline*, *Heavy despeckle*
  - Background removal: *Clear enclosed gaps (global key, tested)*, for edits (04) where gray backdrop is trapped inside a staff ring or between limbs
  - Look: *FFBE-style chibi sprite (tested)*. This is the combination checked end to end in 03 with a style-reference
    sprite: style HD-2D character, 96×120, 40 colours, 64 colour clusters, speck size 2, no outline, no dither.
    Its note gives the subject wording that worked. The other style presets have not been generation-tested.
- Presets and looks live in this browser's `localStorage` (`sk.presets`, `sk.looks`). The built-ins are the
  `BUILTIN_PRESETS` / `BUILTIN_LOOKS` constants in `app.js`.

## Running, progress and cancelling

**Generate** (or ⌘↵ / Ctrl+↵ anywhere) queues the job. You can queue several; each gets a card on the right:

- *Waiting in the ComfyUI queue — N jobs ahead* while other jobs run (ComfyUI runs one job at a time).
- *Running · node · step x/y · elapsed*, with a progress bar. A live latent preview appears only if ComfyUI
  was started with a preview method (e.g. `--preview-method auto`); otherwise you see progress without a picture.
- Results of finished stages show up while the rest is still running (useful for the 40-minute 8-direction sheet).
- Errors: validation errors from `/prompt` (node and input), and `execution_error` from the history
  (`status.messages`) with the failing node, exception and the end of the traceback.
- **Cancel** removes a waiting job from the queue (`POST /queue {delete}`), or stops it while it runs
  (`POST /interrupt {prompt_id}`, which only interrupts if that job is the one running).

Jobs are tracked by prompt id, so you can reload the page or close it and come back: the card
picks the job up again from `/queue` and `/history`.

## Results

- Every image a run outputs is shown: the saved SpriteKiln files first, then ComfyUI's intermediate
  previews (raw model output, tiling checks), and text outputs (direction order, detected grid, colours).
- Pixel art is drawn with `image-rendering: pixelated`. **Zoom** 1×–16× is true pixel zoom of the native
  sprite (the `_x8` copies are scaled back down first); *Fit* fits the card. Large images (raw model output)
  always fit. **Checkerboard** shows transparency.
- Animated GIF / WebP outputs play inline.
- **HD 2D art and Rigging** (22–27, and any workflow with `_hd_` in its name, e.g. 28b) are not pixel art: their results are drawn smoothly (no `pixelated`), always
  fitted (the pixel zoom does not apply), on the checkerboard so the soft RGBA edges show. HD animations list
  only the sheet, so the card adds the APNG next to it (APNG keeps soft alpha; GIF cannot).
- **Rig parts** (`SK_SaveParts`): the card shows `composite.png` and reads `parts.json` from the same folder
  (`output/spritekiln/parts/<name>_NNN/`) to list every part with a thumbnail, its full-canvas layer
  (`NN_<part>.png`, for Spine / DragonBones / Godot) and its cropped version (`cropped/NN_<part>.png`), plus a
  `parts.json` link.
- Downloads: besides the file shown, the card links every companion file `SK_SavePixelArt` writes in
  the same subfolder, if it exists: `<base>_sheet.png` (native), `_sheet.json` (Aseprite), `.gif` and `_x8.gif`,
  `.apng.png` and `_x8.apng.png`, and the native `<base>.png` for single images.
- **Use as input in…** sends a result (the native-size PNG or sheet when there is one) to another tool's first
  image input: generate → animate, edit → rotate, and so on.
- History: the last 80 runs are kept in this browser's `localStorage` (just the run details; the
  images stay in ComfyUI's output folder). **Reuse settings** loads a run's values and seed back into the
  form. **✕** and **Clear history** forget runs; they never delete files.

## Files

| file | |
|---|---|
| `serve.py` | static server + ComfyUI proxy + workflow listing |
| `index.html` | page shell |
| `app.js` | everything else: form builder, run/poll/websocket, results, history |
| `style.css` | layout and theme (follows the system light/dark setting; the ☾/☀ button overrides it) |

## Names

Node classes were renamed from `PF_*` to `SK_*` (the old IDs stay registered as aliases). The UI treats both as the
same node, so workflows using either prefix get the same form, labels and presets. `serve.py` accepts GUI workflow files
named `SpriteKiln_NN_*.json` or the older `PixelForge_NN_*.json`. Output and example-input names come from the
workflows and ComfyUI's history, not from the UI, so the `spritekiln/` output subfolder needs no setting here.
Data saved by the PixelForge-era UI in this browser (`pf.*` keys: history, forms, presets, looks, theme) is moved to
`sk.*` keys on the first load.

