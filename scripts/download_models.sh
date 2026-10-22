#!/usr/bin/env bash
# SpriteKiln model downloader. Resumable; uses aria2c (brew install aria2) when present, else curl. Usage:
#   ./download_models.sh [core] [anim] [hq-rotate] | all     (default: core anim)
#   core ≈ 37 GB (images/edits/rotations/tiles), anim ≈ 36 GB (all animation), hq-rotate ≈ 31 GB,
#   parts ≈ 2.4 GB (character -> parts for cutout rigs, AI normal maps)
# Override the models root with MODELS_DIR=/path/to/models.
set -u
if [ -z "${MODELS_DIR:-}" ]; then
  # Comfy Desktop shared folder, else a git checkout of ComfyUI
  for c in "$HOME/ComfyUI-Shared/models" "$HOME/ComfyUI/models"; do [ -d "$c" ] && MODELS_DIR="$c" && break; done
fi
MODELS_DIR="${MODELS_DIR:-$HOME/ComfyUI-Shared/models}"
HF="https://huggingface.co"
fail=0

get() { # get <subfolder> <filename> <url>
  local dir="$MODELS_DIR/$1" name="$2" url="$3"
  mkdir -p "$dir"
  if [ -f "$dir/$name" ] && [ ! -f "$dir/$name.part" ] && [ ! -f "$dir/$name.aria2" ]; then echo "ok    $1/$name"; return; fi
  echo "fetch $1/$name"
  touch "$dir/$name.part"
  local try
  for try in 1 2 3 4 5 6 7 8 9 10; do
    if command -v aria2c >/dev/null 2>&1; then
      # 8 parallel connections per file, resumable (.aria2 control file)
      if aria2c -q -c -x 8 -s 8 -k 16M --file-allocation=none --max-tries=5 --retry-wait=3 \
           --lowest-speed-limit=20K -d "$dir" -o "$name" "$url"; then
        rm -f "$dir/$name.part"; echo "done  $1/$name"; return
      fi
    else
      # abort stalled transfers (<20 KB/s for 30 s) and resume from where we stopped
      if curl -fL -sS --retry 3 --retry-delay 3 --speed-limit 20000 --speed-time 30 -C - -o "$dir/$name" "$url"; then
        rm -f "$dir/$name.part"; echo "done  $1/$name"; return
      fi
    fi
    echo "retry $1/$name (attempt $try)"; sleep 3
  done
  echo "FAIL  $1/$name  ($url)"; fail=1
}

core() {
  # Z-Image Turbo (fast text->pixel art). Skips files that already exist.
  get diffusion_models z_image_turbo_bf16.safetensors "$HF/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
  get text_encoders    qwen_3_4b.safetensors          "$HF/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
  get vae              ae.safetensors                 "$HF/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
  # FLUX.2 klein 4B (Apache-2.0): reference / edit / rotation / spritesheets. bf16 because MPS has no fp8.
  get diffusion_models flux-2-klein-4b.safetensors      "$HF/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors"
  get diffusion_models flux-2-klein-base-4b.safetensors "$HF/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-base-4b.safetensors"
  get vae              flux2-vae.safetensors            "$HF/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"
  # Background removal (core ComfyUI RemoveBackground node)
  get background_removal birefnet.safetensors "$HF/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors"
  # Pixel-art LoRAs (all Apache-2.0)
  get loras spritekiln_zimage_pixel_art.safetensors      "$HF/tarn59/pixel_art_style_lora_z_image_turbo/resolve/main/pixel_art_style_z_image_turbo.safetensors"
  get loras spritekiln_klein_pixel_art.safetensors       "$HF/Limbicnation/pixel-art-lora/resolve/main/pytorch_lora_weights.comfyui.safetensors"
  get loras spritekiln_klein_base_4walk_sheet.safetensors "$HF/svntax-dev/pixel_spritesheet_4walk_small_lora_v1/resolve/main/pixel_4walk_small_flux2_klein_base_4b_v1.safetensors"
  get loras spritekiln_klein_2x2_multiview.safetensors   "$HF/fal/flux-2-klein-4b-spritesheet-lora/resolve/main/mFtSubaAlDqtHyKk6eIPi_pytorch_lora_weights_comfy_converted.safetensors"
}

anim() {
  # Wan 2.2 I2V 14B (GGUF Q4, needs ComfyUI-GGUF) + pixel sprite animation LoRA + 4-step Lightning LoRAs.
  # (The 5B TI2V model is not used: it produces corrupted colours on Apple MPS and is slower than 14B + Lightning.)
  get unet wan2.2_i2v_high_noise_14B_Q4_K_M.gguf "$HF/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"
  get unet wan2.2_i2v_low_noise_14B_Q4_K_M.gguf  "$HF/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
  get vae  wan_2.1_vae.safetensors "$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
  get text_encoders umt5_xxl_fp16.safetensors "$HF/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors"
  get loras spritekiln_wan22_pixel_animate.safetensors "$HF/styly-agents/Wan2-2-pixel-animate/resolve/main/wan2.2_animate_adapter_model.safetensors"
  get loras wan2.2_i2v_lightning_4step_high.safetensors "$HF/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"
  get loras wan2.2_i2v_lightning_4step_low.safetensors  "$HF/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors"
}

hq_rotate() {
  # Qwen-Image-Edit-2511 (GGUF) + fal Multiple-Angles LoRA (96 camera poses) + Lightning 4-step
  get unet qwen-image-edit-2511-Q4_K_M.gguf "$HF/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf"
  get text_encoders qwen_2.5_vl_7b.safetensors "$HF/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors"
  get vae qwen_image_vae.safetensors "$HF/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"
  get loras qwen-image-edit-2511-multiple-angles-lora.safetensors "$HF/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors"
  get loras Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors "$HF/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
}

parts() {
  # Character -> parts for cutout rigs (SAM3.1 text-prompted segmentation; klein from core does the inpainting)
  # and AI normal maps for HD art (MoGe-2).
  get geometry_estimation moge_2_vitl_normal_fp16.safetensors "$HF/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors"
  get checkpoints sam3.1_multiplex_fp16.safetensors "$HF/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"
}

[ $# -eq 0 ] && set -- core anim
for tier in "$@"; do
  case "$tier" in
    core) core ;; anim|hq-anim) anim ;; hq-rotate) hq_rotate ;; parts) parts ;; all) core; anim; hq_rotate; parts ;;
    *) echo "unknown tier: $tier (core|anim|hq-rotate|parts|all)"; exit 2 ;;
  esac
done
[ $fail -eq 0 ] && echo "ALL DOWNLOADS COMPLETE" || { echo "SOME DOWNLOADS FAILED (re-run to resume)"; exit 1; }
