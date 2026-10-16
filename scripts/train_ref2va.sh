#!/usr/bin/env bash
# Legacy MiniMax-H3/DiffSynth training entry point. The active baseline uses
# scripts/train_stableavatar_lora.sh; keep this file only for old experiments.
set -euo pipefail

# This mirrors DiffSynth-Studio's official two-stage H3 Ref2VA LoRA flow.
# Replace dataset/model paths and launch with torchrun/accelerate on the 8-GPU host.
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-../DiffSynth-Studio}"
DATASET_BASE="${DATASET_BASE:-data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA}"
MODEL_ID="${MODEL_ID:-MiniMax/MiniMax-H3}"

accelerate launch "${DIFFSYNTH_ROOT}/examples/minimax_h3/model_training/train.py" \
  --dataset_base_path "${DATASET_BASE}" \
  --dataset_metadata_path "${DATASET_BASE}/metadata.json" \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 --width 832 --num_frames 56 --dataset_repeat 1 \
  --model_id_with_origin_paths "${MODEL_ID}:Ref2VA/text_encoder/model*.safetensors,${MODEL_ID}:Ref2VA/video_vae/source/model.safetensors,${MODEL_ID}:Ref2VA/audio_vae/model.safetensors" \
  --processor_path "${MODEL_ID}:Ref2VA/processor/" \
  --learning_rate 1e-4 --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "models/train/MiniMax-H3-Ref2VA-split-cache" \
  --lora_base_model dit \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 --use_gradient_checkpointing \
  --task sft:data_process

if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
  accelerate launch "${DIFFSYNTH_ROOT}/examples/minimax_h3/model_training/train.py" \
    --dataset_base_path "models/train/MiniMax-H3-Ref2VA-split-cache" \
    --data_file_keys "video,input_audio,references" \
    --extra_inputs "input_audio,references" \
    --height 480 --width 832 --num_frames 56 --dataset_repeat "${DATASET_REPEAT:-100}" \
    --model_id_with_origin_paths "${MODEL_ID}:Ref2VA/transformer/model*.safetensors" \
    --processor_path "${MODEL_ID}:Ref2VA/processor/" \
    --learning_rate 1e-4 --num_epochs "${EPOCHS:-5}" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "models/train/MiniMax-H3-Ref2VA-split" \
    --lora_base_model dit \
    --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
    --lora_rank 32 --use_gradient_checkpointing --find_unused_parameters \
    --task sft:train
else
  echo "Stage 1 finished. Set RUN_TRAIN=1 to run stage 2 after checking the cache."
fi
