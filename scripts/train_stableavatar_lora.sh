#!/usr/bin/env bash
set -euo pipefail

# LoRA entry point for the official StableAvatar-1.3B trainer.
#
# This wrapper deliberately keeps StableAvatar in its own checkout.  The
# duplex service can use the resulting transformer LoRA, while training still
# follows the upstream data loader and loss.  It does not train MiniCPM-o or
# add user-audio conditioning yet.
#
# Example:
#   STABLEAVATAR_ROOT=/data/StableAvatar \
#   DATA_ROOT=/data/talking_face_data \
#   bash scripts/train_stableavatar_lora.sh
#
# To run only a short smoke job:
#   MAX_STEPS=20 NUM_EPOCHS=1 MAX_TRAIN_SAMPLES=8 \
#   bash scripts/train_stableavatar_lora.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STABLEAVATAR_ROOT="${STABLEAVATAR_ROOT:-${PROJECT_ROOT}/../StableAvatar}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${STABLEAVATAR_ROOT}/checkpoints}"
DATA_ROOT="${DATA_ROOT:-${STABLEAVATAR_ROOT}/talking_face_data}"
# Upstream currently publishes a 1B LoRA trainer for mixed rec/vec data.
# The square trainer is full-finetuning only, so this entry point fails fast
# instead of silently launching a non-LoRA job for VARIANT=square.
VARIANT="${VARIANT:-rec_vec}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OUTPUT_DIR="${OUTPUT_DIR:-${STABLEAVATAR_ROOT}/output_1B_${VARIANT}_lora}"

WAN_MODEL="${WAN_MODEL:-${CHECKPOINT_ROOT}/Wan2.1-Fun-V1.1-1.3B-InP}"
WAV2VEC_PATH="${WAV2VEC_PATH:-${CHECKPOINT_ROOT}/wav2vec2-base-960h}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-}"
# Prefer the released StableAvatar-1.3B transformer when it is present.  If
# the checkpoint has not been downloaded yet, the wrapper remains usable and
# trains a Wan2.1 adapter; the launch summary makes that distinction visible.
if [[ -z "${TRANSFORMER_PATH}" && -f "${CHECKPOINT_ROOT}/StableAvatar-1.3B/transformer3d-rec-vec.pt" ]]; then
  TRANSFORMER_PATH="${CHECKPOINT_ROOT}/StableAvatar-1.3B/transformer3d-rec-vec.pt"
fi
VALIDATION_REFERENCE="${VALIDATION_REFERENCE:-${STABLEAVATAR_ROOT}/validation/reference.png}"
VALIDATION_AUDIO="${VALIDATION_AUDIO:-${STABLEAVATAR_ROOT}/validation/audio.wav}"
CONFIG_PATH="${CONFIG_PATH:-${STABLEAVATAR_ROOT}/deepspeed_config/wan2.1/wan_civitai.yaml}"

NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"${GPU_IDS}")}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
MAX_STEPS="${MAX_STEPS:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
VIDEO_FRAMES="${VIDEO_FRAMES:-81}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VIDEO_REPEAT="${VIDEO_REPEAT:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-500}"
LORA_RANK="${LORA_RANK:-32}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
SEED="${SEED:-42}"

if [[ "${VARIANT}" == "rec_vec" ]]; then
  TRAINER="train_1B_rec_vec_lora.py"
  REC_LIST="${REC_LIST:-${DATA_ROOT}/video_rec_path.txt}"
  VEC_LIST="${VEC_LIST:-${DATA_ROOT}/video_vec_path.txt}"
else
  echo "VARIANT=${VARIANT} has no upstream StableAvatar 1B LoRA trainer; use VARIANT=rec_vec (mixed 480x832/832x480)." >&2
  echo "For square data, run the upstream full trainer separately: train_1B_square.py." >&2
  exit 2
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing directory: $1" >&2
    exit 1
  fi
}

require_dir "${STABLEAVATAR_ROOT}"
require_dir "${WAN_MODEL}"
require_dir "${WAV2VEC_PATH}"
require_file "${CONFIG_PATH}"
require_file "${VALIDATION_REFERENCE}"
require_file "${VALIDATION_AUDIO}"
require_file "${STABLEAVATAR_ROOT}/${TRAINER}"

require_file "${REC_LIST}"
require_file "${VEC_LIST}"

cd "${STABLEAVATAR_ROOT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

ARGS=(
  --config_path="${CONFIG_PATH}"
  --pretrained_model_name_or_path="${WAN_MODEL}"
  --pretrained_wav2vec_path="${WAV2VEC_PATH}"
  --validation_reference_path="${VALIDATION_REFERENCE}"
  --validation_driven_audio_path="${VALIDATION_AUDIO}"
  --video_sample_n_frames="${VIDEO_FRAMES}"
  --train_batch_size="${TRAIN_BATCH_SIZE}"
  --video_repeat="${VIDEO_REPEAT}"
  --gradient_accumulation_steps="${GRAD_ACCUM}"
  --dataloader_num_workers="${NUM_WORKERS}"
  --num_train_epochs="${NUM_EPOCHS}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
  --validation_steps="${VALIDATION_STEPS}"
  --learning_rate="${LEARNING_RATE}"
  --lr_scheduler="constant_with_warmup"
  --lr_warmup_steps="${WARMUP_STEPS}"
  --seed="${SEED}"
  --output_dir="${OUTPUT_DIR}"
  --gradient_checkpointing
  --mixed_precision="${MIXED_PRECISION:-bf16}"
  --adam_weight_decay="${ADAM_WEIGHT_DECAY:-3e-2}"
  --adam_epsilon="${ADAM_EPSILON:-1e-10}"
  --vae_mini_batch="${VAE_MINI_BATCH:-1}"
  --max_grad_norm="${MAX_GRAD_NORM:-0.05}"
  --uniform_sampling
  --motion_sub_loss
  --low_vram
  --train_mode="i2v"
  --rank="${LORA_RANK}"
  --network_alpha="${NETWORK_ALPHA}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  ARGS+=(--max_train_steps="${MAX_STEPS}")
fi
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  ARGS+=(--max_train_samples="${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${TRANSFORMER_PATH}" ]]; then
  require_file "${TRANSFORMER_PATH}"
  # Supplying the released StableAvatar transformer turns LoRA training into
  # LoRA finetuning; leave it empty to train an adapter on Wan2.1 weights.
  ARGS+=(--transformer_path="${TRANSFORMER_PATH}")
fi
if [[ "${USE_DEEPSPEED:-0}" == "1" ]]; then
  ARGS+=(--use_deepspeed)
fi
if [[ "${ALLOW_TF32:-0}" == "1" ]]; then
  ARGS+=(--allow_tf32)
fi

ARGS+=(--train_data_rec_dir="${REC_LIST}" --train_data_vec_dir="${VEC_LIST}")

echo "Launching StableAvatar LoRA training"
echo "  root=${STABLEAVATAR_ROOT} variant=${VARIANT} gpus=${GPU_IDS} output=${OUTPUT_DIR}"
echo "  frames=${VIDEO_FRAMES} rank=${LORA_RANK} max_steps=${MAX_STEPS:-epochs:${NUM_EPOCHS}}"
if [[ -n "${TRANSFORMER_PATH}" ]]; then
  echo "  mode=StableAvatar LoRA finetune transformer=${TRANSFORMER_PATH}"
else
  echo "  mode=Wan2.1 LoRA adapter (StableAvatar transformer not found; set TRANSFORMER_PATH to finetune it)"
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
  --num_processes="${NUM_GPUS}" \
  --mixed_precision="${MIXED_PRECISION:-bf16}" \
  "${TRAINER}" "${ARGS[@]}"
