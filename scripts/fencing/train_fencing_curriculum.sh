#!/bin/bash
# Train fencing curriculum — no AMP discriminator, PULSE VAE motion prior only.
#
# Usage:
#   bash train_fencing_curriculum.sh [version] [stage] [extra hydra args...]
#
# version (optional, default: v2):
#   v1, v2, ...  — curriculum config version from phc/utils/curriculum_configs.py
#
# stage (optional, default: auto):
#   0     — pin to Stage 0 (locomotion only, wins disabled)
#   1     — pin to Stage 1 (engagement)
#   2     — pin to Stage 2 (combat)
#   auto  — step_counter-based progression (resets to Stage 0 on resume)
#   fresh — start from scratch with no checkpoint
#
# Stage thresholds: Stage 0→1 at step 10 000 (~epoch 312), Stage 1→2 at step 30 000 (~epoch 937).
#
# Checkpoint: auto-loads latest checkpoint from the version's output dir (unless "fresh").
#
# Examples:
#   bash train_fencing_curriculum.sh                              # v2, resume latest, auto curriculum
#   bash train_fencing_curriculum.sh v2 fresh                    # v2, start from scratch
#   bash train_fencing_curriculum.sh v2 2                        # v2, pin to Stage 2, resume latest
#   bash train_fencing_curriculum.sh v2 auto max_epochs=50000    # v2, auto + extra hydra arg

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

# --- Parse version argument ---
VERSION="${1:-v2}"
if [[ "$VERSION" =~ ^v[0-9]+$ ]]; then
    shift
else
    VERSION="v2"
fi

EXP_NAME=fencing_curriculum_${VERSION}
OUTPUT_DIR=output/HumanoidIm/${EXP_NAME}

echo "[Version] ${VERSION}  (exp: ${EXP_NAME})"

# --- Parse stage argument ---
STAGE="${1:-auto}"
if [[ "$STAGE" =~ ^[012]$ ]] || [[ "$STAGE" == "auto" ]] || [[ "$STAGE" == "fresh" ]]; then
    shift
else
    STAGE="auto"
fi

# --- Curriculum stage overrides ---
case "$STAGE" in
  0)
    echo "[Stage] Pinned to Stage 0: locomotion"
    CURRICULUM_ARGS="env.curriculum_stage1_steps=2147483647 env.curriculum_stage2_steps=2147483647"
    ;;
  1)
    echo "[Stage] Pinned to Stage 1: engagement"
    CURRICULUM_ARGS="env.curriculum_stage1_steps=0 env.curriculum_stage2_steps=2147483647"
    ;;
  2)
    echo "[Stage] Pinned to Stage 2: combat"
    CURRICULUM_ARGS="env.curriculum_stage1_steps=0 env.curriculum_stage2_steps=0"
    ;;
  auto|fresh)
    echo "[Stage] Auto curriculum (stage advances by step_counter)"
    CURRICULUM_ARGS=""
    ;;
esac

# --- Auto-detect latest checkpoint ---
CHECKPOINT_ARG=""
if [[ "$STAGE" != "fresh" ]]; then
    LATEST_NUMBERED=$(ls ${OUTPUT_DIR}/Humanoid_[0-9]*.pth 2>/dev/null | grep -v "_op\.pth" | sort -V | tail -1)
    if [[ -n "$LATEST_NUMBERED" ]]; then
        CHECKPOINT_ARG="+checkpoint=${LATEST_NUMBERED}"
        echo "[Checkpoint] Resuming from: ${LATEST_NUMBERED}"
    elif [[ -f "${OUTPUT_DIR}/Humanoid.pth" ]]; then
        CHECKPOINT_ARG="+checkpoint=${OUTPUT_DIR}/Humanoid.pth"
        echo "[Checkpoint] Resuming from best: ${OUTPUT_DIR}/Humanoid.pth"
    else
        echo "[Checkpoint] No checkpoint found — starting fresh."
    fi
else
    echo "[Checkpoint] Fresh start (no checkpoint loaded)."
fi

echo ""

python phc/run_hydra.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play_no_disc \
    exp_name=${EXP_NAME} \
    env=env_amp_z \
    env.num_envs=256 \
    env.task=HumanoidFencingZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl \
    headless=True \
    env.episode_length=300 \
    learning.params.config.switch_frequency=250 \
    learning.params.config.task_reward_w=1.0 \
    learning.params.config.disc_reward_w=0.0 \
    +env.curriculum_config=${VERSION} \
    ${CURRICULUM_ARGS} \
    ${CHECKPOINT_ARG} \
    "$@"
