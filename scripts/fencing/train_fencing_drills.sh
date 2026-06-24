#!/bin/bash
# Train drill-conditioned fencing skills (low level of the HRL stack).
#
# Usage:
#   bash train_fencing_drills.sh [phase] [extra hydra args...]
#
# phase (optional, default: A):
#   A     — drills 0-4 only (advance, retreat, stand, lunge_upper, lunge_groin),
#           all against a frozen opponent. Start here.
#   B     — all six drills including dodge. Resume from a phase A checkpoint so
#           the snapshot opponent already knows how to lunge.
#   fresh — phase A from scratch, ignoring existing checkpoints.
#
# NOTE: sitch_frequency is set astronomically high on purpose. Agent 0 is always
# the learner; agent 1 is frozen (its Z-actions are overridden), so its recorded
# experience must never be used for updates.

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

# Shared box: auto-pick the GPU with the most free memory (override by setting
# CUDA_VISIBLE_DEVICES yourself before calling this script).
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
    echo "[GPU] auto-selected GPU ${CUDA_VISIBLE_DEVICES} (most free memory)"
fi

# Drill order: advance retreat stand lunge_upper lunge_groin dodge step_left step_right
PHASE="${1:-A}"
if [[ "$PHASE" =~ ^[ABDLabdl]$ ]] || [[ "$PHASE" == "fresh" ]]; then
    shift
else
    PHASE="A"
fi

# Lunge-only diagnostic gets its OWN experiment dir so it never touches v4.
case "$PHASE" in
  L|l) EXP_NAME=fencing_lunge_only ;;
  *)   EXP_NAME=fencing_drills_v4 ;;
esac
OUTPUT_DIR=output/HumanoidIm/${EXP_NAME}

case "$PHASE" in
  B|b)
    echo "[Phase] B: all drills incl. dodge (vs lunging opponent)"
    DRILL_ARGS="+env.drill_probs=[1,1,1,1,1,1,1,1]"
    ;;
  D|d)
    echo "[Phase] D: emphasize the new/lunge drills (warm-started net)"
    # upweight lunges + lateral footwork; keep a little of the basics for retention
    DRILL_ARGS="+env.drill_probs=[0.3,0.3,0.3,1,1,0.5,1,1]"
    ;;
  L|l)
    echo "[Phase] L: LUNGE ONLY (diagnostic: is an upright lunge reachable in PULSE?)"
    DRILL_ARGS="+env.drill_probs=[0,0,0,1,1,0,0,0]"
    ;;
  *)
    echo "[Phase] A: locomotion + lunge drills + lateral footwork, no dodge"
    DRILL_ARGS="+env.drill_probs=[1,1,1,1,1,0,1,1]"
    ;;
esac

# Training resume is controlled by epoch= (NOT +checkpoint=, which only works in
# test mode). epoch=N loads Humanoid_<N>.pth; epoch=-1 loads best Humanoid.pth;
# epoch=0 starts fresh.
CHECKPOINT_ARG="epoch=0"
if [[ "$PHASE" != "fresh" ]]; then
    LATEST_NUMBERED=$(ls ${OUTPUT_DIR}/Humanoid_[0-9]*.pth 2>/dev/null | grep -v "_op\.pth" | sort -V | tail -1)
    if [[ -n "$LATEST_NUMBERED" ]]; then
        N=$(basename "$LATEST_NUMBERED" .pth | sed 's/Humanoid_//')
        N=$((10#$N))   # strip leading zeros
        CHECKPOINT_ARG="epoch=${N}"
        echo "[Checkpoint] Resuming from epoch ${N}: ${LATEST_NUMBERED}"
    elif [[ -f "${OUTPUT_DIR}/Humanoid.pth" ]]; then
        CHECKPOINT_ARG="epoch=-1"
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
    env.task=HumanoidFencingDrillsZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl \
    headless=True \
    env.episode_length=175 \
    +env.strike_episode_length=50 \
    learning.params.config.switch_frequency=1000000000 \
    learning.params.config.task_reward_w=1.0 \
    learning.params.config.disc_reward_w=0.0 \
    ${DRILL_ARGS} \
    ${CHECKPOINT_ARG} \
    "$@"
