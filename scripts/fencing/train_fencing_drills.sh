#!/bin/bash
# Train drill-conditioned fencing skills (low level of the HRL stack).
#
# Usage:
#   bash train_fencing_drills.sh [phase] [+epochs] [extra hydra args...]
#
#   [+epochs] (optional bare integer, e.g. 6000): train this many epochs ON TOP OF
#   the resumed checkpoint, i.e. stop at max_epochs = resume_epoch + N. Omit to run
#   to the 10M default (until you kill it). Ignored if you pass max_epochs= yourself.
#
# phase (optional, default: A):
#   A     — drills 0-4 only (advance, retreat, stand, lunge_upper, lunge_groin),
#           all against a frozen opponent. Start here.
#   B     — all six drills including dodge. Resume from a phase A checkpoint so
#           the snapshot opponent already knows how to lunge.
#   C     — DODGE-focused: heavily upweight dodge, keep lunges (so the snapshot
#           opponent stays a real threat) + a little basics for retention.
#   fresh — phase A from scratch, ignoring existing checkpoints.
#
# NOTE: sitch_frequency is set astronomically high on purpose. Agent 0 is always
# the learner; agent 1 is frozen (its Z-actions are overridden), so its recorded
# experience must never be used for updates.

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

# smpl_local_robot writes the generated humanoid XML to /tmp/j73liao (hardcoded in
# humanoid.py); a tmp-cleaner wipes it between runs -> FileNotFoundError on env creation. Recreate it.
mkdir -p /tmp/j73liao

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
if [[ "$PHASE" =~ ^[ABCDLabcdl]$ ]] || [[ "$PHASE" == "fresh" ]]; then
    shift
else
    PHASE="A"
fi

# Optional bare-integer epoch INCREMENT (e.g. `... C 6000`): trains +N epochs on top
# of whatever checkpoint we resume from. Consumed here so it isn't passed to hydra.
INCREMENT=""
if [[ "$1" =~ ^[0-9]+$ ]]; then
    INCREMENT="$1"; shift
fi

# Lunge-only diagnostic gets its OWN experiment dir so it never touches v4.
case "$PHASE" in
  L|l) EXP_NAME=fencing_lunge_only ;;
  *)   EXP_NAME=fencing_drills_v14 ;;
esac
OUTPUT_DIR=output/HumanoidIm/${EXP_NAME}

case "$PHASE" in
  B|b)
    echo "[Phase] B: all drills incl. dodge (vs lunging opponent)"
    DRILL_ARGS="+env.drill_probs=[1,1,1,1,1,1,1,1]"
    ;;
  C|c)
    echo "[Phase] C: DODGE-focused (dodge upweighted; lunges kept as the threat)"
    # dodge (idx 5) dominant; lunges (idx 3,4) kept high so the snapshot opponent
    # remains a real lunging threat; basics/lateral low but nonzero for retention.
    DRILL_ARGS="+env.drill_probs=[0.4,0.4,0.4,0.6,0.6,2.0,0.4,0.4]"
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
RESUME_EPOCH=0        # numeric epoch we resume from (for the +epochs feature)
RESUME_UNKNOWN=0     # 1 when resuming from best Humanoid.pth (epoch not knowable here)
if [[ "$PHASE" != "fresh" ]]; then
    LATEST_NUMBERED=$(ls ${OUTPUT_DIR}/Humanoid_[0-9]*.pth 2>/dev/null | grep -v "_op\.pth" | sort -V | tail -1)
    if [[ -n "$LATEST_NUMBERED" ]]; then
        N=$(basename "$LATEST_NUMBERED" .pth | sed 's/Humanoid_//')
        N=$((10#$N))   # strip leading zeros
        CHECKPOINT_ARG="epoch=${N}"
        RESUME_EPOCH=${N}
        echo "[Checkpoint] Resuming from epoch ${N}: ${LATEST_NUMBERED}"
    elif [[ -f "${OUTPUT_DIR}/Humanoid.pth" ]]; then
        CHECKPOINT_ARG="epoch=-1"
        RESUME_UNKNOWN=1
        echo "[Checkpoint] Resuming from best: ${OUTPUT_DIR}/Humanoid.pth"
    else
        echo "[Checkpoint] No checkpoint found — starting fresh."
    fi
else
    echo "[Checkpoint] Fresh start (no checkpoint loaded)."
fi

# Translate the optional +epochs increment into the real (nested) hydra key.
MAX_EPOCHS_ARG=""
if [[ -n "$INCREMENT" ]]; then
    if [[ "$*" == *max_epochs* ]]; then
        echo "[Epochs] max_epochs passed explicitly; ignoring +${INCREMENT} increment."
    elif [[ "$RESUME_UNKNOWN" == "1" ]]; then
        echo "[Epochs] WARN: resuming from best (epoch unknown); cannot add +${INCREMENT}. Pass learning.params.config.max_epochs=<total> yourself."
    else
        TARGET=$((RESUME_EPOCH + INCREMENT))
        MAX_EPOCHS_ARG="learning.params.config.max_epochs=${TARGET}"
        echo "[Epochs] Training +${INCREMENT} from epoch ${RESUME_EPOCH} => stop at max_epochs=${TARGET}."
    fi
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
    +env.strike_episode_length=70 \
    +env.strike_spawn_half_dist=1.0 \
    '+env.lunge_weights={posture:0.30}' \
    +env.arm_back_pen=0.5 \
    +env.dodge_duel=False \
    +env.dodge_pure=True \
    +env.dodge_evade_w=1.0 \
    +env.dodge_evade_gate=1.5 \
    +env.dodge_footback_w=0.5 \
    +env.dodge_imhit_w=5.0 \
    +env.dodge_spawn_jitter=0.2 \
    +env.lateral_spawn_jitter=0.5 \
    +env.heading_spawn_jitter=0.3 \
    +env.hit_use_blade=False \
    +env.hit_dist=0.15 \
    +env.tip_lead_margin=0.06 \
    +env.hit_swept=True \
    learning.params.config.switch_frequency=1000000000 \
    learning.params.config.task_reward_w=1.0 \
    learning.params.config.disc_reward_w=0.0 \
    ${DRILL_ARGS} \
    ${CHECKPOINT_ARG} \
    ${MAX_EPOCHS_ARG} \
    "$@"
