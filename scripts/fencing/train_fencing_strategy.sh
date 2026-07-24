#!/bin/bash
# Train the high-level STRATEGY network on top of a frozen low-level drill policy.
#
# Usage:
#   bash train_fencing_strategy.sh [low_level_checkpoint] [extra hydra args...]
#
# Defaults to the best phase-B drills checkpoint. The strategy net is trained
# with PPO on a sparse win/loss reward; the dense fencing reward is logged to
# W&B for comparison (strategy/dense_return).

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

# smpl_local_robot writes the generated humanoid XML here; a tmp-cleaner can wipe
# it between runs, which crashes env creation with a FileNotFoundError. Recreate it.
mkdir -p /tmp/j73liao

# Shared box: auto-pick the GPU with the most free memory (override by setting
# CUDA_VISIBLE_DEVICES yourself before calling this script).
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
    echo "[GPU] auto-selected GPU ${CUDA_VISIBLE_DEVICES} (most free memory)"
fi

LOW_LEVEL="${1:-output/HumanoidIm/fencing_drills_v8/Humanoid.pth}"
if [[ -f "$LOW_LEVEL" ]] || [[ "$LOW_LEVEL" == output/* ]]; then
    shift 2>/dev/null
else
    LOW_LEVEL="output/HumanoidIm/fencing_drills_v8/Humanoid.pth"
fi

echo "[Strategy] low-level policy: ${LOW_LEVEL}"

python phc/train_fencing_strategy.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play_no_disc \
    exp_name=fencing_strategy_v6 \
    env=env_amp_z \
    env.num_envs=256 \
    env.task=HumanoidFencingStrategyZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl \
    headless=True \
    env.episode_length=175 \
    "+env.low_level_checkpoint=${LOW_LEVEL}" \
    +env.macro_K=15 \
    +strategy.dense_mix=0.05 \
    +env.time_pen_w=0.005 \
    +env.win_frame_hit=5 \
    "$@"
