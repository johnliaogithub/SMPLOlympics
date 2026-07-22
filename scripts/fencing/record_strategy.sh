#!/bin/bash
# Record a strategy-net fencing bout to output/renderings/strategy_viz_bout.mp4.
#
# Usage:
#   bash scripts/fencing/record_strategy.sh [strategy_checkpoint] [low_level_checkpoint]
# Defaults: latest strategy.pth on drills-v7.

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

# smpl_local_robot writes the generated humanoid XML here; a tmp-cleaner can wipe
# it between runs, which crashes env creation with a FileNotFoundError. Recreate it.
mkdir -p /tmp/j73liao

# Shared box: auto-pick the freest GPU (override via CUDA_VISIBLE_DEVICES).
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi

STRATEGY="${1:-output/HumanoidIm/fencing_strategy_v4/strategy.pth}"
LOW_LEVEL="${2:-output/HumanoidIm/fencing_drills_v7/Humanoid.pth}"
echo "[record] strategy: ${STRATEGY}"
echo "[record] low-level: ${LOW_LEVEL}"

python phc/visualize_strategy.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play_no_disc \
    exp_name=strategy_viz \
    env=env_amp_z \
    env.num_envs=1 \
    env.task=HumanoidFencingStrategyZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl \
    headless=True \
    +record_video=True \
    env.episode_length=175 \
    +env.macro_K=15 \
    "+env.low_level_checkpoint=${LOW_LEVEL}" \
    "+env.strategy_checkpoint=${STRATEGY}" \
    +env.clip_len=600
