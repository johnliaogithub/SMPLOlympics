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

LOW_LEVEL="${1:-output/HumanoidIm/fencing_drills_v1/Humanoid.pth}"
if [[ -f "$LOW_LEVEL" ]] || [[ "$LOW_LEVEL" == output/* ]]; then
    shift 2>/dev/null
else
    LOW_LEVEL="output/HumanoidIm/fencing_drills_v1/Humanoid.pth"
fi

echo "[Strategy] low-level policy: ${LOW_LEVEL}"

python phc/train_fencing_strategy.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play_no_disc \
    exp_name=fencing_strategy_v1 \
    env=env_amp_z \
    env.num_envs=256 \
    env.task=HumanoidFencingStrategyZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl \
    headless=True \
    env.episode_length=300 \
    "+env.low_level_checkpoint=${LOW_LEVEL}" \
    +env.macro_K=15 \
    "$@"
