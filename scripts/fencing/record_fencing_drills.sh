#!/bin/bash
# Render all 6 drills into one labeled mp4 using the trained low-level policy.
#
# Usage:
#   bash scripts/fencing/visualize_drills.sh [low_level_checkpoint]
#
# Output: output/renderings/drill_viz_drills_combined.mp4

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

LOW_LEVEL="${1:-output/HumanoidIm/fencing_drills_v4/Humanoid.pth}"
[[ $# -gt 0 ]] && shift   # consume the checkpoint arg so it isn't re-passed to hydra via "$@"
echo "[viz] low-level policy: ${LOW_LEVEL}"

python phc/visualize_drills.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play_no_disc \
    exp_name=drill_viz \
    env=env_amp_z \
    env.num_envs=1 \
    env.task=HumanoidFencingDrillsZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl \
    headless=True \
    +record_video=True \
    env.episode_length=175 \
    +env.strike_episode_length=50 \
    "+env.low_level_checkpoint=${LOW_LEVEL}" \
    +env.clip_len=120 \
    "$@"
