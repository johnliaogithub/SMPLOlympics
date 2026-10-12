#!/bin/bash
# Render all 6 drills into one labeled mp4 using the trained low-level policy.
#
# Usage:
#   bash scripts/fencing/visualize_drills.sh [low_level_checkpoint]
#
# Output: output/renderings/drill_viz_drills_combined.mp4

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

# smpl_local_robot writes the generated humanoid XML to /tmp/j73liao (hardcoded in
# humanoid.py); a tmp-cleaner wipes it between runs -> FileNotFoundError on env creation. Recreate it.
mkdir -p /tmp/j73liao

LOW_LEVEL="${1:-output/HumanoidIm/fencing_drills_v18/Humanoid.pth}"
if [[ ! -f "${LOW_LEVEL}" ]]; then
    echo "[error] low-level checkpoint not found: ${LOW_LEVEL}"
    echo "        (has that version finished training? its first checkpoint saves after 2000 epochs)"
    echo "        available final checkpoints:"
    ls -1t output/HumanoidIm/fencing_drills_v*/Humanoid.pth 2>/dev/null | head -5 | sed 's/^/          /'
    echo "        usage: bash $0 <path/to/Humanoid.pth>"
    exit 1
fi
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
    +env.strike_episode_length=70 \
    +env.strike_spawn_half_dist=1.0 \
    +env.dodge_duel=False \
    +env.dodge_pure=True \
    +env.dodge_evade_w=1.0 \
    +env.dodge_evade_gate=1.5 \
    +env.dodge_footback_w=0.5 \
    +env.lateral_spawn_jitter=0.5 \
    +env.heading_spawn_jitter=0.3 \
    +env.hit_use_blade=False \
    +env.hit_dist=0.15 \
    +env.tip_lead_margin=0.06 \
    +env.hit_swept=True \
    "+env.low_level_checkpoint=${LOW_LEVEL}" \
    +env.clip_len=120 \
    "$@"
