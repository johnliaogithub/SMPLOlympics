#!/bin/bash
# Diagnostic: 10 dodge bouts with randomized spawns (same jitter as training), 5 vs an UPPER
# lunge + 5 vs GROIN, recorded back-to-back (no end pause). Prints the learner's net lateral
# shift per bout so you can see whether every dodge goes the same direction.
#   bash scripts/fencing/record_dodge_eval.sh [low_level_checkpoint]

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics
mkdir -p /tmp/j73liao   # smpl_local_robot writes the humanoid XML here (hardcoded)

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi

LOW_LEVEL="${1:-output/HumanoidIm/fencing_drills_v10/Humanoid.pth}"
[[ $# -gt 0 ]] && shift
echo "[dodge-eval] low-level: ${LOW_LEVEL}"

python phc/visualize_dodge_eval.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play_no_disc \
    exp_name=dodge_eval \
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
    +env.dodge_duel=True \
    +env.dodge_spawn_jitter=0.2 \
    +env.hit_use_blade=False \
    +env.hit_dist=0.15 \
    +env.tip_lead_margin=0.06 \
    +env.n_eval_bouts=10 \
    "+env.low_level_checkpoint=${LOW_LEVEL}" \
    "$@"
