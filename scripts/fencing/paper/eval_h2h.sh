#!/bin/bash
# Head-to-head bouts: strategy (on drills seed <seed>) vs the flat baseline (seed 0). See phc/paper/eval_h2h.py.
#   bash scripts/fencing/paper/eval_h2h.sh <seed> <side 0|1> <tag> [hydra overrides, e.g. +env.contact_term_dist=0]
export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics
mkdir -p /tmp/j73liao
SEED="${1:?usage: $0 <seed> <side> <tag>}"; SIDE="${2:?side}"; TAG="${3:?tag}"
python phc/paper/eval_h2h.py \
    project_name=SMPLOlympics num_agents=2 learning=amp_z_self_play_no_disc exp_name=paper_eval \
    env=env_amp_z env.num_envs=256 env.task=HumanoidFencingStrategyZ env.enableTaskObs=True env.stateInit=Start \
    robot=smpl_humanoid_fencing '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl headless=True env.episode_length=175 \
    +env.macro_K=15 +env.win_frame_hit=2 +env.fall_termination=True +env.fall_head_height=0.5 \
    '+env.drill_probs=[0,0,1,0,0,0,0,0]' +env.lateral_spawn_jitter=0.5 +env.heading_spawn_jitter=0.3 \
    +env.hit_use_blade=False +env.hit_dist=0.15 +env.tip_lead_margin=0.06 +env.hit_swept=True \
    "+env.low_level_checkpoint=output/HumanoidIm/paper_drills_s${SEED}/Humanoid.pth" \
    "+env.h2h_strategy_checkpoint=output/HumanoidIm/paper_strategy_s${SEED}/strategy_00001999.pth" \
    "+env.h2h_flat_checkpoint=output/HumanoidIm/paper_flat_s0/Humanoid.pth" \
    +env.h2h_strategy_side=${SIDE} +env.h2h_tag=${TAG} seed=0 "${@:4}"
