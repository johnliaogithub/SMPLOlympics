#!/bin/bash
# Paper run: strategy policy on top of a frozen v24 drills policy, one per drills seed.
# Same spawn, hit and fall settings as the drills runs. Uses strategy-v10's reward and hit rules.
#   bash scripts/fencing/paper/train_strategy.sh <drills_seed> [iters=2000] [hydra overrides]
# Output: output/HumanoidIm/paper_strategy_s<seed>/ (saves every 500 iters; rerun the same command to resume)
export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics
mkdir -p /tmp/j73liao
SEED="${1:?usage: $0 <drills_seed> [iters]}"
ITERS="${2:-2000}"
LOW_LEVEL=output/HumanoidIm/paper_drills_s${SEED}/Humanoid.pth
[[ -f "${LOW_LEVEL}" ]] || { echo "[error] drills checkpoint not found: ${LOW_LEVEL}"; exit 1; }
python phc/train_fencing_strategy.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play_no_disc \
    exp_name=paper_strategy_s${SEED} \
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
    +strategy.iters=${ITERS} \
    +strategy.resume=True \
    +env.time_pen_w=0.005 \
    +env.offense_only_dense=True \
    +env.win_frame_hit=2 \
    +env.win_frame_hit_start=1 \
    +env.win_frame_hit_ramp_steps=300000 \
    +env.fall_termination=True \
    +env.fall_head_height=0.5 \
    +env.strike_spawn_half_dist=1.0 \
    +env.dodge_spawn_jitter=0.2 \
    +env.lateral_spawn_jitter=0.5 \
    +env.heading_spawn_jitter=0.3 \
    +env.hit_use_blade=False \
    +env.hit_dist=0.15 \
    +env.tip_lead_margin=0.06 \
    +env.hit_swept=True \
    seed=${SEED} \
    "${@:3}"
