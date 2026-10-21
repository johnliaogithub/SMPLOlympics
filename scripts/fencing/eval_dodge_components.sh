#!/bin/bash
# 256-episode dodge evaluation: per-step reward components and state, survived vs got hit.
# Usage: bash scripts/fencing/eval_dodge_components.sh [low_level_checkpoint] [tag]
# Output: output/eval/dodge_components_<tag>.csv and .png
export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics
mkdir -p /tmp/j73liao
LOW_LEVEL="${1:-output/HumanoidIm/fencing_drills_v23/Humanoid.pth}"
TAG="${2:-$(basename $(dirname ${LOW_LEVEL}))}"
[[ -f "${LOW_LEVEL}" ]] || { echo "[error] checkpoint not found: ${LOW_LEVEL}"; exit 1; }
python phc/eval_dodge_components.py project_name=SMPLOlympics num_agents=2 learning=amp_z_self_play_no_disc exp_name=eval_dodge \
  env=env_amp_z env.num_envs=256 env.task=HumanoidFencingDrillsZ env.enableTaskObs=True env.stateInit=Start \
  robot=smpl_humanoid_fencing '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
  env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl headless=True \
  env.episode_length=175 +env.strike_episode_length=70 +env.exchange_episode_length=28 +env.strike_spawn_half_dist=1.0 \
  '+env.lunge_weights={posture:0.30}' +env.arm_back_pen=0.5 \
  +env.dodge_duel=False +env.dodge_pure=False +env.dodge_offline=True +env.dodge_evade_w=0.2 +env.dodge_far_value=0.5 \
  +env.dodge_evade_gate=1.5 +env.dodge_footback_w=0.0 +env.dodge_imhit_w=20.0 +env.dodge_survive_bonus=20.0 \
  +env.dodge_spawn_jitter=0.2 +env.lateral_spawn_jitter=0.5 +env.heading_spawn_jitter=0.3 \
  +env.hit_use_blade=False +env.hit_dist=0.15 +env.tip_lead_margin=0.06 +env.hit_swept=True +env.frozen_opp_policy=stand +env.fall_termination=True \
  "+env.low_level_checkpoint=${LOW_LEVEL}" "+env.eval_tag=${TAG}" seed=0 "${@:3}"
