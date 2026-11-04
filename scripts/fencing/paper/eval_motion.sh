#!/bin/bash
# Same motion metrics for both methods (256 bouts each). Writes a row to output/paper/metrics.csv.
#   bash scripts/fencing/paper/eval_motion.sh ours <drills checkpoint> <tag>
#   bash scripts/fencing/paper/eval_motion.sh flat <flat checkpoint>   <tag>
export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics
mkdir -p /tmp/j73liao
METHOD="${1:?usage: $0 ours|flat <checkpoint> <tag>}"; CKPT="${2:?checkpoint}"; TAG="${3:?tag}"
[[ -f "${CKPT}" ]] || { echo "[error] checkpoint not found: ${CKPT}"; exit 1; }
COMMON="project_name=SMPLOlympics num_agents=2 learning=amp_z_self_play_no_disc exp_name=paper_eval env=env_amp_z
  env.num_envs=256 env.enableTaskObs=True env.stateInit=Start robot=smpl_humanoid_fencing
  +env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth] env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl
  headless=True env.episode_length=300 +env.lateral_spawn_jitter=0.5 +env.heading_spawn_jitter=0.3
  +env.hit_use_blade=False +env.hit_dist=0.15 +env.tip_lead_margin=0.06 +env.hit_swept=True
  +env.fall_termination=True +env.fall_head_height=0.5 seed=0
  +env.eval_method=${METHOD} +env.eval_checkpoint=${CKPT} +env.eval_tag=${TAG}"
if [[ "$METHOD" == "ours" ]]; then
  python phc/paper/eval_motion.py ${COMMON} env.task=HumanoidFencingDrillsZ \
    +env.strike_episode_length=0 +env.exchange_episode_length=0 +env.footwork_episode_length=0 \
    +env.strike_spawn_half_dist=1.0 '+env.lunge_weights={posture:0.30}' +env.arm_back_pen=0.5 \
    +env.stand_engarde=True +env.engarde_split_shape=tri +env.lunge_vs_dodge_prob=0.0 \
    +env.frozen_opp_policy=stand +env.lunge_recover_as_stand=False "${@:4}"
else
  python phc/paper/eval_motion.py ${COMMON} env.task=HumanoidFencingZ +env.curriculum_config=v2 \
    +env.eval_steps=240 "${@:4}"
fi
