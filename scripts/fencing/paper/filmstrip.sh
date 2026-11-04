#!/bin/bash
# Filmstrip figure (4 frames in sequence) + the matching mp4 for one bout.
#   bash scripts/fencing/paper/filmstrip.sh ours <drills checkpoint> <tag> [hydra overrides, e.g. seed=3 +env.film_target=groin]
#   bash scripts/fencing/paper/filmstrip.sh flat <flat checkpoint>   <tag> ['+env.film_steps=[20,60,100,140]']
# Output: output/paper/figures/filmstrip_<tag>.png/.pdf (+ single frames), output/paper/video/bout_<tag>.mp4
export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics
mkdir -p /tmp/j73liao
METHOD="${1:?usage: $0 ours|flat <checkpoint> <tag>}"; CKPT="${2:?checkpoint}"; TAG="${3:?tag}"
[[ -f "${CKPT}" ]] || { echo "[error] checkpoint not found: ${CKPT}"; exit 1; }
COMMON="project_name=SMPLOlympics num_agents=2 learning=amp_z_self_play_no_disc exp_name=paper_film env=env_amp_z
  env.num_envs=1 env.enableTaskObs=True env.stateInit=Start robot=smpl_humanoid_fencing
  +env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth] env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl
  headless=True +record_video=True env.episode_length=300
  +env.hit_use_blade=False +env.hit_dist=0.15 +env.tip_lead_margin=0.06 +env.hit_swept=True
  +env.fall_termination=True +env.fall_head_height=0.5
  +env.film_method=${METHOD} +env.film_checkpoint=${CKPT} +env.film_tag=${TAG}"
if [[ "$METHOD" == "ours" ]]; then
  python phc/paper/filmstrip.py ${COMMON} env.task=HumanoidFencingDrillsZ \
    +env.strike_episode_length=0 +env.exchange_episode_length=0 +env.footwork_episode_length=0 \
    +env.strike_spawn_half_dist=1.0 '+env.lunge_weights={posture:0.30}' +env.arm_back_pen=0.5 \
    +env.stand_engarde=True +env.engarde_split_shape=tri +env.lunge_vs_dodge_prob=0.0 \
    +env.frozen_opp_policy=stand "${@:4}"
else
  python phc/paper/filmstrip.py ${COMMON} env.task=HumanoidFencingZ +env.curriculum_config=v2 "${@:4}"
fi
