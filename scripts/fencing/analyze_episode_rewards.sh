#!/bin/bash
# Run one test episode and plot per-step reward component breakdown.
#
# Usage:
#   bash scripts/fencing/analyze_episode_rewards.sh <checkpoint.pth> [extra hydra args...]
#
# Output:
#   reward_analysis.csv      — per-step log (step, raw components, weights)
#   reward_analysis_plot.png — matplotlib figure

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

CHECKPOINT="${1:?Usage: bash analyze_episode_rewards.sh <checkpoint.pth>}"
shift

rm -f reward_analysis.csv

echo "[Analyze] Running 1 test episode with checkpoint: ${CHECKPOINT}"
echo "[Analyze] Reward data will be saved to reward_analysis.csv on exit."
echo ""

python phc/run_hydra.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play_no_disc \
    exp_name=fencing_curriculum_v1 \
    env=env_amp_z \
    env.num_envs=1 \
    env.task=HumanoidFencingZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/amass_isaac_simple_run_upright_slim.pkl \
    headless=True \
    env.episode_length=300 \
    learning.params.config.switch_frequency=250 \
    learning.params.config.task_reward_w=1.0 \
    learning.params.config.disc_reward_w=0.0 \
    test=True \
    "+checkpoint=${CHECKPOINT}" \
    learning.params.config.player.games_num=1 \
    "$@"

echo ""
if [[ -f reward_analysis.csv ]]; then
    echo "[Analyze] Plotting..."
    python scripts/fencing/plot_episode_rewards.py reward_analysis.csv
else
    echo "[Analyze] ERROR: reward_analysis.csv not found — did the episode complete?"
fi
