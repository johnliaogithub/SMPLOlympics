#!/bin/bash

# Fencing RL training baseline with AMP and self-play
# This trains HumanoidFencingZ agents against each other using:
# - Adversarial Motion Priors (AMP) for natural movement
# - Self-play for competitive training
# - Curriculum learning (warmup period before strict termination)
#
# Usage: bash train_fencing_baseline.sh [additional hydra params...]
# Example: bash train_fencing_baseline.sh env.num_envs=32 learning.params.config.minibatch_size=1024

export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

python phc/run_hydra.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play \
    exp_name=fencing_baseline \
    env=env_amp_z \
    env.num_envs=16 \  # this is overridden
    env.task=HumanoidFencingZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/fencing_all.pkl \
    headless=True \
    env.episode_length=300 \
    learning.params.config.switch_frequency=250 \
    "$@"
