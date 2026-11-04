#!/bin/bash
# Paper run, baseline: flat self-play on the PULSE prior (no drills, no hierarchy), from scratch.
# Same prior, same fencer, same fixed environment as our method: facing-sorted strip obs (E7),
# per-env win counter (E1), head-height fall termination (E4), no discriminator reward or training.
# Reward: the flat env's shaped curriculum (curriculum_config=v2, our best flat-setting reward).
#   bash scripts/fencing/paper/train_flat_baseline.sh <seed> [max_epochs=40000]
# Output: output/HumanoidIm/paper_flat_s<seed>/   (rerun the same command to resume)
export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics
mkdir -p /tmp/j73liao
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
    echo "[GPU] auto-selected GPU ${CUDA_VISIBLE_DEVICES} (most free memory)"
fi
SEED="${1:?usage: $0 <seed> [max_epochs]}"
MAX_EPOCHS="${2:-40000}"
EXP_NAME=paper_flat_s${SEED}
OUT=output/HumanoidIm/${EXP_NAME}
LATEST=$(ls ${OUT}/Humanoid_[0-9]*.pth 2>/dev/null | grep -v "_op\.pth" | sort -V | tail -1)
if [[ -n "$LATEST" ]]; then N=$(basename "$LATEST" .pth | sed 's/Humanoid_//'); CK="epoch=$((10#$N))"; echo "[resume] $LATEST"; else CK="epoch=0"; fi
python phc/run_hydra.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play_no_disc \
    exp_name=${EXP_NAME} \
    env=env_amp_z \
    env.num_envs=256 \
    env.task=HumanoidFencingZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    '+env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth]' \
    env.motion_file=./sample_data/amass_isaac_standing_upright_slim.pkl \
    headless=True \
    env.episode_length=175 \
    +env.curriculum_config=v2 \
    +env.fall_termination=True \
    +env.fall_head_height=0.5 \
    +env.lateral_spawn_jitter=0.5 \
    +env.heading_spawn_jitter=0.3 \
    +env.hit_use_blade=False \
    +env.hit_dist=0.15 \
    +env.tip_lead_margin=0.06 \
    +env.hit_swept=True \
    learning.params.config.switch_frequency=500 \
    learning.params.config.task_reward_w=1.0 \
    learning.params.config.disc_reward_w=0.0 \
    learning.params.config.disc_coef=0.0 \
    learning.params.config.max_epochs=${MAX_EPOCHS} \
    seed=${SEED} \
    ${CK} \
    "${@:3}"
