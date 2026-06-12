export LD_LIBRARY_PATH=/pub0/johnliao/miniconda3/envs/isaac/lib:$LD_LIBRARY_PATH
cd /pub0/johnliao/SMPLOlympics

python phc/run_hydra.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play \
    exp_name=fencing_pulse \
    env=env_amp_z \
    env.num_envs=16 \
    env.task=HumanoidFencingZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_fencing \
    +env.models=[output/HumanoidIm/pulse_vae_iclr/Humanoid.pth] \
    env.motion_file=./sample_data/fencing_all.pkl \
    headless=True \
    env.episode_length=300 \
    learning.params.config.switch_frequency=250 \
    test=True \
    +record_video=True \
    +checkpoint=output/HumanoidIm/fencing_curriculum_v2/Humanoid.pth \
    learning.params.config.player.games_num=5

