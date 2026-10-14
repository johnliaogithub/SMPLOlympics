# Check the exchange window: episode lengths per bout type and the dodge survive bonus timing.
import os, sys
os.environ['OMP_NUM_THREADS'] = "1"; sys.path.append(os.getcwd()); sys.path.append('./SMPLSim')
from isaacgym import gymapi  # noqa
import torch, numpy as np, hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, D_LUNGE_UPPER, D_LUNGE_GROIN, D_DODGE

@torch.no_grad()
def run(task, low, T=220):
    dev = task.device; N = task.num_envs
    p = torch.zeros(NUM_DRILLS, device=dev); p[[D_LUNGE_UPPER, D_LUNGE_GROIN, D_DODGE]] = 1; task.drill_probs = p / p.sum()
    task.step_counter = task.warmup_time + 10; task.reset()
    L = {"dodge": [], "lunge vs dodger": [], "lunge vs still": []}; bonus_steps = []
    for t in range(T):
        kind = torch.where(task.drill_ids == D_DODGE, 0, torch.where(task.opp_drill_ids == D_DODGE, 1, 2))
        obs = task.obs_buf
        task.step(torch.cat([low(obs[:N], task.drill_ids), low(obs[N:2 * N], task.opp_drill_ids)], 0))
        dn = task.reset_buf.nonzero().flatten()
        if len(dn):
            got = task.sword_hit_list[1][:, 0].bool()
            for e in dn.tolist():
                k = ["dodge", "lunge vs dodger", "lunge vs still"][int(kind[e])]
                L[k].append((int(task.progress_buf[e]), bool(got[e])))
            task.reset(dn)
    print("\n=== exchange window check (exchange_episode_length=%d, strike_episode_length=%d) ===" % (task.exchange_episode_length, task.strike_episode_length))
    for k, v in L.items():
        a = np.array(v); 
        if not len(a): continue
        nohit = a[:, 0][a[:, 1] == 0] if k == "dodge" else a[:, 0]
        print("[%-15s] episodes=%4d  length: min=%d median=%d max=%d   %s" % (k, len(a), a[:, 0].min(), np.median(a[:, 0]), a[:, 0].max(),
              ("| not-hit dodge bouts end at: %s" % sorted(set(nohit.tolist()))[:5]) if k == "dodge" else ""))

@hydra.main(version_base=None, config_path="../phc/data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True)); set_np_formatting()
    flags.test = False
    for k in ['debug', 'follow', 'server_mode', 'im_eval', 'no_virtual_display', 'render_o3d', 'fixed', 'divide_group',
              'no_collision_check', 'fixed_path', 'real_path', 'show_traj', 'slow', 'real_traj', 'trigger_input']:
        setattr(flags, k, False)
    flags.add_proj = False; flags.has_eval = False; set_seed(cfg.get("seed", 0), False)
    cfg.learning['params']['config']["num_actors"] = cfg.env.num_envs
    sim_params = parse_sim_params(cfg)
    args = EasyDict({"task": cfg.env.task, "device_id": cfg.device_id, "rl_device": cfg.rl_device,
                     "physics_engine": gymapi.SIM_PHYSX, "headless": cfg.headless, "device": cfg.device})
    task, env = parse_task(args, cfg, cfg_train=cfg.learning, sim_params=sim_params)
    run(task, FrozenLowLevelPolicy(cfg.env["low_level_checkpoint"], NUM_DRILLS, device=task.device))

if __name__ == '__main__':
    main()
