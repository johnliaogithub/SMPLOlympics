# Check two-sided self-play sampling (lunge vs live dodger) and measure the lunge's landing rate
# against a dodging vs a stationary opponent. Both fencers use the same checkpoint, as in training
# after the opponent sync.
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
def run(task, low, T=300):
    dev = task.device; N = task.num_envs
    p = torch.zeros(NUM_DRILLS, device=dev); p[D_LUNGE_UPPER] = p[D_LUNGE_GROIN] = 1; task.drill_probs = p / p.sum()
    task.step_counter = task.warmup_time + 10; task.reset()
    vsd = task.opp_drill_ids == D_DODGE
    print("[sampler] lunge envs facing a live dodger: %.1f%% (target %.0f%%); of those unfrozen: %.0f%%; stationary ones frozen: %.0f%%"
          % (100 * vsd.float().mean(), 100 * task.lunge_vs_dodge_prob, 100 * (~task.opp_frozen[vsd]).float().mean(),
             100 * task.opp_frozen[~vsd].float().mean()))
    landed = torch.zeros(N, dtype=torch.bool, device=dev)
    start1 = task._humanoid_root_states_list[1][:, :2].clone()
    rows = []
    for t in range(T):
        obs = task.obs_buf
        task.step(torch.cat([low(obs[:N], task.drill_ids), low(obs[N:2 * N], task.opp_drill_ids)], 0))
        landed |= task.sword_hit_list[0][:, 0].bool()
        dn = task.reset_buf.nonzero().flatten()
        if len(dn):
            mv = (task._humanoid_root_states_list[1][dn, :2] - start1[dn]).norm(dim=-1)
            for k, e in enumerate(dn.tolist()):
                rows.append((bool(task.opp_drill_ids[e] == D_DODGE), bool(landed[e]), float(mv[k])))
            landed[dn] = False
            task.reset(dn)
            start1[dn] = task._humanoid_root_states_list[1][dn, :2]
    a = np.array(rows, dtype=float)
    for name, m in [("vs live DODGER", a[:, 0] > 0), ("vs STATIONARY", a[:, 0] == 0)]:
        if m.any():
            print("[%-15s] episodes=%4d  lunge landed a hit: %5.1f%%   opponent moved: mean %.2f m" % (name, m.sum(), 100 * a[m, 1].mean(), a[m, 2].mean()))

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
