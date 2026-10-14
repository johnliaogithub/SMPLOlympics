# Role-symmetry test: the same network plays attacker/defender as agent 0 or agent 1 in identical
# fixed-length windows (no drill-based episode endings). If the env is symmetric, swapping sides
# must not change outcomes.
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
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, D_LUNGE_UPPER, D_DODGE, D_STAND

@torch.no_grad()
def window(task, low, d0, d1, still0, still1, T, rounds):
    dev = task.device; N = task.num_envs; res = []
    p = torch.zeros(NUM_DRILLS, device=dev); p[D_LUNGE_UPPER] = p[D_DODGE] = 1; task.drill_probs = p / p.sum()
    for r in range(rounds):
        torch.manual_seed(100 + r); task.reset()
        a_idx = 0 if d0 == D_LUNGE_UPPER else 1                       # attacker index
        hit = torch.zeros(N, dtype=torch.bool, device=dev); t_hit = torch.full((N,), -1, device=dev)
        s_def = task._humanoid_root_states_list[1 - a_idx][:, :2].clone(); s_att = task._humanoid_root_states_list[a_idx][:, :2].clone()
        away = torch.nn.functional.normalize(s_def - s_att, dim=-1)
        for t in range(T):
            task.drill_ids[:] = d0; task.opp_drill_ids[:] = d1; task.opp_frozen[:] = False
            obs = task.obs_buf
            z0 = low(obs[:N], task.drill_ids); z1 = low(obs[N:2 * N], task.opp_drill_ids)
            if still0: z0 = torch.zeros_like(z0)
            if still1: z1 = torch.zeros_like(z1)
            task.step(torch.cat([z0, z1], 0))
            h = task.sword_hit_list[a_idx][:, 0].bool()
            t_hit = torch.where(h & ~hit, torch.full_like(t_hit, t), t_hit); hit |= h
        back = ((task._humanoid_root_states_list[1 - a_idx][:, :2] - s_def) * away).sum(-1)
        res.append((hit.float().cpu(), t_hit.float().cpu(), back.cpu()))
    h = torch.cat([r[0] for r in res]); th = torch.cat([r[1] for r in res]); b = torch.cat([r[2] for r in res])
    return h.mean().item(), (th[th >= 0].median().item() if (th >= 0).any() else float('nan')), b.mean().item(), len(h)

def run(task, low, T=70, rounds=4):
    task.step_counter = task.warmup_time + 10
    cases = [("A0 lunge vs A1 STILL ", D_LUNGE_UPPER, D_STAND, False, True),
             ("A0 STILL vs A1 lunge ", D_STAND, D_LUNGE_UPPER, True, False),
             ("A0 lunge vs A1 dodge ", D_LUNGE_UPPER, D_DODGE, False, False),
             ("A0 dodge vs A1 lunge ", D_DODGE, D_LUNGE_UPPER, False, False)]
    print("\n=== role symmetry, %d-step windows, %d bouts each ===" % (T, rounds * task.num_envs))
    for name, d0, d1, s0, s1 in cases:
        hr, th, back, n = window(task, low, d0, d1, s0, s1, T, rounds)
        print("[%s] attacker lands: %5.1f%%   median step of first hit: %4.0f   defender moved away: %+.2f m" % (name, 100 * hr, th, back))

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
