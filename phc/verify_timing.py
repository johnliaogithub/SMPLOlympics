# Timing measurements for choosing the dodge/lunge exchange window (symmetric env).
#  - lunge vs STILL target: first-hit step distribution (how fast a full-speed lunge lands)
#  - lunge vs live DODGER: defender displacement over time, first-hit step, attacker falls
#  - landing rate if the window were truncated at W steps
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
def bouts(task, low, d1, still1, T=70, rounds=4):
    dev = task.device; N = task.num_envs
    p = torch.zeros(NUM_DRILLS, device=dev); p[D_LUNGE_UPPER] = p[D_DODGE] = 1; task.drill_probs = p / p.sum()
    head = int(task._build_key_body_ids_tensor(["Head"])[0]); cids = task._contact_body_ids
    FH, GAP0, AWAY, FALL, FALLT = [], [], [], [], []
    for r in range(rounds):
        torch.manual_seed(200 + r); task.reset()
        s_att = task._humanoid_root_states_list[0][:, :2].clone(); s_def = task._humanoid_root_states_list[1][:, :2].clone()
        away_dir = torch.nn.functional.normalize(s_def - s_att, dim=-1)
        GAP0.append((s_def - s_att).norm(dim=-1).cpu())
        fh = torch.full((N,), -1, device=dev); fall_t = torch.full((N,), -1, device=dev); away = []
        for t in range(T):
            task.drill_ids[:] = D_LUNGE_UPPER; task.opp_drill_ids[:] = d1; task.opp_frozen[:] = False
            obs = task.obs_buf; z0 = low(obs[:N], task.drill_ids); z1 = low(obs[N:2 * N], task.opp_drill_ids)
            if still1: z1 = torch.zeros_like(z1)
            task.step(torch.cat([z0, z1], 0))
            h = task.sword_hit_list[0][:, 0].bool(); fh = torch.where(h & (fh < 0), torch.full_like(fh, t + 1), fh)
            z = task._rigid_body_pos_list[0][..., 2].clone(); z[:, cids] = 1e3
            fallen = (z < 0.15).any(-1) | (task._rigid_body_pos_list[0][:, head, 2] < 0.6)
            fall_t = torch.where(fallen & (fall_t < 0), torch.full_like(fall_t, t + 1), fall_t)
            away.append(((task._humanoid_root_states_list[1][:, :2] - s_def) * away_dir).sum(-1).cpu())
        FH.append(fh.cpu()); FALL.append((fall_t > 0).cpu()); FALLT.append(fall_t.cpu()); AWAY.append(torch.stack(away, 1))
    return torch.cat(FH).numpy(), torch.cat(GAP0).numpy(), torch.cat(AWAY).numpy(), torch.cat(FALL).numpy(), torch.cat(FALLT).numpy()

def run(task, low):
    task.step_counter = task.warmup_time + 10
    print("\n================ exchange-window timing (symmetric env, 1024 bouts each) ================")
    fh, gap, _, fall, _ = bouts(task, low, D_STAND, True)
    ok = fh > 0
    print("[lunge vs STILL]  spawn gap %.2f-%.2f m (mean %.2f) | landed %.1f%% | first-hit step p10=%d p50=%d p90=%d (x33ms) | attacker fell: %.1f%%"
          % (gap.min(), gap.max(), gap.mean(), 100 * ok.mean(), np.percentile(fh[ok], 10), np.median(fh[ok]), np.percentile(fh[ok], 90), 100 * fall.mean()))
    for lo, hi in [(1.5, 1.8), (1.8, 2.0), (2.0, 2.2), (2.2, 2.6)]:
        m = ok & (gap >= lo) & (gap < hi)
        if m.any(): print("      spawn gap %.1f-%.1f m: median first-hit step %d (n=%d)" % (lo, hi, np.median(fh[m]), m.sum()))
    still_fh = fh.copy()
    fh, gap, away, fall, fall_t = bouts(task, low, D_DODGE, False)
    ok = fh > 0
    print("[lunge vs DODGER] landed %.1f%% | first-hit step p10=%d p50=%d p90=%d | attacker fell in %.1f%% of bouts"
          % (100 * ok.mean(), np.percentile(fh[ok], 10), np.median(fh[ok]), np.percentile(fh[ok], 90), 100 * fall.mean()))
    fb = fall & ((fh < 0) | (fall_t < fh))
    print("      attacker fell BEFORE landing (or without landing): %.1f%% of bouts" % (100 * fb.mean()))
    ft = fall_t[fall]
    if ft.size: print("      fall step: p10=%d p50=%d p90=%d | falls within the first 28 steps: %.1f%% of all bouts" %
                      (np.percentile(ft, 10), np.median(ft), np.percentile(ft, 90), 100 * (fall & (fall_t <= 28)).mean()))
    print("      dodger distance moved away from the attacker (mean m) at step: " +
          "  ".join("t%d=%.2f" % (t, away[:, t - 1].mean()) for t in (5, 10, 15, 20, 25, 30, 40, 70)))
    print("[if the window were W steps] landing rate vs STILL / vs DODGER:")
    for W in (15, 18, 20, 22, 25, 28, 30, 35, 40, 70):
        print("      W=%2d (%.2f s): still %5.1f%%   dodger %5.1f%%" % (W, W / 30, 100 * ((still_fh > 0) & (still_fh <= W)).mean(), 100 * (ok & (fh <= W)).mean()))

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
