# Survey: how often does the current policy produce a usable "ready" (en-garde-like) state?
# A frame qualifies only if ALL hold for the fencer:
#   both feet planted   - each foot (ankle+toe) carries > FOOT_N of contact force
#   feet split          - front (right) foot ahead of rear (left) foot by SPLIT_MIN..SPLIT_MAX along facing
#   sword up and aimed  - tip height in TIP_Z, and the blade points at the opponent (cos > AIM_COS)
#   nearly still        - root horizontal speed < SPEED_MAX
# Reports the qualifying fraction per drill and per bout-phase, and which criteria fail most.
import os, sys
os.environ['OMP_NUM_THREADS'] = "1"; sys.path.append(os.getcwd()); sys.path.append('./SMPLSim')
from isaacgym import gymapi  # noqa
import torch, numpy as np, hydra
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, DRILL_NAMES

FOOT_N, SPLIT_MIN, SPLIT_MAX, TIP_Z, AIM_COS, SPEED_MAX = 50.0, 0.25, 0.70, (0.7, 1.6), 0.8, 0.3

@torch.no_grad()
def run(task, low, T=300):
    dev = task.device; N = task.num_envs
    task.drill_probs = torch.tensor([0.4, 0.4, 0.4, 0.6, 0.6, 2.0, 0.4, 0.4], device=dev); task.drill_probs /= task.drill_probs.sum()
    task.step_counter = task.warmup_time + 10; task.reset()
    ids = lambda names: task._build_key_body_ids_tensor(names)
    lf, rf = ids(["L_Ankle", "L_Toe"]), ids(["R_Ankle", "R_Toe"])
    la, ra = int(ids(["L_Ankle"])[0]), int(ids(["R_Ankle"])[0]); hand = int(task._hand_ids[0])
    stats = {d: np.zeros(6) for d in range(NUM_DRILLS)}           # n, planted, split, sword, still, ALL
    phase = {}                                                     # (drill, progress bin) -> [n, all]
    for t in range(T):
        obs = task.obs_buf; d_now = task.drill_ids.clone(); prog = task.progress_buf.clone()
        prev = task._humanoid_root_states_list[0][:, :2].clone()
        task.step(torch.cat([low(obs[:N], task.drill_ids), low(obs[N:2 * N], task.opp_drill_ids)], 0))
        bp = task._rigid_body_pos_list[0]; cf = task._contact_forces_list[0]
        planted = (cf[:, lf].norm(dim=-1).sum(-1) > FOOT_N) & (cf[:, rf].norm(dim=-1).sum(-1) > FOOT_N)
        root = task._humanoid_root_states_list[0][:, :2]; opp = task._humanoid_root_states_list[1][:, :2]
        fwd = F.normalize(opp - root, dim=-1)
        split = ((bp[:, ra, :2] - bp[:, la, :2]) * fwd).sum(-1)
        split_ok = (split > SPLIT_MIN) & (split < SPLIT_MAX)
        tip = task.get_sword_tip_pos()[0][:, 0]
        blade = F.normalize(tip - bp[:, hand], dim=-1)
        to_opp = F.normalize(task._rigid_body_pos_list[1][:, task._target_ids].mean(1) - bp[:, hand], dim=-1)
        sword_ok = (tip[:, 2] > TIP_Z[0]) & (tip[:, 2] < TIP_Z[1]) & ((blade * to_opp).sum(-1) > AIM_COS)
        still = ((root - prev).norm(dim=-1) / task.dt) < SPEED_MAX
        allok = planted & split_ok & sword_ok & still
        for d in range(NUM_DRILLS):
            m = d_now == d
            if m.any():
                stats[d] += np.array([m.sum().item(), (m & planted).sum().item(), (m & split_ok).sum().item(),
                                      (m & sword_ok).sum().item(), (m & still).sum().item(), (m & allok).sum().item()])
                for b in (0, 10, 20, 30, 50, 70):
                    mb = m & (prog >= b) & (prog < b + (10 if b < 30 else 20))
                    if mb.any():
                        k = (d, b); v = phase.setdefault(k, [0, 0]); v[0] += mb.sum().item(); v[1] += (mb & allok).sum().item()
        dn = task.reset_buf.nonzero().flatten()
        if len(dn): task.reset(dn)
    print("\n================ ready-stance survey (agent 0, %d frames) ================" % int(sum(s[0] for s in stats.values())))
    print("criteria: both feet >%.0f N | split %.2f-%.2f m | tip z %.1f-%.1f m & aimed cos>%.1f | speed <%.1f m/s"
          % (FOOT_N, SPLIT_MIN, SPLIT_MAX, TIP_Z[0], TIP_Z[1], AIM_COS, SPEED_MAX))
    print("  %-12s %7s %8s %8s %8s %8s %8s" % ("drill", "frames", "planted", "split", "sword", "still", "ALL"))
    for d in range(NUM_DRILLS):
        s = stats[d]
        if s[0]: print("  %-12s %7d %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.2f%%  (%d frames)" % ((DRILL_NAMES[d], s[0]) + tuple(100 * s[1:] / s[0]) + (s[5],)))
    print("\n  qualifying fraction by bout phase (progress step bin), drills with any hits:")
    for d in range(NUM_DRILLS):
        row = [(b, v) for (dd, b), v in sorted(phase.items()) if dd == d]
        if any(v[1] for _, v in row):
            print("  %-12s " % DRILL_NAMES[d] + "  ".join("t%d+: %.1f%%" % (b, 100 * v[1] / max(v[0], 1)) for b, v in row))

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
