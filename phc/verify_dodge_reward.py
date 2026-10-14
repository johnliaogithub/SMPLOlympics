# Verify the v15 dodge_offline reward is live and computes what we intend, and show the reward
# landscape the dodger actually experiences. No env code is modified.
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
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, D_DODGE

@torch.no_grad()
def run(task, low, T=200):
    dev = task.device; N = task.num_envs
    assert task.dodge_offline and not task.dodge_pure, "v15 config not active"
    p = torch.zeros(NUM_DRILLS, device=dev); p[D_DODGE] = 1; task.drill_probs = p
    task.step_counter = task.warmup_time + 10; task.reset()
    errs, gate_on, off_g, rew_g, rew_ng, evade_g, gaps0 = [], [], [], [], [], [], []
    for t in range(T):
        obs = task.obs_buf
        task.step(torch.cat([low(obs[:N], task.drill_ids), low(obs[N:2 * N], task.opp_drill_ids)], 0))
        # --- independent recomputation of the evade term (agent 0 = dodger) ---
        r0 = task._humanoid_root_states_list[0][:, :2]; r1 = task._humanoid_root_states_list[1][:, :2]
        gap = (r0 - r1).norm(dim=-1); gate = (gap < task.dodge_evade_gate).float()
        opp_tip = task.get_sword_tip_pos()[1]; opp_hand = task._rigid_body_pos_list[1][:, task._hand_ids]
        d = opp_tip - opp_hand; d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ap = task._rigid_body_pos_list[0][:, task._target_ids] - opp_hand
        perp = (ap - (ap * d).sum(-1, keepdim=True) * d).norm(dim=-1).min(-1).values
        mine = task.dodge_evade_w * torch.clamp(perp / task.dodge_offline_scale, 0, 1) * gate
        # --- env's own reward with and without the evade term, same state ---
        keep_rew = task.rew_buf.clone(); keep_ll = task._lunge_landed.clone(); w = task.dodge_evade_w
        task.dodge_evade_w = 0.0; task._compute_reward(task.actions); r_off = task.rew_buf[:N].clone()
        task.dodge_evade_w = w;   task._compute_reward(task.actions); r_on = task.rew_buf[:N].clone()
        task.rew_buf[:] = keep_rew; task._lunge_landed[:] = keep_ll
        m = task.drill_ids == D_DODGE
        errs.append(((r_on - r_off) - mine)[m].abs())
        gate_on.append(gate[m]); off_g.append(torch.clamp(perp / task.dodge_offline_scale, 0, 1)[m & (gate > 0)])
        rew_g.append(r_on[m & (gate > 0)]); rew_ng.append(r_on[m & (gate == 0)]); evade_g.append(mine[m & (gate > 0)])
        first = task.progress_buf == 1
        if first.any(): gaps0.append(gap[first])
        dn = task.reset_buf.nonzero().flatten()
        if len(dn): task.reset(dn)
    c = lambda L: torch.cat(L).cpu().numpy()
    e = c(errs); g = c(gate_on); og = c(off_g); rg = c(rew_g); rn = c(rew_ng); eg = c(evade_g); g0 = c(gaps0)
    print("\n================ v15 dodge reward verification ================")
    print("[R1] env reward difference (evade term on - off) vs my independent offline_r*gate: max |err| = %.2e over %d dodge frames" % (e.max(), e.size))
    print("[R2] gap at episode start: min=%.2f  mean=%.2f  max=%.2f m   (gate pays only when gap < %.1f m)" % (g0.min(), g0.mean(), g0.max(), task.dodge_evade_gate))
    print("[R3] fraction of dodge frames with the gate ON: %.1f%%" % (100 * g.mean()))
    print("[R4] when gate ON: offline_r mean=%.2f (0 = on the line, 1 = >=%.1f m off)  -> evade reward mean=%.2f/step" % (og.mean(), task.dodge_offline_scale, eg.mean()))
    print("[R5] total dodge reward per step: gate ON mean=%.3f   gate OFF mean=%.3f   (difference = pay for being within %.1f m)" % (rg.mean(), rn.mean(), task.dodge_evade_gate))


SIG = [0.0, 0.0]
@torch.no_grad()
def episode_budget(task, low, T=300, low_opp=None):
    low_opp = low_opp or low
    """Per dodge EPISODE: sum of the gated evade term, the hit penalty, and everything else."""
    dev = task.device; N = task.num_envs
    p = torch.zeros(NUM_DRILLS, device=dev); p[D_DODGE] = 1; task.drill_probs = p
    task.step_counter = task.warmup_time + 10; task.reset()
    tot = torch.zeros(N, device=dev); ev = torch.zeros(N, device=dev); hit = torch.zeros(N, device=dev); L = torch.zeros(N, device=dev); sv = torch.zeros(N, device=dev)
    rows = []; start = task._humanoid_root_states_list[0][:, :2].clone(); fwd0 = torch.nn.functional.normalize(task._humanoid_root_states_list[1][:, :2] - start, dim=-1)
    for t in range(T):
        obs = task.obs_buf
        z0 = low(obs[:N], task.drill_ids); z1 = low_opp(obs[N:2 * N], task.opp_drill_ids)
        if SIG[0] > 0: z0 = z0 + SIG[0] * torch.randn_like(z0)
        if SIG[1] > 0: z1 = z1 + SIG[1] * torch.randn_like(z1)
        task.step(torch.cat([z0, z1], 0))
        r0 = task._humanoid_root_states_list[0][:, :2]; r1 = task._humanoid_root_states_list[1][:, :2]
        gate = ((r0 - r1).norm(dim=-1) < task.dodge_evade_gate).float()
        opp_tip = task.get_sword_tip_pos()[1]; opp_hand = task._rigid_body_pos_list[1][:, task._hand_ids]
        d = opp_tip - opp_hand; d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ap = task._rigid_body_pos_list[0][:, task._target_ids] - opp_hand
        perp = (ap - (ap * d).sum(-1, keepdim=True) * d).norm(dim=-1).min(-1).values
        disp = r0 - start; fw = (disp * fwd0).sum(-1); lat = (disp - fw[:, None] * fwd0).norm(dim=-1)
        ev += task.dodge_evade_w * torch.where(gate > 0, torch.clamp(perp / task.dodge_offline_scale, 0, 1), torch.full_like(perp, task.dodge_far_value))
        got = task.sword_hit_list[1][:, 0].float()
        hit += -task.dodge_imhit_w * got; tot += task.rew_buf[:N]; L += 1
        sv += task.dodge_survive_bonus * ((task.progress_buf >= task._strike_len()) & (got < 0.5)).float()
        dn = task.reset_buf.nonzero().flatten()
        if len(dn):
            oob = task.out_bound[dn]
            for k, e in enumerate(dn.tolist()):
                rows.append((float(tot[e]), float(ev[e]), float(hit[e]), float(L[e]), bool(got[e]), bool(oob[k]), float(sv[e]), float(fw[e]), float(lat[e]), float(perp[e])))
            tot[dn] = 0; ev[dn] = 0; hit[dn] = 0; L[dn] = 0; sv[dn] = 0
            task.reset(dn)
            start[dn] = task._humanoid_root_states_list[0][dn, :2]
            fwd0[dn] = torch.nn.functional.normalize(task._humanoid_root_states_list[1][dn, :2] - start[dn], dim=-1)
    a = np.array(rows, dtype=float)
    H = a[:, 4] > 0; S = (~H) & (a[:, 5] == 0)
    print("\n================ dodge reward budget per EPISODE (v15 weights) ================")
    print("episodes: %d  (got hit %d = %.0f%%, survived to timeout %d, OOB %d)" % (len(a), H.sum(), 100 * H.mean(), S.sum(), int(a[:, 5].sum())))
    for name, m in [("got hit", H), ("survived", S)]:
        if m.any():
            t, e, h, l, v = a[m, 0].mean(), a[m, 1].mean(), a[m, 2].mean(), a[m, 3].mean(), a[m, 6].mean()
            print("  %-9s len=%5.1f steps | total=%6.2f = evade %6.2f + hit penalty %6.2f + survive bonus %6.2f + other shaping %6.2f" % (name, l, t, e, h, v, t - e - h - v))
    if H.any() and S.any():
        gapr = a[S, 0].mean() - a[H, 0].mean(); outc = -a[H, 2].mean() + a[S, 6].mean()
        print("  return(survive) - return(hit) = %.2f ; explicit win/lose part = %.2f (%.0f%%)" % (gapr, outc, 100 * outc / gapr))
    print("  movement at episode end (from spawn): forward toward opp mean=%+.2f m | sideways |lat| mean=%.2f m, >0.3m=%.0f%% | off-line dist at hit mean=%.2f m" %
          (a[:, 7].mean(), a[:, 8].mean(), 100 * (a[:, 8] > 0.3).mean(), a[H, 9].mean() if H.any() else float('nan')))

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
    low = FrozenLowLevelPolicy(cfg.env["low_level_checkpoint"], NUM_DRILLS, device=task.device)
    SIG[0] = cfg.env.get('sig_learner', 0.0); SIG[1] = cfg.env.get('sig_opp', 0.0)
    print('[noise] learner sigma=%.3f opponent sigma=%.3f' % (SIG[0], SIG[1]))
    oc = cfg.env.get('opp_checkpoint', None)
    low_opp = FrozenLowLevelPolicy(oc, NUM_DRILLS, device=task.device) if oc else None
    print('[opponent] %s' % (oc or 'same network as learner'))
    episode_budget(task, low, low_opp=low_opp) if cfg.env.get('budget', False) else run(task, low)

if __name__ == '__main__':
    main()
