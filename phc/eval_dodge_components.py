# 256-episode dodge evaluation: every reward component and key state variables, per step,
# split by outcome (dodge survived vs got hit). One episode per env (no resets), so exactly
# num_envs episodes. The dodger is agent 0; the opponent lunges with the same network.
#
# Reward components are extracted EXACTLY: for each weighted term the reward is recomputed on the
# same state with that weight set to 0 (difference = the term). Facing and posture (hard-coded
# weights) are computed directly, and a residual check confirms the parts sum to the env reward.
#
# Outputs (output/eval/): dodge_components_<tag>.csv (one row per episode-step) and
# dodge_components_<tag>.png (per-step means by outcome).
import os, sys
os.environ['OMP_NUM_THREADS'] = "1"; sys.path.append(os.getcwd()); sys.path.append('./SMPLSim')
from isaacgym import gymapi  # noqa
import torch, numpy as np, hydra, csv
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.utils import torch_utils
from phc.run_hydra import parse_sim_params
from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, D_DODGE

COMPONENTS = ["evade", "facing", "posture", "arm_back", "foot_back", "hit_penalty", "survive_bonus"]
FEATURES = ["gap", "off_line", "back_disp", "lat_disp", "opp_tip_dist", "speed"]

@torch.no_grad()
def components(task):
    """Exact per-term decomposition of the dodger's (agent 0) reward on the current state."""
    N = task.num_envs
    keep_rew = task.rew_buf.clone(); keep_ll = task._lunge_landed.clone()
    def recompute(**zero):
        old = {k: getattr(task, k) for k in zero}
        for k, v in zero.items(): setattr(task, k, v)
        task._compute_reward(task.actions); out = task.rew_buf[:N].clone()
        for k, v in old.items(): setattr(task, k, v)
        return out
    full = recompute()
    c = {"evade": full - recompute(dodge_evade_w=0.0),
         "hit_penalty": full - recompute(dodge_imhit_w=0.0),
         "survive_bonus": full - recompute(dodge_survive_bonus=0.0),
         "arm_back": full - recompute(arm_back_w=0.0),
         "foot_back": full - recompute(dodge_footback_w=0.0)}
    task.rew_buf[:] = keep_rew; task._lunge_landed[:] = keep_ll
    # hard-coded terms, replicated from _compute_drill_reward
    rs = task._humanoid_root_states_list[0]; root_pos, root_rot = rs[:, 0:3], rs[:, 3:7]
    opp = task._humanoid_root_states_list[1][:, 0:3]
    d3 = opp - root_pos; d3[:, 2] = 0; d3 = F.normalize(d3, dim=-1)
    loc = torch_utils.my_quat_rotate(torch_utils.calc_heading_quat_inv(root_rot), d3)
    c["facing"] = 0.2 * torch.exp(-2.0 * (1 - loc[:, 0]))
    pel = task._rigid_body_pos_list[0][:, task._pelvis_id[0]]; ch = task._rigid_body_pos_list[0][:, task._chest_id[0]]
    c["posture"] = 0.15 * torch.clamp(F.normalize(ch - pel, dim=-1)[:, 2], 0.0, 1.0)
    resid = full - sum(c[k] for k in COMPONENTS)
    return c, full, resid

@torch.no_grad()
def run(cfg, task, low, tag, out_dir="output/eval"):
    dev = task.device; N = task.num_envs; T = int(task.exchange_episode_length or task.strike_episode_length)
    p = torch.zeros(NUM_DRILLS, device=dev); p[D_DODGE] = 1; task.drill_probs = p
    task.step_counter = task.warmup_time + 10; task.reset()
    start = task._humanoid_root_states_list[0][:, :2].clone()
    away = F.normalize(start - task._humanoid_root_states_list[1][:, :2], dim=-1)
    lat_dir = torch.stack([-away[:, 1], away[:, 0]], -1)
    alive = torch.ones(N, dtype=torch.bool, device=dev); outcome = torch.zeros(N, dtype=torch.long, device=dev)  # 1 = got hit
    length = torch.zeros(N, dtype=torch.long, device=dev)
    rows, max_resid, max_env_err = [], 0.0, 0.0
    for t in range(1, T + 3):
        if not alive.any(): break
        obs = task.obs_buf
        task.step(torch.cat([low(obs[:N], task.drill_ids), low(obs[N:2 * N], task.opp_drill_ids)], 0))
        r_env = task.rew_buf[:N].clone()
        c, full, resid = components(task)
        max_env_err = max(max_env_err, (full - r_env)[alive].abs().max().item())
        max_resid = max(max_resid, resid[alive].abs().max().item())
        r0 = task._humanoid_root_states_list[0][:, :2]; r1 = task._humanoid_root_states_list[1][:, :2]
        opp_tip = task.get_sword_tip_pos()[1]; opp_hand = task._rigid_body_pos_list[1][:, task._hand_ids]
        my_tg = task._rigid_body_pos_list[0][:, task._target_ids]
        dl = F.normalize(opp_tip - opp_hand, dim=-1); ap = my_tg - opp_hand
        disp = r0 - start
        feat = {"gap": (r0 - r1).norm(dim=-1),
                "off_line": (ap - (ap * dl).sum(-1, keepdim=True) * dl).norm(dim=-1).min(-1).values,
                "back_disp": (disp * away).sum(-1), "lat_disp": (disp * lat_dir).sum(-1),
                "opp_tip_dist": (opp_tip - my_tg).norm(dim=-1).min(-1).values,
                "speed": ((task._humanoid_root_states_list[0][:, :2] - task._prev_root_pos_list[0][:, :2]).norm(dim=-1) / task.dt)}
        done = task.reset_buf.bool() & alive
        got = task.sword_hit_list[1][:, 0].bool()
        for e in alive.nonzero().flatten().tolist():
            rows.append([e, t] + [float(c[k][e]) for k in COMPONENTS] + [float(r_env[e])] + [float(feat[k][e]) for k in FEATURES])
        outcome[done & got] = 1; length[done] = t; alive &= ~done
    length[alive] = T
    # ---- write CSV (outcome joined per episode) ----
    os.makedirs(out_dir, exist_ok=True)
    oc = outcome.cpu().numpy(); ln = length.cpu().numpy()
    path_csv = os.path.join(out_dir, "dodge_components_%s.csv" % tag)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "step"] + COMPONENTS + ["total"] + FEATURES + ["outcome", "episode_len"])
        for r in rows: w.writerow(r + ["got_hit" if oc[r[0]] else "survived", int(ln[r[0]])])
    A = np.array([r for r in rows], dtype=float)
    ep, st = A[:, 0].astype(int), A[:, 1].astype(int)
    hitmask = oc[ep] == 1
    ci = {k: 2 + i for i, k in enumerate(COMPONENTS)}; ti = 2 + len(COMPONENTS); fi = {k: ti + 1 + i for i, k in enumerate(FEATURES)}
    # ---- printed summary ----
    nh, ns = int(oc.sum()), int((oc == 0).sum())
    print("\n================ dodge component eval [%s]: %d episodes, window %d steps ================" % (tag, N, T))
    print("checks: env reward vs recomputed max|err| = %.1e ; components sum residual max = %.1e" % (max_env_err, max_resid))
    print("outcome: survived %d (%.1f%%) | got hit %d (%.1f%%), median hit step %s" %
          (ns, 100 * ns / N, nh, 100 * nh / N, int(np.median(ln[oc == 1])) if nh else "-"))
    print("\nper-episode TOTAL of each component (mean):")
    print("  %-14s %10s %10s" % ("component", "survived", "got hit"))
    for k in COMPONENTS + ["total"]:
        col = ci.get(k, ti)
        def tot(m):
            if not m.any(): return float('nan')
            s = np.bincount(ep[m], weights=A[m, col], minlength=N); e = np.unique(ep[m]); return s[e].mean()
        print("  %-14s %10.2f %10.2f" % (k, tot(~hitmask), tot(hitmask)))
    print("\nper-step means (survived | got hit) at selected steps; n = episodes still running:")
    hdr = ["step", "n_s|n_h"] + ["evade", "facing", "posture", "arm_back", "gap", "off_line", "back_disp", "lat_disp", "opp_tip_dist"]
    print("  " + "  ".join("%-17s" % h if i > 1 else "%-7s" % h for i, h in enumerate(hdr)))
    for t in [1, 5, 10, 15, 20, 24, 26, T]:
        ms, mh = (st == t) & ~hitmask, (st == t) & hitmask
        vals = []
        for k in hdr[2:]:
            col = ci.get(k, fi.get(k))
            a = A[ms, col].mean() if ms.any() else float('nan'); b = A[mh, col].mean() if mh.any() else float('nan')
            vals.append("%7.3f | %-7.3f" % (a, b))
        print("  %-7d %-7s " % (t, "%d|%d" % (ms.sum(), mh.sum())) + "  ".join(vals))
    # ---- chart ----
    try:
        plot(A, st, hitmask, ci, fi, T, os.path.join(out_dir, "dodge_components_%s.png" % tag), tag, ns, nh)
    except Exception as ex:
        print("[plot] skipped:", ex)
    print("\nwrote %s" % path_csv)

def plot(A, st, hitmask, ci, fi, T, path, tag, ns, nh):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    SURV, HIT = "#2a78d6", "#eb6834"          # validated categorical slots 1-2 (light surface)
    INK, MUTED, GRID, SURF = "#2b2b29", "#6b6b66", "#e7e6e1", "#fcfcfb"
    panels = [("evade", "off-line reward (evade term)"), ("facing", "facing reward"), ("posture", "posture bonus"),
              ("arm_back", "free-arm penalty"), ("gap", "distance to attacker (m)"), ("off_line", "off the attack line (m)"),
              ("back_disp", "moved back from spawn (m)"), ("lat_disp", "moved sideways from spawn (m)")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor=SURF)
    steps = np.arange(1, T + 1)
    for ax, (k, title) in zip(axes.flat, panels):
        col = ci.get(k, fi.get(k))
        for m, color, name in [(~hitmask, SURV, "survived (n=%d)" % ns), (hitmask, HIT, "got hit (n=%d)" % nh)]:
            y = [A[m & (st == t), col].mean() if (m & (st == t)).sum() >= 5 else np.nan for t in steps]
            ax.plot(steps, y, color=color, lw=2, label=name, solid_capstyle="round")
        ax.set_facecolor(SURF); ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8); ax.set_xlim(1, T)
    for ax in axes[1]: ax.set_xlabel("step in bout (x33 ms)", fontsize=9, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=10, labelcolor=INK, ncol=2)
    fig.suptitle("Dodge drill [%s]: per-step means by outcome (got-hit lines stop as bouts end; points need >=5 bouts)" % tag,
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=130, facecolor=SURF); plt.close(fig)
    print("wrote %s" % path)

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
    run(cfg, task, low, cfg.env.get("eval_tag", "eval"))

if __name__ == '__main__':
    main()
