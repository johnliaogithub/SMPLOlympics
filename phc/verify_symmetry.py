# Mirror test: force agent 1 to be the exact 180-degree rotation (about the vertical axis through the
# origin) of agent 0, then (1) compare the two agents' observations and (2) step forward with the
# same deterministic policy and check whether they stay mirror images.
import os, sys
os.environ['OMP_NUM_THREADS'] = "1"; sys.path.append(os.getcwd()); sys.path.append('./SMPLSim')
from isaacgym import gymapi  # noqa
import torch, numpy as np, hydra
from isaacgym.torch_utils import quat_mul
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, D_STAND, D_LUNGE_UPPER

def rot_v(v):  # 180 deg about z: (x, y, z) -> (-x, -y, z)
    out = v.clone(); out[..., 0] = -v[..., 0]; out[..., 1] = -v[..., 1]; return out
def rot_q(q):
    q180 = torch.zeros_like(q); q180[..., 2] = 1.0   # xyzw
    return quat_mul(q180.reshape(-1, 4), q.reshape(-1, 4)).reshape(q.shape)

@torch.no_grad()
def run(task, low, drill, T=60):
    dev = task.device; N = task.num_envs
    orig = task._set_env_state
    def mirrored(env_ids, root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list,
                 rigid_body_pos_list=None, rigid_body_rot_list=None, rigid_body_vel_list=None, rigid_body_ang_vel_list=None):
        root_pos_list[1] = rot_v(root_pos_list[0]); root_rot_list[1] = rot_q(root_rot_list[0])
        root_vel_list[1] = rot_v(root_vel_list[0]); root_ang_vel_list[1] = rot_v(root_ang_vel_list[0])
        dof_pos_list[1] = dof_pos_list[0].clone(); dof_vel_list[1] = dof_vel_list[0].clone()
        if rigid_body_pos_list is not None:
            rigid_body_pos_list[1] = rot_v(rigid_body_pos_list[0]); rigid_body_rot_list[1] = rot_q(rigid_body_rot_list[0])
            rigid_body_vel_list[1] = rot_v(rigid_body_vel_list[0]); rigid_body_ang_vel_list[1] = rot_v(rigid_body_ang_vel_list[0])
        return orig(env_ids, root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list,
                    rigid_body_pos_list, rigid_body_rot_list, rigid_body_vel_list, rigid_body_ang_vel_list)
    task._set_env_state = mirrored
    p = torch.zeros(NUM_DRILLS, device=dev); p[drill] = 1; task.drill_probs = p
    task.step_counter = task.warmup_time + 10; task.reset()
    task._set_env_state = orig
    task.drill_ids[:] = drill; task.opp_drill_ids[:] = drill; task.opp_frozen[:] = False
    task._compute_observations()
    S = task.get_self_obs_size(); o = task.obs_buf
    d = (o[:N] - o[N:2 * N]).abs()
    r0 = task._humanoid_root_states_list[0][:, :3]; r1 = task._humanoid_root_states_list[1][:, :3]
    print("\n=== mirror test, both agents running drill '%s' ===" % ["advance","retreat","stand","lunge_upper","lunge_groin","dodge","step_left","step_right"][drill])
    print("[t=0] state mirror error: root pos %.2e m" % (r1 - rot_v(r0)).norm(dim=-1).max())
    print("[t=0] obs |agent0 - agent1|: self-obs part max=%.3e   task-obs part max=%.3e" % (d[:, :S].max(), d[:, S:].max()))
    col = d.max(0).values
    top = torch.topk(col, 8)
    print("      largest differing obs columns (index: diff):", ", ".join("%d: %.3f" % (i - S if i >= S else i, v) + (" [task]" if i >= S else " [self]") for v, i in zip(top.values.tolist(), top.indices.tolist())))
    errs = []
    C0 = S + 3 + 6 + 3 + 72 + 18          # start of self_contact_norm block (24), then opp block (24)
    def contact_report(t):
        o = task.obs_buf; a0 = o[:N, C0:C0 + 24]; a1 = o[N:2 * N, C0:C0 + 24]
        f0 = task._contact_forces_list[0].norm(dim=-1); f1 = task._contact_forces_list[1].norm(dim=-1)
        print("   t=%-3d own-contact obs: agent0 sum=%7.1f agent1 sum=%7.1f |diff| max=%6.1f   | sim contact tensor: agent0 sum=%7.1f agent1 sum=%7.1f"
              % (t, a0.sum(-1).mean(), a1.sum(-1).mean(), (a0 - a1).abs().max(), f0.sum(-1).mean(), f1.sum(-1).mean()))
    contact_report(0)
    for t in range(T):
        obs = task.obs_buf
        z = low(obs, torch.cat([task.drill_ids, task.opp_drill_ids]))
        task.step(z)
        if t + 1 in (1, 2, 3, 5, 10, 30): contact_report(t + 1)
        r0 = task._humanoid_root_states_list[0][:, :3]; r1 = task._humanoid_root_states_list[1][:, :3]
        errs.append((r1 - rot_v(r0)).norm(dim=-1).mean().item())
    print("[steps] mean root mirror error at t=1,5,10,30,%d: %s m" % (T, ", ".join("%.3f" % errs[i] for i in [0, 4, 9, 29, T - 1])))


@torch.no_grad()
def matchup(task, low, attacker, T=70, rounds=4, zero_contact=False):
    """Mirrored starts; one agent lunges (policy), the other stands still (Z=0)."""
    dev = task.device; N = task.num_envs; orig = task._set_env_state
    def mirrored(env_ids, root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list,
                 rigid_body_pos_list=None, rigid_body_rot_list=None, rigid_body_vel_list=None, rigid_body_ang_vel_list=None):
        rp, rr, dp, rv, rav, dv = root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list
        rbp, rbr, rbv, rbav = rigid_body_pos_list, rigid_body_rot_list, rigid_body_vel_list, rigid_body_ang_vel_list
        rp[1] = rot_v(rp[0]); rr[1] = rot_q(rr[0]); rv[1] = rot_v(rv[0]); rav[1] = rot_v(rav[0])
        dp[1] = dp[0].clone(); dv[1] = dv[0].clone()
        if rbp is not None:
            rbp[1] = rot_v(rbp[0]); rbr[1] = rot_q(rbr[0]); rbv[1] = rot_v(rbv[0]); rbav[1] = rot_v(rbav[0])
        return orig(env_ids, rp, rr, dp, rv, rav, dv, rbp, rbr, rbv, rbav)
    p = torch.zeros(NUM_DRILLS, device=dev); p[D_LUNGE_UPPER] = 1; task.drill_probs = p
    land = []; first = []; MIND = []
    for r in range(rounds):
        torch.manual_seed(7 + r); task._set_env_state = mirrored; task.reset(); task._set_env_state = orig
        hit = torch.zeros(N, dtype=torch.bool, device=dev); th = torch.full((N,), -1., device=dev)
        mind = torch.full((N,), 9.0, device=dev)
        for t in range(T):
            task.drill_ids[:] = D_LUNGE_UPPER if attacker == 0 else D_STAND
            task.opp_drill_ids[:] = D_LUNGE_UPPER if attacker == 1 else D_STAND
            task.opp_frozen[:] = False
            obs = task.obs_buf
            if zero_contact:
                C0 = task.get_self_obs_size() + 102; obs = obs.clone(); obs[:, C0:C0 + 48] = 0
            z = low(obs, torch.cat([task.drill_ids, task.opp_drill_ids]))
            z[(1 - attacker) * N:(2 - attacker) * N] = 0
            task.step(z)
            h = task.sword_hit_list[attacker][:, 0].bool(); th = torch.where(h & ~hit, torch.full_like(th, t), th); hit |= h
            tip = task.get_sword_tip_pos()[attacker][:, 0]; tg = task._rigid_body_pos_list[1 - attacker][:, task._target_ids]
            mind = torch.minimum(mind, (tip[:, None] - tg).norm(dim=-1).min(-1).values)
        land.append(hit.float()); first.append(th); MIND.append(mind)
    L = torch.cat(land); F = torch.cat(first)
    M = torch.cat(MIND)
    print("[mirrored starts%s] agent %d lunges at still agent %d: lands %.1f%%  median first-hit step %.0f  (%d bouts)"
          % (", CONTACT OBS ZEROED" if zero_contact else "", attacker, 1 - attacker, 100 * L.mean(), F[F >= 0].median(), L.numel()))
    print("      closest tip approach to a target point: median %.3f m | within hit_dist (%.2f m): %.1f%% | of those, registered: %.1f%%"
          % (M.median(), task.hit_dist, 100 * (M < task.hit_dist).float().mean(), 100 * L[M < task.hit_dist].mean()))


@torch.no_grad()
def compare_traces(task, low, T=30):
    """Run 'A0 lunges at still A1' and its mirror 'A1 lunges at still A0' from the same mirrored start;
    compare the attacker's observation step by step."""
    dev = task.device; N = task.num_envs; orig = task._set_env_state; S = task.get_self_obs_size()
    def mirrored(env_ids, root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list,
                 rigid_body_pos_list=None, rigid_body_rot_list=None, rigid_body_vel_list=None, rigid_body_ang_vel_list=None):
        rp, rr, dp, rv, rav, dv = root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list
        rbp, rbr, rbv, rbav = rigid_body_pos_list, rigid_body_rot_list, rigid_body_vel_list, rigid_body_ang_vel_list
        rp[1] = rot_v(rp[0]); rr[1] = rot_q(rr[0]); rv[1] = rot_v(rv[0]); rav[1] = rot_v(rav[0])
        dp[1] = dp[0].clone(); dv[1] = dv[0].clone()
        if rbp is not None:
            rbp[1] = rot_v(rbp[0]); rbr[1] = rot_q(rbr[0]); rbv[1] = rot_v(rbv[0]); rbav[1] = rot_v(rbav[0])
        return orig(env_ids, rp, rr, dp, rv, rav, dv, rbp, rbr, rbv, rbav)
    p = torch.zeros(NUM_DRILLS, device=dev); p[D_LUNGE_UPPER] = 1; task.drill_probs = p
    traces = {}
    for attacker in (0, 1):
        torch.manual_seed(11); np.random.seed(11)
        task._set_env_state = mirrored; task.reset(); task._set_env_state = orig
        tr = []
        for t in range(T):
            task.drill_ids[:] = D_LUNGE_UPPER if attacker == 0 else D_STAND
            task.opp_drill_ids[:] = D_LUNGE_UPPER if attacker == 1 else D_STAND
            task.opp_frozen[:] = False
            task._compute_observations()
            obs = task.obs_buf.clone()
            tr.append(obs[attacker * N:(attacker + 1) * N, :-NUM_DRILLS].clone())   # attacker's obs (drill one-hot excluded)
            z = low(obs, torch.cat([task.drill_ids, task.opp_drill_ids])); z[(1 - attacker) * N:(2 - attacker) * N] = 0
            task.step(z)
        traces[attacker] = tr
    names = (["self:%d" % i for i in range(S)] + ["root_pos_diff"] * 3 + ["root_rot"] * 6 + ["opp_root_vel"] * 3 +
             ["opp_body_pos"] * 72 + ["tip_to_target"] * 18 + ["own_contact"] * 24 + ["opp_contact"] * 24 + ["bbox"] * 4)
    print("\n=== attacker observation: case 'A0 attacks' vs its mirror 'A1 attacks' (same start) ===")
    for t in (0, 1, 2, 3, 5, 8, 12, 20, T - 1):
        d = (traces[0][t] - traces[1][t]).abs()
        col = d.max(0).values; top = torch.topk(col, 4)
        groups = {}
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            g = names[i] if i < len(names) else "col%d" % i
            groups[g] = max(groups.get(g, 0), v)
        print("   t=%-3d max|diff| self-obs=%.3e  task-obs=%.3e   top: %s" % (t, d[:, :S].max(), d[:, S:].max(),
              ", ".join("%s=%.3f" % kv for kv in groups.items())))


@torch.no_grad()
def bias_test(task, low, drill, T=40, rounds=4, swap=False):
    """Mirrored starts, both run the same drill; signed difference in how far each travels along its own
    facing direction and in hand/tip height, to detect a systematic (not random) side bias."""
    dev = task.device; N = task.num_envs; orig = task._set_env_state
    def mirrored(env_ids, root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list,
                 rigid_body_pos_list=None, rigid_body_rot_list=None, rigid_body_vel_list=None, rigid_body_ang_vel_list=None):
        rp, rr, dp, rv, rav, dv = root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list
        rbp, rbr, rbv, rbav = rigid_body_pos_list, rigid_body_rot_list, rigid_body_vel_list, rigid_body_ang_vel_list
        rp[1] = rot_v(rp[0]); rr[1] = rot_q(rr[0]); rv[1] = rot_v(rv[0]); rav[1] = rot_v(rav[0])
        dp[1] = dp[0].clone(); dv[1] = dv[0].clone()
        if rbp is not None:
            rbp[1] = rot_v(rbp[0]); rbr[1] = rot_q(rbr[0]); rbv[1] = rot_v(rbv[0]); rbav[1] = rot_v(rbav[0])
        if swap:   # put agent 0 on the +y side and agent 1 on the -y side
            for L in (rp, rr, rv, rav, dp, dv) + ((rbp, rbr, rbv, rbav) if rbp is not None else ()):
                L[0], L[1] = L[1], L[0]
        return orig(env_ids, rp, rr, dp, rv, rav, dv, rbp, rbr, rbv, rbav)
    p = torch.zeros(NUM_DRILLS, device=dev); p[drill] = 1; task.drill_probs = p
    D = []
    for r in range(rounds):
        torch.manual_seed(31 + r); task._set_env_state = mirrored; task.reset(); task._set_env_state = orig
        s0 = task._humanoid_root_states_list[0][:, :2].clone(); s1 = task._humanoid_root_states_list[1][:, :2].clone()
        for t in range(T):
            task.drill_ids[:] = drill; task.opp_drill_ids[:] = drill; task.opp_frozen[:] = False
            obs = task.obs_buf; task.step(low(obs, torch.cat([task.drill_ids, task.opp_drill_ids])))
        sg = -1.0 if swap else 1.0   # agent 0 faces +y normally, -y when swapped
        f0 = sg * (task._humanoid_root_states_list[0][:, 1] - s0[:, 1]); f1 = -sg * (task._humanoid_root_states_list[1][:, 1] - s1[:, 1])
        D.append(torch.stack([f0, f1], -1))
    D = torch.cat(D); diff = D[:, 0] - D[:, 1]
    se = diff.std() / diff.numel() ** 0.5
    print("[bias%s, both '%s', %d steps] forward travel: agent0 %.3f m, agent1 %.3f m | paired diff (a0-a1) mean %+.4f m (std err %.4f, %d pairs)"
          % (" SIDES SWAPPED (agent0 at +y)" if swap else " (agent0 at -y)", ["advance","retreat","stand","lunge_upper","lunge_groin","dodge","step_left","step_right"][drill], T, D[:, 0].mean(), D[:, 1].mean(), diff.mean(), se, diff.numel()))


@torch.no_grad()
def bbox_check(task, low):
    dev = task.device; N = task.num_envs; orig = task._set_env_state; S = task.get_self_obs_size()
    def mirrored(env_ids, root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list,
                 rigid_body_pos_list=None, rigid_body_rot_list=None, rigid_body_vel_list=None, rigid_body_ang_vel_list=None):
        rp, rr, dp, rv, rav, dv = root_pos_list, root_rot_list, dof_pos_list, root_vel_list, root_ang_vel_list, dof_vel_list
        rbp, rbr, rbv, rbav = rigid_body_pos_list, rigid_body_rot_list, rigid_body_vel_list, rigid_body_ang_vel_list
        rp[1] = rot_v(rp[0]); rr[1] = rot_q(rr[0]); rv[1] = rot_v(rv[0]); rav[1] = rot_v(rav[0])
        dp[1] = dp[0].clone(); dv[1] = dv[0].clone()
        if rbp is not None:
            rbp[1] = rot_v(rbp[0]); rbr[1] = rot_q(rbr[0]); rbv[1] = rot_v(rbv[0]); rbav[1] = rot_v(rbav[0])
        return orig(env_ids, rp, rr, dp, rv, rav, dv, rbp, rbr, rbv, rbav)
    p = torch.zeros(NUM_DRILLS, device=dev); p[D_STAND] = 1; task.drill_probs = p
    torch.manual_seed(3); task._set_env_state = mirrored; task.reset(); task._set_env_state = orig
    task._compute_observations(); o = task.obs_buf
    B0 = S + 150   # bounding-box block: 2 corners x (forward, left) in the fencer's own frame
    print("\n[bbox obs, mirrored pair, env 0] (corner0 fwd, corner0 left, corner1 fwd, corner1 left)")
    print("   agent 0 (at y=%+.2f): %s" % (task._humanoid_root_states_list[0][0, 1], [round(v, 2) for v in o[0, B0:B0 + 4].tolist()]))
    print("   agent 1 (at y=%+.2f): %s" % (task._humanoid_root_states_list[1][0, 1], [round(v, 2) for v in o[N, B0:B0 + 4].tolist()]))
    print("   max |agent0 - agent1| over all envs in this block: %.3f" % (o[:N, B0:B0 + 4] - o[N:2 * N, B0:B0 + 4]).abs().max())
    new0 = o[:N].clone(); task.bbox_obs_facing_sorted = False; task._compute_observations(); old0 = task.obs_buf[:N].clone(); task.bbox_obs_facing_sorted = True
    print("   agent 0 full obs, fix ON vs OFF: max |diff| = %.2e  (should be 0: agent 0 unchanged)" % (new0 - old0).abs().max())

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
    task.step_counter = task.warmup_time + 10
    from phc.env.tasks.humanoid_fencing_drills import D_ADVANCE, D_RETREAT
    bbox_check(task, low)
    from phc.env.tasks.humanoid_fencing_drills import D_RETREAT
    bias_test(task, low, D_RETREAT); bias_test(task, low, D_RETREAT, swap=True)

if __name__ == '__main__':
    main()
