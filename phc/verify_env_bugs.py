# Definitive runtime verification of suspected fencing-env bugs. Nothing in the env is modified;
# all interventions happen inside this probe process (instance-level wrappers, obs masking,
# or a temporary asset passed via robot.asset.assetFileName).
#   env.verify_mode=drills    -> T1 wipe unit test, T5 obs-invariance, T6 no-auto-reset,
#                                 T2 lag (pos+force), T3 geometry validation + air hits,
#                                 T4 fall termination A/B/C, T7 cane (load share + conditions A,C)
#   env.verify_mode=noblade   -> T7 condition B (run with the blade-removed temp asset)
#   env.verify_mode=strategy  -> T1 in the strategy env + wipe impact via a shadow counter
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
from phc.env.tasks.humanoid_fencing import compute_fencing_observation
from phc.env.tasks.humanoid_fencing_drills import (NUM_DRILLS, DRILL_NAMES, D_STAND, D_RETREAT,
                                                    D_STEP_LEFT, D_DODGE, D_LUNGE_UPPER, D_LUNGE_GROIN)
from phc.audit_env import parse_target_geoms, qrot
from isaacgym.torch_utils import quat_conjugate

XML = 'phc/data/assets/mjcf/smpl_humanoid_neutral_fencing.xml'
def P(*a): print(*a, flush=True)

def per_target_dist(Pts, bpos, brot, geoms):
    """Pts (N,S,3) -> (N,K) min distance from the samples to each target's TRUE geometry surface."""
    N, S, _ = Pts.shape; dev = Pts.device; out = []
    for k, (typ, a, b, r) in enumerate(geoms):
        pos, q = bpos[:, k], brot[:, k]
        if typ == 'cap':
            A = pos + qrot(q, torch.tensor(a, device=dev).expand(N, 3)); B = pos + qrot(q, torch.tensor(b, device=dev).expand(N, 3))
            ab = (B - A)[:, None]; ap = Pts - A[:, None]
            t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)).clamp(0, 1)
            d = (Pts - (A[:, None] + t[..., None] * ab)).norm(dim=-1) - r
        else:
            C = pos + qrot(q, torch.tensor(a, device=dev).expand(N, 3))
            qi = quat_conjugate(q)[:, None].expand(N, S, 4).reshape(-1, 4)
            loc = qrot(qi, (Pts - C[:, None]).reshape(-1, 3)).reshape(N, S, 3)
            d = (loc.abs() - torch.tensor(b, device=dev)).clamp_min(0).norm(dim=-1)
        out.append(d.clamp_min(0).min(1).values)
    return torch.stack(out, -1)

def step(task, low, learner_z=None, zero_col=None):
    N = task.num_envs; obs = task.obs_buf; o0 = obs[:N]
    if zero_col is not None: o0 = o0.clone(); o0[:, zero_col] = 0
    z0 = low(o0, task.drill_ids) if learner_z is None else learner_z
    task.step(torch.cat([z0, low(obs[N:2 * N], task.opp_drill_ids)], 0))

def reset_done(task):
    d = task.reset_buf.nonzero().flatten()
    if len(d): task.reset(d)
    return d

def onehot(task, d):
    p = torch.zeros(NUM_DRILLS, device=task.device); p[d] = 1.0; return p

# ---------------------------------------------------------------- tests
def t1_wipe(task, tag):
    h = task.sword_hit_history; h[:] = 1.5
    task.reset(torch.tensor([0], device=task.device, dtype=torch.long))
    o = h[1:]
    P("[T1 %s] set hit-history=1.5 for all %d envs, reset ONLY env 0 -> other envs now: min=%.2f max=%.2f  (%.1f%% wiped to 0)"
      % (tag, task.num_envs, o.min(), o.max(), 100 * (o == 0).all(-1).float().mean()))

def t5_obs(task):
    sw = task.get_sword_tip_pos(); cf = task._contact_forces_list
    args = [task._humanoid_root_states_list[0], task._rigid_body_pos_list[0], task._humanoid_root_states_list[1],
            task._rigid_body_pos_list[1], task._rigid_body_rot_list[1], task._dof_pos_list[1], task._dof_vel_list[1],
            cf[0].norm(dim=-1), cf[1].norm(dim=-1), task.bounding_box_points, sw[0], task._target_ids]
    o1 = compute_fencing_observation(*args)
    a2 = list(args)
    q = torch.randn_like(args[4]); a2[4] = q / q.norm(dim=-1, keepdim=True)
    a2[5] = torch.randn_like(args[5]) * 3; a2[6] = torch.randn_like(args[6]) * 10
    o2 = compute_fencing_observation(*a2)
    a3 = list(args); a3[3] = args[3] + 0.1   # control: perturb a USED input
    o3 = compute_fencing_observation(*a3)
    P("[T5] task-obs dim=%d. Randomizing opponent body ROTATIONS + DOF pos + DOF vel changes obs by max |d|=%.3g ; "
      "control (shift opponent body POSITIONS 10cm) changes it by max |d|=%.3g" % (o1.shape[-1], (o1 - o2).abs().max(), (o1 - o3).abs().max()))

def t6_noreset(task, low, steps=250):
    task.drill_probs = onehot(task, D_LUNGE_UPPER); task.reset()
    L = task.max_episode_length; flagged_step = None
    for t in range(steps):
        step(task, low)
        if flagged_step is None and task.reset_buf.any(): flagged_step = t
    pb = task.progress_buf
    P("[T6] step_z loop with NO manual reset: reset_buf first raised at step %s; after %d steps progress_buf min=%d max=%d "
      "(max_episode_length=%d) -> %.0f%% of envs still running past their episode end"
      % (flagged_step, steps, pb.min(), pb.max(), L, 100 * (task.reset_buf.bool()).float().mean()))

def t2_lag(task, low, steps=60):
    probs = torch.zeros(NUM_DRILLS, device=task.device); probs[[D_LUNGE_UPPER, D_LUNGE_GROIN, D_DODGE]] = 1
    task.drill_probs = probs / probs.sum(); task.reset()
    ids = torch.cat([task._target_ids, task._hand_ids])
    orig = task.check_sword_hit
    def wrapped():
        task._seen = (task.get_sword_tip_pos()[0][:, 0].clone(), task._contact_forces_list[1][:, ids].clone())
        return orig()
    task.check_sword_hit = wrapped
    prev = None; prev_reset = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    dp_prev, dp_now, df_prev, df_now = [], [], [], []
    for t in range(steps):
        step(task, low)
        fresh = (task.get_sword_tip_pos()[0][:, 0].clone(), task._contact_forces_list[1][:, ids].clone())
        if prev is not None:
            ok = ~prev_reset
            dp_prev.append((task._seen[0][ok] - prev[0][ok]).norm(dim=-1).max()); dp_now.append((task._seen[0][ok] - fresh[0][ok]).norm(dim=-1).mean())
            df_prev.append((task._seen[1][ok] - prev[1][ok]).abs().max()); df_now.append((task._seen[1][ok] - fresh[1][ok]).abs().mean())
        prev = fresh; prev_reset = torch.zeros_like(prev_reset); prev_reset[reset_done(task)] = True
    task.check_sword_hit = orig
    P("[T2] hit detector's inputs vs the simulator state:")
    P("     sword tip:      max |seen - previous step| = %.2e m   mean |seen - current step| = %.4f m" % (max(dp_prev), torch.stack(dp_now).mean()))
    P("     contact forces: max |seen - previous step| = %.2e N   mean |seen - current step| = %.3f N" % (max(df_prev), torch.stack(df_now).mean()))

def t3_geom(task, low, steps=250):
    geoms = parse_target_geoms(XML); tgt = task._target_ids; hid = task._hand_ids[0]; dev = task.device
    task.drill_probs = onehot(task, D_STAND); task.reset(); step(task, low)
    bp, br = task._rigid_body_pos_list[0][:, tgt], task._rigid_body_rot_list[0][:, tgt]
    selfd = torch.stack([per_target_dist(bp[:, k:k + 1], bp, br, geoms)[:, k] for k in range(6)], -1)
    P("[T3a] reconstruction sanity: each target joint-origin's distance to its OWN reconstructed shape (0 = inside): " +
      ", ".join("%s=%.3f" % (n, v) for n, v in zip(["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"], selfd.median(0).values.tolist())))
    probs = torch.zeros(NUM_DRILLS, device=dev); probs[[D_LUNGE_UPPER, D_LUNGE_GROIN, D_DODGE]] = 1
    task.drill_probs = probs / probs.sum(); task.reset()
    orig = task.check_sword_hit
    def wrapped():
        s = dict(tips=[p[:, 0].clone() for p in task.get_sword_tip_pos()], bpos=[b.clone() for b in task._rigid_body_pos_list],
                 brot=[b.clone() for b in task._rigid_body_rot_list], cf=[c.clone() for c in task._contact_forces_list],
                 prev=None if task._swept_prev_tip_list is None else [p[:, 0].clone() for p in task._swept_prev_tip_list],
                 pp=None if task._swept_prev_progress is None else task._swept_prev_progress.clone(), pr=task.progress_buf.clone())
        out = orig(); s['hit'] = [o[:, 0].clone() for o in out]; task._snap = s; return out
    task.check_sword_hit = wrapped
    val, air, ex = [], [], []
    w = torch.linspace(0, 1, 21, device=dev)
    for t in range(steps):
        step(task, low)
        # (b) PhysX ground truth: physical sword-on-target contact (both sides loaded, blade clear of the floor)
        for i in range(2):
            j = 1 - i; hp = task._rigid_body_pos_list[i][:, hid]; tp = task.get_sword_tip_pos()[i][:, 0]
            tf = task._contact_forces_list[j][:, tgt].norm(dim=-1); hf = task._contact_forces_list[i][:, hid].norm(dim=-1)
            m = (tf.max(-1).values > 30) & (hf > 30) & (hp[:, 2] > 0.3) & (tp[:, 2] > 0.2)
            if m.any():
                blade = hp[:, None] + w[None, :, None] * (tp - hp)[:, None]
                d = per_target_dist(blade, task._rigid_body_pos_list[j][:, tgt], task._rigid_body_rot_list[j][:, tgt], geoms)
                k = tf.argmax(-1)
                val.append(d.gather(1, k[:, None])[:, 0][m])
        # (c) air hits in the swept branch, using exactly what the detector saw
        s = task._snap
        for i in range(2):
            j = 1 - i; tip = s['tips'][i]
            a = tip if s['prev'] is None else torch.where((s['pr'] != s['pp'] + 1)[:, None], tip, s['prev'][i])
            td = (tip[:, None] - s['bpos'][j][:, tgt]).norm(dim=-1); force = s['cf'][j][:, tgt].norm(dim=-1)
            inst = ((td < task.hit_dist) & (force > 50)).any(-1); so = s['hit'][i] & ~inst
            if so.any():
                path = a[:, None] + w[None, :, None] * (tip - a)[:, None]
                d = per_target_dist(path, s['bpos'][j][:, tgt], s['brot'][j][:, tgt], geoms).min(-1).values
                ab = (tip - a)[:, None]; ap = s['bpos'][j][:, tgt] - a[:, None]
                tt = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)).clamp(0, 1)
                od = (s['bpos'][j][:, tgt] - (a[:, None] + tt[..., None] * ab)).norm(dim=-1).min(-1).values
                air.append(d[so])
                for e in so.nonzero().flatten().tolist()[:1]:
                    if len(ex) < 4 and d[e] > 0.03: ex.append((float(od[e]), float(d[e]), float((tip[e] - a[e]).norm())))
        reset_done(task)
    task.check_sword_hit = orig
    v = torch.cat(val).cpu().numpy(); a = torch.cat(air).cpu().numpy()
    P("[T3b] PhysX ground truth: in %d frames where the sword and a target were physically loaded (>30 N both), blade distance "
      "to that target's reconstructed shape: median=%.3f m, within 3 cm=%.1f%%" % (v.size, np.median(v), 100 * (v <= 0.03).mean()))
    P("[T3c] swept-only hits: n=%d, tip path >3 cm from every real target shape: %.1f%%" % (a.size, 100 * (a > 0.03).mean()))
    for od, d, disp in ex:
        P("      example air hit: path came %.3f m from a joint origin (<0.15 scores) but stayed %.3f m from the body surface; tip moved %.2f m that step" % (od, d, disp))

def t4_falls(task, low, steps=150, zscale=4.0):
    task.drill_probs = onehot(task, D_STAND)
    cids = task._contact_body_ids; th = task._termination_heights
    zdim = low(task.obs_buf[:task.num_envs], task.drill_ids).shape[-1]
    for name, sc, test in [("pre-warmup (first 7.5k steps)", 0, False), ("post-warmup (rest of training)", task.warmup_time + 10, False), ("eval/recording (flags.test)", 0, True)]:
        task.step_counter = sc; flags.test = test; torch.manual_seed(0); task.reset()
        ff = tf = 0; run = torch.zeros(task.num_envs, device=task.device); longest = 0
        for t in range(steps):
            step(task, low, learner_z=torch.randn(task.num_envs, zdim, device=task.device) * zscale)
            z = task._rigid_body_pos_list[0][..., 2].clone(); z[:, cids] = 1e3
            fallen = (z < th).any(-1) & (task.progress_buf > 1); rb = task.reset_buf.bool()
            ff += int(fallen.sum()); tf += int((fallen & rb).sum())
            run = torch.where(fallen & ~rb, run + 1, torch.zeros_like(run)); longest = max(longest, int(run.max()))
            reset_done(task)
        P("[T4] %-32s fallen frames=%5d  episode ended on the fall frame=%5.1f%%  longest time lying down without termination=%d steps"
          % (name, ff, 100 * tf / max(ff, 1), longest))
    flags.test = False; task.step_counter = task.warmup_time + 10

def find_hand_obs_col(task, low):
    task.drill_probs = onehot(task, D_STAND); task.reset()
    for _ in range(5): step(task, low)
    N = task.num_envs; hid = task._hand_ids[0]
    target = task._contact_forces_list[0][:, hid].norm(dim=-1)
    diff = (task.obs_buf[:N] - target[:, None]).abs().max(0).values
    cols = (diff < 1e-3).nonzero().flatten().tolist()
    P("[T7] obs column(s) equal to the learner's own sword-hand contact force: %s (hand force spread across envs: %.1f..%.1f N)" % (cols, target.min(), target.max()))
    return cols[0] if len(cols) == 1 else None

def t7_cane(task, low, label, zero_col=None, steps=150):
    hid = int(task._hand_ids[0]); head = int(task._build_key_body_ids_tensor(["Head"])[0])
    feet = task._build_key_body_ids_tensor(["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]); cids = task._contact_body_ids
    for d in [D_STAND, D_RETREAT, D_STEP_LEFT, D_DODGE]:
        task.drill_probs = onehot(task, d); torch.manual_seed(1); task.reset()
        n = floor = fallen = 0; fz_s, share, headz, spd, dirz = [], [], [], [], []
        prev_root = task._humanoid_root_states_list[0][:, :2].clone()
        for t in range(steps):
            step(task, low, zero_col=zero_col)
            cf = task._contact_forces_list[0]; tip = task.get_sword_tip_pos()[0][:, 0]
            on = tip[:, 2] < 0.05; F = cf[:, hid]
            tot = cf[..., 2].sum(-1)
            if on.any():
                fz_s.append(F[on, 2]); share.append((F[on, 2] / tot[on].clamp_min(1)))
                dirz.append(F[on, 2] / F[on].norm(dim=-1).clamp_min(1e-6))
            z = task._rigid_body_pos_list[0][..., 2].clone(); z[:, cids] = 1e3
            fallen += int((z < task._termination_heights).any(-1).sum()); floor += int(on.sum()); n += task.num_envs
            headz.append(task._rigid_body_pos_list[0][:, head, 2].clone())
            r = task._humanoid_root_states_list[0][:, :2]; spd.append(((r - prev_root).norm(dim=-1) / task.dt).clone())
            reset_done(task); prev_root = task._humanoid_root_states_list[0][:, :2].clone()
        c = lambda L: torch.cat(L).cpu().numpy() if L else np.array([np.nan])
        hz = c(headz); fz = c(fz_s); sh = c(share); dz = c(dirz); sp = c(spd)
        P("[T7 %s] %-10s tip on floor=%5.1f%% | when on floor: sword vertical support median=%5.1f N (%.1f%% of all vertical ground force), force direction z-share=%.2f | "
          "fallen frames=%.2f%%  head height mean=%.3f m, head<1.0m=%.1f%%  root speed mean=%.2f m/s"
          % (label, DRILL_NAMES[d], 100 * floor / n, np.nanmedian(fz), 100 * np.nanmedian(sh), np.nanmedian(dz),
             100 * fallen / n, hz.mean(), 100 * (hz < 1.0).mean(), np.nanmean(sp)))

def masses(task):
    env0 = task.envs[0]; h = task.humanoid_handles_list[0][0]
    nm = task.gym.get_actor_rigid_body_names(env0, h); pr = task.gym.get_actor_rigid_body_properties(env0, h)
    m = {a: p.mass for a, p in zip(nm, pr)}
    P("[mass] from the simulator: R_Hand(+sword)=%.5f kg  L_Hand=%.3f kg  whole body=%.2f kg  (weight=%.0f N)" % (m['R_Hand'], m['L_Hand'], sum(m.values()), 9.81 * sum(m.values())))


def t3_geom_v2(task, low, steps=250):
    geoms = parse_target_geoms(XML); tgt = task._target_ids; hid = task._hand_ids[0]; dev = task.device
    probs = torch.zeros(NUM_DRILLS, device=dev); probs[[D_LUNGE_UPPER, D_LUNGE_GROIN, D_DODGE]] = 1
    task.drill_probs = probs / probs.sum(); task.reset()
    # physical blade axis from the mesh, in the R_Hand frame (x~0, z~0.01, y from +0.109 to -0.961); half-thickness ~7 mm
    ys = torch.linspace(0.109, -0.961, 41, device=dev)
    axis_local = torch.stack([torch.zeros_like(ys), ys, torch.full_like(ys, 0.01)], -1)          # (41,3)
    w = torch.linspace(0, 1, 21, device=dev)
    orig = task.check_sword_hit
    def wrapped():
        s = dict(tips=[p[:, 0].clone() for p in task.get_sword_tip_pos()], bpos=[b.clone() for b in task._rigid_body_pos_list],
                 brot=[b.clone() for b in task._rigid_body_rot_list], cf=[c.clone() for c in task._contact_forces_list],
                 prev=None if task._swept_prev_tip_list is None else [p[:, 0].clone() for p in task._swept_prev_tip_list],
                 pp=None if task._swept_prev_progress is None else task._swept_prev_progress.clone(), pr=task.progress_buf.clone())
        out = orig(); s['hit'] = [o[:, 0].clone() for o in out]; task._snap = s; return out
    task.check_sword_hit = wrapped
    phys, virt, air = [], [], []
    for t in range(steps):
        step(task, low)
        for i in range(2):
            j = 1 - i; N = task.num_envs
            hp = task._rigid_body_pos_list[i][:, hid]; hr = task._rigid_body_rot_list[i][:, hid]
            ax = hp[:, None] + qrot(hr[:, None].expand(N, 41, 4).reshape(-1, 4), axis_local[None].expand(N, 41, 3).reshape(-1, 3)).reshape(N, 41, 3)
            tp = task.get_sword_tip_pos()[i][:, 0]
            tf = task._contact_forces_list[j][:, tgt].norm(dim=-1); hf = task._contact_forces_list[i][:, hid].norm(dim=-1)
            k = tf.argmax(-1); tk = tf.gather(1, k[:, None])[:, 0]
            pair = (tk > 30) & (hf > 30) & ((tk - hf).abs() <= 0.25 * torch.maximum(tk, hf)) & (ax[..., 2].min(-1).values > 0.1)
            if pair.any():
                dP = per_target_dist(ax, task._rigid_body_pos_list[j][:, tgt], task._rigid_body_rot_list[j][:, tgt], geoms).gather(1, k[:, None])[:, 0] - 0.007
                vl = hp[:, None] + w[None, :, None] * (tp - hp)[:, None]
                dV = per_target_dist(vl, task._rigid_body_pos_list[j][:, tgt], task._rigid_body_rot_list[j][:, tgt], geoms).gather(1, k[:, None])[:, 0]
                phys.append(dP[pair]); virt.append(dV[pair])
        s = task._snap
        for i in range(2):
            j = 1 - i; tip = s['tips'][i]
            a = tip if s['prev'] is None else torch.where((s['pr'] != s['pp'] + 1)[:, None], tip, s['prev'][i])
            td = (tip[:, None] - s['bpos'][j][:, tgt]).norm(dim=-1); force = s['cf'][j][:, tgt].norm(dim=-1)
            so = s['hit'][i] & ~((td < task.hit_dist) & (force > 50)).any(-1)
            if so.any():
                path = a[:, None] + w[None, :, None] * (tip - a)[:, None]
                air.append(per_target_dist(path, s['bpos'][j][:, tgt], s['brot'][j][:, tgt], geoms).min(-1).values[so])
        reset_done(task)
    task.check_sword_hit = orig
    p = torch.cat(phys).cpu().numpy(); v = torch.cat(virt).cpu().numpy(); a = torch.cat(air).cpu().numpy()
    P("[T3b'] Newton-matched sword<->target contacts (|F_sword| ~= |F_target| >30 N, blade off floor): n=%d" % p.size)
    P("       PHYSICAL blade surface -> reconstructed target surface: median=%.3f m  p90=%.3f m  within 1cm=%.1f%%  within 2cm=%.1f%%" %
      (np.median(p), np.percentile(p, 90), 100 * (np.abs(p) <= 0.01).mean(), 100 * (np.abs(p) <= 0.02).mean()))
    P("       (virtual hand->tip line, for comparison: median=%.3f m)" % np.median(v))
    P("[T3c'] swept-only hits n=%d; tip path distance to the real body: p50=%.3f p75=%.3f p90=%.3f m | >3cm=%.1f%%  >5cm=%.1f%%  >10cm=%.1f%%" %
      (a.size, np.percentile(a, 50), np.percentile(a, 75), np.percentile(a, 90), 100 * (a > .03).mean(), 100 * (a > .05).mean(), 100 * (a > .10).mean()))

# ---------------------------------------------------------------- strategy
def strategy(task, low, macro_steps=120):
    task.low_level_policy = low
    t1_wipe(task, "strategy env")
    task.reset()
    N = task.num_envs; dev = task.device; W = float(task.win_frame_hit)
    shadow = torch.zeros(N, 2, device=dev); s_win = torch.zeros(N, dtype=torch.bool, device=dev); r_win = torch.zeros_like(s_win)
    bouts = dict(n=0, real=0, shadow=0, swallowed=0); steps_with_reset = [0, 0]
    ocgs = task.check_game_state
    def cgs():
        ob, rw, gw = ocgs()
        shadow.add_(torch.cat([task.sword_hit_list[0], task.sword_hit_list[1]], -1).float())
        s_win.logical_or_((shadow > W).any(-1)); r_win.logical_or_(rw.bool() | gw.bool())
        return ob, rw, gw
    task.check_game_state = cgs
    orst = task.reset
    def rst(env_ids=None):
        if env_ids is not None and len(env_ids) < N:
            steps_with_reset[0] += 1
            bouts['n'] += len(env_ids); bouts['real'] += int(r_win[env_ids].sum()); bouts['shadow'] += int(s_win[env_ids].sum())
            bouts['swallowed'] += int((s_win[env_ids] & ~r_win[env_ids]).sum())
            shadow[env_ids] = 0; s_win[env_ids] = False; r_win[env_ids] = False
        return orst(env_ids)
    task.reset = rst
    for _ in range(macro_steps):
        ld = torch.randint(0, NUM_DRILLS, (N,), device=dev); od = torch.randint(0, NUM_DRILLS, (N,), device=dev)
        task.macro_step(ld, od)
    total_steps = macro_steps * task.macro_K
    P("[T1 impact, strategy env, %d envs, win = hit-frames > %.0f, random drills] physics steps=%d, steps where some env reset=%d (%.0f%%)"
      % (N, W, total_steps, steps_with_reset[0], 100 * steps_with_reset[0] / total_steps))
    P("     finished bouts=%d | bouts with a win by the per-env ('correct') counter=%d | wins the env actually registered=%d | "
      "wins swallowed by the global wipe=%d" % (bouts['n'], bouts['shadow'], bouts['real'], bouts['swallowed']))

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
    mode = cfg.env.get("verify_mode", "drills")
    task.step_counter = task.warmup_time + 10     # the regime training spends >99% of its time in
    with torch.no_grad():
        masses(task)
        if mode == "drills":
            t1_wipe(task, "drills env"); t5_obs(task); t6_noreset(task, low); t2_lag(task, low); t3_geom(task, low)
            t4_falls(task, low)
            col = find_hand_obs_col(task, low)
            t7_cane(task, low, "A normal       ")
            if col is not None: t7_cane(task, low, "C hand-obs=0   ", zero_col=col)
        elif mode == "drills2":
            t2_lag(task, low); t3_geom_v2(task, low)
        elif mode == "noblade":
            col = cfg.env.get("hand_obs_col", None)
            P("[T7] using hand-contact obs column from the normal-asset run: %s" % col)
            t7_cane(task, low, "B no blade     ", zero_col=col)
        elif mode == "strategy":
            strategy(task, low)

if __name__ == '__main__':
    main()
