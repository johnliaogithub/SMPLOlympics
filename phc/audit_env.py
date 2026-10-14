# Runtime audit of the fencing drills env. Resets finished envs every step (like
# training), lets the env sample drills itself, and wraps check_sword_hit to see
# exactly what the hit detector saw. Probes:
#  P1 hit-check lag (stale tensors)      P2 sword_hit_history global wipe
#  P3 fall termination                   P4 hit force attribution (sword vs other)
#  P5 masses/inertia                     P6 spawn facing
#  P7 detector vs true geometry (air hits / missed real touches)
#  P8 own-blade self-collision           P9 termination causes (OOB etc.)
import os, sys, re
os.environ['OMP_NUM_THREADS']="1"; sys.path.append(os.getcwd()); sys.path.append('./SMPLSim')
from isaacgym import gymapi  # noqa
import torch, numpy as np, hydra
from isaacgym.torch_utils import quat_rotate, quat_conjugate
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from phc.utils import torch_utils
from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, DRILL_NAMES, D_LUNGE_UPPER, D_LUNGE_GROIN, D_DODGE

TARGETS = ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"]

def parse_target_geoms(xml):
    txt = open(xml).read(); out = []
    for b in TARGETS:
        seg = txt[txt.index('<body name="%s"' % b):][:1500]
        g = re.search(r'<geom [^>]*name="%s"[^>]*/>' % b, seg).group(0)
        if 'fromto' in g:
            ft = list(map(float, re.search(r'fromto="([^"]+)"', g).group(1).split()))
            r = float(re.search(r'size="([^"]+)"', g).group(1).split()[0])
            out.append(('cap', ft[:3], ft[3:], r))
        else:
            c = list(map(float, re.search(r' pos="([^"]+)"', g).group(1).split())) if ' pos=' in g else [0, 0, 0]
            h = list(map(float, re.search(r'size="([^"]+)"', g).group(1).split()))
            out.append(('box', c, h, None))
    return out

def qrot(q, v):  # q (M,4) xyzw, v (M,3)
    return quat_rotate(q, v)

def path_to_surface(P, bpos, brot, geoms, dev):
    """P (N,S,3) path samples; bpos/brot (N,K,3)/(N,K,4) target body frames.
    Returns min over samples & targets of distance to the TRUE target geometry surface (0 = inside)."""
    N, S, _ = P.shape; best = torch.full((N,), 1e9, device=dev)
    for k, (typ, a, b, r) in enumerate(geoms):
        pos, q = bpos[:, k], brot[:, k]
        if typ == 'cap':
            A = pos + qrot(q, torch.tensor(a, device=dev).expand(N, 3))
            B = pos + qrot(q, torch.tensor(b, device=dev).expand(N, 3))
            ab = (B - A)[:, None]; ap = P - A[:, None]
            t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)).clamp(0, 1)
            d = (P - (A[:, None] + t[..., None] * ab)).norm(dim=-1) - r
        else:
            C = pos + qrot(q, torch.tensor(a, device=dev).expand(N, 3))
            qi = quat_conjugate(q)[:, None].expand(N, S, 4).reshape(-1, 4)
            loc = qrot(qi, (P - C[:, None]).reshape(-1, 3)).reshape(N, S, 3)
            qq = loc.abs() - torch.tensor(b, device=dev)
            d = qq.clamp_min(0).norm(dim=-1)
        best = torch.minimum(best, d.clamp_min(0).min(dim=1).values)
    return best

@torch.no_grad()
def run(cfg, task, low_level, T):
    dev = task.device; N = task.num_envs
    geoms = parse_target_geoms('phc/data/assets/mjcf/smpl_humanoid_neutral_fencing.xml')
    hand_id = task._hand_ids[0]; tgt = task._target_ids
    contact_ids = task._contact_body_ids
    head_id = task._build_key_body_ids_tensor(["Head"])[0]

    # ---- P5 masses (static) ----
    env0 = task.envs[0]; h = task.humanoid_handles_list[0][0]
    names = task.gym.get_actor_rigid_body_names(env0, h)
    props = task.gym.get_actor_rigid_body_properties(env0, h)
    m = {n: p.mass for n, p in zip(names, props)}
    I_hand = props[names.index("R_Hand")].inertia
    print("\n[P5] masses (kg): R_Hand(sword)=%.5f  R_Wrist=%.3f  R_Elbow=%.3f  total body=%.2f" %
          (m["R_Hand"], m["R_Wrist"], m["R_Elbow"], sum(m.values())))
    print("     R_Hand inertia diag: %.2e %.2e %.2e" % (I_hand.x.x, I_hand.y.y, I_hand.z.z))

    # ---- wrap the hit check ----
    orig = task.check_sword_hit
    def patched():
        s = {'tips': [p.clone() for p in task.get_sword_tip_pos()],
             'bpos': [b.clone() for b in task._rigid_body_pos_list],
             'brot': [b.clone() for b in task._rigid_body_rot_list],
             'cf': [c.clone() for c in task._contact_forces_list],
             'prev': None if task._swept_prev_tip_list is None else [p.clone() for p in task._swept_prev_tip_list],
             'prevprog': None if task._swept_prev_progress is None else task._swept_prev_progress.clone(),
             'prog': task.progress_buf.clone()}
        out = orig(); s['hit'] = [o[:, 0].clone() for o in out]; task._probe = s
        return out
    task.check_sword_hit = patched

    probs = torch.zeros(NUM_DRILLS, device=dev); probs[D_LUNGE_UPPER] = probs[D_LUNGE_GROIN] = probs[D_DODGE] = 1.0
    task.drill_probs = probs / probs.sum()
    if cfg.env.get('audit_post_warmup', True):
        task.step_counter = task.warmup_time + 10   # match the regime training spends >99% of its time in
    print('[cfg] step_counter=%d warmup_time=%d flags.test=%s hit_swept=%s' % (task.step_counter, task.warmup_time, flags.test, task.hit_swept))
    task.reset()

    st = dict(lag_now=[], lag_prev=[], wipes=0, wipe_opps=0, fallen_steps=0, fallen_reset=0, steps=0,
              down_run=torch.zeros(N, 2, device=dev), down_runs=[], hits_inst=0, hits_swept=0,
              inst_handF=[], air_inst=[], air_swept=[], real_touch=0, real_touch_missed=0,
              selfcol=[], face_err=[], causes={})
    prev_fresh = None; prev_reset = torch.zeros(N, dtype=torch.bool, device=dev)
    for t in range(T):
        obs = task.obs_buf
        z0 = low_level(obs[:N], task.drill_ids); z1 = low_level(obs[N:2 * N], task.opp_drill_ids)
        drills_now = task.drill_ids.clone()
        task.step(torch.cat([z0, z1], 0))
        s = task._probe; fresh = [p.clone() for p in task.get_sword_tip_pos()]
        # P1 lag: what the detector saw vs fresh-this-step vs fresh-last-step
        if prev_fresh is not None:
            ok = ~prev_reset
            st['lag_now'].append((s['tips'][0][ok, 0] - fresh[0][ok, 0]).norm(dim=-1))
            st['lag_prev'].append((s['tips'][0][ok, 0] - prev_fresh[0][ok, 0]).norm(dim=-1))
        # P6 spawn facing (first step of an episode)
        first = task.progress_buf == 1
        if first.any():
            for i in range(2):
                rr = task._humanoid_root_states_list[i][first, 3:7]
                fwd = qrot(torch_utils.calc_heading_quat(rr), torch.tensor([1., 0, 0], device=dev).expand(len(rr), 3))[:, :2]
                to = (task._humanoid_root_states_list[1 - i][first, :2] - task._humanoid_root_states_list[i][first, :2])
                cos = (torch.nn.functional.normalize(fwd, dim=-1) * torch.nn.functional.normalize(to, dim=-1)).sum(-1)
                st['face_err'].append(torch.rad2deg(torch.acos(cos.clamp(-1, 1))))
        # P3 falls (fresh state)
        for i in range(2):
            z = task._rigid_body_pos_list[i][..., 2].clone(); z[:, contact_ids] = 1e3
            fallen = (z < 0.15).any(-1)
            down = task._rigid_body_pos_list[i][:, head_id, 2] < 0.5
            st['fallen_steps'] += int(fallen.sum()); st['fallen_reset'] += int((fallen & task.reset_buf.bool()).sum())
            st['down_run'][:, i] = torch.where(down, st['down_run'][:, i] + 1, torch.zeros_like(st['down_run'][:, i]))
        st['steps'] += N
        # P4/P7 detector vs geometry, per attacker
        for i in range(2):
            j = 1 - i; hit = s['hit'][i]
            tip = s['tips'][i][:, 0]
            if s['prev'] is not None:
                fr = s['prog'] != s['prevprog'] + 1
                a = torch.where(fr[:, None], tip, s['prev'][i][:, 0])
            else:
                a = tip
            w = torch.linspace(0, 1, 21, device=dev)
            P = a[:, None] + w[None, :, None] * (tip - a)[:, None]
            dsurf = path_to_surface(P, s['bpos'][j][:, tgt], s['brot'][j][:, tgt], geoms, dev)
            # instantaneous condition (as coded) to split inst vs swept-only hits
            td = (tip[:, None] - s['bpos'][j][:, tgt]).norm(dim=-1)
            force = s['cf'][j][:, tgt].norm(dim=-1)
            inst = ((td < task.hit_dist) & (force > 50)).any(-1)
            swept_only = hit & ~inst
            st['hits_inst'] += int((hit & inst).sum()); st['hits_swept'] += int(swept_only.sum())
            st['inst_handF'].append(s['cf'][i][hit & inst, hand_id].norm(dim=-1))
            st['air_inst'].append(dsurf[hit & inst]); st['air_swept'].append(dsurf[swept_only])
            real = dsurf <= 0.0
            st['real_touch'] += int(real.sum()); st['real_touch_missed'] += int((real & ~hit).sum())
            miss = real & ~hit
            if miss.any():
                disp = (tip - a).norm(dim=-1)
                blade = tip - s['bpos'][i][:, hand_id]
                align = ((tip - a) * blade).sum(-1) / (disp.clamp_min(1e-9) * blade.norm(dim=-1).clamp_min(1e-9))
                bd = st.setdefault('miss_bd', dict(n=0, slow=0, sideways=0, origin_far=0, low_force=0, far_and_slow=0))
                slow = disp <= task.swept_min_disp; side = (~slow) & (align <= task.swept_align_min)
                ofar = td.min(-1).values >= task.hit_dist; lowf = force.max(-1).values <= 50
                bd['n'] += int(miss.sum()); bd['slow'] += int((miss & slow).sum()); bd['sideways'] += int((miss & side).sum())
                bd['origin_far'] += int((miss & ofar).sum()); bd['low_force'] += int((miss & lowf).sum())
                bd['far_and_slow'] += int((miss & ofar & slow).sum())
        # P8 own-blade self-collision: learner far from opponent and hand above ground
        hp = task._rigid_body_pos_list[0][:, hand_id]; tp = fresh[0][:, 0]
        ab = (tp - hp)[:, None]; ap = task._rigid_body_pos_list[1] - hp[:, None]
        tt = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)).clamp(0, 1)
        blade_to_opp = (task._rigid_body_pos_list[1] - (hp[:, None] + tt[..., None] * ab)).norm(dim=-1).min(-1).values
        far = (blade_to_opp > 0.4) & (hp[:, 2] > 0.4) & (tp[:, 2] > 0.15)
        if far.any(): st['selfcol'].append(task._contact_forces_list[0][far, hand_id].norm(dim=-1))
        # P9 termination causes
        done = task.reset_buf.bool()
        if done.any():
            got = task.sword_hit_list[1][:, 0].bool(); lh = task.sword_hit_list[0][:, 0].bool()
            for e in done.nonzero().flatten().tolist():
                d = DRILL_NAMES[int(drills_now[e])]
                c = 'oob' if bool(task.out_bound[e]) else ('got_hit' if bool(got[e]) else ('landed' if bool(lh[e]) else 'timeout'))
                st['causes'].setdefault(d, {}).setdefault(c, 0); st['causes'][d][c] += 1
            # P2: wipe test — history before/after a partial reset
            hb = task.sword_hit_history.clone()
            ids = done.nonzero().flatten()
            for i in range(2): st['down_runs'] += st['down_run'][ids, i].tolist(); st['down_run'][ids, i] = 0
            task.reset(ids)
            others = ~done
            st['wipe_opps'] += int(((hb > 0).any(-1) & others).sum())
            st['wipes'] += int(((hb > 0).any(-1) & others & (task.sword_hit_history == 0).all(-1)).sum())
        prev_fresh = fresh; prev_reset = done.clone()

    cat = lambda L: torch.cat(L).cpu().numpy() if len(L) else np.array([])
    ln, lp = cat(st['lag_now']), cat(st['lag_prev'])
    print("\n[P1] tip seen by hit check vs fresh tip:  |seen-fresh(t)| mean=%.4f m   |seen-fresh(t-1)| mean=%.4f m" % (ln.mean(), lp.mean()))
    print("     => fraction of steps where detector used LAST step's tip (exact match <1mm): %.1f%%" % (100 * (lp < 1e-3).mean()))
    print("[P2] envs holding hit-history>0 that did NOT reset, at a moment some other env reset: %d ; wiped to 0: %d" % (st['wipe_opps'], st['wipes']))
    dr = np.array(st['down_runs'] + st['down_run'].flatten().tolist())
    print("[P3] env-steps with a non-foot body <0.15m (fall-termination condition): %d / %d ; of those, env reset that step: %d" % (st['fallen_steps'], st['steps'] * 2, st['fallen_reset']))
    print("     'down' (head<0.5m) run lengths: max=%d steps, runs>=15 steps: %d" % (dr.max() if dr.size else 0, int((dr >= 15).sum())))
    hf = cat(st['inst_handF'])
    print("[P4] instantaneous hits: %d ; attacker sword(R_Hand) contact force at hit: median=%.1fN, frac<1N (force not from sword)=%.1f%%" %
          (st['hits_inst'], np.median(hf) if hf.size else float('nan'), 100 * (hf < 1).mean() if hf.size else float('nan')))
    ai, asw = cat(st['air_inst']), cat(st['air_swept'])
    print("[P7] hits vs TRUE target geometry (tip path distance to body surface):")
    if ai.size: print("     inst hits:  n=%d  median=%.3f m  frac >3cm from body (air hit)=%.1f%%" % (ai.size, np.median(ai), 100 * (ai > 0.03).mean()))
    if asw.size: print("     swept-only: n=%d  median=%.3f m  frac >3cm from body (air hit)=%.1f%%" % (asw.size, np.median(asw), 100 * (asw > 0.03).mean()))
    print("     frames where tip path truly penetrated a target body: %d ; of those NOT registered as a hit: %d (%.1f%%)" %
          (st['real_touch'], st['real_touch_missed'], 100 * st['real_touch_missed'] / max(st['real_touch'], 1)))
    bd = st.get('miss_bd')
    if bd: print("     why true touches went unscored (non-exclusive, of %d): tip nearly still=%.0f%%  moving sideways(non-thrust)=%.0f%%  nearest joint-origin >= hit_dist=%.0f%%  target force<=50N=%.0f%%  (origin-far AND still=%.0f%%)" %
                 (bd['n'], 100*bd['slow']/bd['n'], 100*bd['sideways']/bd['n'], 100*bd['origin_far']/bd['n'], 100*bd['low_force']/bd['n'], 100*bd['far_and_slow']/bd['n']))
    sc = cat(st['selfcol'])
    if sc.size: print("[P8] learner sword(R_Hand) force while blade >0.4m from every opponent body & off the ground: n=%d  median=%.1fN  frac>5N=%.1f%%" % (sc.size, np.median(sc), 100 * (sc > 5).mean()))
    fe = cat(st['face_err'])
    print("[P6] spawn facing error vs direction-to-opponent: mean=%.1f deg  p95=%.1f deg  max=%.1f deg" % (fe.mean(), np.percentile(fe, 95), fe.max()))
    print("[P9] termination causes per learner drill:")
    for d, c in st['causes'].items(): print("     %-12s %s" % (d, c))

@hydra.main(version_base=None, config_path="../phc/data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True)); set_np_formatting()
    flags.test = cfg.env.get("audit_test_flag", False)
    for k in ['debug', 'follow', 'server_mode', 'im_eval', 'no_virtual_display', 'render_o3d', 'fixed', 'divide_group',
              'no_collision_check', 'fixed_path', 'real_path', 'show_traj', 'slow', 'real_traj', 'trigger_input']:
        setattr(flags, k, False)
    flags.add_proj = False; flags.has_eval = False
    set_seed(cfg.get("seed", 0), False)
    cfg.learning['params']['config']["num_actors"] = cfg.env.num_envs
    sim_params = parse_sim_params(cfg)
    args = EasyDict({"task": cfg.env.task, "device_id": cfg.device_id, "rl_device": cfg.rl_device,
                     "physics_engine": gymapi.SIM_PHYSX, "headless": cfg.headless, "device": cfg.device})
    task, env = parse_task(args, cfg, cfg_train=cfg.learning, sim_params=sim_params)
    low_level = FrozenLowLevelPolicy(cfg.env["low_level_checkpoint"], NUM_DRILLS, device=task.device)
    run(cfg, task, low_level, T=cfg.env.get("audit_steps", 300))

if __name__ == '__main__':
    main()
