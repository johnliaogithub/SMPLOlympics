# Headless measurement: how fast does the sword TIP actually move during a real
# lunge, and does its per-control-step displacement put it in the tunneling regime?
# Drives the frozen low-level with drill=LUNGE_UPPER (learner) vs STAND (frozen opp),
# logs per 30Hz control step: tip speed, per-step displacement, the instantaneous
# tip->target distance the detector uses, whether a hit registered, and whether a
# SWEPT path (prev_tip->cur_tip) came within hit_dist while the instant check missed.
import os, sys
os.environ['OMP_NUM_THREADS'] = "1"
sys.path.append(os.getcwd()); sys.path.append('./SMPLSim')
from isaacgym import gymapi  # noqa
import torch, numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, D_STAND, D_LUNGE_UPPER

def seg_min_dist(a, b, pts):
    # a,b: (N,3) segment endpoints; pts: (N,k,3). return min over k of point-to-seg dist -> (N,)
    ab = (b - a).unsqueeze(1)                      # (N,1,3)
    ap = pts - a.unsqueeze(1)                      # (N,k,3)
    t = (ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)
    t = t.clamp(0, 1).unsqueeze(-1)
    proj = a.unsqueeze(1) + t * ab                 # (N,k,3)
    return torch.linalg.norm(pts - proj, dim=-1).min(dim=1).values

@torch.no_grad()
def run(cfg, task, low_level, n_steps):
    N = task.num_envs
    dt = float(task.dt)
    hit_dist = float(task.hit_dist)
    probs = torch.zeros(NUM_DRILLS, device=task.device); probs[D_LUNGE_UPPER] = 1.0
    task.drill_probs = probs
    task.reset()
    prev_tip = None; prev_prog = None
    SP, DISP, INST, SWEPT, HIT, TUN, PASS, COMMIT = [], [], [], [], [], [], [], []
    for step in range(n_steps):
        task.drill_ids[:] = D_LUNGE_UPPER
        task.opp_drill_ids[:] = D_STAND
        task.opp_frozen[:] = True
        obs = task.obs_buf
        z0 = low_level(obs[:N], task.drill_ids)
        z1 = low_level(obs[N:2*N], task.opp_drill_ids)
        task.step(torch.cat([z0, z1], dim=0))

        tip = task.get_sword_tip_pos()[0][:, 0, :]                       # (N,3) learner tip
        hand = task._rigid_body_pos_list[0][:, task._hand_ids[0], :]     # (N,3) learner hand
        targets = task._rigid_body_pos_list[1][:, task._target_ids]     # (N,k,3) opp targets
        inst = torch.linalg.norm(tip.unsqueeze(1) - targets, dim=-1).min(dim=1).values  # (N,)
        hit = task.sword_hit_list[0][:, 0]                              # (N,) env's own per-step hit (swept-aware)
        prog = task.progress_buf.clone()
        if prev_tip is not None:
            valid = (prog == prev_prog + 1)                             # no reset between frames
            disp = torch.linalg.norm(tip - prev_tip, dim=-1)
            speed = disp / dt
            swept = seg_min_dist(prev_tip, tip, targets)
            # blade axis and tip-motion alignment (a committed thrust moves tip-first)
            v = tip - prev_tip                                         # (N,3)
            blade = tip - hand                                        # (N,3)
            align = (v*blade).sum(-1) / (v.norm(dim=-1).clamp_min(1e-9)*blade.norm(dim=-1).clamp_min(1e-9))
            passthru = valid & (swept < hit_dist) & (inst >= hit_dist)          # tip path crossed sphere, both samples outside
            commit  = valid & (swept < hit_dist) & (align > 0.5) & (disp > 0.05)  # committed thrust that reached a target
            tun = passthru & (~hit)
            m = valid
            SP.append(speed[m].cpu()); DISP.append(disp[m].cpu()); INST.append(inst[m].cpu())
            SWEPT.append(swept[m].cpu()); HIT.append(hit[m].cpu()); TUN.append(tun[m].cpu())
            PASS.append(passthru[m].cpu()); COMMIT.append(commit[m].cpu())
        prev_tip = tip.clone(); prev_prog = prog
    cat = lambda L: torch.cat(L).numpy()
    sp, disp, inst, swept, hit, tun, passthru, commit = map(cat, (SP, DISP, INST, SWEPT, HIT, TUN, PASS, COMMIT))
    p = lambda a, q: float(np.percentile(a, q))
    print("\n================ MEASURED TIP KINEMATICS (real LUNGE_UPPER drill) ================")
    print(f"envs={N}  steps={n_steps}  dt={dt*1000:.2f} ms  hit_dist={hit_dist} m  samples={sp.size}")
    print(f"tunneling threshold: per-step disp > 2*hit_dist = {2*hit_dist*100:.0f} cm  (tip speed > {2*hit_dist/dt:.2f} m/s)")
    print(f"\nTIP SPEED (m/s):    mean={sp.mean():.2f}  p50={p(sp,50):.2f}  p90={p(sp,90):.2f}  p99={p(sp,99):.2f}  MAX={sp.max():.2f}")
    print(f"PER-STEP DISP (cm): mean={disp.mean()*100:.1f}  p90={p(disp,90)*100:.1f}  p99={p(disp,99)*100:.1f}  MAX={disp.max()*100:.1f}")
    thr = 2*hit_dist
    print(f"\nframes with disp > 2*hit_dist ({thr*100:.0f} cm): {(disp>thr).mean()*100:.2f}%  "
          f"(these can skip a dead-center capture sphere in one step)")
    print(f"frames with disp > hit_dist ({hit_dist*100:.0f} cm):  {(disp>hit_dist).mean()*100:.2f}%")
    print(f"\nEMPIRICAL TUNNEL FRAMES (swept path within {hit_dist*100:.0f}cm of a target, "
          f"but sampled tip outside AND no hit): {tun.sum()} frames ({tun.mean()*100:.3f}% of all)")
    if tun.sum() > 0:
        print(f"   among tunnel frames: tip speed mean={sp[tun].mean():.2f}  max={sp[tun].max():.2f} m/s;"
              f"  swept-approach mean={swept[tun].mean()*100:.1f}cm  (< {hit_dist*100:.0f}cm = a real pass-through)")
    reg = hit
    if reg.sum() > 0:
        print(f"\nfor comparison, REGISTERED-hit frames ({reg.sum()}): tip speed mean={sp[reg].mean():.2f}  "
              f"p90={p(sp[reg],90):.2f}  max={sp[reg].max():.2f} m/s  <- hits happen at LOW tip speed")
    import numpy as _np
    def rate(mask): 
        return (hit[mask].mean()*100 if mask.sum() else float("nan")), int(mask.sum())
    pr, pn = rate(passthru); cr, cn = rate(commit)
    print(f"\nCATCH RATE (fraction that REGISTER a hit):")
    print(f"  pass-throughs (tip path crossed a target sphere, both samples outside): {pr:5.1f}%  of {pn} frames")
    print(f"  committed thrusts that reached a target (aligned, moving):              {cr:5.1f}%  of {cn} frames")
    print("="*82)

@hydra.main(version_base=None, config_path="../phc/data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    set_np_formatting()
    flags.test = True
    flags.debug = flags.follow = flags.server_mode = False
    flags.im_eval = flags.no_virtual_display = flags.render_o3d = False
    flags.fixed = flags.divide_group = flags.no_collision_check = flags.fixed_path = False
    flags.real_path = flags.show_traj = flags.slow = flags.real_traj = flags.trigger_input = False
    flags.add_proj = cfg.get("add_proj", False); flags.has_eval = cfg.get("has_eval", False)
    set_seed(cfg.get("seed", 0), False)
    cfg.learning['params']['config']["num_actors"] = cfg.env.num_envs
    sim_params = parse_sim_params(cfg)
    args = EasyDict({"task": cfg.env.task, "device_id": cfg.device_id, "rl_device": cfg.rl_device,
                     "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
                     "headless": cfg.headless, "device": cfg.device})
    task, env = parse_task(args, cfg, cfg_train=cfg.learning, sim_params=sim_params)
    low_level = FrozenLowLevelPolicy(cfg.env["low_level_checkpoint"], NUM_DRILLS, device=task.device)
    run(cfg, task, low_level, n_steps=cfg.env.get("measure_steps", 120))

if __name__ == '__main__':
    main()
