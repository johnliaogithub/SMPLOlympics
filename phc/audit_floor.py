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
    dev=task.device; N=task.num_envs; hid=int(task._build_key_body_ids_tensor(["R_Hand"])[0])
    task.drill_probs=torch.tensor([0.4,0.4,0.4,0.6,0.6,2.0,0.4,0.4],device=dev); task.drill_probs/=task.drill_probs.sum()
    task.step_counter=task.warmup_time+10; task.reset()
    R={d:{'n':0,'floor':0,'F':[]} for d in range(NUM_DRILLS)}
    for t in range(T):
        obs=task.obs_buf; dr=task.drill_ids.clone()
        task.step(torch.cat([low_level(obs[:N],task.drill_ids),low_level(obs[N:2*N],task.opp_drill_ids)],0))
        tip=task.get_sword_tip_pos()[0][:,0]; F=task._contact_forces_list[0][:,hid].norm(dim=-1)
        fl=tip[:,2]<0.05
        for d in range(NUM_DRILLS):
            m=dr==d
            R[d]['n']+=int(m.sum()); R[d]['floor']+=int((m&fl).sum()); R[d]['F'].append(F[m&fl])
        dn=task.reset_buf.nonzero().flatten()
        if len(dn): task.reset(dn)
    print("\n[SWORD-ON-FLOOR] learner, phase-C drill mix, per drill:")
    for d in range(NUM_DRILLS):
        f=torch.cat(R[d]['F']).cpu().numpy()
        print("   %-12s frames=%6d  tip on floor(<5cm)=%5.1f%%  force into floor when there: median=%5.1f N"%(DRILL_NAMES[d],R[d]['n'],100*R[d]['floor']/max(R[d]['n'],1),np.median(f) if f.size else float('nan')))

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
    run(cfg, task, low_level, T=200)

if __name__ == '__main__':
    main()
