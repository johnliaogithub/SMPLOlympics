# Test whether the dodge's avoid_r term is gameable / misleading.
# Drive learner=DODGE vs a live LUNGE across many envs; every control step log the
# EXACT avoid_r the reward uses, plus threat_dist, a proposed off-the-thrust-line
# metric, gap, progress, and got-hit. Then for each got-hit episode, analyze how
# avoid_r behaves in the steps leading into the hit.
import os, sys
os.environ['OMP_NUM_THREADS']="1"; sys.path.append(os.getcwd()); sys.path.append('./SMPLSim')
from isaacgym import gymapi  # noqa
import torch, numpy as np, hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, D_DODGE, D_LUNGE_UPPER, D_LUNGE_GROIN

@torch.no_grad()
def run(cfg, task, low_level, n_steps):
    N = task.num_envs
    # half the envs face an UPPER lunge, half a GROIN lunge
    opp = torch.full((N,), D_LUNGE_UPPER, device=task.device, dtype=torch.long)
    opp[N//2:] = D_LUNGE_GROIN
    task.reset()
    T = n_steps
    AV = np.zeros((T, N), np.float32)   # avoid_r
    TD = np.zeros((T, N), np.float32)   # threat_dist
    OL = np.zeros((T, N), np.float32)   # off-the-line perpendicular dist (proposed)
    GAP= np.zeros((T, N), np.float32)
    PR = np.zeros((T, N), np.int32)     # progress_buf
    HIT= np.zeros((T, N), np.bool_)     # learner GOT HIT this step
    for t in range(n_steps):
        task.drill_ids[:] = D_DODGE
        task.opp_drill_ids[:] = opp
        task.opp_frozen[:] = False
        obs = task.obs_buf
        z0 = low_level(obs[:N], task.drill_ids)
        z1 = low_level(obs[N:2*N], task.opp_drill_ids)
        task.step(torch.cat([z0, z1], dim=0))

        # --- exact avoid_r (mirror _compute_drill_reward) ---
        opp_tip = task.get_sword_tip_pos()[1]                       # (N,1,3) opponent tip
        my_tg   = task._rigid_body_pos_list[0][:, task._target_ids] # (N,k,3) learner targets
        threat  = torch.linalg.norm(opp_tip - my_tg, dim=-1).min(dim=-1).values  # (N,)
        avoid_r = 1.0 - torch.exp(-2.0 * threat)
        # --- proposed: perpendicular distance of learner's nearest target to the opp THRUST LINE ---
        opp_hand = task._rigid_body_pos_list[1][:, task._hand_ids]  # (N,1,3)
        d = (opp_tip - opp_hand)                                    # (N,1,3) line dir (hand->tip)
        dn = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ap = my_tg - opp_hand                                       # (N,k,3)
        proj = (ap * dn).sum(-1, keepdim=True) * dn                 # (N,k,3)
        perp = torch.linalg.norm(ap - proj, dim=-1).min(dim=-1).values  # (N,) nearest target's dist off the line
        gap = torch.linalg.norm(task._humanoid_root_states_list[0][...,0:2]
                                - task._humanoid_root_states_list[1][...,0:2], dim=-1)
        got = task.sword_hit_list[1][:, 0].bool()                   # opponent hit the learner
        AV[t]=avoid_r.cpu(); TD[t]=threat.cpu(); OL[t]=perp.cpu(); GAP[t]=gap.cpu()
        PR[t]=task.progress_buf.cpu(); HIT[t]=got.cpu()
    return AV, TD, OL, GAP, PR, HIT

def analyze(AV, TD, OL, GAP, PR, HIT, W=8):
    T, N = AV.shape
    hit_slopes=[]; hit_at=[]; hit_before=[]; hit_epavg=[]; surv_epavg=[]
    td_at=[]; td_before=[]; ol_at=[]; ol_before=[]
    rose=0; fell=0; n_hit=0
    for e in range(N):
        # segment episodes by progress resets (progress drops => new episode)
        starts=[0]
        for t in range(1,T):
            if PR[t,e] <= PR[t-1,e]:   # reset boundary
                starts.append(t)
        starts.append(T)
        for si in range(len(starts)-1):
            a,b = starts[si], starts[si+1]           # episode [a,b)
            seg_hit = np.where(HIT[a:b,e])[0]
            epavg = float(AV[a:b,e].mean())
            if len(seg_hit)>0:
                h = a + seg_hit[0]                    # first hit step in this episode
                n_hit+=1
                lo = max(a, h-W)
                if h-lo>=2:
                    xs=np.arange(h-lo+1); ys=AV[lo:h+1,e]
                    slope=np.polyfit(xs,ys,1)[0]; hit_slopes.append(slope)
                    if slope>0: rose+=1
                    else: fell+=1
                    hit_at.append(AV[h,e]); hit_before.append(AV[lo,e])
                    td_at.append(TD[h,e]); td_before.append(TD[lo,e])
                    ol_at.append(OL[h,e]); ol_before.append(OL[lo,e])
                hit_epavg.append(epavg)
            else:
                surv_epavg.append(epavg)
    p=lambda x: (np.mean(x) if len(x) else float('nan'))
    print("\n================ avoid_r GAMEABILITY TEST (dodge vs live lunge) ================")
    print(f"episodes: {n_hit} ended in GOT-HIT, {len(surv_epavg)} survived (timeout/no-hit)")
    print(f"window into the hit: last W={W} steps\n")
    print("avoid_r trajectory INTO the hit:")
    print(f"  rose vs fell approaching the hit: {rose} rose / {fell} fell  ({100*rose/max(rose+fell,1):.0f}% rising)")
    print(f"  mean slope over last {W} steps: {p(hit_slopes):+.4f} / step   (>0 => reward RISES into the hit)")
    print(f"  avoid_r  {W} steps before hit -> at hit:  {p(hit_before):.3f} -> {p(hit_at):.3f}")
    print(f"  threat_dist(m) {W} before -> at hit:      {p(td_before):.3f} -> {p(td_at):.3f}   (tip distance to body)")
    print(f"  off-LINE(m)   {W} before -> at hit:       {p(ol_before):.3f} -> {p(ol_at):.3f}   (proposed metric)")
    print(f"\nper-episode MEAN avoid_r:  got-hit episodes={p(hit_epavg):.3f}   survived episodes={p(surv_epavg):.3f}")
    print("  (if got-hit episodes earn avoid_r >= survivors, the term fails to distinguish good from bad)")
    print("="*80)

@hydra.main(version_base=None, config_path="../phc/data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True)); set_np_formatting()
    flags.test=True
    for k in ['debug','follow','server_mode','im_eval','no_virtual_display','render_o3d','fixed',
              'divide_group','no_collision_check','fixed_path','real_path','show_traj','slow','real_traj','trigger_input']:
        setattr(flags,k,False)
    flags.add_proj=cfg.get("add_proj",False); flags.has_eval=cfg.get("has_eval",False)
    set_seed(cfg.get("seed",0),False)
    cfg.learning['params']['config']["num_actors"]=cfg.env.num_envs
    sim_params=parse_sim_params(cfg)
    args=EasyDict({"task":cfg.env.task,"device_id":cfg.device_id,"rl_device":cfg.rl_device,
                   "physics_engine":gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
                   "headless":cfg.headless,"device":cfg.device})
    task,env=parse_task(args,cfg,cfg_train=cfg.learning,sim_params=sim_params)
    low_level=FrozenLowLevelPolicy(cfg.env["low_level_checkpoint"],NUM_DRILLS,device=task.device)
    out=run(cfg,task,low_level,n_steps=cfg.env.get("measure_steps",200))
    analyze(*out, W=cfg.env.get("avoid_window",8))

if __name__=='__main__':
    main()
