# Shared pieces for the paper scripts (phc/paper/*): env construction, a loader for the flat
# baseline policy, and the motion metrics, so both methods are measured by the same code.
import os
import sys

os.environ['OMP_NUM_THREADS'] = "1"
sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'phc'))
sys.path.append('./SMPLSim')

from isaacgym import gymapi  # noqa: F401  (before torch)
import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
from omegaconf import OmegaConf

from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
from phc.run_hydra import parse_sim_params
from phc.learning.network_loader import load_mcp_mlp

METHOD_COLORS = {"ours": "#2a78d6", "flat": "#eb6834", "mocap": "#1d9e75"}   # validated (dataviz)
METHOD_LABELS = {"ours": "Drills (ours)", "flat": "Flat self-play", "mocap": "Fencing mocap"}

# en-garde criteria, identical to phc/survey_engarde.py
FOOT_N, SPLIT_MIN, SPLIT_MAX, TIP_Z, AIM_COS, SPEED_MAX = 50.0, 0.25, 0.70, (0.7, 1.6), 0.8, 0.3
FALL_HEAD = 0.5


def make_task(cfg_hydra, test=True, seed=0):
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    set_np_formatting()
    for k in ['debug', 'follow', 'server_mode', 'im_eval', 'no_virtual_display', 'render_o3d', 'fixed',
              'divide_group', 'no_collision_check', 'fixed_path', 'real_path', 'show_traj', 'slow',
              'real_traj', 'trigger_input']:
        setattr(flags, k, False)
    flags.test = test
    flags.add_proj = False
    flags.has_eval = False
    set_seed(seed, False)
    cfg.learning['params']['config']["num_actors"] = cfg.env.num_envs
    args = EasyDict({"task": cfg.env.task, "device_id": cfg.device_id, "rl_device": cfg.rl_device,
                     "physics_engine": gymapi.SIM_PHYSX, "headless": cfg.headless, "device": cfg.device})
    task, _ = parse_task(args, cfg, cfg.learning, parse_sim_params(cfg))
    return cfg, task


class FlatPolicy:
    """Deterministic actor (obs -> Z) of a flat HumanoidFencingZ checkpoint, with its obs normalization.
    Same reconstruction as FrozenLowLevelPolicy, without the drill one-hot."""

    def __init__(self, ckpt_path, activation="silu", device="cuda:0"):
        ckpt = torch_ext.load_checkpoint(ckpt_path)
        self.actor = load_mcp_mlp(ckpt, activation=activation, mlp_name="actor_mlp").to(device)
        self.actor.eval()
        rms = ckpt['running_mean_std']
        self.mean = rms['running_mean'].float().to(device)
        self.var = rms['running_var'].float().to(device)
        print(f"[Flat] loaded {ckpt_path}  (obs_dim={self.mean.shape[0]})")

    @torch.no_grad()
    def __call__(self, obs):
        obs = torch.clamp(obs, -5.0, 5.0)
        # rl_games clamps again after normalizing (RunningMeanStd.forward); match training
        return self.actor(torch.clamp((obs - self.mean) / torch.sqrt(self.var + 1e-5), -5.0, 5.0))


class MotionMetrics:
    """Per-step motion metrics for agent 0 over the envs still in their first bout (`alive`)."""

    def __init__(self, task):
        self.t = task
        ids = task._build_key_body_ids_tensor
        self.lfoot, self.rfoot = ids(["L_Ankle", "L_Toe"]), ids(["R_Ankle", "R_Toe"])
        self.feet = ids(["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"])
        self.la, self.ra = int(ids(["L_Ankle"])[0]), int(ids(["R_Ankle"])[0])
        self.head, self.chest, self.pelvis = int(ids(["Head"])[0]), int(ids(["Chest"])[0]), int(ids(["Pelvis"])[0])
        self.hand = int(task._hand_ids[0])
        N = task.num_envs
        self.alive = torch.ones(N, dtype=torch.bool, device=task.device)
        self.fell = torch.zeros(N, dtype=torch.bool, device=task.device)
        self.fall_step = torch.full((N,), -1, dtype=torch.long, device=task.device)
        self.hist = []                    # last body positions, for jerk
        self.prev_hit = torch.zeros(N, dtype=torch.bool, device=task.device)
        self.touched = torch.zeros(N, dtype=torch.bool, device=task.device)
        self.acc = {k: [] for k in ["slide", "jerk", "engarde", "split", "spine", "touch", "still"]}
        self.steps = 0

    @torch.no_grad()
    def update(self):
        t, a = self.t, self.alive
        bp = t._rigid_body_pos_list[0]
        dt = t.dt
        cf = t._contact_forces_list[0]
        head_z = bp[:, self.head, 2]
        new_fall = a & (head_z < FALL_HEAD) & ~self.fell
        self.fall_step[new_fall] = self.steps
        self.fell |= new_fall
        # foot sliding: horizontal speed of a foot body while it carries load
        if self.hist:
            v = (bp[:, self.feet, :2] - self.hist[-1][:, self.feet, :2]).norm(dim=-1) / dt     # (N, 4)
            loaded = cf[:, self.feet].norm(dim=-1) > FOOT_N
            m = loaded & a[:, None]
            self.acc["slide"].append(v[m].cpu().numpy())
        # jerk: third finite difference of all body positions, mean over bodies
        self.hist.append(bp.clone())
        if len(self.hist) > 4:
            self.hist.pop(0)
        if len(self.hist) == 4:
            p3, p2, p1, p0 = self.hist
            j = ((p0 - 3 * p1 + 3 * p2 - p3) / dt ** 3).norm(dim=-1).mean(-1)
            self.acc["jerk"].append(j[a].cpu().numpy())
        # en garde (same criteria as survey_engarde.py)
        opp = t._rigid_body_pos_list[1]
        r0, r1 = t._humanoid_root_states_list[0], t._humanoid_root_states_list[1]
        fwd = F.normalize(r1[:, :2] - r0[:, :2], dim=-1)
        planted = (cf[:, self.lfoot].norm(dim=-1).sum(-1) > FOOT_N) & (cf[:, self.rfoot].norm(dim=-1).sum(-1) > FOOT_N)
        split = ((bp[:, self.ra, :2] - bp[:, self.la, :2]) * fwd).sum(-1)
        tip = t.get_sword_tip_pos()[0][:, 0]
        blade = F.normalize(tip - bp[:, self.hand], dim=-1)
        to_opp = F.normalize(opp[:, self.chest] - bp[:, self.hand], dim=-1)
        sword_ok = (tip[:, 2] > TIP_Z[0]) & (tip[:, 2] < TIP_Z[1]) & ((blade * to_opp).sum(-1) > AIM_COS)
        still = r0[:, 7:9].norm(dim=-1) < SPEED_MAX
        eg = planted & (split > SPLIT_MIN) & (split < SPLIT_MAX) & sword_ok & still
        up = bp[:, self.chest] - bp[:, self.pelvis]
        self.acc["engarde"].append(eg[a].float().cpu().numpy())
        self.acc["split"].append(split[a].cpu().numpy())
        self.acc["spine"].append((up[:, 2] / up.norm(dim=-1))[a].cpu().numpy())
        self.acc["still"].append(still[a].float().cpu().numpy())
        hit = t.sword_hit_list[0].squeeze(-1).bool() if hasattr(t, "sword_hit_list") else torch.zeros_like(a)
        self.acc["touch"].append((hit & ~self.prev_hit)[a].float().cpu().numpy())
        self.touched |= hit & a
        self.prev_hit = hit.clone()
        self.steps += 1

    def end_bouts(self, done):
        """Envs whose first bout ended stop counting (no resets, so each env is one bout)."""
        self.alive &= ~done.bool()

    def summary(self):
        cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.acc.items()}
        alive_steps = len(cat["engarde"])
        minutes = alive_steps * self.t.dt / 60.0
        return {
            "bouts": int(self.t.num_envs),
            "fall_rate": float(self.fell.float().mean().item()),
            "engarde_frac": float(cat["engarde"].mean()) if alive_steps else float("nan"),
            "foot_slide_cms": 100.0 * float(np.mean(cat["slide"])) if len(cat["slide"]) else float("nan"),
            "jerk_median": float(np.median(cat["jerk"])) if len(cat["jerk"]) else float("nan"),
            "spine_up_mean": float(cat["spine"].mean()) if alive_steps else float("nan"),
            "split_median_m": float(np.median(cat["split"])) if alive_steps else float("nan"),
            "touches_per_min": float(cat["touch"].sum() / max(minutes, 1e-9)),
            "bouts_with_touch": float(self.touched.float().mean().item()),
            "agent_steps": int(alive_steps),
        }
