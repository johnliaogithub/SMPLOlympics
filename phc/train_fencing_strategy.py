# Train the high-level STRATEGY network for hierarchical fencing.
#
# Architecture:
#   strategy net (trained here)  --every macro_K steps-->  drill choice (discrete, 6)
#         |                                                       |
#         v                                                       v
#   FROZEN low-level drill policy (obs + drill one-hot --> Z) --> PULSE decoder --> joints
#
# The strategy net is trained with PPO on a SPARSE win/loss reward. The original
# dense fencing reward is computed by the env and logged to W&B for comparison
# only (it never enters the gradient).
#
# This is a self-contained training loop (it does NOT use rl_games), mirroring
# phc/run_hydra.py only for env construction. It is a research prototype: the
# physics/loader integration should be sanity-checked on the first GPU run (see
# the DEBUG CHECKLIST in the chat answer).
#
# Example:
#   bash scripts/fencing/train_fencing_strategy.sh

import os
import sys

os.environ['OMP_NUM_THREADS'] = "1"
sys.path.append(os.getcwd())
sys.path.append('./SMPLSim')

from isaacgym import gymapi, gymutil  # noqa: F401  (must precede torch)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
import wandb

from phc.utils.config import set_np_formatting, set_seed
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from phc.run_hydra import parse_sim_params
from rl_games.algos_torch import torch_ext
from phc.learning.network_loader import load_mcp_mlp
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, DRILL_NAMES


# --------------------------------------------------------------------------- #
# Frozen low-level drill policy: obs (+ one-hot drill) -> Z latent
# --------------------------------------------------------------------------- #
class FrozenLowLevelPolicy:
    """Reconstructs the trained drills actor (obs -> Z) from a checkpoint using
    weight-shape inference (no rl_games rebuild), and applies the matching obs
    normalization. The one-hot drill id occupies the final NUM_DRILLS obs dims."""

    def __init__(self, ckpt_path, num_drills, activation="silu", device="cuda:0"):
        ckpt = torch_ext.load_checkpoint(ckpt_path)
        # actor_mlp + mu head -> deterministic Z (matches is_determenistic=True)
        self.actor = load_mcp_mlp(ckpt, activation=activation, mlp_name="actor_mlp").to(device)
        self.actor.eval()
        rms = ckpt['running_mean_std']
        self.mean = rms['running_mean'].float().to(device)
        self.var = rms['running_var'].float().to(device)
        self.num_drills = num_drills
        self.device = device
        print(f"[LowLevel] loaded {ckpt_path}  (obs_dim={self.mean.shape[0]}, "
              f"Z_dim={self.actor[-1].out_features})")

    @torch.no_grad()
    def __call__(self, obs, drill_ids):
        obs = obs.clone()
        # overwrite the drill one-hot so the executed drill matches the command,
        # regardless of when the env last recomputed its observation
        obs[:, -self.num_drills:] = F.one_hot(drill_ids, self.num_drills).float()
        obs = torch.clamp(obs, -5.0, 5.0)
        obs_n = (obs - self.mean) / torch.sqrt(self.var + 1e-5)
        return self.actor(obs_n)


# --------------------------------------------------------------------------- #
# Strategy network: obs -> (drill logits, value)
# --------------------------------------------------------------------------- #
class StrategyNet(nn.Module):
    def __init__(self, obs_dim, num_drills, hidden=512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.pi = nn.Linear(hidden, num_drills)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        return self.pi(h), self.v(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# PPO over macro (options) timesteps
# --------------------------------------------------------------------------- #
def train(cfg, task, env):
    device = task.device
    N = task.num_envs
    obs_dim = task.get_obs_size()

    sp = cfg.get("strategy", {})
    T = sp.get("rollout_macrosteps", 32)      # macro steps per PPO iteration
    total_iters = sp.get("iters", 20000)
    gamma = sp.get("gamma", 0.99)
    lam = sp.get("gae_lambda", 0.95)
    clip = sp.get("clip", 0.2)
    lr = sp.get("lr", 3e-4)
    epochs = sp.get("update_epochs", 5)
    num_mb = sp.get("num_minibatches", 4)
    ent_coef = sp.get("entropy_coef", 0.01)
    vf_coef = sp.get("value_coef", 0.5)
    save_every = sp.get("save_every", 500)
    out_dir = cfg.output_path

    net = StrategyNet(obs_dim, NUM_DRILLS, sp.get("hidden", 512)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def get_self_obs():
        return torch.clamp(task.obs_buf[:N].detach(), -5.0, 5.0)

    def get_opp_obs():
        return torch.clamp(task.obs_buf[N:2 * N].detach(), -5.0, 5.0)

    env.reset()  # populate obs_buf

    for it in range(total_iters):
        obs_b = torch.zeros(T, N, obs_dim, device=device)
        act_b = torch.zeros(T, N, dtype=torch.long, device=device)
        logp_b = torch.zeros(T, N, device=device)
        val_b = torch.zeros(T, N, device=device)
        rew_b = torch.zeros(T, N, device=device)
        done_b = torch.zeros(T, N, device=device)

        ep_dense = torch.zeros(N, device=device)
        win_count = loss_count = end_count = 0
        drill_hist = torch.zeros(NUM_DRILLS, device=device)

        for t in range(T):
            obs_self = get_self_obs()
            logits, value = net(obs_self)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            with torch.no_grad():
                opp_logits, _ = net(get_opp_obs())
                opp_action = torch.distributions.Categorical(logits=opp_logits).sample()

            sparse, dense, done = task.macro_step(action, opp_action)

            obs_b[t], act_b[t], logp_b[t], val_b[t] = obs_self, action, logp.detach(), value.detach()
            rew_b[t], done_b[t] = sparse, done.float()

            ep_dense += dense
            drill_hist += torch.bincount(action, minlength=NUM_DRILLS).float()
            win_count += (sparse > 0).sum().item()
            loss_count += (sparse < 0).sum().item()
            end_count += done.sum().item()

        # bootstrap value of the final state
        with torch.no_grad():
            _, last_val = net(get_self_obs())

        # GAE
        adv = torch.zeros(T, N, device=device)
        last_gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else val_b[t + 1]
            next_nonterminal = 1.0 - done_b[t]
            delta = rew_b[t] + gamma * next_val * next_nonterminal - val_b[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + val_b

        b_obs = obs_b.reshape(T * N, obs_dim)
        b_act = act_b.reshape(T * N)
        b_logp = logp_b.reshape(T * N)
        b_adv = adv.reshape(T * N)
        b_ret = ret.reshape(T * N)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        mb_size = (T * N) // num_mb
        idx = torch.randperm(T * N, device=device)
        for _ in range(epochs):
            for s in range(0, T * N, mb_size):
                mb = idx[s:s + mb_size]
                logits, value = net(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * torch.clamp(ratio, 1 - clip, 1 + clip)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = F.mse_loss(value, b_ret[mb])
                ent = dist.entropy().mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        if wandb.run is not None:
            log = {
                "strategy/sparse_return": rew_b.sum(0).mean().item(),
                "strategy/dense_return": ep_dense.mean().item(),
                "strategy/win_rate": win_count / max(end_count, 1),
                "strategy/loss_rate": loss_count / max(end_count, 1),
                "strategy/episodes_ended": end_count,
                "strategy/policy_entropy": ent.item(),
                "strategy/value_loss": v_loss.item(),
            }
            dh = drill_hist / drill_hist.sum().clamp_min(1)
            for d, name in enumerate(DRILL_NAMES):
                log[f"strategy_drill_usage/{name}"] = dh[d].item()
            wandb.log(log, step=it)

        if it % 20 == 0:
            print(f"[it {it}] sparse_ret={rew_b.sum(0).mean():.3f} "
                  f"win_rate={win_count / max(end_count,1):.3f} ended={end_count}")

        if it > 0 and it % save_every == 0:
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"strategy_net": net.state_dict(), "iter": it},
                       os.path.join(out_dir, f"strategy_{it:08d}.pth"))
            torch.save({"strategy_net": net.state_dict(), "iter": it},
                       os.path.join(out_dir, "strategy.pth"))


@hydra.main(version_base=None, config_path="../phc/data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    set_np_formatting()

    flags.debug, flags.follow, flags.test, flags.server_mode = cfg.debug, False, False, cfg.server_mode
    flags.im_eval = flags.no_virtual_display = flags.render_o3d = False
    flags.fixed = flags.divide_group = flags.no_collision_check = flags.fixed_path = False
    flags.real_path = flags.show_traj = flags.slow = flags.real_traj = flags.trigger_input = False
    flags.add_proj = cfg.get("add_proj", False)
    flags.has_eval = cfg.get("has_eval", False)

    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    project_name = cfg.get("project_name", "SMPLOlympics")
    if (not cfg.get("no_log", False)) and (not cfg.debug):
        wandb.init(entity=cfg.wandb_entity, project=project_name, resume='allow',
                   notes=cfg.get("notes", "strategy training"))
        wandb.config.update(cfg, allow_val_change=True)
        wandb.run.name = cfg.exp_name

    cfg_train = cfg.learning
    cfg_train['params']['config']["num_actors"] = cfg.env.num_envs

    sim_params = parse_sim_params(cfg)
    args = EasyDict({
        "task": cfg.env.task,
        "device_id": cfg.device_id,
        "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
        "headless": cfg.headless,
        "device": cfg.device,
    })
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    low_level_ckpt = cfg.env["low_level_checkpoint"]
    task.low_level_policy = FrozenLowLevelPolicy(low_level_ckpt, NUM_DRILLS, device=task.device)

    os.makedirs(cfg.output_path, exist_ok=True)
    train(cfg, task, env)


if __name__ == '__main__':
    main()
