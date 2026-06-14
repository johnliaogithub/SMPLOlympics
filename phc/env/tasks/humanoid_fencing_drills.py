# Drill-conditioned fencing task for hierarchical RL.
#
# A single policy is conditioned on a one-hot drill ID (appended to the task obs)
# and trained on isolated fencing skills. Agent 0 is always the learner; agent 1
# is the opponent. The opponent is frozen (Z-actions overridden to 0, which with
# the PULSE VAE prior decodes to neutral idle motion) for all drills except
# "dodge", where the opponent executes a lunge drill via its own one-hot.
#
# IMPORTANT (training setup): the self-play algo only uses the *updating* agent's
# experience, and the opponent runs a snapshot model. Train with
# learning.params.config.switch_frequency set very large so only agent 0 is ever
# updated — agent 1's recorded actions are overridden when frozen and must never
# be used for gradient updates. For the dodge drill, resume from a checkpoint
# where the lunge drills are already trained so the snapshot opponent can lunge
# (phase A: drills 0-4, phase B: add dodge).
#
# Drill switching mid-episode is intentionally supported by design: rewards and
# observations are computed from the live drill_ids tensors every step, so
# sequencing drills within an episode only requires resampling drill_ids at a
# chosen step (see _resample_drills).

import torch
import torch.nn.functional as F

from isaacgym.torch_utils import *

import phc.env.tasks.humanoid_amp as humanoid_amp
from phc.env.tasks.humanoid_fencing import (
    HumanoidFencing,
    compute_humanoid_reset,
    compute_humanoid_reset_z,
)
from phc.utils import torch_utils
from phc.utils.flags import flags

DRILL_NAMES = ["advance", "retreat", "stand", "lunge_upper", "lunge_groin", "dodge"]
NUM_DRILLS = len(DRILL_NAMES)
D_ADVANCE, D_RETREAT, D_STAND, D_LUNGE_UPPER, D_LUNGE_GROIN, D_DODGE = range(NUM_DRILLS)


class HumanoidFencingDrills(HumanoidFencing):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        assert self.num_agents == 2, "Drills require exactly 2 agents (learner + opponent)"

        self._head_id = self._build_key_body_ids_tensor(["Head"])
        self._upper_target_ids = self._build_key_body_ids_tensor(["Chest", "Neck", "Head"])
        self._groin_target_ids = self._build_key_body_ids_tensor(["Pelvis"])

        # Per-env drill assignment. Agent 0 = learner drill, agent 1 = opponent drill.
        self.drill_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.opp_drill_ids = torch.full((self.num_envs,), D_STAND, dtype=torch.long, device=self.device)
        self.opp_frozen = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Sampling distribution over drills, e.g. +env.drill_probs=[1,1,1,1,1,0] for phase A (no dodge)
        drill_probs = cfg["env"].get("drill_probs", [1.0] * NUM_DRILLS)
        self.drill_probs = torch.tensor(drill_probs, dtype=torch.float, device=self.device)
        assert len(drill_probs) == NUM_DRILLS

        # Phase C: resample each env's drill every N steps WITHIN an episode (0 = off).
        # Forces the policy to chain skills (e.g. advance -> stand) without losing
        # balance at the transition, and pre-trains for the strategy net's switches.
        self.drill_switch_interval = cfg["env"].get("drill_switch_interval", 0)

        self._prev_head_pos_list = [torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
                                    for _ in range(self.num_agents)]

    def get_task_obs_size(self):
        obs_size = super().get_task_obs_size()
        if self._enable_task_obs:
            obs_size += NUM_DRILLS
        return obs_size

    def _compute_task_obs(self, env_ids=None):
        obs_list = super()._compute_task_obs(env_ids)
        if env_ids is None:
            env_ids = self.env_ids_all
        oh_learner = F.one_hot(self.drill_ids[env_ids], NUM_DRILLS).float()
        oh_opp = F.one_hot(self.opp_drill_ids[env_ids], NUM_DRILLS).float()
        return [torch.cat([obs_list[0], oh_learner], dim=-1),
                torch.cat([obs_list[1], oh_opp], dim=-1)]

    def _resample_drills(self, env_ids):
        n = len(env_ids)
        new_drills = torch.multinomial(self.drill_probs, n, replacement=True)
        self.drill_ids[env_ids] = new_drills

        opp = torch.full((n,), D_STAND, dtype=torch.long, device=self.device)
        dodge_mask = new_drills == D_DODGE
        lunge_choice = torch.where(torch.rand(n, device=self.device) < 0.5,
                                   torch.tensor(D_LUNGE_UPPER, device=self.device),
                                   torch.tensor(D_LUNGE_GROIN, device=self.device))
        opp[dodge_mask] = lunge_choice[dodge_mask]
        self.opp_drill_ids[env_ids] = opp
        self.opp_frozen[env_ids] = ~dodge_mask

    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self._resample_drills(env_ids)
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            for i in range(self.num_agents):
                self._prev_head_pos_list[i][env_ids] = self._rigid_body_pos_list[i][env_ids, self._head_id[0]]

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        for i in range(self.num_agents):
            self._prev_head_pos_list[i] = self._rigid_body_pos_list[i][:, self._head_id[0]].clone()

    def post_physics_step(self):
        super().post_physics_step()
        # Phase C: mid-episode drill switching. progress_buf is per-env (num_envs,).
        if self.drill_switch_interval > 0:
            due = (self.progress_buf > 0) & (self.progress_buf % self.drill_switch_interval == 0)
            due_ids = due.nonzero(as_tuple=False).flatten()
            if len(due_ids) > 0:
                self._resample_drills(due_ids)
        return

    def _check_hit_subset(self, attacker_idx, target_body_ids):
        tip = self.get_sword_tip_pos()[attacker_idx]  # (N, 1, 3)
        target_pos = self._rigid_body_pos_list[1 - attacker_idx][:, target_body_ids]
        dist = torch.linalg.norm(tip - target_pos, dim=-1)
        contact = torch.norm(self._contact_forces_list[1 - attacker_idx][:, target_body_ids], dim=-1) > 50
        return torch.any(torch.logical_and(dist < 0.1, contact), dim=-1)

    def _compute_drill_reward(self, i, drill_ids, sword_tip_pos_list):
        dt = self.dt
        root_state = self._humanoid_root_states_list[i]
        root_pos = root_state[..., 0:3]
        root_rot = root_state[..., 3:7]
        prev_root_pos = self._prev_root_pos_list[i]
        opp_root_pos = self._humanoid_root_states_list[1 - i][..., 0:3]
        root_vel = (root_pos - prev_root_pos) / dt

        # velocity toward / away from opponent (same shaping as compute_fencing_reward)
        tar_dir = F.normalize(opp_root_pos[..., 0:2] - root_pos[..., 0:2], dim=-1)
        speed_toward = torch.sum(tar_dir * root_vel[..., 0:2], dim=-1)

        def speed_shaping(speed, tar_speed=1.0):
            err = torch.clamp_min(tar_speed - speed, 0.0)
            rwd = torch.exp(-4.0 * err * err)
            rwd[speed <= 0] = 0
            return rwd

        vel_toward_r = speed_shaping(speed_toward)
        vel_away_r = speed_shaping(-speed_toward)

        # facing the opponent
        heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)
        tar_dir_3d = opp_root_pos - root_pos
        tar_dir_3d[..., -1] = 0.0
        tar_dir_3d = F.normalize(tar_dir_3d, dim=-1)
        tar_dir_local = torch_utils.my_quat_rotate(heading_rot_inv, tar_dir_3d)
        facing_r = torch.exp(-2.0 * (1 - tar_dir_local[..., 0]))

        # head vertical stability: penalize vertical head motion since the LAST frame
        head_pos = self._rigid_body_pos_list[i][:, self._head_id[0]]
        head_vert_speed = (head_pos[:, 2] - self._prev_head_pos_list[i][:, 2]).abs() / dt
        head_stab_r = torch.exp(-5.0 * head_vert_speed)

        # standing still
        still_r = torch.exp(-2.0 * torch.linalg.norm(root_vel, dim=-1))

        # lunge proximity + hit on specific target regions
        tip = sword_tip_pos_list[i]
        opp_body_pos = self._rigid_body_pos_list[1 - i]
        upper_dist = torch.linalg.norm(tip - opp_body_pos[:, self._upper_target_ids], dim=-1).min(dim=-1).values
        groin_dist = torch.linalg.norm(tip - opp_body_pos[:, self._groin_target_ids], dim=-1).min(dim=-1).values
        upper_prox_r = torch.exp(-10.0 * upper_dist)
        groin_prox_r = torch.exp(-10.0 * groin_dist)
        hit_upper = self._check_hit_subset(i, self._upper_target_ids).float()
        hit_groin = self._check_hit_subset(i, self._groin_target_ids).float()

        # dodge: keep opponent's sword tip away from own target bodies
        opp_tip = sword_tip_pos_list[1 - i]
        my_targets = self._rigid_body_pos_list[i][:, self._target_ids]
        threat_dist = torch.linalg.norm(opp_tip - my_targets, dim=-1).min(dim=-1).values
        avoid_r = 1.0 - torch.exp(-2.0 * threat_dist)
        im_hit = self.sword_hit_list[1 - i].squeeze(-1).float()

        r_advance = 0.50 * vel_toward_r + 0.20 * facing_r + 0.30 * head_stab_r
        r_retreat = 0.50 * vel_away_r + 0.20 * facing_r + 0.30 * head_stab_r
        r_stand = 0.60 * still_r + 0.20 * facing_r + 0.20 * head_stab_r
        r_lunge_u = 0.35 * upper_prox_r + 0.15 * facing_r + 0.20 * vel_toward_r + 0.30 * hit_upper
        r_lunge_g = 0.35 * groin_prox_r + 0.15 * facing_r + 0.20 * vel_toward_r + 0.30 * hit_groin
        r_dodge = 0.50 * avoid_r + 0.20 * facing_r + 0.30 * head_stab_r - 1.00 * im_hit

        all_r = torch.stack([r_advance, r_retreat, r_stand, r_lunge_u, r_lunge_g, r_dodge], dim=-1)
        return all_r.gather(-1, drill_ids[:, None]).squeeze(-1)

    def _compute_reward(self, actions):
        sword_tip_pos_list = self.get_sword_tip_pos()
        for i in range(self.num_agents):
            drill_ids = self.drill_ids if i == 0 else self.opp_drill_ids
            reward = self._compute_drill_reward(i, drill_ids, sword_tip_pos_list)
            if i == 1:
                # frozen opponents take no reward; their experience is never used anyway
                reward = torch.where(self.opp_frozen, torch.zeros_like(reward), reward)
            self.rew_buf[i * self.num_envs:(i + 1) * self.num_envs] = reward

            if i == 0:
                self._log_drill_rewards(reward)
        return

    def _log_drill_rewards(self, reward):
        # accumulate per-drill mean reward; flush to W&B every 32 steps (~1 epoch)
        if not hasattr(self, '_drill_rew_sum'):
            self._drill_rew_sum = torch.zeros(NUM_DRILLS, device=self.device)
            self._drill_rew_count = torch.zeros(NUM_DRILLS, device=self.device)
            self._drill_log_n = 0
        self._drill_rew_sum.scatter_add_(0, self.drill_ids, reward.detach())
        self._drill_rew_count.scatter_add_(0, self.drill_ids, torch.ones_like(reward))
        self._drill_log_n += 1
        if self._drill_log_n >= 32:
            try:
                import wandb
                if wandb.run is not None:
                    means = (self._drill_rew_sum / self._drill_rew_count.clamp_min(1)).cpu()
                    log_data = {f"drills/{name}": means[d].item()
                                for d, name in enumerate(DRILL_NAMES) if self._drill_rew_count[d] > 0}
                    wandb.log(log_data, step=self.step_counter // 32)
            except Exception:
                pass
            self._drill_rew_sum.zero_()
            self._drill_rew_count.zero_()
            self._drill_log_n = 0

    def _compute_reset(self):
        # drills have no win conditions — only out-of-bounds and falls end an episode
        game_done = self.out_bound
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                       self._contact_forces_list, self._contact_body_ids,
                                                       self._rigid_body_pos_list,
                                                       self._strike_body_ids, self.max_episode_length,
                                                       self._enable_early_termination, self._termination_heights, self.num_agents)
        self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, game_done), torch.logical_or(self._terminate_buf, game_done)
        return


class HumanoidFencingDrillsZ(HumanoidFencingDrills):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return

    def step(self, actions):
        # Freeze the opponent (agent 1) by zeroing its Z-actions. With the PULSE
        # VAE prior, Z=0 decodes to the prior mean — neutral, balanced idle motion.
        actions = actions.clone()
        agent1 = actions[self.num_envs:2 * self.num_envs]
        agent1[self.opp_frozen] = 0
        super().step_z(actions)
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()
        return

    def _compute_reset(self):
        game_done = self.out_bound
        if self.step_counter > self.warmup_time or flags.test:
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset_z(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        else:
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces_list, self._contact_body_ids,
                                                           self._rigid_body_pos_list,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self.num_agents)
        self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, game_done), torch.logical_or(self._terminate_buf, game_done)
        return
