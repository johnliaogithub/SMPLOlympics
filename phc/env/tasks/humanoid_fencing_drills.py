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

# New drills are appended (never inserted) so indices 0-5 stay fixed — this lets
# an old 6-drill checkpoint be warm-started into the 8-drill net (see
# scripts/fencing/expand_drills_checkpoint.py).
DRILL_NAMES = ["advance", "retreat", "stand", "lunge_upper", "lunge_groin", "dodge",
               "step_left", "step_right"]
NUM_DRILLS = len(DRILL_NAMES)
(D_ADVANCE, D_RETREAT, D_STAND, D_LUNGE_UPPER, D_LUNGE_GROIN, D_DODGE,
 D_STEP_LEFT, D_STEP_RIGHT) = range(NUM_DRILLS)


class HumanoidFencingDrills(HumanoidFencing):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        assert self.num_agents == 2, "Drills require exactly 2 agents (learner + opponent)"

        self._head_id = self._build_key_body_ids_tensor(["Head"])
        self._upper_target_ids = self._build_key_body_ids_tensor(["Chest", "Neck", "Head"])
        self._groin_target_ids = self._build_key_body_ids_tensor(["Pelvis"])
        self._right_foot_id = self._build_key_body_ids_tensor(["R_Ankle"])  # front/lunging foot

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

        # The lunge "thrust line" is captured this many steps into the episode, so
        # the agent has time to orient/choose a target before the reference is set
        # (agents spawn with the sword pointing at the ground).
        self.sword_ref_step = cfg["env"].get("sword_ref_step", 15)

        self._prev_head_pos_list = [torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
                                    for _ in range(self.num_agents)]
        # Sword-tip position last frame, for the lunge thrust-speed reward.
        self._prev_sword_tip_list = [torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
                                     for _ in range(self.num_agents)]
        # Sword direction (hand->tip) captured at episode start. The lunge rewards
        # keeping the sword aligned to this "thrust line" so a straight thrust
        # scores high while a slash (which rotates the blade) does not.
        self._sword_dir_0_list = [torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
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

        # Strike drills (lunge_upper/groin/dodge) spawn IN RANGE (1.5 m apart) so
        # the proximity/hit reward is reachable; otherwise the agent farms the
        # thrust term by waving the sword in place from 3 m away. Footwork/locomotion
        # drills keep the default 1.5 half-distance (3 m apart).
        in_range = (new_drills == D_LUNGE_UPPER) | (new_drills == D_LUNGE_GROIN) | (new_drills == D_DODGE)
        self._spawn_half_dist[env_ids] = torch.where(
            in_range, torch.full_like(self._spawn_half_dist[env_ids], 1.25),   # 2.5 m apart
            torch.full_like(self._spawn_half_dist[env_ids], 1.5))

    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self._resample_drills(env_ids)
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            tips = self.get_sword_tip_pos()
            for i in range(self.num_agents):
                self._prev_head_pos_list[i][env_ids] = self._rigid_body_pos_list[i][env_ids, self._head_id[0]]
                self._prev_sword_tip_list[i][env_ids] = tips[i][env_ids, 0]
                # Invalidate the thrust line; it is (re)captured at step sword_ref_step.
                # While zero, thrust_align_r = dot(dir, 0) = 0 (no thrust reward yet).
                self._sword_dir_0_list[i][env_ids] = 0.0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        tips = self.get_sword_tip_pos()
        for i in range(self.num_agents):
            self._prev_head_pos_list[i] = self._rigid_body_pos_list[i][:, self._head_id[0]].clone()
            self._prev_sword_tip_list[i] = tips[i][:, 0].clone()

    def post_physics_step(self):
        super().post_physics_step()
        # Capture the lunge "thrust line" once the agent has had time to aim.
        cap_ids = (self.progress_buf == self.sword_ref_step).nonzero(as_tuple=False).flatten()
        if len(cap_ids) > 0:
            tips = self.get_sword_tip_pos()
            for i in range(self.num_agents):
                hand = self._rigid_body_pos_list[i][cap_ids, self._hand_ids[0]]
                self._sword_dir_0_list[i][cap_ids] = F.normalize(tips[i][cap_ids, 0] - hand, dim=-1)
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

        # --- lunge: reward a straight THRUST that lands, NOT a slash or a walk-in ---
        tip = sword_tip_pos_list[i][:, 0]                  # (N, 3)
        opp_body_pos = self._rigid_body_pos_list[1 - i]
        opp_xy = opp_root_pos[..., 0:2]

        # Thrust quality (professor's idea): how aligned the sword still is with the
        # "thrust line" captured at episode start. A thrust keeps the blade pointed
        # the same way (align ~1); a slash rotates it (align drops).
        hand = self._rigid_body_pos_list[i][:, self._hand_ids[0]]
        sword_dir = F.normalize(tip - hand, dim=-1)
        thrust_align_r = torch.clamp(torch.sum(sword_dir * self._sword_dir_0_list[i], dim=-1), 0.0, 1.0)

        # Right (front) foot approach: the lunging foot drives toward the opponent.
        foot_xy = self._rigid_body_pos_list[i][:, self._right_foot_id[0], 0:2]
        foot_approach = torch.exp(-0.7 * torch.linalg.norm(foot_xy - opp_xy, dim=-1))

        def lunge_reward(target_ids):
            tgts = opp_body_pos[:, target_ids]             # (N, k, 3)
            dist = torch.linalg.norm(tip[:, None, :] - tgts, dim=-1).min(dim=-1).values
            # approach: tip-to-target (primary) + front-foot-to-opponent (secondary),
            # dense from 2.5 m. exp(-0.7*d): 0.17 @ 2.5 m, 0.50 @ 1 m, 0.81 @ 0.3 m.
            tip_approach = torch.exp(-0.7 * dist)
            approach_r = 0.7 * tip_approach + 0.3 * foot_approach
            force = torch.linalg.norm(self._contact_forces_list[1 - i][:, target_ids], dim=-1).max(dim=-1).values
            hit = self._check_hit_subset(i, target_ids).float()
            hit_r = hit * (0.5 + 0.5 * torch.clamp(force / 300.0, 0.0, 1.0))  # landed + strength
            return 0.25 * approach_r + 0.25 * thrust_align_r + 0.10 * facing_r + 0.40 * hit_r

        r_lunge_u = lunge_reward(self._upper_target_ids)
        r_lunge_g = lunge_reward(self._groin_target_ids)

        # --- lateral footwork: step left / right while staying square to opponent ---
        left_dir = torch.stack([-tar_dir[..., 1], tar_dir[..., 0]], dim=-1)   # rotate +90 deg in xy
        lat_left = torch.sum(root_vel[..., 0:2] * left_dir, dim=-1)
        r_step_left = 0.50 * speed_shaping(lat_left, 0.8) + 0.20 * facing_r + 0.30 * head_stab_r
        r_step_right = 0.50 * speed_shaping(-lat_left, 0.8) + 0.20 * facing_r + 0.30 * head_stab_r

        # --- dodge: keep opponent's sword tip away from own target bodies ---
        opp_tip = sword_tip_pos_list[1 - i]
        my_targets = self._rigid_body_pos_list[i][:, self._target_ids]
        threat_dist = torch.linalg.norm(opp_tip - my_targets, dim=-1).min(dim=-1).values
        avoid_r = 1.0 - torch.exp(-2.0 * threat_dist)
        im_hit = self.sword_hit_list[1 - i].squeeze(-1).float()

        r_advance = 0.50 * vel_toward_r + 0.20 * facing_r + 0.30 * head_stab_r
        r_retreat = 0.50 * vel_away_r + 0.20 * facing_r + 0.30 * head_stab_r
        r_stand = 0.60 * still_r + 0.20 * facing_r + 0.20 * head_stab_r
        # dodge: NO head-stability term — it would penalize the very evasive motion
        # dodging requires. Reward keeping the opponent's tip away + staying oriented.
        r_dodge = 0.60 * avoid_r + 0.40 * facing_r - 1.00 * im_hit

        all_r = torch.stack([r_advance, r_retreat, r_stand, r_lunge_u, r_lunge_g,
                             r_dodge, r_step_left, r_step_right], dim=-1)
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
            # Stash into _tb_scalars; the RLGPUAlgoObserver drains this into the
            # rl_games SummaryWriter each epoch, so it lands on the SAME wandb step
            # axis as reward/loss and is never dropped (logging directly via
            # wandb.log fights rl_games' tensorboard step and gets truncated).
            means = (self._drill_rew_sum / self._drill_rew_count.clamp_min(1)).cpu()
            if not hasattr(self, '_tb_scalars'):
                self._tb_scalars = {}
            for d, name in enumerate(DRILL_NAMES):
                if self._drill_rew_count[d] > 0:
                    self._tb_scalars[f"drills/{name}"] = means[d].item()
            self._drill_rew_sum.zero_()
            self._drill_rew_count.zero_()
            self._drill_log_n = 0

    def _drill_hit_done(self):
        # End the episode on a decisive sword contact so the agent commits to ONE
        # clean strike instead of farming reward by continuously poking the sword.
        #   lunge drills : end when the LEARNER lands a hit on the opponent
        #   dodge drill  : end when the OPPONENT lands a hit on the learner (dodge failed)
        if not hasattr(self, 'sword_hit_list'):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        learner_hit = self.sword_hit_list[0].squeeze(-1).bool()   # agent 0 hit opponent
        got_hit = self.sword_hit_list[1].squeeze(-1).bool()       # opponent hit agent 0
        lunge_mask = (self.drill_ids == D_LUNGE_UPPER) | (self.drill_ids == D_LUNGE_GROIN)
        dodge_mask = self.drill_ids == D_DODGE
        return (lunge_mask & learner_hit) | (dodge_mask & got_hit)

    def _compute_reset(self):
        # drills have no win conditions — only out-of-bounds, falls, and decisive hits
        game_done = torch.logical_or(self.out_bound, self._drill_hit_done())
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
        game_done = torch.logical_or(self.out_bound, self._drill_hit_done())
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
