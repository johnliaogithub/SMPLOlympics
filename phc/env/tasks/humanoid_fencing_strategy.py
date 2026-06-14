# High-level strategy task for hierarchical fencing RL.
#
# This is a REAL fencing match (win/loss resets enabled), but the per-step Z
# actions come from a FROZEN low-level drill policy (trained via
# humanoid_fencing_drills). A separate strategy network (trained by
# scripts/fencing/train_fencing_strategy.py) chooses, every macro_K physics
# steps, which drill each agent executes. The environment exposes `macro_step`,
# which runs macro_K low-level steps for a fixed pair of drill choices and
# returns the sparse win/loss reward plus the dense fencing reward (for logging).
#
# Buffer-shape conventions (verified against the codebase):
#   obs_buf  : (num_envs * num_agents, obs_dim)  rows [0:N]=agent0, [N:2N]=agent1
#   rew_buf  : (num_envs * num_agents,)
#   reset_buf, progress_buf, green_win, red_win : (num_envs,)  (a match resets as a whole)
#   green_win == agent 0 (learner) scored;  red_win == agent 1 (opponent) scored.

import torch
import torch.nn.functional as F

from phc.env.tasks.humanoid_fencing import (
    HumanoidFencing,
    compute_humanoid_reset,
    compute_humanoid_reset_z,
)
from phc.env.tasks.humanoid_fencing_drills import HumanoidFencingDrills, NUM_DRILLS
from phc.utils.flags import flags


class HumanoidFencingStrategy(HumanoidFencingDrills):
    """Drill-conditioned obs (from drills) + real win-condition resets + dense
    fencing reward in rew_buf for comparison logging."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        assert self.obs_v == 1, "Strategy low-level loader assumes flat obs (obs_v=1)"
        # Strategy matches are full combat: use the original dense fencing weights for
        # the comparison reward we log. (Sparse win/loss is computed in macro_step.)
        self.reward_weights = {"reward_f": 0.10, "reward_v": 0.10, "reward_s": 0.20,
                               "reward_t": 1.00, "reward_h": 0.60}
        # Filled in by the training script before rollouts begin.
        self.low_level_policy = None
        self.macro_K = cfg["env"].get("macro_K", 15)

    def _compute_reward(self, actions):
        # Dense fencing reward (original formulation) for logging/comparison only.
        HumanoidFencing._compute_reward(self, actions)

    def _compute_reset(self):
        # Real match: out-of-bounds, a scored win, or a fall ends the episode.
        game_done = torch.logical_or(self.out_bound, torch.logical_or(self.red_win, self.green_win))
        reset_fn = compute_humanoid_reset_z if (self.step_counter > self.warmup_time or flags.test) else compute_humanoid_reset
        self.reset_buf[:], self._terminate_buf[:] = reset_fn(self.reset_buf, self.progress_buf,
                                                       self._contact_forces_list, self._contact_body_ids,
                                                       self._rigid_body_pos_list,
                                                       self._strike_body_ids, self.max_episode_length,
                                                       self._enable_early_termination, self._termination_heights, self.num_agents)
        self.reset_buf[:], self._terminate_buf[:] = torch.logical_or(self.reset_buf, game_done), torch.logical_or(self._terminate_buf, game_done)
        return


class HumanoidFencingStrategyZ(HumanoidFencingStrategy):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return

    def step(self, actions):
        # Z actions for BOTH agents come pre-computed from the frozen low-level
        # policy (see macro_step); no opponent freezing here.
        self.step_z(actions)
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()
        return

    @torch.no_grad()
    def macro_step(self, learner_drills, opp_drills):
        """Run macro_K low-level physics steps with fixed drill choices.

        Returns (sparse_reward, dense_reward, done), each shape (num_envs,):
          sparse_reward : +1 learner scored, -1 opponent scored, 0 otherwise
                          (captured at the step the match ends; 0 if no end in window)
          dense_reward  : summed original dense fencing reward for the learner
          done          : episode ended within this window
        """
        assert self.low_level_policy is not None, "Attach env.task.low_level_policy first"
        N = self.num_envs
        self.drill_ids[:] = learner_drills
        self.opp_drill_ids[:] = opp_drills

        sparse = torch.zeros(N, device=self.device)
        dense = torch.zeros(N, device=self.device)
        done = torch.zeros(N, dtype=torch.bool, device=self.device)

        for _ in range(self.macro_K):
            obs = self.obs_buf  # (2N, obs_dim)
            z0 = self.low_level_policy(obs[:N], self.drill_ids)
            z1 = self.low_level_policy(obs[N:2 * N], self.opp_drill_ids)
            self.step(torch.cat([z0, z1], dim=0))

            # green_win/red_win hold the pre-reset outcome until the next physics
            # step recomputes them, so they are valid to read here even though the
            # env has already auto-reset the finished matches.
            step_done = self.reset_buf.bool()
            newly = step_done & (~done)
            outcome = self.green_win.float() - self.red_win.float()  # +1 / -1 / 0(timeout|oob)
            sparse = torch.where(newly, outcome, sparse)
            dense = dense + self.rew_buf[:N] * (~done).float()
            done = done | step_done

        return sparse, dense, done
