# Registry of curriculum learning configurations for HumanoidFencing.
#
# Each entry documents:
#   description  — what this version tests
#   hypothesis   — what we expected would happen
#   outcome      — what actually happened (fill in after training)
#   stages       — list of curriculum stages in order

CONFIGS = {
    "v1": {
        "description": "Baseline curriculum: locomotion → engagement → combat",
        "hypothesis": "Gradually introducing combat signals prevents the agents getting stuck on zero reward.",
        "outcome": (
            "Agents fall to the ground in Stage 2 (mean episode length ~50, vs ~300 no-curriculum). "
            "Root cause: vel/facing downweighted to 0.10 in Stage 2, weakening the locomotion gradient "
            "and letting the PULSE Z-latent drift toward a falling pose. terminate_reward weight of 1.0 "
            "also dominated (zero-sum), adding noise without useful signal."
        ),
        "stages": [
            {
                "step_threshold": 0,
                "label": "locomotion",
                "reward_weights": {"reward_f": 0.35, "reward_v": 0.50, "reward_s": 0.00, "reward_t": 0.00, "reward_h": 0.15},
                "enable_win_conditions": False,
            },
            {
                "step_threshold": 10000,
                "label": "engagement",
                "reward_weights": {"reward_f": 0.15, "reward_v": 0.25, "reward_s": 0.10, "reward_t": 0.50, "reward_h": 0.50},
                "enable_win_conditions": True,
            },
            {
                "step_threshold": 30000,
                "label": "combat",
                "reward_weights": {"reward_f": 0.10, "reward_v": 0.25, "reward_s": 0.20, "reward_t": 1.00, "reward_h": 0.60},
                "enable_win_conditions": True,
            },
        ],
    },

    "v2": {
        "description": "Fix locomotion collapse: maintain vel/facing throughout all stages",
        "hypothesis": (
            "v1 decimated vel (0.50→0.10) and facing (0.35→0.10) in Stage 2, removing the gradient "
            "that kept agents upright. v2 keeps vel>=0.35 and facing>=0.25 in all stages so the "
            "PULSE Z-latent never loses its locomotion signal. terminate_reward halved to reduce "
            "zero-sum noise dominating the gradient."
        ),
        "outcome": None,  # fill in after training
        "stages": [
            {
                "step_threshold": 0,
                "label": "locomotion",
                "reward_weights": {"reward_f": 0.35, "reward_v": 0.50, "reward_s": 0.00, "reward_t": 0.00, "reward_h": 0.15},
                "enable_win_conditions": False,
            },
            {
                "step_threshold": 10000,
                "label": "engagement",
                "reward_weights": {"reward_f": 0.25, "reward_v": 0.35, "reward_s": 0.05, "reward_t": 0.30, "reward_h": 0.50},
                "enable_win_conditions": True,
            },
            {
                "step_threshold": 30000,
                "label": "combat",
                "reward_weights": {"reward_f": 0.25, "reward_v": 0.35, "reward_s": 0.10, "reward_t": 0.50, "reward_h": 0.60},
                "enable_win_conditions": True,
            },
        ],
    },
}
