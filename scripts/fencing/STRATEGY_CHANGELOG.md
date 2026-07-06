# Fencing Strategy (High-Level Policy) — Version Changelog

Research log of the HIGH-LEVEL policy (`HumanoidFencingStrategyZ` +
`phc/train_fencing_strategy.py`). This is the top of the HRL stack: it picks a
DRILL every `macro_K` steps; a FROZEN low-level drill policy executes it. Versioned
independently of the drills — each entry records which `drills-v*` it was built on.
Newest version on top.

---

## strategy-v1 — current

**Built on:** `drills-v6` (`output/HumanoidIm/fencing_drills_v6/Humanoid.pth`).

**Architecture.** A small MLP observes the game state and outputs a discrete choice
over the drills, re-decided every `macro_K=15` steps (0.5 s @ 30 Hz). The frozen
low-level (loaded via `FrozenLowLevelPolicy`, weight-shape reconstruction of the
drills actor) converts (obs + drill one-hot) → Z each physics step; PULSE decodes Z.
Trained with a self-contained minimal PPO (`train_fencing_strategy.py`), NOT rl_games.

**Reward.** SPARSE win/loss (+1 learner scores, −1 opponent scores, 0 else) — a real
fencing match with win conditions. The original dense fencing reward is computed and
logged (`strategy/dense_return`) for comparison only; it never enters the gradient.

**Setup.** Self-play (both fencers use the strategy net; opponent runs no-grad).
`env.task=HumanoidFencingStrategyZ`, `episode_length=175`, `macro_K=15`.

**Known limitations / caveats:**
- **Offense-only.** drills-v6 was trained phase A (no dodge), so the low-level cannot
  dodge; the strategy net will learn dodge is a dead action and avoid it. Add dodge
  via v6 phase B, then retrain the strategy.
- **First real execution of the strategy PPO** — treat the first run as a debug pass:
  verify the `[LowLevel] loaded (obs_dim=…, Z_dim=32)` print, the `macro_step` buffer
  shapes, and that win_rate climbs against a fixed-drill opponent before trusting
  self-play.
- Watch for self-play collapse to a degenerate equilibrium (both standing / both
  spamming one drill).

**Success signal:** `strategy/win_rate` climbing = the drills COMPOSE into fencing
(the research payoff). Flat win_rate = either a PPO bug or genuine non-composition
(isolated drills failing in a live bout) — the latter is itself a finding.

**Command:** `bash scripts/fencing/train_fencing_strategy.sh
output/HumanoidIm/fencing_drills_v6/Humanoid.pth +strategy.iters=10000`

**Outcome:** _(fill in after training)_
