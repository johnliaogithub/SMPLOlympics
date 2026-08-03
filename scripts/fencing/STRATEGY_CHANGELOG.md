# Fencing Strategy (High-Level Policy) — Version Changelog

Research log of the HIGH-LEVEL policy (`HumanoidFencingStrategyZ` +
`phc/train_fencing_strategy.py`). This is the top of the HRL stack: it picks a
DRILL every `macro_K` steps; a FROZEN low-level drill policy executes it. Versioned
independently of the drills — each entry records which `drills-v*` it was built on.
Newest version on top.

---

## ⚠️ CRITICAL BUG FIX (invalidates strategy-v1 … v6.2) — missing env reset

**Every strategy version up to and including v6.2 trained on broken environments.** The custom
PPO loop (`train_fencing_strategy.py`) calls `env.reset()` exactly once at startup; `macro_step`
then steps via `step_z()`, which — unlike the rl_games play loop that trains the drills
(`self.env_reset(done_indices)` after every step, see `amp_agent.py:243` / `common_agent.py:324`)
— **never resets finished envs.** So after the first bout ended in each env (within the first
rollout), that humanoid was never repositioned: it stayed out-of-bounds / fallen / overlapping,
`reset_buf` recomputed to True every subsequent step, and from ~iteration 2 onward essentially
ALL training experience was post-terminal garbage. Signatures this produced: `mean_bout_len`
pinned at 1, `time_pen` constant at 32, old `episodes_ended` constant at 8192 (= 256×32), OOB
end-rate dominating (a stuck OOB body reports OOB forever), win/loss ≈ 0.01. Recordings looked
fine ONLY because the recorder stops at the first termination, so it never ran long enough to
expose the missing reset.

**Fix:** `macro_step` now resets the just-ended envs each physics step
(`reset_ids = step_done.nonzero(); if len: self.reset(reset_ids)`), exactly mirroring the
rl_games loop. All outcome/length capture stays before the reset (reads the pre-reset state).

**Implication:** all v1–v6.2 conclusions (passive equilibrium, "attack↔defense deadlock",
reward-shaping effects) are CONFOUNDED and should be treated as unreliable. Re-run from a clean
baseline. Recommended: re-run the v6.2 config (identical reward) now that the env actually resets
— that is the first *valid* strategy training. Watch `mean_bout_len` climb well above 1 as the
first confirmation the fix works.

---

## strategy-v7 — current: offense-only dense + win_frame_hit curriculum

**Built on:** `drills-v8`, on the reset-fixed env (v6.3 was the first valid run). v6.3's clean
data exposed the real problem: `win_rate = loss_rate = 0` for the ENTIRE run — the policy
converged to a passive face-and-wait-for-timeout (timeout_rate → 0.86), and `sparse_return` rose
2.8 → 3.9 purely from **dense-farming**: `dense_mix·dense` ≈ 7/rollout (mostly the always-on
`facing` term) vs a time penalty that can only claw back ~2.3 and is capped by the suicide ceiling
(`time_pen_w < 1/bout_len`). No `time_pen_w` can beat the facing farm — the dense reward itself
was paying for standing still.

**Change 1 — offense-only dense (`+env.offense_only_dense=True`).** The dense signal mixed into
the PPO reward is now ONLY the components that require acting on the opponent:
`reward_s·strike + reward_t·terminate + reward_h·hit` — strike = contact force on THEIR target
zones minus force on yours (offense+defense), terminate = the score, hit = tip→target proximity.
Dropped: `vel` (raw approach velocity — farmable locomotion that drove the walk-into-contact) and
`facing` (~1 just for standing and facing). Standing-and-facing now earns ~0 dense; the only way
to raise it is to act on the opponent. Plumbing: `humanoid_fencing._compute_reward` stashes
`self._last_reward_raw` (`[vel,facing,strike,terminate,hit]`); `macro_step` builds the sum from it. `dense_mix` stays 0.05 (kept low so a tip-hover farm can't out-earn a
real touch once the time penalty is netted in); watch `dense_return` vs `win_rate` for a hover farm.

**Change 2 — win_frame_hit curriculum (`+env.win_frame_hit_start=1`,
`win_frame_hit_ramp_steps=700000`, end `win_frame_hit=5`).** Starts lenient (a touch scores after
~2 on-target forceful frames) so the agent gets EARLY scoring signal, then ramps linearly to 5
(a committed touch) over 700k physics steps (~iter 1460 at 480 steps/iter). Ramps on
`step_counter` inside `macro_step`; recording sets no `_start`, so it uses a constant 5. New
`strategy/win_frame_hit` metric logs the live value.

**Why both:** offense-only dense gives a gradient toward attacking; the curriculum makes the first
touches actually reachable so that gradient gets reinforced by real +1s instead of dying in the
"never lands 6 frames" gap. Time + contact penalties are unchanged — but now the time penalty
finally works, because with no facing farm a passive bout is genuinely net-negative.

**Command:** `bash scripts/fencing/train_fencing_strategy.sh +strategy.iters=3000`
(writes `fencing_strategy_v7/`). **Record:** `bash scripts/fencing/record_strategy.sh`
(constant `win_frame_hit=5`, 5 consecutive bouts).

**Outcome:** _(fill in — the key question: does `win_rate` finally leave 0? Watch it against the
`win_frame_hit` ramp — expect touches to appear while it's low (1–2), then a possible dip as it
climbs to 5. If `dense_return` rises but `win_rate` stays 0, it's hovering the tip, not scoring →
lower `dense_mix` or gate hit on closing. If touches vanish entirely once win_frame_hit hits 5,
5 is unreachable for this low-level and épée-targets / a lower end value is the next lever.)_

---

## strategy-v6.3: FIRST VALID run (v6 config + the env-reset fix)

**Identical configuration to v6 / v6.2** (`win_frame_hit=5`, `dense_mix=0.05`, `time_pen_w=0.005`,
contact penalty + body-contact termination, drills-v8 low-level, fresh, 3000 iters) — written to
its own dir `fencing_strategy_v6.3/`. The ONLY substantive difference vs v6.2 is the **critical
env-reset fix** above: `macro_step` now resets finished envs each step, so this is the **first
strategy run that trains on a correctly-resetting environment**. Treat v6.3 as the real baseline;
v1–v6.2 are confounded.

**First thing to check when it starts:** `mean_bout_len` should be well above 1 (was pinned at 1),
and the per-cause split should spread out instead of OOB-dominating. If `mean_bout_len` is still 1,
the fix didn't take — stop and investigate before anything else.

The strategy recorder (`visualize_strategy.py`) now plays **5 consecutive bouts** back-to-back
(configurable via `+env.n_record_bouts`), resetting between them (it needs the same explicit reset,
since `step_z()` doesn't self-reset), each labeled `BOUT k/5` with its verdict.

**Command:** `bash scripts/fencing/train_fencing_strategy.sh +strategy.iters=3000`
(writes `fencing_strategy_v6.3/`). **Record:** `bash scripts/fencing/record_strategy.sh`.

**Outcome:** _(fill in — this is the first run whose numbers are trustworthy.)_

---

## strategy-v6.2: re-run of v6 (fresh) with the new instrumentation

**strategy-v6.2 is the EXACT SAME configuration as strategy-v6** — identical reward and
task: `win_frame_hit=5`, `dense_mix=0.05`, `time_pen_w=0.005`, contact penalty + body-contact
termination, on the `drills-v8` low-level. Trained **fresh** for 3000 iterations. It writes to
a SEPARATE dir (`fencing_strategy_v6.2/`) so the original `fencing_strategy_v6/` run is not
overwritten. The ONLY difference from v6 is added **logging instrumentation** — `mean_bout_len`
(unconfounded per-env step counter) and the per-cause termination split (`contact_end_rate`,
`oob_end_rate`, `timeout_rate`, real `loss_rate`); removed the constant `episodes_ended` /
`dense_mix_w` graphs (`dense_mix` remains in the W&B run config). Logging changes do NOT affect
training — the policy dynamics are identical to v6. See the v6 entry below for all rationale.

**Command:** `bash scripts/fencing/train_fencing_strategy.sh +strategy.iters=3000`
(writes `fencing_strategy_v6.2/`, fresh). **Record:** `bash scripts/fencing/record_strategy.sh`.

**Outcome:** _(fill in — first read `mean_bout_len` + `contact_end_rate` vs `oob_end_rate` to
settle whether bouts are short and whether contact or OOB dominates the endings.)_

---

## strategy-v6: WINNABLE touch (lower win_frame_hit) + metric split

**Built on:** `drills-v8` (dodge-focused fine-tune of v7). Carries all of v5 (contact penalty,
`dense_mix=0.05`, body-contact termination, `time_pen_w=0.005`). Fixes a fundamental,
previously-invisible bug in the win condition that affected **every** strategy version so far.

**Also this version:** the loss metric is now split — `strategy/loss_rate` counts only a REAL
opponent touch (`red_win`), while `strategy/bad_end_rate` counts out-of-bounds + body-contact
ends, and `strategy/draw_rate` the timeouts. v5's `loss_rate` lumped all three, which is why it
looked alarmingly high when it was mostly the fencers wandering off the 2 m strip, not being
touched. `macro_step` stashes `_last_win/_last_redloss/_last_badend`; the trainer tallies each.

**Is `win_frame_hit=5` cheating? No — if anything it's stricter than real fencing.** Electronic
scoring registers a touch in ~2–15 ms (épée ~2–10, foil ~14); at 30 Hz one frame is 33 ms, so
even ONE frame already exceeds a real touch's minimum contact time. 45 frames (1.5 s) is ~100×
too long; 5 frames (167 ms) is a conservative "committed touch." The real anti-cheat guard is
unchanged: tip <0.1 m AND >50 N force — a genuine forceful on-target contact. (The rigid,
non-bending blade makes SUSTAINED contact physically harder than a real flexing foil, another
reason 45 was unreachable; adding blade flex is a soft-body modeling change we're not doing.)

**The bug.** A "win" was never a touch — [`humanoid_fencing.py`](../../phc/env/tasks/humanoid_fencing.py)
scores it as `sword_hit_history > win_frame_hit` with `win_frame_hit = 45`: **more than 45
cumulative hit-frames**, where a hit-frame = sword tip within 0.1 m of a target AND >50 N
contact force. That's ~1.5 s of forceful on-target contact. A quick thrust lands a few frames
and prints "Green Hit"/"Red Hit" but accumulates maybe 2–5 of the 46 frames needed, so
`green_win`/`red_win` **almost never fire**. Consequences seen at v5 iter 2999:
- `win_rate` ~0 — real touches never reach the threshold.
- `loss_rate` high — but mostly the v5 `bad_end`s (out-of-bounds on the 2 m strip + body
  contact), which are counted as losses, NOT opponent touches.
- `episodes_ended` pinned at 8192 = 256 envs × 32 macro-steps — every env terminates within
  every 15-step window, i.e. bouts are extremely short (walking out of the narrow strip fast).
- Recording: "Red Hit" then "Green Hit", neither ends the bout, then OUT OF BOUNDS at step 62.

This likely explains a lot of the whole passivity saga: **if a touch can't win, attacking has
no payoff, so passivity/standoff is rational.** The 45-frame threshold was an invisible ceiling
under v1–v5.

**Change.** `win_frame_hit` is now a config (default unchanged at 45); the strategy trains and
records with `+env.win_frame_hit=5` — a committed touch (≥6 on-target forceful frames) scores.
Aligns the win with fencing (a solid touch = a point) and finally gives the strategy an
achievable win signal. Base task and the drills are untouched (they keep the 45 default / use
their own drill-hit logic). **Tuning:** if `win_rate` is still ~0, lower toward 2–3; if wins
look flukey/grazing, raise. Watch that `green_win` vs `red_win` are now both reachable (a real
duel), and that `loss_rate` split shifts from `bad_end`s toward genuine opponent touches.

**Command:** `bash scripts/fencing/train_fencing_strategy.sh +strategy.iters=10000`
(writes `fencing_strategy_v6/`; `dense_mix=0.05`, `time_pen_w=0.005`, `win_frame_hit=5` baked in).
**Record:** `bash scripts/fencing/record_strategy.sh` (also uses `win_frame_hit=5` so hits end bouts).

**Outcome:** _(fill in — does `win_rate` finally lift off 0? read the new split:
`bad_end_rate` should fall (they stop wandering off / clinching), `loss_rate` now means real
opponent touches, `draw_rate` = timeouts. Tune `win_frame_hit` (down to 2–3 if wins stay ~0,
up if wins look like grazes).)_

---

## strategy-v5: cost-of-existence (per-step time penalty)

**Built on:** `drills-v7`. Carries v4 (contact penalty + `dense_mix=0.05` + body-contact
termination); adds a per-step existence penalty. Separate version — changes the reward baseline.

**Motivation (from watching v4 ~iter 2500–3000).** The hard body-contact wall stopped the
clinching, but they found a NEW passive equilibrium: one fencer stands, the other approaches
and **retreats just before entering range**, neither committing. Diagnosis (confirmed in
`compute_fencing_reward`): the dense reward pays for *inaction* — `facing_reward ≈ 1.0` just
for facing the opponent (`0.1/step`), `vel_reward` pays the approacher, `hit_reward` pays for
hovering the tip near the target. Over a 175-step bout ×`dense_mix=0.05` the facing term alone
farms ≈ +0.6–0.9 — comparable to a whole win — so the standoff is *positive-EV*. The bout has
no clock pressure, so dragging it out is free.

**Change: `time_pen_w=0.005` subtracted per LIVE physics step** (accumulated in `macro_step`,
like the contact penalty). This is the same "cost of existence" trick that made the lunge drill
explosive instead of a slow creep: a passive full bout now nets slightly negative (≈ −0.2 to
−0.3 after the dense farm offset), while a quick *scoring* bout stays clearly positive (≈ +0.9)
because it ends before the penalty accumulates. Only ending the bout by SCORING comes out ahead.

**Two escape hatches this could open — both closed:**
1. **Fleeing.** The strip is only 2 m wide (`x∈[−1,1]`, spawn at x=0) and out-of-bounds ended
   the bout at outcome 0 — so a per-step penalty would teach "step 1 m sideways and leave." Now
   the **learner going out of bounds is treated as a loss** (−`contact_term_pen`), folded into
   the same `bad_end` override as body contact.
2. **Suicide.** If the existence cost over a full bout exceeded the loss magnitude (−1), the
   agent would prefer to be touched fast. `time_pen_w=0.005 × 175 ≈ 0.875 < 1.0`, so a passive
   drag stays *better* than a loss — no suicide incentive. **Do not raise `time_pen_w` above
   ~0.0057** without also raising the loss magnitude, or this flips. New `strategy/time_penalty`
   metric logs the mean accrued cost; watch `loss_rate` for a suicide signature (spikes with a
   short episode length).

**Knobs:** `+env.time_pen_w=` (existence cost/step). **Tuning tension to watch:** the facing
farm (~0.005/step after `dense_mix`) and the suicide ceiling (0.0057/step) bracket this term
tightly. If they still won't attack at 0.005, the cleaner next move is to cut the *inaction
farm at the source* — lower `dense_mix`, or mix only the OFFENSIVE dense components (hit +
terminate, dropping facing/vel/strike) so the dense reward can't pay for standing at all. That
would be strategy-v6 if needed.

**Command:** `bash scripts/fencing/train_fencing_strategy.sh +strategy.iters=10000`
(writes `fencing_strategy_v5/`; `dense_mix=0.05`, `time_pen_w=0.005` baked in).
**Record:** `bash scripts/fencing/record_strategy.sh`.

**Outcome:** _(fill in — does `win_rate` finally climb / bouts get shorter and more decisive?
watch `strategy/time_penalty` (should fall as bouts shorten), `loss_rate` (suicide check), and
whether they now flee — if `bad_end`s dominate, tighten or reconsider.)_

---

## strategy-v4: body-contact TERMINATION (hard wall)

**Built on:** `drills-v7`. Carries v3 (contact penalty + `dense_mix=0.05`); adds a hard
terminal on body contact. Separate version because it changes the episode dynamics, not just
a weight — v3 kept running (dense mix alone didn't stop the closing).

**Motivation.** v3's soft contact penalty + dense mix still let the fencers walk into each
other: the video showed the opponent *start* a lunge, then both close past blade distance,
collapse together, and topple. The soft penalty only nudges; closing was still the policy's
move. So make body contact **terminal and costly**, not merely discouraged.

**Change.** If the two fencers' horizontal root gap drops below `contact_term_dist=0.4` m
(bodies colliding), the bout **ends immediately** and **both** get `−contact_term_pen=1.0`
(≈ as bad as being touched). Implemented in `_compute_reset` (adds `_body_contact` to the
reset condition so the env actually resets) and `macro_step` (a body-contact end with no
valid touch overrides the outcome to `−1`; a genuine touch still scores normally). The v3
soft penalty (`contact_pen_w=0.05` from 0.6 m) is kept as a smooth approach gradient *before*
the wall. Knobs: `+env.contact_term_dist=`, `+env.contact_term_pen=`. Net intent: milling into
a clinch is now a loss, so the only positive-EV behavior left is to score from blade distance
(which `dense_mix` rewards) — the passive standoff and the slugging match are both punished.

**Recording fix (`visualize_strategy.py`).** The clip now **ends at the first bout
termination** and freezes ~1 s on the last live frame, labeled with the verdict (GREEN/RED
scores, BODY CONTACT, OUT OF BOUNDS, or TIMEOUT). Before, it ran a fixed 600 frames and the
env auto-reset through many bouts, so a hit looked like a teleport rather than an ending —
which is why v2/v3 clips "didn't end" on a hit even though training did terminate on it.

**Note — sparse vs. original reward:** the ±1 the strategy optimizes is a *wrapper I built*
around SMPLOlympics' original touch detection (`green_win`/`red_win`); the original fencing
task itself trained on the DENSE reward (`0.1·vel + 0.1·facing + 0.2·strike + 1.0·terminate
+ 0.6·hit`), which is exactly what `dense_mix` now folds back in.

**Command:** `bash scripts/fencing/train_fencing_strategy.sh +strategy.iters=10000`
(writes `fencing_strategy_v4/`; resume with `+strategy.resume=True`).
**Record:** `bash scripts/fencing/record_strategy.sh`.

**Outcome:** _(fill in — watch: does `win_rate` finally climb, or do body-contact terminations
just replace draws (check the new `strategy/contact_penalty` and how often bouts end in
contact)? if they end every bout in a clinch, `contact_term_dist=0.4` may be too generous or
they can't attack from range — try tightening it or raising `dense_mix`.)_

---

## strategy-v3: contact penalty + dense-reward mix

**Built on:** `drills-v7` (`output/HumanoidIm/fencing_drills_v7/Humanoid.pth`). Trained
**fresh** (contact penalty carried over; dense mix is new).

**Motivation (from strategy-v2's result + video).** v2 (contact penalty alone) converged to
`win_rate ≈ 0.02` — a passive standoff. Video: the fencers circle and slowly close (much
slower than the pre-penalty shoving, so the contact penalty *did* dampen the collision) until
their blades overlap to the armpit, "dance" together, and topple. Crucially they get **close
enough to hit but decline to thrust** — under pure sparse win/loss a thrust risks a
counter-touch (a loss) while milling guarantees a safe draw, so non-aggression is the
equilibrium. The blades pass each other rather than tips-to-target — they aren't even aiming.

**Change: mix the dense fencing reward into the PPO objective.** The env already computes the
original dense reward (`0.1·vel + 0.1·facing + 0.2·strike + 1.0·terminate + 0.6·hit`) and
returns it from `macro_step`; the trainer now adds it to the sparse reward with weight
`dense_mix=0.05`:  `reward = (win/loss − contact_pen) + 0.05·dense`. The `strike`/`hit`/
`terminate` terms give a gradient toward putting the tip on target and actually landing the
touch, so aggression pays even when a clean win doesn't materialize — directly attacking the
"close enough but won't thrust" behavior. `dense_mix` is the primary knob (`+strategy.dense_mix=`).

**NOT included:** fall penalty — deliberately skipped. The v3 video shows falling is a
*byproduct* of over-closing, not a strategic dive, so fixing non-aggression should remove it;
no need to add a separate fall term (revisit only if falls persist as an evasion).

**Also this version (infra):**
- **Resume-from-checkpoint** (`+strategy.resume=True` → auto `strategy.pth`, or a path):
  restores net + optimizer + iter. v2 died to a native segfault (GPU contention) after ~4
  days and lost everything past the last 500-step save; resume prevents that.
- **Optimizer state is now saved** in every checkpoint (was net + iter only).
- **Clean logging:** `win_rate`/`loss_rate` now count the penalty-free outcome (v2 bundled the
  contact penalty into the counted reward, contaminating loss_rate). New `strategy/contact_penalty`
  logs the mean contact penalty separately, and `strategy/dense_mix_w` records the mix weight.
- Scripts `mkdir -p /tmp/j73liao` defensively (a tmp-cleaner wiped it, crashing env XML write).

**Command:** `bash scripts/fencing/train_fencing_strategy.sh +strategy.iters=10000`
(low-level defaults to drills-v7; writes `fencing_strategy_v3/`; `dense_mix=0.05` baked in).
Resume: append `+strategy.resume=True`.

**Record:** `bash scripts/fencing/record_strategy.sh`.

**Outcome:** _(fill in after training — watch: does `win_rate` climb above v2's 0.02 plateau?
does `dense_return` rising coincide with `win_rate` rising, or does dense get farmed without
wins? does `contact_penalty` stay low? if they start circling harder — the dense `vel/facing`
terms can reinforce orbiting — cut `dense_mix` or switch to an offense-only dense.)_

---

## strategy-v2: contact penalty (isolated fix)

**Built on:** `drills-v7` (`output/HumanoidIm/fencing_drills_v7/Humanoid.pth`) — the
dodge-added low-level. Trained **fresh** (not warm-started from strategy-v1, which was a
degenerate passive policy — warm-starting would carry that habit).

**Diagnosis (revised, from watching strategy-v1 bouts).** The collapse is not (only) an
abstract "passive standoff." Empirically the fencers *close all the way in and shove each
other* — a body-to-body slugging match — instead of keeping blade distance and thrusting.
The low-level drills reward closing, and nothing penalized the two torsos overlapping, so
the high-level policy's cheapest way to interact was to walk in and push.

**Fix (isolated on purpose).** A single new penalty: while a match is live, each physics
step where the two fencers' **horizontal root gap < `contact_pen_dist=0.6` m** accrues a
cost, summed over the `macro_K` window and subtracted from the sparse reward with weight
`contact_pen_w=0.05`. Only bites at body overlap (~0.6 m); a legit lunge extends the blade
~1 m while torsos stay >1 m apart, so real attacks are untouched. Implemented in
`HumanoidFencingStrategyZ.macro_step`. Overrides: `+env.contact_pen_w=`, `+env.contact_pen_dist=`.

**Deliberately NOT included (deferred unless still needed):** the two other anti-passivity
fixes proposed for the sparse collapse — (1) mixing a small dense reward into the sparse
objective, and (2) making a fall end-the-bout-with-a-penalty. Isolating the contact penalty
first keeps attribution clean: if pushing was the real cause, this alone should restore
blade-distance fencing. Add the other two only if the standoff persists.

**Command:** `bash scripts/fencing/train_fencing_strategy.sh +strategy.iters=10000`
(low-level defaults to drills-v7; writes to `fencing_strategy_v2/`).

**Record:** `bash scripts/fencing/record_strategy.sh` (defaults to
`fencing_strategy_v2/strategy.pth` on drills-v7).

**Outcome:** Converged to a low-win equilibrium, then the process died (native segfault
around it 8860 — no Python traceback, ran ~4 days; last checkpoint `strategy_00008500`,
likely GPU contention on the shared box, not a code bug). Trajectory: `win_rate` rose
0 → ~0.02 by it ~4780 and **flatlined there for the next 4000 iters**; `sparse_return`
climbed −0.73 → +0.355 and plateaued. So the objective improved and the policy *converged*,
but the improvement did **not** turn into wins — the fencers essentially stop scoring.
Read: isolating the contact penalty answered the question — **the shoving was NOT the whole
story.** Removing the collision incentive did not produce a fencing match; it left (or
produced) a passive standoff. The `sparse_return` climb is consistent with the policy
learning to stop incurring the contact penalty (stop overlapping) while *not* learning to
attack. => the contact penalty alone is insufficient; the two deferred fixes (dense-reward
mix + fall penalty) are now warranted. Visual confirmation of "did they at least stop
shoving?" pending a recording of `strategy_00008500` (contact-penalty vs win_rate are not
logged separately, so the video is the disambiguator). NOTE: `sparse_return` bundles the
contact penalty into the outcome, so its absolute value isn't a clean win/loss read — log
`contact_pen` separately next run.

---

## strategy-v1

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

**DEPENDENCY (pinned):** built on `drills-v6` **phase A — NO dodge**, low-level
checkpoint `output/HumanoidIm/fencing_drills_v6/Humanoid.pth` (v6 trained to ~40000
epochs). `fencing_drills_v6/` is frozen; the dodge-added model is a SEPARATE version
(`drills-v7`) so this dependency stays unambiguous. Also pinned in the run's W&B config
(`low_level_checkpoint`).

**Outcome:** win_rate rose to ~0.21 then collapsed to ~0.05 and plateaued — the classic
sparse-self-play PASSIVE equilibrium (both fencers learn to *not lose* → mutual standoff,
mostly timeouts). Two contributing gaps identified: (1) pure sparse win/loss gives no
gradient toward aggression; (2) falls are neither penalized nor terminal, so a downed
fencer is hard to touch — "fall to avoid losing" is a safe outcome. Planned fixes for
strategy-v2: mix a small dense reward into the sparse objective, and make a fall
end-the-bout-with-a-penalty. Also: strategy-v1 ran WITHOUT dodge (forgot phase B), so it
was offense-only.
