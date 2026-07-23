# Fencing Strategy (High-Level Policy) — Version Changelog

Research log of the HIGH-LEVEL policy (`HumanoidFencingStrategyZ` +
`phc/train_fencing_strategy.py`). This is the top of the HRL stack: it picks a
DRILL every `macro_K` steps; a FROZEN low-level drill policy executes it. Versioned
independently of the drills — each entry records which `drills-v*` it was built on.
Newest version on top.

---

## strategy-v5 — current: cost-of-existence (per-step time penalty)

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
