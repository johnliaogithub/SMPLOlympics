# Fencing Drills — Version Changelog

Research log of the drill-conditioned low-level policy (`HumanoidFencingDrillsZ`).
Newest version on top. Append a new section per trained version.

NOTE on reproducibility: code was iterated in the working tree without a commit per
version, so exact v1-v4 reward code is NOT bit-recoverable from git — these notes are
the human record. Going forward: commit + tag per version, and behavior changes are
config flags (logged to W&B) rather than code replacements. See v5.

---

## v14 — retrain under the FIXED (swept) hit detector; v13 warm-start

**Motivation:** v1–v13 all trained under the tunneling hit detector (see the ENVIRONMENT FIX entry
below): fast committed thrusts didn't register, so the lunge learned SLOW contact. v14 is the first
drills version trained with the detector fixed on BOTH paths — `hit_swept=True` (episode/win
scoring) AND the lunge reward's own hit signal `_check_hit_subset` routed through the same swept,
thrust-gated, geometric test (gated to `hit_dist=0.15`, was a hardcoded 0.1). So a fast thrust that
reaches a target now earns `hit_r`.

- **Only change vs v13: the hit detector (env), not the reward weights.** All drill rewards, phase-C
  `drill_probs=[0.4,0.4,0.4,0.6,0.6,2.0,0.4,0.4]`, spawn jitter, foot/arm penalties are unchanged.
- **Warm-start:** from v13 @ 84000 (weights + opponent `_op` snapshot copied into the v14 dir);
  cumulative epoch counter continues. Launch: `bash scripts/fencing/train_fencing_drills.sh C
  max_epochs=90000` (= +6000; raise for longer).
- **Expectation / how to verify:** the lunge should shift from a slow reach toward a faster committed
  thrust (its `explosive_r` + per-step time penalty no longer fight an unrewarding hit signal).
  Confirm post-train with `phc/measure_tip_speed.py` — the registered-hit tip-speed distribution
  should climb above the old ≤5 m/s ceiling. If the lunge stays stuck slow (warm-start basin), retry
  warm-starting from v11 (pre-dodge-specialization) or a fresh phase A.

## ⚠️ ENVIRONMENT FIX — sword-tip TUNNELING in hit detection (affects ALL prior versions)

**What:** `check_sword_hit` (base `humanoid_fencing.py`) sampled the virtual sword tip ONCE per
30 Hz control step (`self.dt = control_freq_inv(2) × 1/60 = 33.3 ms`) and required that single tip
snapshot to be within `hit_dist` (0.15 m, measured to the target CENTER) AND a >50 N PhysX contact —
with NO CCD anywhere. So a fast thrust can cross a target's capture sphere entirely between two
samples and register nothing ("tunneling"). BOTH gates leak: the 30 Hz distance snapshot (33 ms) and
the 120 Hz contact force (thin blade, no CCD, 0.02 m contact offset).

**Proof (measured — LUNGE_UPPER drill, drills-v13, 128 envs; `phc/measure_tip_speed.py`):**
- Dead-center tunnel threshold: tip speed > 2·hit_dist/dt = **9.0 m/s** (lower for off-axis grazes).
- Real lunge tip speed reaches **p99 ≈ 22 m/s**; **2.2%** of frames move the tip > one full capture
  diameter (30 cm) in a single step.
- Every REGISTERED hit happened at tip speed **≤ ~5 m/s** — the detector was BLIND to fast committed
  thrusts and only "saw" slow lingering contact. This is the root cause of the blade-slap / slow-slug
  reward-hacking we kept patching downstream (thrust-only `tip_lead_margin`, etc.).

**Fix — `hit_swept` (single config bool; default False, scripts now set True):** the distance gate is
also evaluated against the tip's SWEPT PATH over the last control step (point-to-segment,
`prev_tip → cur_tip`), reset-safe (falls back to the instantaneous tip on any non-consecutive frame,
so a segment never spans an episode reset). A thrust is required — tip moved > `swept_min_disp`
(0.02 m) roughly along the blade axis (`align > swept_align_min` = 0.5) — so a broadside swing whose
tip grazes past does NOT score, and a STILL sword near the body has ~zero tip motion so it stays
silent (no phantom hits). The swept branch is geometric (no >50 N gate), so it also closes the
contact-force leak for committed thrusts. Control rate is untouched → PULSE/drills unaffected.

**Verified (same measurement, `hit_swept` False→True):** committed thrusts that reach a target and
register a hit: **2.3% → 55.2%**; total registered hit-frames **+112%**.

**Implication:** every scoring-dependent result before this (strategy scoring; the blade-slap→thrust
saga; dodge "safety", since the opponent's fast lunges also tunneled) is confounded and should be
re-evaluated with `hit_swept=True`. The env-reset-bug fix, passivity fixes, and footwork/spawn-
randomization work are orthogonal and stand.

## v13 — current: pure dodge via gated SLIP (not parry); v11 warm-start; 3000-epoch test

**Motivation:** v12's parry reward made the dodge reach its blade forward (see v12 outcome). Replace
it with a signal that rewards moving the BODY off-line, never the blade forward.

- **`dodge` reward (pure):** `0.2*facing + dodge_evade_w * avoid_r * gate - dodge_imhit_w*im_hit
  - dodge_footback_w*foot_back`. `avoid_r = 1 - exp(-2*threat_dist)` rewards the opponent's tip being
  FAR from your targets (i.e. you slipped the body off-line). **Gated** (`gate = gap <
  dodge_evade_gate=1.5 m`) so it only pays when the opponent is actually in range — fixes the old
  `avoid_r` far-farm (standing far no longer pays) AND, unlike the parry, it rewards moving the body,
  not the blade. Parry term removed (`dodge_parry_*` -> `dodge_evade_w=1.0`, `dodge_evade_gate=1.5`).
  Retreating turns the gate OFF (no reward) and trips the rear-foot penalty, so a slip beats a retreat.
- **Warm-start from `fencing_drills_v11`** (NOT v12): v12 collapsed to forward-lunge, so it's a bad
  base; v11's dodge (duel) is less committed to reaching forward. New dir `fencing_drills_v13`
  seeded from v11 @ 82000.
- **TEST RUN: 3000 epochs only** — just to see if the policy is trending the right way (slip, not
  reach), before committing to a full run.

**Command:** `bash scripts/fencing/train_fencing_drills.sh C
learning.params.config.max_epochs=85000`  (+3000 from epoch 82000).
**Verify:** `bash scripts/fencing/record_dodge_eval.sh` — success direction = blade does NOT extend
forward, body slips off-line, `GOT-HIT` trending down, no rear-foot retreat.

**Note:** if v11's forward-reach inertia is too strong to shed in 3000 epochs, warm-start instead
from a pre-duel checkpoint (v9-era, where the dodge wasn't attacking at all).

**Outcome:** _(fill in after the test run)_

---

## v12: PURE defensive dodge (parry/slip, no retreat; riposte = composition)

**Motivation:** the v11 "dodge" was a malformed riposte — the agent stepped the rear (left) foot
BACK while reaching the arm forward to touch. That's baked into the dueling reward: offense pulls
the tip forward, `-im_hit` is most cheaply satisfied by retreating, and applied *simultaneously*
they produce "back up + reach". A real riposte is sequential (parry, then step in). Fix: split the
roles. Make `dodge` a PURE defensive action; the riposte becomes the STRATEGY composing
`dodge -> lunge` (no new drill, so no obs-size change — and a pure dodge can't score, so it no
longer dominates `lunge` in the strategy's choices).

- **`dodge` reward (`+env.dodge_pure=True`), offense removed:**
  `0.2*facing + dodge_parry_w*parry - dodge_imhit_w*im_hit - dodge_footback_w*foot_back`.
  - **parry** (sparse, `dodge_parry_w=1.5`): +1/frame when the learner's blade (sampled hand/mid/
    tip points) comes within `dodge_parry_dist=0.15 m` of the opponent's blade — an active,
    non-farmable block signal (replaces the old farmable `avoid_r` distance reward).
  - **rear-foot-back penalty** (`dodge_footback_w=0.5`): penalizes the LEFT (rear) foot drifting
    backward from its spawn anchor (in the away-from-opponent direction). The torso may still lean
    back a little and the agent may move forward — but it can't retreat by stepping the rear foot,
    so the defense must be a parry/slip in place. Re-adds a per-agent `_left_foot_anchor_list`.
  - `im_hit` (`dodge_imhit_w=5.0`): got-touched penalty (defense signal).
- **More spawn randomization:** `lateral_spawn_jitter` 0.3->0.5 and new `heading_spawn_jitter=0.3`
  (yaw) so lunges arrive from more varied lines — so the dodges aren't all identical.
- **New dir `fencing_drills_v12`**, seeded from `fencing_drills_v11/*.pth` (v11 at epoch 82000,
  FROZEN). Phase C, warm-start.

**Command:** `bash scripts/fencing/train_fencing_drills.sh C
learning.params.config.max_epochs=88000`  (+6000 from epoch 82000).
**Verify:** `bash scripts/fencing/record_dodge_eval.sh` — success = defends WITHOUT stepping the
rear foot back (parry/slip, ideally blade-on-blade), varied across the randomized spawns, and NOT
reaching-while-retreating. (Riposte quality is then the strategy's job, tested later.)

**Watch:** (1) parry-farming — holding blades together to farm `parry_r` (gate on the opponent
lunging if seen); (2) pure-sparse learnability — if it won't learn from parry+im_hit alone, add a
*gated* "keep opp tip off my targets when close" term (not the old far-farm). Warm-start should carry it.

**Outcome:** FAILED — both fencers lunge forward. The offense removal + learner-hit-termination
removal were correct, but the **parry reward backfired**: rewarding "your blade near the opponent's
blade" (proximity) incentivizes extending your blade FORWARD to meet the incoming (lunging) blade,
and with the rear-foot-back penalty blocking retreat, forward is the only way to close that gap. So
the dodge reaches its sword out to "parry" — visually a lunge. Proximity is the wrong parry signal
(a real parry deflects sideways, not a forward poke). Fixed in v13.

---

## v11: dodge robustness — lateral spawns + randomized duel target (from v10)

**Motivation:** the dodge-eval diagnostic confirmed the v10 dodge is *one-directional* — a
consistent backward retreat to the same side, and weak against the LOW lunge. The retreat itself
is fine (fencing is linear; breaking distance is the textbook defense, so we do NOT penalize it or
force sideways dodging). The problem is it's overfit to one exact centerline, head-high lunge. v11
makes the defense generalize.

- **Lateral spawn randomization** (`+env.lateral_spawn_jitter=0.3`, base-env change): each fencer
  spawns with a per-env sideways x-offset, so the incoming lunge arrives off the centerline / at a
  varied angle (up to ~15-20 deg). The retreat/slip must then work from varied geometry instead of
  the identical head-shot. Gated by the config (default 0 = old centerline spawn).
- **Randomized learner duel target** (`_duel_target`, per env): the dueling-dodge offense is now
  `where(duel_target, lunge_reward(groin), lunge_reward(upper))` instead of always upper — so the
  counter isn't one canned high thrust, and (with the opponent already lunging up/low 50/50) both
  engage-high and engage-low get trained. Fixes "no defense against lunge lower."
- **New dir `fencing_drills_v11`**, seeded from `fencing_drills_v10/*.pth` (v10 at epoch 76000,
  left FROZEN). Phase C (dodge-focused), warm-start.

**Command:** `bash scripts/fencing/train_fencing_drills.sh C
learning.params.config.max_epochs=82000`  (+6000 from the copied epoch-76000 checkpoint).
**Verify:** `bash scripts/fencing/record_dodge_eval.sh` — want the defense to hold against BOTH
upper and groin lunges across the randomized spawns (not a fixed one-sided retreat that only beats
the head-shot).

**Known, NOT fixed here (documented):** the lunge target shift — the *reward* targets the nearest
of a body set while the *hit* uses the full torso column, so the upper lunge converges on the HEAD
(nearest along the rising arc) and the low lunge scores on the TORSO (the pelvis is too low to
reach). Emergent + a reward-target/hit-target mismatch, not a bug; narrow the target sets if a
foil-style chest target is wanted later.

**Outcome:** _(fill in after training)_

---

## v10: DUELING dodge (dodge + lunge in conjunction, warm-started from v9)

**Motivation:** dodge was the weakest drill — in isolation it's a one-sided, sparse reactive
task and the old reward (`0.6*avoid_r + 0.4*facing - im_hit`) let the agent farm `avoid_r` by
just standing far. Fix: make the dodge drill a real EXCHANGE where the learner both attacks and
defends, so the dodge is learned in context.

- **Reward (dodge drill, `+env.dodge_duel=True`):** `r_dodge = lunge_reward(upper_targets)
  - dodge_imhit_w * im_hit`. Dense **offense** = the existing lunge reward (keeps it committing
  to a real thrust), **sparse defense** = a per-frame "got touched" penalty (`dodge_imhit_w=5.0`).
  The dodge EMERGES from avoiding the opponent's touch while attacking — no farmable distance
  reward (the old `avoid_r`/`facing` are dropped for the duel). Matches real fencing: you read the
  opponent's whole-body commitment (in the obs), not the sword tip.
- **Opponent lunges** (option A): `_resample_drills` already sets the dodge opponent to lunge, so
  there's a competent incoming threat from step 0 (v9's lunge is good; warm-started).
- **Episode length 70** (`strike_episode_length` 90->70). VERIFIED: an isolated lunge lands at
  ~episode_step 44 (median; range 41-61, measured from a lunge-only recording). 70 is long enough
  for the incoming lunge to land + a counter, short enough that stalling/fleeing can't win (with
  the OOB penalty kept). The episode also ends on a touch by EITHER fencer (a point ends it).
- **Spawn-gap jitter** `dodge_spawn_jitter=0.2` (+/- 0.2 m) so the duel can't memorize one fixed
  timing/exchange.
- **New dir `fencing_drills_v10`**, seeded from `fencing_drills_v9/*.pth` (v9 at epoch 70000, left
  FROZEN). Phase C (dodge-focused) so the duel is heavily sampled; lunges kept for the opponent's
  threat + retention.

**Command:** `bash scripts/fencing/train_fencing_drills.sh C
learning.params.config.max_epochs=76000`  (resumes from the copied epoch-70000 checkpoint, +6000).
**Record:** `bash scripts/fencing/record_fencing_drills.sh`.

**Reward note (open fork):** offense is DENSE (lunge reward), defense is SPARSE (-im_hit). Dense
offense is deliberate — a fully-sparse duel (+hit - got_hit) risks the passive standoff that sank
the sparse strategy runs. To try fully-sparse instead, drop the lunge offense and reward only the
sparse hit events.

**Watch:** does a dodge motion actually emerge (agent evades the incoming lunge while landing its
own), or does it just trade hits recklessly (raise `dodge_imhit_w`) / turtle and never attack
(lower it)? Guard retention of the other drills.

**Outcome:** _(fill in after training)_

---

## v9: non-sword arm BACK (fencing form fix, warm-started from v8)

**Motivation:** strategy-v7 recordings showed a bad low-level habit — the agents **raise the
non-sword (left) arm forward** and use it to block; the only touches that landed hit that raised
arm, not the torso. In real fencing the free arm stays *behind*. This is a low-level form problem,
so we fix it in the drills rather than the strategy.

- **New experiment dir `fencing_drills_v9`**, seeded by copying all of `fencing_drills_v8/*.pth`
  (v8 trained to epoch 56000). Resumes from 56000. `fencing_drills_v8/` left FROZEN.
- **New reward term `arm_back_pen` (`+env.arm_back_pen=0.5`), applied to EVERY drill.** Penalizes
  the FORWARD-MOST of the non-sword arm chain (`L_Shoulder`, `L_Elbow`, `L_Hand`; sword is on
  `R_Hand`) being forward of the chest toward the opponent:
  `pen = 0.5 · clamp(max_j dot(joint_j−Chest, tar_dir) / 0.3, 0, 1)`. Zero when the free arm is
  at/behind the chest (natural stance), ramps to full when any joint is ≥0.3 m forward (raised into
  the strike zone / blocking). Uses the whole chain (not just the hand) because a hand-only penalty
  was gamed by tucking the hand into the armpit while the elbow juts forward. A real lunge throws
  the rear arm back too, so this reinforces form across all drills.
- Same as v8 otherwise (warm-start; checkpoint/config version, not a code-structure change).

**Command:** `bash scripts/fencing/train_fencing_drills.sh B
learning.params.config.max_epochs=62000`  (phase B = all drills, so the arm form is fixed across
the whole set; resumes from the copied epoch-56000 checkpoint, +6000).

**Watch:** does the free arm actually tuck back in a v9 recording? Guard that the arm penalty
doesn't distort the lunge (the sword arm is R_Hand, unaffected) or make dodge stiff. If the arm
stays raised, raise `arm_back_pen`; if the agent contorts to keep it back, lower it.

**Outcome:** _(fill in after training)_

---

## v8: dodge-focused fine-tune (warm-started from v7)

**Motivation:** dodge never learned under v7 (phase B, dodge at equal weight). Note: this is
NOT related to the strategy-net `win_frame_hit=45` bug — the drills use the *instantaneous*
`sword_hit_list` (single-frame) for the dodge hit/penalty (`humanoid_fencing_drills.py` lines
362, 438), never the 45-frame win accumulation. So dodge's failure is its own problem: it's a
hard reactive skill, `avoid_r ≈ 1` whenever the opponent is far (standing is rewarded most of
the time), and — structurally — the observation does NOT include the opponent's sword tip, so
the learner is partly blind to the incoming blade. v8 attacks the training-time budget half of
that (far more dodge reps); the obs blindness remains a known cap.

- **New experiment dir `fencing_drills_v8`**, seeded by copying all of `fencing_drills_v7/*.pth`
  (v7 trained to epoch 50000). Training resumes from epoch 50000. `fencing_drills_v7/` is left
  FROZEN (it was strategy-v1…v5's pinned low-level).
- **New phase C** (dodge-focused): `drill_probs=[0.4,0.4,0.4,0.6,0.6,2.0,0.4,0.4]` — dodge (idx 5)
  ~40% of samples, lunges kept high so the snapshot opponent stays a real lunging threat,
  basics/lateral low-but-nonzero for retention. Same reward code as v6/v7 (checkpoint/config
  version, not a code version).

**Command:** `bash scripts/fencing/train_fencing_drills.sh C
learning.params.config.max_epochs=56000`  (phase C; resumes from the copied epoch-50000
checkpoint, so max_epochs must exceed 50000 — 56000 = +6000).

**Watch:** dodge is the target, but guard retention — if advance/retreat/stand/step or the
lunges visibly degrade in a v8 recording, raise their `drill_probs` share. Dodge may still cap
low due to the opponent-sword-tip obs gap (a bigger, obs-size-changing fix, deferred).

**Outcome:** _(fill in after training)_

---

## v7: add dodge (warm-started from v6)

**Motivation:** v6 was trained phase A (no dodge) — `strategy-v1` was accidentally run on
it, so it was offense-only. v7 adds dodge WITHOUT touching v6 (which `strategy-v1` depends
on) by warm-starting a new experiment dir from v6's weights and training phase B.

- **New experiment dir `fencing_drills_v7`**, seeded by copying all of
  `fencing_drills_v6/*.pth` into it (v6 trained to ~epoch 40000). Training resumes from
  epoch 40000 and adds dodge (phase B). `fencing_drills_v6/` is left FROZEN as
  `strategy-v1`'s pinned dependency.
- **Same code as v6** (reward/env already support dodge). This is a checkpoint/model
  version, not a code version — reuses the `drills-v6` git tag; only the training config
  (phase B + warm-start) differs. Dodge can now learn against v6's competent lunge.

**Command:** `bash scripts/fencing/train_fencing_drills.sh B
learning.params.config.max_epochs=50000`  (phase B = all 8 drills incl dodge; resumes
from the copied epoch-40000 checkpoint, so max_epochs must exceed 40000).

**Outcome:** _(fill in after training)_

---

## v6: reintegration (full 8-drill set)

**Motivation:** the isolated lunge (v5) proved the motion is achievable. v6 puts the
v5 lunge reward back into the SHARED multi-drill policy to (a) confirm the lunge fixes
hold when the net also serves locomotion, and (b) test the real research question —
do the drills COMPOSE into fencing under the strategy net?

- Same reward code as v5-final (lunge reward: approach/explosive/thrust_align/aim/
  posture/split/hit/time/low_sword, two-phase recovery). No reward changes — only the
  drill distribution differs (full set vs lunge-only).
- New experiment dir `fencing_drills_v6` (v4/v5 checkpoints untouched).
- Phasing: `fresh` (phase A, no dodge) → `B` (add dodge, once lunge is a real threat).
- Per-drill strike settings carried over: `strike_spawn_half_dist=1.0`,
  `strike_episode_length=90`, `lunge_weights={posture:0.30}` (these only affect the
  lunge/dodge envs; locomotion drills use the global 1.5 spawn / 175-step episode).

**Watch:** reward-scale mismatch — the lunge's terminal +5 / two-phase dynamics share a
PPO batch with the smooth ~1/step locomotion drills; if locomotion degrades when lunge
is present, that's the cause (fix: per-drill advantage norm or reward rescale).

**Outcome:** _(fill in after training)_

---

## v5

**Lunge-isolated diagnostic** (separate experiment dir `fencing_lunge_only`, phase `L`,
`drill_probs=[0,0,0,1,1,0,0,0]`). Run to answer: is the lunge a reward problem or a
PULSE action-space limit? Iterations and findings:

- **Posture is reachable in PULSE.** With `+env.lunge_posture_weight=1.0`, the agent
  kept a straight back. => the upright lunge is NOT off-manifold; remaining issues are
  reward-shapeable. (`lunge_posture_weight` is now a config knob, default 0.20.)
- **Explosiveness term added.** Nothing rewarded strike SPEED, so the lunge was a slow
  reach/topple. Added `+0.30·explosive`: fast sword-tip speed toward the target, gated
  by `exp(-1.5·dist)` so it's the committed strike, not a wave from afar. Rebalanced
  lunge: `0.40·approach + 0.30·explosive + 0.15·thrust_align·gate + 0.10·aim +
  lunge_posture_weight·posture + 5.0·hit − 0.20`.
- **Two-phase lunge (recovery).** Observed: the agent reaches with the HAND, not a
  foot-forward lunge, and topples in. Fix: `lunge_two_phase=True` (default) — the
  episode no longer ends on the hit; it latches `_lunge_landed` and switches that env
  to a recovery reward `0.5·still + 0.3·posture + 0.2·facing` (return to balanced
  en-garde). A hand-reach-and-topple cannot recover, so requiring recovery pressures
  the agent into a real lunge stance. Episode ends on `strike_episode_length` timeout
  (use ~90 to fit lunge + recovery) or a fall.

- **Rear-foot pin.** Observed: with `lunge_posture_weight=1.0` it reaches+crouches
  (straight back, but no lunge step) — too close (1.75 m) to need a step, and posture
  at 1.0 over-rewards staying upright. Fixes:
  - Rear (left) foot PENALTY: `−0.30·(1 − exp(−5·foot_drift))`, drift from the foot's
    episode-start anchor. A penalty (not reward) so it can't be farmed by standing;
    it only bites on crouch-shuffle. Plants the back foot => front-foot lunge stance.
    (`L_Ankle`; flip to `R_Ankle` if the agent leads with the left foot.)
  - Training config (vs the `=1.0` diagnostic): `strike_spawn_half_dist=1.0` (2.0 m,
    so a stationary reach can't connect — must step) and `lunge_posture_weight=0.30`
    (upright still pays but doesn't dominate the time penalty / become a stand-farm).

- **Low-sword penalty.** With the foot pin + 2 m spawn, the agent learned to plant the
  sword as a "third leg" support strut to lean in. Penalty `−0.40·clamp((0.5 − tip_z)/0.5)`
  — bites only when the tip drops below 0.5 m (a ground-prop; pelvis target is ~0.9 m so
  a real groin thrust is unaffected). A penalty, so no new farm.
- **Foot SPLIT (replaces the rear-pin + front-forward pair).** The agent was leading with
  the LEFT foot (backwards for a right-arm lunge). Replaced both foot terms with one:
  `split·clamp((right_foot − left_foot)·tar_dir / 0.7, −1, 1)` — positive when the right
  foot is ahead (correct stance), negative when the left is. Net simplification: removed
  `_left_foot_anchor_list` + `_prev_foot_pos_list` tracking. **Result: a real lunge motion.**
- **Config consolidation.** All lunge reward weights now live in one dict
  `DEFAULT_LUNGE_WEIGHTS` (approach/explosive/thrust_align/aim/posture/split/hit/time/
  low_sword). Override a subset from the CLI: `+env.lunge_weights="{posture:0.3,split:0.5}"`.
  Replaces the scattered `lunge_posture_weight` / `lunge_split_weight` flags. The whole
  weight set is logged to W&B config as one object.

**Training command (post-consolidation):** `bash scripts/fencing/train_fencing_drills.sh L
learning.params.config.max_epochs=<cur+10000> +env.strike_spawn_half_dist=1.0
+env.strike_episode_length=90 +env.lunge_weights="{posture:0.30}"`

**Reproducibility:** set `+env.lunge_two_phase=False` to recover the v4-and-earlier
behavior (lunge episode ends immediately on the hit). The flag is logged in W&B config.

**Training command:** `bash scripts/fencing/train_fencing_drills.sh L
learning.params.config.max_epochs=<cur+10000> +env.strike_spawn_half_dist=1.0
+env.strike_episode_length=90 +env.lunge_posture_weight=0.30`

**Outcome:** _(fill in after training)_

---

## v4

**Motivation:** v3 collapsed to standing for all drills; the lunge walked-and-hovered
instead of thrusting.

**Reward Changes**
- Removed `head_stab` from locomotion drills (advance/retreat/step_left/step_right).
  It made standing a local optimum (a small step loses more head-stability than it
  gains in velocity). Now `0.70·velocity + 0.30·facing`. `stand` keeps `head_stab`.
  *→ fixes the standing collapse.*
- Lunge approach is now POTENTIAL-BASED shaping `Φ(s')−Φ(s)`, `Φ=exp(−0.7·dist)`
  (tip 0.7 + front-foot 0.3), replacing absolute `exp(−dist)`. Telescopes, so
  hovering near the target earns ~0. *→ fixes walk-and-hover.*
- Lunge hit is a large TERMINAL bonus `1.0 + clamp(force/300)` (1.0–2.0) at weight
  1.0, up from `0.5 + 0.5·force` at weight 0.4. Landing the strike now dominates.
- Sword "thrust line" captured at step `sword_ref_step=15` instead of at reset
  (agents spawn with the sword pointing at the ground).
- Small upright-POSTURE bonus (0.15) on every drill: `clamp((chest-pelvis).z, 0, 1)`
  = 1 when the spine is vertical, dropping as the torso folds forward. Fixes the
  bent-over torso when retreating / lunging (was toppling backward).
- Lunge AIM term (0.10), active only for the first `sword_ref_step` steps: rewards
  pointing the blade at the nearest target point, so the thrust line captured at
  step 15 is on-target. Hands off to `thrust_align_r` after step 15.
- Anti-farm rebalance of the lunge (the agent was standing and posing — pointing
  the blade at the low groin target and holding it — because dense per-step rewards
  accumulated over the episode beat the one-time hit, and hitting ends the episode):
  - lunge = `0.50·approach + 0.20·thrust_align·close_gate + 0.10·aim + 5.0·hit`.
  - removed `facing` from the lunge; removed `posture` from the lunges (it was an
    always-positive standing payout).
  - `thrust_align` is GATED to pay only while closing (`approach_r > 0`).
  - hit weight raised 1.0 → 5.0 so the strike dominates.
  - Net effect: standing earns ~0 (not negative — no suicide incentive), closing +
    hitting earns ~10. Only striking is profitable.
- Walk-and-drag fix (at 4k epochs the net defaulted to "walk forward + drag the
  sword" for every drill except stand):
  - Lunge approach is now TIP-ONLY (removed `foot_prog`). The foot term let the
    lunge farm approach by walking the foot toward the opponent with the blade
    dragging — never using the sword — which (together with advance) made
    "walk forward" the net's default. Now approach is earned only by bringing the
    sword TIP to the (raised) target, so the blade must be lifted and aimed.
  - Removed `facing` from step_left/step_right: rewarding facing while stepping
    laterally is the definition of orbiting. Pure lateral velocity only.
- Lunge MOTION quality (it landed hits but by hunching the back + slow creep, not a
  lunge): added `+0.20·posture` (upright spine — penalizes folding over to reach)
  and a `-0.20` per-step time cost (rewards hitting FAST = explosive). They cancel
  for an upright stand (~0, no farm/suicide) but a fast upright lunge scores far
  above a slow hunch.
- Sidestep still spiralled inward: gated the lateral reward by
  `exp(-3·speed_toward²)` so any forward/backward drift kills it → straight-line
  sidestep instead of walking forward around the opponent.

**Env / episode**
- `strike_spawn_half_dist=0.875` → strike drills spawn **1.75 m apart** (was 2.5 m).
- `strike_episode_length=50` → lunge/dodge time out fast (forces a lunge, not a walk-in).
- Global `episode_length=175` (was 200).

**Training:** fresh (no warm-start), for clean attribution of the redefined reward.

**Outcome:** 

---

## v3

8-drill net (added `step_left`, `step_right`), trained fresh.

**Outcome:** collapsed to standing for ALL drills (head_stab standing-trap); lunge
walked forward and hovered the tip just outside hit range to farm the absolute
approach reward without triggering the hit-termination.

---

## v2

6→8 drill net via warm-start surgery (`expand_drills_checkpoint.py`) from v1.

**Outcome:** lunge waved the sword in place (farmed the old velocity-thrust term from
distance); reward-component logging fixes; dodge didn't move (non-threatening
opponent + head_stab penalizing evasion).

---

## v1

First drills net: 6 drills (advance, retreat, stand, lunge_upper, lunge_groin, dodge).
Phase A (drills 0–4 vs frozen opponent) → Phase B (+dodge vs lunging opponent).

**Outcome:** locomotion drills worked; lunge "just walked forward" (old reward
rewarded body velocity toward opponent).
