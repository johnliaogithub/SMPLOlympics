# Fencing Drills — Version Changelog

Research log of the drill-conditioned low-level policy (`HumanoidFencingDrillsZ`).
Newest version on top. Append a new section per trained version.

---

## v4 — current

**Motivation:** v3 collapsed to standing for all drills; the lunge walked-and-hovered
instead of thrusting.

**Reward**
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
- Lunge AIM term (0.20), active only for the first `sword_ref_step` steps: rewards
  pointing the blade at the nearest target point, so the thrust line captured at
  step 15 is on-target. Hands off to `thrust_align_r` after step 15.

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
