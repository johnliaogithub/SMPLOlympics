# Fencing Drills — Version Changelog

Research log of the drill-conditioned low-level policy (`HumanoidFencingDrillsZ`).
Newest version on top. Append a new section per trained version.

---

## v4 — current

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
