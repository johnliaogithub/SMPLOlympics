# Dodge / lunge exchange investigation (from 2026-10-13)

A running log of the tests behind the planned v20: a short exchange window so the lunge must be
explosive and the dodge is one explosive step back, plus fall termination with a penalty for a
lunger that falls.
Each entry records the date, what was run, and the result.

## Plan
1. Reconcile two measurements of v19 that disagree:
   - `record_dodge_eval.sh` (agent 0 dodges): dodger hit in **1/10** bouts.
   - `phc/verify_selfplay.py` (agent 0 lunges, agent 1 dodges, 256 envs): lunge landed in **99.6%**.
2. Measure how long a full-speed lunge takes to land on a stationary target.
3. Measure how far a dodger can get from the attacker in N steps.
4. Pick the exchange window from 2–3, add fall termination and a fall penalty (audit E4), and set up v20.

---

## 2026-10-13 — Step 1: the dodge eval's bout loop
`phc/visualize_dodge_eval.py` runs one env per bout and forces the roles every step
(learner = dodge, opponent = live lunge, `opp_frozen=False`). It counts any opponent hit during the
bout, and ends the bout when the env sets `reset_buf` (got hit, or the 70-step strike timeout), or
after a maximum of 90 steps. No problem found in the loop itself.

The next suspect is the role split (who is agent 0 vs agent 1), tested below.

## 2026-10-13 — Step 1 result: the dodge eval is right; the env is asymmetric between sides
- **Budget probe with the dodge eval's roles** (`verify_dodge_reward.py`, learner dodges as agent 0,
  256 envs, v19): dodger hit **4%**, retreat **2.70 m**. This matches the dodge eval (1/10, ~3 m).
- **2×2 side-swap test** (`phc/verify_roles.py`, v19 for both fencers, fixed 70-step windows, no
  drill-based endings, 1,024 bouts per case):

| match-up | attacker lands | median first-hit step | defender moves away |
|---|---|---|---|
| agent 0 lunges vs agent 1 still (Z=0) | **99.8%** | 20 | −0.12 m |
| agent 1 lunges vs agent 0 still (Z=0) | **79.1%** | 22 | −0.18 m |
| agent 0 lunges vs agent 1 dodging | **99.7%** | 22 | 1.13 m |
| agent 1 lunges vs agent 0 dodging | **15.6%** | 26 | 2.64 m |

- **Conclusion:** the same network performs better as agent 0 in both roles, even against a target that
  never moves. So agent 1's body or observations differ from agent 0's. The learner is always agent 0,
  so v19's "lunge beats dodge 99.6%" was agent 0 beating a handicapped agent 1, and vice versa.
  Neither earlier number measured the policy fairly.
- **Next:** find the side-specific difference (actor creation, per-actor physics properties,
  collision filters, observations).

## 2026-10-13 — Narrowing the side asymmetry (`phc/verify_symmetry.py`)
- **Mirrored-state test (both fencers running the same drill):**
  - Setup: agent 1 is forced to be the exact 180° rotation of agent 0 at reset.
  - Self-observations are identical (Δ ≈ 4e-5), and the two stay mirror images over 60 steps
    (root drift ≤ 7 cm).
  - The only observation difference is contact-force readings: ±hundreds of N of step-to-step
    PhysX contact noise, with no consistent bias toward either side.
- **Mirrored starts, asymmetric matchup (one lunges, the other still), 1,024 bouts each:**
  - agent 0 lunging lands **100%**; agent 1 lunging lands **77%**.
  - This holds even from exactly mirrored starts, so spawn differences are ruled out.
  - The hit detector is fine: when the tip gets within 0.15 m it registers 97–100% for either side.
  - The difference is behavioral: agent 1's tip gets within range only 77% of the time (median
    closest approach 0.109 m vs 0.089 m).
- **Contact observations are not the cause:** zeroing them for both fencers left the gap unchanged
  (100% vs 75%). My first hypothesis (contact obs) was wrong.
- **Implication:** with contacts zeroed, both attackers get identical observations, so the network
  outputs identical actions. The asymmetry must be downstream of the policy: the PULSE latent decoder,
  the PD/torque controller, or the physics.

## 2026-10-13 — Root cause of the side asymmetry: the strip-boundary observation is world-ordered
- **No-contact bias test** (mirrored starts, both fencers run the same drill, 40 steps):
  - advance: the −y fencer travels +7.9 cm further (SE 0.3 cm).
  - retreat: the −y fencer retreats **0.51 m further** (1.22 vs 0.72 m, SE 0.4 cm).
- **Side swap:** putting agent 0 at +y flips the result exactly (±0.5056 m). The bias follows the
  **side of the strip**, not the actor.
- **Not GPU overlap or contact overflow:** identical with `num_envs=1` (−0.49 / +0.50 m) and 8.
  No overflow warnings in the training logs.
- **Cause, confirmed:** the bounding-box observation (`compute_fencing_observation`) encodes the strip
  with only two corners, `[-1,-7]` and `[1,7]`, in a fixed world order. For a mirrored pair:
  - agent 0 at y=−1.5: `[−5.5, +1.0, +8.5, −1.0]` (behind-corner first);
  - agent 1 at y=+1.5: `[+8.5, −1.0, −5.5, +1.0]` (ahead-corner first; slots swapped).
  - Max difference 14. After the policy's ±5 obs clip, agent 1 effectively reads "the wall behind me
    is in front of me".
  - My first mirrored test missed this because contact noise (~78–231 N) dominated its
    largest-difference list.
- **Impact:** the learner always trains as agent 0 (−y), so every self-play opponent so far ran with
  flipped boundaries:
  - the dodge's training attacker in v17–v19;
  - v19's dodging target;
  - agent 1 in every eval.
  - This explains the 100% vs 77% lunge, the 4% vs 99.6% dodge results, and the retreat-distance gap.
- **Planned fix:** order the corners by the fencer's own facing (behind-corner first). Agent 0's
  observation is unchanged, so checkpoints stay valid; agent 1 gets the layout the network expects.

## 2026-10-13 — Fix applied and verified: `bbox_obs_facing_sorted` (uncommitted, awaiting OK)
- **Change** (`phc/env/tasks/humanoid_fencing.py`): the two strip corners are ordered by each fencer's
  own facing (behind first) before building the observation. New config `bbox_obs_facing_sorted`,
  default **True**. The compiled observation function is unchanged.
- **Verified:**
  - Mirrored pair: boundary observations now identical (max |Δ| = 0.000).
  - Agent 0's full observation is bit-identical with the fix on vs off (max |Δ| = 0), so existing
    checkpoints stay valid.
  - Retreat side bias gone: both sides 1.134 m. Paired difference −0.0005 m (was −0.506 m); swapped
    +0.0001 m.
  - 2×2 side test (v19, 1,024 bouts each):

| match-up | before fix | after fix |
|---|---|---|
| agent 0 lunges vs agent 1 still | 99.8% | 99.8% |
| agent 1 lunges vs agent 0 still | 79.1% | **100.0%** |
| agent 0 lunges vs agent 1 dodging | 99.7% | **45.2%** (dodger retreats 2.71 m) |
| agent 1 lunges vs agent 0 dodging | 15.6% | **42.7%** (dodger retreats 2.72 m) |

- **True v19 state:** the lunge lands ~44% of 70-step bouts against a live dodger, and the dodger
  walks back ~2.7 m. This matches what was seen in the video.
- **Still to do:** past self-play training (v17–v19 dodge attacker, v19 dodging target) used the
  flipped-boundary opponent. v20 is the first version to train against a correctly-observing opponent.

## 2026-10-13 — Steps 2–3: exchange timing (`phc/verify_timing.py`, v19, symmetric env, 1,024 bouts each)
- **Lunge vs still target:**
  - Lands **99.9%**. First-hit step **p10 16 / p50 21 / p90 25** (≈0.53 / 0.70 / 0.83 s).
  - Scales with spawn gap: 1.5–1.8 m → 16 steps; 1.8–2.0 → 19; 2.0–2.2 → 22; 2.2–2.6 → 25.
- **Lunge vs live dodger:**
  - Lands **46.9%**, only late: first hit at step p10 25 / p50 28 / p90 31. No hits after ~35.
  - Dodger distance moved away (mean): t5 0.07, t10 0.15, t15 0.25, t20 0.40, **t25 0.60, t30 0.84**,
    t40 1.38, **t70 2.71 m**. It walks back at ~0.9 m/s after step 20.
- **Attacker falls** (strict: a non-foot body <0.15 m or head <0.6 m) are rare in 70-step windows:
  0.4% vs still, 0.1% vs dodger. The "lunger falls" seen in the video was recorded *before* the
  boundary fix, when the lunger was the handicapped +y fencer. To recheck on the fixed env.
- **Landing rate if the bout were cut at W steps** (vs still / vs dodger):

| W (steps) | 15 | 18 | 20 | 22 | 25 | 28 | 30 | 35 | 70 |
|---|---|---|---|---|---|---|---|---|---|
| vs still | 6.6% | 26.3% | 41.3% | 59.6% | 90.9% | 99.9% | 99.9% | 99.9% | 99.9% |
| vs dodger | 0.4% | 0.6% | 0.7% | 0.9% | 8.3% | 30.2% | 40.7% | 46.9% | 46.9% |

- **Window candidates:** W≈25–28 steps (0.83–0.93 s).
  - At W=28 a full-speed lunge on a still target lands 99.9%, while the dodger has time for only
    ~0.6–0.8 m, about one step, not a 2.7 m walk.
  - W=25 is stricter: it pushes the lunge to speed up (only 91% vs still at current speed).

## 2026-10-13 — Decision for v20
- **Exchange window W = 28 steps.**
- **No fall penalty:** strict-definition falls are 0.1–0.4% of bouts.
- **Rear-foot penalty stays off.**
- The boundary fix (`bbox_obs_facing_sorted`) is committed with v20.
- **Verified before launch** (`phc/verify_exchange.py`):
  - dodge bouts end at ≤28 steps (28 when not hit);
  - lunge-vs-dodger bouts end at 28;
  - lunge-vs-still bouts end at 70;
  - the +20 survival bonus is paid at step 28.
  - v19 under these rules: the lunge lands only ~10% against the dodger within 28 steps.

## 2026-10-20 — v20 evaluation
- **Symmetry** (`verify_roles.py`, 1,024 bouts each) holds:
  - lunge vs still 100.0% / 100.0% (attacker as agent 0 / agent 1);
  - lunge vs dodger 14.5% / 13.0%;
  - the dodger retreats ~3.1 m over 70 steps.
- **Timing** (`verify_timing.py`, v20):
  - **Lunge vs still:** landed 100%; first hit p10 16 / p50 21 / p90 24. It's faster than v19:
    97.0% landed by step 25 (v19 90.9%).
  - **Lunge vs dodger:** landed **11.6% by step 28** (v19: 30.2%) and 13.9% by step 70.
  - Dodger moved away: t10 0.13, t20 0.46, **t25 0.69, t30 0.99**, t70 3.12 m.
  - **Attacker falls** in 37% of 70-step bouts vs a dodger, but at **median step 60** (p10 51) and
    **0% within the first 28 steps**, i.e. only once the chase outlasts anything it trains on.
    No fall penalty is needed for the 28-step drills.
- **256-episode reward-component eval** (`phc/eval_dodge_components.py`,
  `scripts/fencing/eval_dodge_components.sh`):
  - Checks: the env reward matches the recomputation exactly, and the component residual is 4e-6.
  - Survived 89.1%; got hit 10.9% (median hit step 26).
  - **The off-line term is dead:** it's a constant 0.100/step in every episode. The 1.5 m gate never
    opens because the fencers stay 1.9–2.3 m apart for the whole bout, so it always pays
    `far_value × evade_w`.
  - Facing (≈0.19 vs 0.19), posture (≈0.15 vs 0.15) and free-arm (≈0) are indistinguishable between
    outcomes. **Within a bout, only the terminal ±20 separates success from failure.**
  - **How it survives:** it moves back **0.91 m** by step 28 (hit bouts: 0.28–0.35 m). Sideways
    movement ≈ 0 (−0.035 m), and it stays on the attack line (off-line ≈ 0.1 m in both outcomes).
    It keeps the gap at ~2.15 m; in hit bouts the attacker closes it to ~1.9 m.
  - The attacker's sword comes up and points at the dodger within ~5–10 steps: off-line distance
    1.93 m (t1) → 0.52 (t5) → 0.11 m (t10).

---

## 2026-10-21 — v22 eval: stand narrow and fidgety, lunge sags forward after landing
Probe: 256 bouts each, one episode per env, v22 env (stand-policy opponent, head-height fall ending).
Also traced the exact recorder setup (1 env, test mode; `+env.viz_trace=True` now covers the lunges).

**Stand:**

| | v21 | v22 |
|---|---|---|
| foot split, median at t60 | 0.00 m | 0.28 m (target 0.45) |
| bouts with split > 0.25 m | 0% | 69% |
| left foot ahead | 19% | 0% |
| sword term | 0.92 | 0.87 |
| still term | 0.95 | ~0.60 |
| both feet planted | 1.00 | 0.73–0.93 |

- `drills/stand` plateaued at ~0.90 from ~60% of the run.
- Recorder bout: split 0.33 m by t10, then slides back to ~0.18 m while drifting 0.4 m backward.
  It takes a half stance and then shuffles.

**Lunge** (stationary opponent):
- Still lands 100% at step ~20 with a 0.8–0.9 m split. No falls (head never < 0.5 m) in 100 steps,
  in the probe or the recorder.
- After landing it never recovers. The spine tilts forward over the rest of the bout, and the clip
  ends at step 70 mid-lean, which looks like a fall:

  | | v21 | v22 |
  |---|---|---|
  | spine_up at t65, upper | 0.94 | 0.88 |
  | spine_up at t65, groin | 0.92 | 0.80 (~37°) |

- The groin lunge keeps creeping in after the hit (recorder: gap 1.65 → 0.95 m).
- Vs a dodger: lands 16% (v21 in the same env: 23%). `drills/lunge_*` drifted down 0.71 → 0.69 over the run.
- **Cause (reward):** the recovery reward is `0.5·still + 0.3·posture + 0.2·facing`. "Still" pays for
  freezing in the deep lunge, so nothing asks it to bring the feet back to guard.
