# Fencing Environment Audit (2026-09-30)

Audit of the SMPLOlympics fencing env (`HumanoidFencing*`, drills + strategy). Every item says
**how** it was verified. Nothing below has been fixed yet except E0, which was fixed on 2026-09-28.

- **Probes:** `phc/verify_env_bugs.py` (definitive tests T1–T7), `phc/audit_env.py` and `phc/audit_floor.py` (surveys).
- **Setup:** drills-v14 low-level, 256 envs. Unless noted, runs are in the post-warmup regime that training spends more than 99% of its time in.
- **Origin:** "upstream" means the issue is present in the original SMPLOlympics release commit `3d99e8e` (Luo et al., July 2024).
- **Paper:** SMPLOlympics, arXiv 2407.00187 (fencing described in §4 and supplement B).

## Summary

| ID | Issue | Origin | Verified by | Severity |
|----|-------|--------|-------------|----------|
| E1 | Win counter wiped for all envs whenever any env resets | upstream | runtime unit test + impact test (drills and strategy envs) | **Critical** for strategy |
| E2 | Hit detector reads last step's positions and forces (1-step lag) | upstream | runtime test (exact equality) | Medium |
| E3 | Sword + right hand weigh 0.25 g; stance uses the sword as a load-bearing cane | upstream asset | sim mass readout + force measurement + counterfactual | **High** (stand), mild (dodge) |
| E4 | Fall termination off after warmup and in eval, contradicting the paper | upstream | runtime A/B/C test | Medium (low for current policies) |
| E5 | Swept-hit branch scores some air hits | **ours** (2026-09-28) | runtime, against validated body geometry | Low |
| E6 | `measure_tip_speed.py` / `measure_avoid_r.py` never reset finished envs | **ours** (tooling) | runtime test | Affects past numbers only |
| E0 | Fast thrusts tunnel past the 30 Hz tip check (no CCD) | upstream | runtime (fixed via `hit_swept`) | Fixed |
| T1 | Drills dodge trains against a frozen v2/v3-era opponent that never lands (added 2026-10-06) | ours (training setup) | weight comparison + runtime | **Fixed in v17** (`opponent_sync_frequency`) |
| E7 | Strip-boundary observation is world-ordered: the +y fencer sees the corners swapped (added 2026-10-13) | upstream | mirrored-state runtime tests | **Fixed in v20** (`bbox_obs_facing_sorted`) |

The following were checked and ruled out:
- **Virtual tip placement:** it sits within 2 cm of the blade mesh end.
- **Hit force attribution:** instantaneous hits are real sword contacts (median 46 N on the sword; 0% under 1 N).
- **Spawn facing:** errors stay within the configured jitter (mean 12°, max 45°).

---

## E1 — Win counter wiped globally (critical)
- **What:** `_reset_ref_state_init` runs `self.sword_hit_history[:] = 0`. This zeroes every env's win counter, not just the envs being reset. The win condition is `green_win/red_win = hit_frames > win_frame_hit`.
- **Verified (T1):** set every env's counter to 1.5, then reset only env 0. **All 255 other envs dropped to 0**, in both the drills env and the strategy env.
- **Impact (strategy env, 256 envs, win = hit-frames > 2, random drills, 5,966 finished bouts):** a correct per-env counter records **1,850 wins**, but the env registered **61**, so **1,789 wins (97%) were swallowed**. Every strategy result that uses wins as reward or termination is confounded. The drills env has no win condition, so its reward is unaffected.
- **Proposed fix (not applied):** `self.sword_hit_history[env_ids] = 0`.

## E2 — Hit detector lags one step
- **What:** `HumanoidFencing.post_physics_step` calls `check_game_state()` (the hit check) *before* `super().post_physics_step()`, which is where `_refresh_sim_tensors()` runs. `_physics_step` only refreshes DOF state.
- **Verified (T2), comparing what the detector saw against the simulator:**
  - Tip position: the maximum difference from the previous step is **0.0**; the mean difference from the current step is 8.2 cm.
  - Contact forces: the maximum difference from the previous step is **0.0**; the mean difference from the current step is 6.5 N.
- **Impact:**
  - Hit flags (`sword_hit_list`, the dodge's `im_hit`, got-hit termination) arrive one step (33 ms) late.
  - They disagree with the reward, which reads fresh state.
  - `_check_hit_subset`, the lunge's `hit_r`, runs inside the reward and is fresh. So the two hit signals are offset by a frame.
- **Proposed fix:** run `check_game_state()` after the tensor refresh.

## E3 — Weightless sword used as a cane
- **What:** upstream replaced the right hand's box geom with sword meshes at `density="1"`.
  - Simulator masses: **R_Hand + sword = 0.00025 kg**; L_Hand = 0.159 kg; a real foil is about 0.5 kg.
  - The paper says only "a sword (replacing the right hand)"; mass isn't discussed.
- **Load-bearing (T7):** in 85% of `stand` frames the tip is on the floor.
  - The floor pushes the blade straight up (z-share 1.00) with a median **68 N ≈ 10% of all vertical ground force**.
  - Retreat, step-left and dodge show 5–9% when the tip is down.
  - Tip-on-floor rates: stand 85%, dodge 40%, step_left 21%, retreat 23%.
- **Relied on (counterfactual):** B = blade removed from the asset; C = blade kept but the hand-contact observation (column 483) zeroed. They differ only in physical support. Same policy and seeds.
  - **stand:** fallen frames 0.45% → **11.4%**; head height 1.48 → **1.24 m**; root speed 0.12 → **0.50 m/s**.
  - **dodge:** fallen 0.02% → 0.71% (mild). Retreat and step_left: no dependence.
  - A (normal) vs C: identical, so the policy isn't keying off the contact observation. The effect is physical support.
- **Not tested, not claimed:** that weightlessness causes the extreme tip-speed spikes.
- **Proposed fix (changes dynamics; drills must be retrained):**
  - realistic sword and hand mass;
  - penalize or terminate blade–floor contact.
  - Open question: the PULSE prior was trained without a sword. Check whether it tolerates the added hand mass.

## E4 — Fall termination disabled after warmup and in eval
- **What:** the Z, Drills and Strategy `_compute_reset` use `compute_humanoid_reset_z`, which is timeout-only.
  - This applies when `step_counter > warmup_time`, with `warmup_time = 250/dt ≈ 7,500 steps ≈ 234 epochs`, or whenever `flags.test` is on.
  - The base fencing reward accepts its fall flags as arguments but never uses them.
  - **The paper states: "The episode terminates if either of the humanoids falls or steps out of bounds."** The strategy env's own comment says the same, but the code doesn't do it.
- **Verified (T4, falls induced with random latent actions):**
  - Pre-warmup: 100% of fall frames end the episode.
  - Post-warmup: 0.3% (coincident with other terminations); fencers lie down for up to 142 steps.
  - Eval: same as post-warmup.
- **Impact:** low for current policies (falls in 77 of 153,600 agent-steps), but the rule is unenforced for all training after the first ~234 epochs.
- **Proposed fix:** always use `compute_humanoid_reset`, which terminates on falls.

## E5 — Swept branch scores some air hits (ours)
- **What:** the swept branch (`hit_swept`) measures the tip path against target *joint origins* within `hit_dist`, with no force gate. So a path can score while missing the actual body.
- **Geometry validation:** I rebuilt the target shapes from the XML.
  - Each origin lies inside its own shape.
  - In Newton-matched sword↔target contacts, the physical blade is within 1 cm of the rebuilt surface in 49% of cases (median 1.5 cm). The rest are contacts that weren't actually blade-on-that-target.
  - Conclusion: the rebuild is accurate to about 1–2 cm.
- **Result (3,941 swept-only hits):** distance of the tip path from the real body:
  - median 0; p90 3.2 cm;
  - more than 3 cm: 11.0%; **more than 5 cm: 3.7%**; more than 10 cm: 0%.
  - Instantaneous hits: 1.9% beyond 3 cm.
- **Proposed fix:** test the swept segment against the true shapes (the distance code is in `verify_env_bugs.py`).

## E6 — My diagnostics skipped resets (tooling)
- **What:** `step_z` never resets finished envs (T6: after 250 steps every env had `progress_buf = 250` versus an episode length of 175). `measure_tip_speed.py` and `measure_avoid_r.py` never called `task.reset()`.
- **Invalid:** tip-speed percentiles and per-bout mean avoid_r (0.806 vs 0.835) both include post-episode frames.
- **Still valid:**
  - avoid_r falls into the *first* hit (214/215);
  - the tunneling proof;
  - the detector catch-rate comparisons, since both conditions are affected equally;
  - the dodge-eval, which resets per bout.
- **Fix:** add a manual reset of done envs to those tools, then re-measure.

---

## Design notes (not bugs)
- **Hit rule matches the paper:** the tip must be within 0.1 m of a target body point with contact force ≥ 50 N. It's borrowed from NCP boxing, and we widened the distance to 0.15.
  - 62% of frames where the tip truly penetrates a target aren't scored.
  - Nearly all of those are the tip held still (67%) or sliding sideways (33%) with target force ≤ 50 N (96%). Joint-origin distance explains only 2%.
  - Real foil and épée tips register at 4.9 N and 7.35 N, so 50 N is 7–10× stricter.
- **Observation matches the paper's design:** opponent body positions, no rotations.
  - The obs function takes opponent rotations, DOF positions and DOF velocities as arguments but never uses them (T5: randomizing them changes the obs by exactly 0). They are leftover arguments.
  - Two deviations from the paper: the code includes only the opponent's *root* velocity (the paper lists all 24 body velocities), and it uses 6 targets (the code adds Neck; the paper lists 5).
- **Sword–floor contact** doesn't count as a fall: only non-foot body *origins* below 0.15 m trigger termination, and the blade never does.

## Were these compute-motivated?
- **Plausibly compute or simplicity driven (and documented):** 30 Hz policy / 60 Hz sim, no CCD (Isaac default), and the point-plus-force hit rule (cheap, from NCP). The paper notes each model trains on a single RTX 3090 in 1–3 days.
- **Not compute driven:**
  - **E1:** an indexing bug with no cost difference. The win counter doesn't appear in the paper.
  - **E2:** an ordering bug. The refresh runs anyway.
  - **E4:** contradicts the paper's own termination rule.
  - **E3:** mass costs nothing to simulate. Keeping the pretrained controller's dynamics unchanged is a plausible reason, but that's an inference; the paper is silent.

---

## T1 — The dodge's training opponent is frozen at drills-v3 and never lands (added 2026-10-06)

- **What:** in drills training, the dodge's lunging opponent is rl_games' separate `agent_1_model`.
  - With `switch_frequency=1e9`, the learner's weights are never copied into it.
  - `save()` writes it back out unchanged as `Humanoid_*_op.pth`, and each warm-start restores it.
  - Every non-dodge drill overrides the opponent's actions with Z=0, so only the dodge uses this network.
- **Verified (weights):** `Humanoid_op.pth` is bit-identical (max |Δw| = 0.0000) in every version from v3 to v16. It differs from v2's only by 0.088.
- **Verified (behavior), v16 dodger, v16 reward:**
  - Against its real training opponent: hit in **0 of 1,024** bouts (2 of 1,024 with training-time action noise). That is ≈0.74 reward/step, matching the W&B training log (+0.75).
  - Against the current learner's lunge, which is what `record_dodge_eval.sh` uses: hit **99%**.
- **Impact:** every dodge result from v3 to v16 was trained without a real threat.
  - v14/v15's forward drift (farming the distance gate) and v16's freeze (standing still is optimal when nothing lands) are both rational responses to that.
  - Drills that use a frozen Z=0 opponent (lunges, footwork) are unaffected.
- **Proposed fix (not applied):** make the opponent the learner.
  - **Simplest:** seed the next version's `_op` checkpoint from the learner checkpoint, which has the same keys.
  - **Proper:** periodically copy agent 0's weights into agent 1 (snapshot self-play).

---

## E7 — Strip-boundary observation is world-ordered (added 2026-10-13; fixed in v20)
- **What:** `compute_fencing_observation` encodes the strip with two corners, `[-1,-7]` and `[1,7]`, in
  a fixed world order. A mirrored pair therefore gets swapped slots:
  - agent at −y: `[−5.5, +1.0, +8.5, −1.0]`;
  - agent at +y: `[+8.5, −1.0, −5.5, +1.0]`.
  - After the ±5 clip, the +y fencer reads "the wall behind me is in front".
- **Verified:**
  - From mirrored starts, the same network lunges 100% (−y) vs 77% (+y) and retreats 1.22 vs 0.72 m.
  - The bias follows the side, not the actor index.
  - It is identical with `num_envs=1`, so it isn't caused by GPU overlap.
- **Impact:** the learner always trains at −y, so every self-play opponent and agent 1 in every eval was
  handicapped. This includes the strategy env's opponent.
- **Fix:** `bbox_obs_facing_sorted` (default True) orders the corners by the fencer's facing.
  - Agent 0's observation is bit-identical; the +y fencer now gets the expected layout.
  - Verified: retreat bias 0.506 → 0.0005 m, and side-swap match-ups symmetric.
  - Full log in `DODGE_INVESTIGATION.md`.
