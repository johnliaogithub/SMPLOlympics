# Paper evaluation: the same motion metrics for our drills policy and the flat self-play baseline.
#
# ours: each env runs one scripted bout (env.eval_schedule, a list of [drill, steps]); a landed lunge
#       hands over to the stand drill for the rest of its segment, as in training (lunge_recover_as_stand).
#       The opponent holds en garde with the stand drill.
# flat: each env runs one free self-play bout; both fencers use the flat policy.
# One bout per env, no resets; a bout ends at the schedule's end, a fall, or a win (flat).
#
#   bash scripts/fencing/paper/eval_motion.sh ours output/HumanoidIm/paper_drills_s0/Humanoid.pth ours_s0
#   bash scripts/fencing/paper/eval_motion.sh flat output/HumanoidIm/paper_flat_s0/Humanoid.pth flat_s0
# Appends (or replaces) a row in output/paper/metrics.csv.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_task, FlatPolicy, MotionMetrics  # noqa: E402  (sets up paths + isaacgym)

import csv
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from phc.train_fencing_strategy import FrozenLowLevelPolicy
from phc.env.tasks.humanoid_fencing_drills import NUM_DRILLS, DRILL_NAMES, D_STAND, D_LUNGE_UPPER, D_LUNGE_GROIN

DEFAULT_SCHEDULE = [["stand", 30], ["advance", 25], ["lunge_upper", 35], ["stand", 25], ["retreat", 25],
                    ["step_left", 20], ["step_right", 20], ["lunge_groin", 35], ["stand", 25]]


def write_row(row, path="output/paper/metrics.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r["tag"] != row["tag"]]
    rows.append({k: str(v) for k, v in row.items()})
    keys = list(row.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[eval] wrote {row['tag']} -> {path}")


@torch.no_grad()
def run_ours(cfg, task, ckpt):
    low = FrozenLowLevelPolicy(ckpt, NUM_DRILLS, device=task.device)
    sched = [(DRILL_NAMES.index(d), int(n)) for d, n in cfg.env.get("eval_schedule", DEFAULT_SCHEDULE)]
    N = task.num_envs
    probs = torch.zeros(NUM_DRILLS, device=task.device)
    # spawn as for this drill (strike drills spawn in lunge range); default = the first scheduled drill
    probs[DRILL_NAMES.index(cfg.env.get("eval_spawn_drill", DRILL_NAMES[sched[0][0]]))] = 1.0
    task.drill_probs = probs
    task.step_counter = task.warmup_time + 10
    task.reset()
    M = MotionMetrics(task)
    for d, n in sched:
        task._lunge_landed[:] = False        # each lunge segment starts a fresh strike
        for _ in range(n):
            lunge = d in (D_LUNGE_UPPER, D_LUNGE_GROIN)
            task.drill_ids[:] = d
            if lunge:
                task.drill_ids[task._lunge_landed] = D_STAND
            task.opp_drill_ids[:] = D_STAND
            task.opp_frozen[:] = True
            obs = task.obs_buf
            task.step(torch.cat([low(obs[:N], task.drill_ids), low(obs[N:2 * N], task.opp_drill_ids)], 0))
            M.update()
            M.end_bouts(task.reset_buf)
    # where in the schedule the falls happen
    ft = M.fall_step.cpu().numpy()
    edges = np.cumsum([0] + [n for _, n in sched])
    parts = []
    for k, (d, n) in enumerate(sched):
        c = int(((ft >= edges[k]) & (ft < edges[k + 1])).sum())
        parts.append(f"{DRILL_NAMES[d]}[{edges[k]}-{edges[k + 1]}):{c}")
    print("[eval] falls by segment: " + "  ".join(parts))
    return M.summary(), sum(n for _, n in sched)


@torch.no_grad()
def run_flat(cfg, task, ckpt):
    pol0 = FlatPolicy(ckpt, device=task.device)
    op = ckpt.replace(".pth", "_op.pth")
    pol1 = FlatPolicy(op, device=task.device) if os.path.exists(op) else pol0
    N, T = task.num_envs, int(cfg.env.get("eval_steps", 240))
    # the baseline env picks its curriculum stage from step_counter; stage 0 disables win endings, so
    # start in the final stage, as at the end of training
    task.step_counter = max(task.warmup_time + 10, task.curriculum_stages[-1]["step_threshold"] + 1)
    task.reset()
    M = MotionMetrics(task)
    for _ in range(T):
        obs = task.obs_buf
        task.step(torch.cat([pol0(obs[:N]), pol1(obs[N:2 * N])], 0))
        M.update()
        M.end_bouts(task.reset_buf)
    return M.summary(), T


@hydra.main(version_base=None, config_path="../data/cfg", config_name="config")
def main(cfg_hydra: DictConfig) -> None:
    method = cfg_hydra.env.eval_method
    cfg, task = make_task(cfg_hydra, test=True, seed=int(cfg_hydra.get("seed", 0)))
    ckpt = cfg.env.eval_checkpoint
    summ, T = (run_ours if method == "ours" else run_flat)(cfg, task, ckpt)
    row = {"tag": cfg.env.eval_tag, "method": method, "checkpoint": ckpt, "steps_per_bout": T, **summ}
    for k, v in row.items():
        print(f"  {k:18s} {v}")
    write_row(row)


if __name__ == "__main__":
    main()
