#!/usr/bin/env python3
"""
Warm-start a larger drill net from a smaller one by expanding the policy's input
layer to accommodate extra drill one-hot dimensions.

Because new drills are APPENDED to DRILL_NAMES, the extra one-hot dims sit at the
very end of the observation vector. So we:
  - append `n_extra` ZERO columns to every first-layer weight whose in-features
    match the old obs size (actor_mlp.0, critic_mlp.0). Zero columns => the new
    drills contribute nothing until trained, and all existing drills behave
    EXACTLY as before.
  - append `n_extra` entries to running_mean_std (mean=0, var=1) so the new dims
    pass through normalization unchanged.
  - reset epoch/frame to 0 so the warm-started run is a fresh experiment.

Usage (run inside the isaac env):
  python scripts/fencing/expand_drills_checkpoint.py \
      --src output/HumanoidIm/fencing_drills_v1 \
      --dst output/HumanoidIm/fencing_drills_v2 \
      --n_extra 2

Then train the new experiment warm-started:
  bash scripts/fencing/train_fencing_drills.sh D \
      exp_name=fencing_drills_v2 learning.params.config.max_epochs=20000

VERIFY before training: run scripts/fencing/visualize_drills.sh on the expanded
net (pointed at fencing_drills_v2/Humanoid.pth) and confirm drills 0-5 still look
identical to v1. If they do, the surgery preserved the learned behavior.
"""

import argparse
import os
import torch


def expand_checkpoint(path_in, path_out, n_extra):
    ckpt = torch.load(path_in, map_location="cpu")

    rms = ckpt.get("running_mean_std", None)
    if rms is None or "running_mean" not in rms:
        raise RuntimeError(f"{path_in}: no running_mean_std/running_mean to size from")
    old_obs = rms["running_mean"].shape[0]
    new_obs = old_obs + n_extra
    print(f"\n=== {os.path.basename(path_in)} : obs {old_obs} -> {new_obs} ===")

    # 1) expand first-layer weights (in-features == old_obs)
    model = ckpt["model"]
    for k, v in list(model.items()):
        if v.dim() == 2 and v.shape[1] == old_obs:
            pad = torch.zeros(v.shape[0], n_extra, dtype=v.dtype)
            model[k] = torch.cat([v, pad], dim=1)
            print(f"  expanded {k}: {tuple(v.shape)} -> {tuple(model[k].shape)}")

    # 1b) expand Adam momentum buffers for the same first-layer params, or they
    #     mismatch the resized weights at optimizer.step() (restore loads them).
    opt = ckpt.get("optimizer", None)
    if opt is not None and "state" in opt:
        for pstate in opt["state"].values():
            for bkey in ("exp_avg", "exp_avg_sq"):
                t = pstate.get(bkey, None)
                if torch.is_tensor(t) and t.dim() == 2 and t.shape[1] == old_obs:
                    pad = torch.zeros(t.shape[0], n_extra, dtype=t.dtype)
                    pstate[bkey] = torch.cat([t, pad], dim=1)
                    print(f"  expanded optimizer {bkey}: {tuple(t.shape)} -> {tuple(pstate[bkey].shape)}")

    # 2) expand the obs normalizer (mean -> 0, var -> 1 for new dims)
    for key in list(rms.keys()):
        t = rms[key]
        if torch.is_tensor(t) and t.dim() == 1 and t.shape[0] == old_obs:
            fill = 1.0 if "var" in key else 0.0
            rms[key] = torch.cat([t, torch.full((n_extra,), fill, dtype=t.dtype)])
            print(f"  expanded running_mean_std/{key} (fill={fill})")

    # 3) reset counters so the warm-started run is a fresh experiment
    for key in ("epoch", "frame"):
        if key in ckpt:
            print(f"  reset {key} {ckpt[key]} -> 0")
            ckpt[key] = 0

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    torch.save(ckpt, path_out)
    print(f"  saved -> {path_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source experiment dir (old drills)")
    ap.add_argument("--dst", required=True, help="destination experiment dir (new drills)")
    ap.add_argument("--n_extra", type=int, default=2, help="number of new drills added")
    ap.add_argument("--name", default="Humanoid", help="checkpoint base name")
    args = ap.parse_args()

    # Expand both the learner ('Humanoid.pth') and the self-play opponent snapshot
    # ('Humanoid_op.pth'); both must match the new obs size.
    for suffix in ("", "_op"):
        fn = f"{args.name}{suffix}.pth"
        src = os.path.join(args.src, fn)
        if not os.path.exists(src):
            print(f"[skip] {src} not found")
            continue
        expand_checkpoint(src, os.path.join(args.dst, fn), args.n_extra)

    print("\nDone. Train with epoch=-1 resume (the script auto-detects the best "
          "checkpoint) and a fresh max_epochs.")


if __name__ == "__main__":
    main()
