#!/bin/bash
# After training: evaluate every paper checkpoint, then draw all figures.
#   bash scripts/fencing/paper/make_figures.sh
# Output: output/paper/metrics.csv, output/paper/metrics_table.tex, output/paper/figures/*, output/paper/video/*
cd /pub0/johnliao/SMPLOlympics
export PATH=/pub0/johnliao/miniconda3/envs/isaac/bin:$PATH
for d in output/HumanoidIm/paper_drills_s*; do
  s=${d##*_s}; bash scripts/fencing/paper/eval_motion.sh ours $d/Humanoid.pth ours_s$s | grep -E "^\s+(fall|engarde|foot|jerk|bouts_with)|\[eval\]"
done
for d in output/HumanoidIm/paper_flat_s*; do
  s=${d##*_s}; bash scripts/fencing/paper/eval_motion.sh flat $d/Humanoid.pth flat_s$s | grep -E "^\s+(fall|engarde|foot|jerk|bouts_with)|\[eval\]"
done
bash scripts/fencing/paper/filmstrip.sh ours output/HumanoidIm/paper_drills_s0/Humanoid.pth ours +env.film_guard_steps=30 | grep "\[film\]"
bash scripts/fencing/paper/filmstrip.sh flat output/HumanoidIm/paper_flat_s0/Humanoid.pth flat | grep "\[film\]"
python phc/paper/plot_paper.py
python phc/paper/method_diagram.py
