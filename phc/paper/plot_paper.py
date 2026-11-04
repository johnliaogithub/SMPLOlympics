# Paper plots, read from local files only (no W&B API):
#   curves  : training curves from the local wandb runs (found by exp_name), x = environment steps
#   metrics : small multiples of output/paper/metrics.csv (from eval_motion.py), one panel per metric
#
#   python phc/paper/plot_paper.py                       # both, default paper runs
#   python phc/paper/plot_paper.py --ours-prefix fencing_drills_v23 --flat-prefix fencing_curriculum_v2
# Output: output/paper/figures/{curves,metrics}.{pdf,png}, output/paper/metrics_table.tex
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_COLORS = {"ours": "#2a78d6", "flat": "#eb6834", "mocap": "#1d9e75"}   # validated, same as common.py
METHOD_LABELS = {"ours": "Drills (ours)", "flat": "Flat self-play", "mocap": "Fencing mocap"}
DRILL_COLORS = {"stand": "#6250d6", "lunge_upper": "#c98500", "dodge": "#e87ba4"}   # validated; direct-labeled
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
STEPS_PER_EPOCH = 256 * 32          # num_envs x horizon_length

METRICS = [  # key, panel title, scale, unit, lower_is_better
    ("fall_rate", "Falls", 100, "% of bouts", True),
    ("engarde_frac", "Time in en garde", 100, "% of steps", False),
    ("foot_slide_cms", "Foot sliding", 1, "cm/s", True),
    ("jerk_median", "Jerk (median)", 1, "m/s³", True),
    ("spine_up_mean", "Torso upright", 1, "cos(lean)", False),
]


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42, "figure.dpi": 150,
    })


# ---------------------------------------------------------------- wandb (local files)
def wandb_history(run_dir):
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb
    f = glob.glob(os.path.join(run_dir, "*.wandb"))[0]
    ds = datastore.DataStore()
    ds.open_for_scan(f)
    rows = []
    while True:
        try:
            d = ds.scan_data()
        except Exception:
            break
        if d is None:
            break
        r = pb.Record()
        r.ParseFromString(d)
        if r.WhichOneof('record_type') == 'history':
            row = {}
            for i in r.history.item:
                k = i.key or '/'.join(i.nested_key)
                try:
                    row[k] = json.loads(i.value_json)
                except Exception:
                    pass
            rows.append(row)
    return rows


def find_runs(prefix):
    """Local wandb runs whose exp_name starts with prefix, grouped by exp_name (latest run per name)."""
    out = {}
    for cfg in sorted(glob.glob("wandb/run-*/files/config.yaml")):
        m = re.search(r"exp_name=(\S+)", open(cfg).read())
        if m and m.group(1).startswith(prefix):
            out[m.group(1)] = os.path.dirname(os.path.dirname(cfg))
    return out


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and v == v:
            xs.append(r.get("epoch", r.get("_step")))
            ys.append(v)
    x, y = np.array(xs, float), np.array(ys, float)
    return x - x.min() if len(x) else x, y


def smooth(y, w):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def band(ax, runs, key, color, label, w=200):
    """Mean over seeds (line) with min-max across seeds (band), x in millions of env steps."""
    curves = [series(wandb_history(d), key) for d in runs.values()]
    curves = [(x, smooth(y, w)) for x, y in curves if len(x)]
    if not curves:
        return None
    n = min(len(x) for x, _ in curves)
    x = curves[0][0][:n] * STEPS_PER_EPOCH / 1e6
    Y = np.stack([y[:n] for _, y in curves])
    if len(curves) > 1:
        ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.15, linewidth=0)
    ax.plot(x, Y.mean(0), color=color, linewidth=2, label=label)
    return x[-1], Y.mean(0)[-1]


def plot_curves(args):
    ours, flat = find_runs(args.ours_prefix), find_runs(args.flat_prefix)
    print(f"[curves] ours: {sorted(ours)} | flat: {sorted(flat)}")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    for drill, col in DRILL_COLORS.items():
        end = band(ax, ours, f"drills/{drill}", col, drill.replace("_", " "))
        if end:
            ax.annotate(drill.replace("_", " "), end, xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK)
    ax.set_title("Drills (ours): reward per step, by drill", loc="left", pad=22)
    ax.set_xlabel("fine-tuning environment steps (M)")
    ax.set_ylabel("reward / step")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7, handlelength=1.5)
    ax.margins(x=0.12)
    ax = axes[1]
    band(ax, flat, "mean_rewards", METHOD_COLORS["flat"], METHOD_LABELS["flat"])
    ax.set_title("Flat self-play: mean episode return", loc="left", pad=22)
    ax.set_xlabel("environment steps (M)")
    ax.set_ylabel("return")
    fig.tight_layout()
    save(fig, "curves")


# ---------------------------------------------------------------- metrics (eval_motion.py output)
def method_of(row):
    return row.get("method", "ours")


def plot_metrics(args):
    rows = [r for r in csv.DictReader(open(args.metrics))]
    if args.tags:
        rows = [r for r in rows if r["tag"] in args.tags.split(",")]
    methods = [m for m in ["ours", "flat", "mocap"] if any(method_of(r) == m for r in rows)]
    print(f"[metrics] {len(rows)} rows: " + ", ".join(f"{r['tag']}({method_of(r)})" for r in rows))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.0, 1.9))
    table = {}
    for ax, (key, title, scale, unit, lower) in zip(axes, METRICS):
        for j, m in enumerate(methods):
            vals = np.array([float(r[key]) * scale for r in rows if method_of(r) == m and r.get(key, "") != ""])
            if not len(vals):
                continue
            mean = vals.mean()
            table.setdefault(m, {})[key] = (mean, vals.std() if len(vals) > 1 else None)
            ax.bar(j, mean, width=0.6, color=METHOD_COLORS[m], edgecolor="white", linewidth=2, zorder=2)
            if len(vals) > 1:   # seeds as dots
                ax.scatter(np.full(len(vals), j), vals, s=10, color=INK, zorder=3, linewidths=0)
            ax.annotate(f"{mean:.1f}" if abs(mean) >= 10 else f"{mean:.2f}", (j, mean), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_title(f"{title} {'↓' if lower else '↑'}", fontsize=7.5)
        ax.set_ylabel(unit, fontsize=7)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS[m].replace(" (ours)", "\n(ours)").replace(" self-play", "\nself-play")
                            for m in methods], fontsize=6.5)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.18)
    fig.text(0.0, -0.02, "↓ lower is better, ↑ higher is better. Bars: mean over seeds; dots: seeds.", fontsize=6.5, color=MUTED)
    fig.tight_layout(w_pad=1.0)
    save(fig, "metrics")
    write_table(table, methods)


def write_table(table, methods):
    lines = ["\\begin{tabular}{l" + "c" * len(METRICS) + "}", "\\toprule",
             "Method & " + " & ".join(f"{t} ({u})" for _, t, _, u, _ in METRICS) + " \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for key, *_ in METRICS:
            mean, sd = table.get(m, {}).get(key, (float("nan"), None))
            cells.append(f"{mean:.2f}" + (f" $\\pm$ {sd:.2f}" if sd is not None else ""))
        lines.append(f"{METHOD_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open("output/paper/metrics_table.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[metrics] output/paper/metrics_table.tex")


def save(fig, name):
    os.makedirs("output/paper/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"output/paper/figures/{name}.{ext}", bbox_inches="tight", dpi=300)
    print(f"[plot] output/paper/figures/{name}.pdf/.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["curves", "metrics"])
    p.add_argument("--ours-prefix", default="paper_drills_s")
    p.add_argument("--flat-prefix", default="paper_flat_s")
    p.add_argument("--metrics", default="output/paper/metrics.csv")
    p.add_argument("--tags", default="", help="comma-separated metric rows to include (default: all)")
    a = p.parse_args()
    style()
    if a.only != "metrics":
        plot_curves(a)
    if a.only != "curves":
        plot_metrics(a)
