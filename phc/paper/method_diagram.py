# Paper figure: method overview (strategy -> drills policy -> frozen motion prior -> simulator).
#   python phc/paper/method_diagram.py   ->  output/paper/figures/method.{pdf,png}
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED = "#1a1a1a", "#6b6b6b"
TRAINED, TRAINED_FILL = "#2a78d6", "#e9f1fb"     # learned by us
FROZEN, FROZEN_FILL = "#8a8a8a", "#f2f2f2"       # pretrained / fixed

BOXES = {  # name: (x, y, w, h, title, subtitle, kind)
    "strategy": (0.000, 0.38, 0.195, 0.32, "Strategy policy", "picks a drill\nevery few steps", "trained"),
    "drills": (0.265, 0.38, 0.195, 0.32, "Drills policy", "one network,\nconditioned on drill id", "trained"),
    "prior": (0.530, 0.38, 0.195, 0.32, "PULSE decoder", "general motion prior\n(no fencing data)", "frozen"),
    "sim": (0.795, 0.38, 0.205, 0.32, "Physics simulation", "two SMPL fencers\n(Isaac Gym)", "frozen"),
}
ARROWS = [("strategy", "drills", "drill id"), ("drills", "prior", "latent z"), ("prior", "sim", "PD targets")]
DRILLS = "advance · retreat · en garde · lunge (high / low) · dodge · step left / right"


def main():
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, y, w, h, title, sub, kind in BOXES.values():
        edge, fill = (TRAINED, TRAINED_FILL) if kind == "trained" else (FROZEN, FROZEN_FILL)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.025",
                                    linewidth=1.4, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=7.3, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.33, sub, ha="center", va="center", fontsize=6.0, color=MUTED, linespacing=1.3)
    for a, b, lab in ARROWS:
        xa, ya, wa, ha = BOXES[a][:4]
        xb = BOXES[b][0]
        yc = ya + ha / 2
        ax.add_patch(FancyArrowPatch((xa + wa + 0.008, yc), (xb - 0.008, yc), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.2, color=INK))
        ax.text((xa + wa + xb) / 2, yc + 0.04, lab, ha="center", va="bottom", fontsize=5.8, color=INK)
    # observation feedback from the simulator to both policies
    xs, ys, ws, _ = BOXES["sim"][:4]
    y_fb = 0.22
    ax.add_patch(FancyArrowPatch((xs + ws / 2, 0.38), (xs + ws / 2, y_fb), arrowstyle="-", linewidth=1.0, color=MUTED))
    for k in ("strategy", "drills"):
        x, y, w, _ = BOXES[k][:4]
        ax.add_patch(FancyArrowPatch((x + w / 2, y_fb), (x + w / 2, 0.38 - 0.008), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.0, color=MUTED))
    ax.plot([BOXES["strategy"][0] + BOXES["strategy"][2] / 2, xs + ws / 2], [y_fb, y_fb], color=MUTED, linewidth=1.0)
    ax.text(0.62, y_fb - 0.03, "observations: own body, opponent, sword, strip", ha="center", va="top", fontsize=6.0, color=MUTED)
    # drill library + legend
    x, y, w, h = BOXES["drills"][:4]
    ax.text(0.0, y + h + 0.07, "8 drills, one reward each: " + DRILLS, ha="left", va="bottom", fontsize=6.0, color=INK)
    for i, (lab, edge, fill) in enumerate([("trained (ours)", TRAINED, TRAINED_FILL), ("frozen", FROZEN, FROZEN_FILL)]):
        lx = 0.0 + i * 0.15
        ax.add_patch(FancyBboxPatch((lx, 0.94), 0.016, 0.045, boxstyle="round,pad=0,rounding_size=0.006",
                                    linewidth=1.0, edgecolor=edge, facecolor=fill))
        ax.text(lx + 0.022, 0.962, lab, va="center", fontsize=6.0, color=INK)
    os.makedirs("output/paper/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"output/paper/figures/method.{ext}", bbox_inches="tight", dpi=300)
    print("[diagram] output/paper/figures/method.pdf/.png")


if __name__ == "__main__":
    main()
