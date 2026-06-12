#!/usr/bin/env python3
"""
Plot per-step reward component breakdown from reward_analysis.csv.

Usage:
    python scripts/fencing/plot_episode_rewards.py [path/to/reward_analysis.csv]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "reward_analysis.csv"

df = pd.read_csv(CSV_PATH)

# Detect episode boundaries: episode_step resets to 0 or 1 after being high
boundaries = [0] + list(df.index[df['episode_step'].diff() < 0]) + [len(df)]
episodes = [df.iloc[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries)-1)]
print(f"Loaded {len(df)} steps across {len(episodes)} episode(s) from {CSV_PATH}")

# Use only first episode for clarity
ep = episodes[0].reset_index(drop=True)
steps = ep['episode_step'].values

components = ['vel', 'facing', 'strike', 'terminate', 'hit']
weights_cols = ['w_vel', 'w_facing', 'w_strike', 'w_terminate', 'w_hit']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f"Episode Reward Breakdown (Episode 1, {len(ep)} steps)", fontsize=14, fontweight='bold')

# --- Panel 1: Raw component values ---
ax = axes[0]
ax.set_title("Raw Component Values (unweighted)")
ax.set_ylabel("Value")
for comp, color in zip(components, colors):
    ax.plot(steps, ep[comp].values, label=comp, color=color, linewidth=0.8, alpha=0.85)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(-5, 5)  # strike can be large; clip for readability

# Note if strike clipped
strike_max = ep['strike'].abs().max()
if strike_max > 5:
    ax.text(0.01, 0.97, f"strike range: [{ep['strike'].min():.1f}, {ep['strike'].max():.1f}] (clipped for display)",
            transform=ax.transAxes, fontsize=7, va='top', color='#C44E52')

# --- Panel 2: Weighted contributions (w * raw value) per step ---
ax = axes[1]
ax.set_title("Weighted Contributions per Step  (weight × raw)")
ax.set_ylabel("Weighted Value")
weighted = {}
for comp, wcol, color in zip(components, weights_cols, colors):
    weighted[comp] = ep[comp].values * ep[wcol].values
    ax.plot(steps, weighted[comp], label=f"{comp} × w", color=color, linewidth=0.8, alpha=0.85)
ax.plot(steps, ep['total'].values, label='total', color='black', linewidth=1.2, linestyle='--')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.legend(loc='upper right', fontsize=8)

# --- Panel 3: Stacked positive/negative contributions + cumulative reward ---
ax = axes[2]
ax.set_title("Cumulative Reward & Per-Step Total")
ax.set_ylabel("Cumulative Reward")
ax2 = ax.twinx()
ax2.set_ylabel("Per-Step Total", color='gray')

cumulative = ep['total'].cumsum().values
ax.plot(steps, cumulative, color='black', linewidth=1.5, label='cumulative reward')
ax2.bar(steps, ep['total'].values, color=['#4C72B0' if v >= 0 else '#C44E52' for v in ep['total'].values],
        alpha=0.3, width=1.0, label='per-step reward')
ax.legend(loc='upper left', fontsize=8)

# Print summary
print("\n--- Summary (Episode 1) ---")
print(f"{'Component':<12} {'Raw mean':>10} {'Raw std':>10} {'Weight':>8} {'Weighted mean':>14}")
print("-" * 58)
for comp, wcol in zip(components, weights_cols):
    w = ep[wcol].iloc[0]
    raw_mean = ep[comp].mean()
    raw_std = ep[comp].std()
    print(f"{comp:<12} {raw_mean:>10.4f} {raw_std:>10.4f} {w:>8.3f} {raw_mean*w:>14.4f}")
print(f"\n{'Total reward':<12} {ep['total'].mean():>10.4f}  (mean per step)")
print(f"{'Episode return':<12} {ep['total'].sum():>10.4f}  (sum over episode)")

plt.tight_layout()
out_path = CSV_PATH.replace('.csv', '_plot.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")
plt.show()
