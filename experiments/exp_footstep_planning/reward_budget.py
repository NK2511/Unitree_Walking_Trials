"""Reward Budget & Impact Analysis for Angad Footstep Training.

Visualises:
  1. Maximum potential per-step impact (positive for rewards, negative for worst-case penalties).
  2. Absolute Impact Ranking (sorted by total magnitude |weight * peak_val * dt|).
  3. Share of positive reward budget (pie chart).
  4. Episode return summary.

Run from Angad_Footstep_Planning_Trials:
    python reward_budget.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

DT            = 0.02    # seconds per physics step (50 Hz)
STEPS_PER_ENV = 88      # rollout length
EPISODE_STEPS = 1000    # typical episode length (~20s)

# ─── Definition of Reward Terms & Peak/Max Values ─────────────────────────────
# (name, weight, max_peak_term_value, typical_term_value, kind)
#
# max_peak_term_value:
#   - For positive rewards: maximum attainable value of func (usually 1.0)
#   - For penalties: worst-case peak violation magnitude during active movement / failure
#
REWARDS = [
    # ── POSITIVE REWARDS (bounded [0, 1]) ─────────────────────────────────────
    ("imitation",          +1.00,  1.00,  0.70, "primary"),   # Trajectory matching (double support)
    ("footstep_hit",       +1.00,  1.00,  0.50, "primary"),   # Target marker precision
    ("foot_clock",         +0.50,  1.00,  0.60, "primary"),   # Rohan phase-synchronised contact/vel
    ("footstep_progress",  +0.09,  1.00,  0.40, "positive"),
    ("upright",            +0.05,  1.00,  0.90, "positive"),
    ("pelvis_height",      +0.05,  1.00,  0.85, "positive"),
    ("footstep_orient",    +0.05,  1.00,  0.80, "positive"),
    ("air_time",            0.00,  1.00,  0.00, "positive"),   # Disabled (replaced by foot_clock)

    # ── PENALTIES (weight < 0, max_peak_term_value = peak worst-case violation) ─
    ("dof_pos_limits",     -1.00,  0.80,  0.05, "penalty"),   # 0.8 rad overshoot across joints
    ("foot_clearance",     -0.30,  0.25,  0.05, "penalty"),   # 0.08m err * 1.5m/s vel = 0.12 * 2 feet
    ("foot_slip",          -0.30,  1.50,  0.05, "penalty"),   # 1.5 m²/s² sliding foot speed
    ("foot_swing_height",  -0.25,  0.50,  0.10, "penalty"),   # 50% peak height error
    ("body_ang_vel",       -0.08,  4.00,  0.30, "penalty"),   # 2.0 rad/s roll/pitch spin = 4.0
    ("angular_momentum",   -0.03, 10.00,  0.50, "penalty"),   # 10.0 kg·m²/s² body spin
    ("action_rate_l2",     -0.01,  8.00,  1.50, "penalty"),   # 8.0 sum of squared action changes
    ("soft_landing",       -1e-5, 300.0, 50.00, "penalty"),   # 300 N hard impact landing
]

names          = [r[0] for r in REWARDS]
weights        = np.array([r[1] for r in REWARDS])
peak_terms     = np.array([r[2] for r in REWARDS])
typical_terms  = np.array([r[3] for r in REWARDS])
kinds          = [r[4] for r in REWARDS]

# ─── Calculations ─────────────────────────────────────────────────────────────
# Peak per-step impact (signed): +val for rewards, -val for peak worst-case penalties
peak_impact_per_step = np.where(
    weights > 0,
    weights * peak_terms * DT,
    weights * peak_terms * DT  # weights < 0, so result is negative
)

# Absolute magnitude per step (to rank pure importance regardless of sign)
abs_importance_per_step = np.abs(peak_impact_per_step)

# Episode scale
peak_impact_episode = peak_impact_per_step * EPISODE_STEPS

pos_mask = weights > 0
theoretical_max_ep = float(np.sum(peak_impact_episode[pos_mask]))
rollout_ceiling    = float(np.sum(peak_impact_per_step[pos_mask]) * STEPS_PER_ENV)

# ─── Print Table ──────────────────────────────────────────────────────────────
print("=" * 75)
print("  REWARD BUDGET & MAXIMUM IMPACT ANALYSIS")
print("=" * 75)
print(f"  {'Term':<20} {'Weight':>8}  {'Peak Term':>10}  {'Peak Impact/Step':>18}  {'Abs Rank':>8}")
print("-" * 75)
rank_indices = np.argsort(-abs_importance_per_step)
for rank, idx in enumerate(rank_indices, 1):
    n = names[idx]
    w = weights[idx]
    pt = peak_terms[idx]
    imp = peak_impact_per_step[idx]
    print(f"  {n:<20} {w:>+8.3f}  {pt:>10.2f}  {imp:>+18.4f}  #{rank:<7}")
print("=" * 75)
print(f"  Theoretical Max Episode Return (Positive terms): {theoretical_max_ep:+.2f}")
print(f"  Rollout-level ceiling (88 steps)              : {rollout_ceiling:+.2f}")
print("=" * 75)

# ─── Plotting ─────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":  "#6C63FF",   # purple — main footstep rewards
    "positive": "#3EC6A6",   # teal — stability rewards
    "penalty":  "#FF5252",   # bright red — penalties
}

fig = plt.figure(figsize=(18, 12), facecolor="#0F1117")
fig.suptitle(
    "Reward & Penalty Peak Impact Analysis (Angad Footstep Task)",
    fontsize=18, fontweight="bold", color="white", y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# ── 1. Peak Per-Step Impact (Bidirectional Horizontal Bar) ─────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor("#1A1D2E")

# Sort by signed peak impact
sorted_idx = np.argsort(peak_impact_per_step)
snames  = [names[i] for i in sorted_idx]
svals   = [peak_impact_per_step[i] for i in sorted_idx]
scolors = [PALETTE[kinds[i]] for i in sorted_idx]

bars = ax1.barh(snames, svals, color=scolors, edgecolor="none", height=0.65)
ax1.axvline(0, color="white", linewidth=1.0, alpha=0.5)
ax1.set_xlabel("Peak Impact per Physics Step (Weight × Peak Value × DT)", color="white", fontsize=11)
ax1.set_title("Per-Step Peak Impact: Max Reward (+ / Right) vs Max Penalty (- / Left)", color="white", fontsize=13, pad=10)
ax1.tick_params(colors="white", labelsize=9.5)
ax1.spines[:].set_color("#333355")

for bar, val in zip(bars, svals):
    x = bar.get_width()
    offset = 0.0008 if x >= 0 else -0.0008
    ha = "left" if x >= 0 else "right"
    ax1.text(
        x + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.4f}",
        va="center", ha=ha,
        color="white", fontsize=8.5, fontweight="bold"
    )

# ── 2. Share of Positive Reward Budget ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor("#1A1D2E")

pos_names  = [n for n, m in zip(names, pos_mask) if m]
pos_vals   = [v for v, m in zip(peak_impact_episode, pos_mask) if m]
pos_colors = [PALETTE[k] for k, m in zip(kinds, pos_mask) if m]

wedges, texts, autotexts = ax2.pie(
    pos_vals,
    labels=pos_names,
    autopct="%1.1f%%",
    colors=pos_colors,
    startangle=140,
    wedgeprops={"edgecolor": "#0F1117", "linewidth": 1.5},
    textprops={"color": "white", "fontsize": 8.5},
)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_color("#0F1117")
    at.set_fontweight="bold"

ax2.set_title("Share of Positive Reward Ceiling", color="white", fontsize=12, pad=12)

# ── 3. Absolute Impact Ranking (Sorted by Pure Magnitude) ──────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_facecolor("#1A1D2E")

rank_names  = [names[i] for i in rank_indices]
rank_vals   = [abs_importance_per_step[i] for i in rank_indices]
rank_colors = [PALETTE[kinds[i]] for i in rank_indices]

bars3 = ax3.bar(rank_names, rank_vals, color=rank_colors, edgecolor="none", width=0.6)
ax3.set_ylabel("Absolute Per-Step Magnitude  |Weight × Peak × DT|", color="white", fontsize=10)
ax3.set_title("Overall Importance Ranking (Rewards & Penalties combined)", color="white", fontsize=13, pad=10)
ax3.set_xticks(np.arange(len(rank_names)))
ax3.set_xticklabels(rank_names, rotation=35, ha="right", color="white", fontsize=9.5)
ax3.tick_params(colors="white", labelsize=9)
ax3.spines[:].set_color("#333355")

for bar, val in zip(bars3, rank_vals):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.0005,
        f"{val:.4f}",
        ha="center", va="bottom",
        color="white", fontsize=8, alpha=0.9
    )

# ── 4. Summary Card ────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor("#1A1D2E")
ax4.axis("off")

lines = [
    ("Theoretical MAX (Positive)", f"{theoretical_max_ep:+.1f}", "#6C63FF"),
    ("Max Penalty Potential",      "-64.0",                   "#FF5252"),
    ("Rollout Ceiling (88 steps)",  f"{rollout_ceiling:.1f}",  "#FFD166"),
    ("Target TensorBoard Mean",     "2.5 – 4.0",              "#3EC6A6"),
]

ax4.set_title("Environment Ceiling Summary", color="white", fontsize=12, pad=10)

for i, (label, val, color) in enumerate(lines):
    y = 0.85 - i * 0.22
    ax4.add_patch(FancyBboxPatch(
        (0.05, y - 0.07), 0.90, 0.18,
        boxstyle="round,pad=0.02",
        facecolor=color + "22",
        edgecolor=color,
        linewidth=1.5,
        transform=ax4.transAxes,
        clip_on=False,
    ))
    ax4.text(0.10, y + 0.03, label, transform=ax4.transAxes,
             color="white", fontsize=9, va="center")
    ax4.text(0.90, y + 0.03, val,   transform=ax4.transAxes,
             color=color, fontsize=12.5, va="center", ha="right", fontweight="bold")

legend_patches = [
    plt.Rectangle((0,0), 1, 1, fc=PALETTE["primary"],  label="Primary Footstep Reward"),
    plt.Rectangle((0,0), 1, 1, fc=PALETTE["positive"], label="Stability Reward"),
    plt.Rectangle((0,0), 1, 1, fc=PALETTE["penalty"],  label="Penalty (Max Violation Impact)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           facecolor="#0F1117", labelcolor="white", fontsize=10,
           framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

plt.savefig("reward_budget.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("  Saved updated: reward_budget.png")
