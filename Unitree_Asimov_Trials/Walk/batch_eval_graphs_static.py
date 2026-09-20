#!/usr/bin/env python3
"""
Angad Gait Evaluation — Static Torque Graph Generator
======================================================
Runs every model at commanded speed = 0.0 m/s (standing still).
Useful for measuring static holding torques and understanding
the robot's equilibrium joint loads.

Output goes into:
    graphs/<No_Torso|With_Torso>/<model_name>/static_torques/

HOW TO USE:
    python generate_graphs_static.py
"""

import os
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ██  USER CONFIGURATION  ██
# ─────────────────────────────────────────────────────────────────────────────

# Python interpreter (mjlab virtual environment)
PYTHON_EXE = "/home/nandhith/Python/Humanoid_Xterra_IITK/mjlab_env/bin/python"

# Path to the evaluator script
EVALUATOR_SCRIPT = str(Path(__file__).resolve().parent / "policy_evaluator.py")

# Speed commanded to the robot (0.0 = standing still)
STATIC_SPEED = 0.0      # m/s

# How long to hold the robot at speed 0 (longer = more stable average)
STATIC_DURATION = 10.0  # seconds

# ── Model Runs ────────────────────────────────────────────────────────────────
# Each entry:  (category, abs_path_to_run_folder, short_name_for_output_subfolder)

_LOGS_ROOT = str(Path(__file__).parent / "logs" / "rsl_rl" / "unitree_velocity")

MODEL_RUNS = [
    # ── No Torso ──────────────────────────────────────────────────────────────
    (
        "No_Torso",
        f"{_LOGS_ROOT}/No_Torso/2026-06-20_18-09-13_Dynamic_Stride_Gait_with_Hip_Sway",
        "Dynamic_Stride_Gait_with_Hip_Sway",
    ),
    (
        "No_Torso",
        f"{_LOGS_ROOT}/No_Torso/2026-06-21_11-42-22_Dynamic_Stride_Gait_no_Hip_Sway",
        "Dynamic_Stride_Gait_no_Hip_Sway",
    ),
    (
        "No_Torso",
        f"{_LOGS_ROOT}/No_Torso/2026-06-25_12-36-12_Dynamic_Stride_COT_no_Hip_Sway",
        "Dynamic_Stride_COT_no_Hip_Sway",
    ),

    # ── With Torso ────────────────────────────────────────────────────────────
    (
        "With_Torso",
        f"{_LOGS_ROOT}/With_Torso/2026-06-26_12-44-42_Torso_Dynamic_Stride_COT_no_Hip_Sway",
        "Torso_Dynamic_Stride_COT_no_Hip_Sway",
    ),
    (
        "With_Torso",
        f"{_LOGS_ROOT}/With_Torso/2026-06-29_12-53-34_Torso_Dynamic_Stride_COT_no_Hip_Sway_Softfoot_1",
        "Torso_Dynamic_Stride_COT_no_Hip_Sway_Softfoot_1",
    ),
    (
        "With_Torso",
        f"{_LOGS_ROOT}/With_Torso/2026-06-29_15-54-23_Raw_RL",
        "Raw_RL",
    ),
    (
        "With_Torso",
        f"{_LOGS_ROOT}/With_Torso/2026-06-30_10-48-00",
        "2026-06-30_10-48-00",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# ██  END OF USER CONFIGURATION  ██
# ─────────────────────────────────────────────────────────────────────────────

GRAPHS_ROOT = Path(__file__).resolve().parent / "graphs"


def run_static_eval(run_path: str, output_dir: Path) -> bool:
    """Run the evaluator at speed=0 and move output into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXE,
        EVALUATOR_SCRIPT,
        run_path,
        "--no-render",
        "--speed", str(STATIC_SPEED),
        "--duration", str(STATIC_DURATION),
    ]

    print(f"    ▶  speed=0.0 m/s (static)  →  {output_dir}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"    ✗  Evaluator exited with code {result.returncode}")
        return False

    # Move files from the evaluator's default output dir into output_dir
    run_folder_name = Path(run_path).name
    category = "With_Torso" if "With_Torso" in run_path else "No_Torso"
    evaluator_default_out = GRAPHS_ROOT / category / run_folder_name

    if evaluator_default_out.exists() and evaluator_default_out != output_dir:
        for item in evaluator_default_out.iterdir():
            dest = output_dir / item.name
            if dest.exists():
                import shutil
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            item.rename(dest)
        try:
            evaluator_default_out.rmdir()
        except OSError:
            pass

    print(f"    ✓  Saved to {output_dir}")
    return True


def main() -> None:
    print("=" * 70)
    print("  Angad Static Torque Graph Generator")
    print(f"  Commanded speed : {STATIC_SPEED} m/s  (standing still)")
    print(f"  Duration        : {STATIC_DURATION}s")
    print(f"  Models          : {len(MODEL_RUNS)}")
    print("=" * 70)

    for idx, (category, run_path, short_name) in enumerate(MODEL_RUNS, 1):
        if not os.path.exists(run_path):
            print(f"\n[{idx}/{len(MODEL_RUNS)}] ✗ SKIP (path not found): {run_path}")
            continue

        print(f"\n[{idx}/{len(MODEL_RUNS)}] ══ {category} / {short_name} ══")

        out_dir = GRAPHS_ROOT / category / short_name / "static_torques"
        run_static_eval(run_path=run_path, output_dir=out_dir)

    print("\n" + "=" * 70)
    print("  ✅  Static torque evaluations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
