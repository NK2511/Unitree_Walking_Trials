#!/usr/bin/env python3
"""
Angad Gait Evaluation — Graph Generation Script
================================================
This is the STANDARD script to regenerate all graphs for all models.

HOW TO USE:
    python generate_graphs.py

WHAT IT DOES:
    1. For each model run defined below, runs the Angad_Gait_Evaluator at each
       speed in EVAL_SPEEDS and saves graphs into a subfolder per speed:
           graphs/<No_Torso|With_Torso>/<model_name>/speed_<X.X>/
    2. Additionally runs one velocity-sweep evaluation per model and saves into:
           graphs/<No_Torso|With_Torso>/<model_name>/velocity_sweep/

EDIT THE SECTION BELOW TO CONFIGURE EVERYTHING:
"""

import os
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ██  USER CONFIGURATION — Edit everything here  ██
# ─────────────────────────────────────────────────────────────────────────────

# Python interpreter (mjlab virtual environment)
PYTHON_EXE = "/home/nandhith/Python/Humanoid_Xterra_IITK/mjlab_env/bin/python"

# Path to the evaluator script
EVALUATOR_SCRIPT = str(Path(__file__).resolve().parent / "policy_evaluator.py")

# Speeds to evaluate at (one subfolder per speed will be created)
EVAL_SPEEDS = [0.5, 1.0, 1.5, 2.0, 3.0]  # m/s

# Max speed for the velocity sweep run (0 → SWEEP_MAX_SPEED ramp over SWEEP_DURATION s)
SWEEP_MAX_SPEED = 3.0    # m/s
SWEEP_DURATION  = 20.0   # seconds

# Fixed-speed evaluation duration
FIXED_DURATION = 12.0    # seconds

# ── Model Runs ────────────────────────────────────────────────────────────────
# Each entry:  (category, abs_path_to_run_folder, short_name_for_output_subfolder)
#   category   : "No_Torso" or "With_Torso" (determines output parent folder)
#   abs_path   : full absolute path to the training run logs folder
#   short_name : name used as the output subfolder under graphs/<category>/

_DEFAULT_LOGS = Path(__file__).resolve().parents[1] / "Unitree_Asimov_Trials" / "Walk" / "logs" / "rsl_rl" / "unitree_velocity"
_ALT_LOGS = Path(__file__).resolve().parent / "logs" / "rsl_rl" / "unitree_velocity"
_LOGS_ROOT = str(_DEFAULT_LOGS if _DEFAULT_LOGS.exists() else _ALT_LOGS)

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


def run_eval(run_path: str, output_dir: Path, speed: float, duration: float,
             sweep: bool = False) -> bool:
    """Run the evaluator and move the output into output_dir.

    The evaluator script saves its output directly inside:
        graphs/<No_Torso|With_Torso>/<run_folder_name>/
    We rename/move that to output_dir after completion so that every speed
    gets its own clean subfolder.

    Returns True if the evaluator exited cleanly, False otherwise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXE,
        EVALUATOR_SCRIPT,
        run_path,
        "--no-render",
        "--speed", str(speed),
        "--duration", str(duration),
    ]
    if sweep:
        cmd.append("--sweep")

    label = f"sweep 0→{speed} m/s" if sweep else f"{speed} m/s"
    print(f"    ▶  {label}  →  {output_dir}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"    ✗  Evaluator exited with code {result.returncode}")
        return False

    # The evaluator writes directly into graphs/<category>/<run_folder_name>/
    # We need to move those files into our speed subfolder.
    run_folder_name = Path(run_path).name
    # Detect category from path
    category = "With_Torso" if "With_Torso" in run_path else "No_Torso"
    evaluator_default_out = GRAPHS_ROOT / category / run_folder_name

    if evaluator_default_out.exists() and evaluator_default_out != output_dir:
        # Move every file from evaluator_default_out into output_dir
        for item in evaluator_default_out.iterdir():
            dest = output_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    import shutil
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            item.rename(dest)
        # Remove the now-empty default output folder if empty
        try:
            evaluator_default_out.rmdir()
        except OSError:
            pass  # Not empty — leave it

    print(f"    ✓  Saved to {output_dir}")
    return True


def main() -> None:
    print("=" * 70)
    print("  Angad Gait Graph Generator")
    print(f"  Fixed speeds : {EVAL_SPEEDS} m/s  ×  {FIXED_DURATION}s each")
    print(f"  Velocity sweep: 0 → {SWEEP_MAX_SPEED} m/s over {SWEEP_DURATION}s")
    print(f"  Models       : {len(MODEL_RUNS)}")
    print("=" * 70)

    for idx, (category, run_path, short_name) in enumerate(MODEL_RUNS, 1):
        if not os.path.exists(run_path):
            print(f"\n[{idx}/{len(MODEL_RUNS)}] ✗ SKIP (path not found): {run_path}")
            continue

        print(f"\n[{idx}/{len(MODEL_RUNS)}] ══ {category} / {short_name} ══")

        model_out_root = GRAPHS_ROOT / category / short_name

        # ── Fixed-speed evaluations ───────────────────────────────────────────
        for speed in EVAL_SPEEDS:
            speed_label = f"speed_{speed:.1f}".replace(".", "_")  # e.g. speed_0_5
            out_dir = model_out_root / speed_label
            run_eval(
                run_path=run_path,
                output_dir=out_dir,
                speed=speed,
                duration=FIXED_DURATION,
                sweep=False,
            )

        # ── Velocity sweep evaluation ─────────────────────────────────────────
        sweep_out_dir = model_out_root / "velocity_sweep"
        run_eval(
            run_path=run_path,
            output_dir=sweep_out_dir,
            speed=SWEEP_MAX_SPEED,
            duration=SWEEP_DURATION,
            sweep=True,
        )

    print("\n" + "=" * 70)
    print("  ✅  All evaluations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
