"""Footstep Path Generator for Angad Bipedal Robot.

This script provides utilities to generate custom footstep plans (sequences of
left and right foot targets) along arbitrary 2D/3D continuous paths (e.g.,
straight lines, circles, sine waves, or waypoint-defined curves).

It generates alternating footsteps offset by `step_width` perpendicular to the
path direction, and spaced by `step_length` along the path.

The generated footsteps are saved as standard text plans (x, y, z, theta, side)
that can be dynamically loaded by the FootstepManager.
"""

import argparse
import math
import os
import numpy as np


class PathGenerator:
    """Generates continuous geometric paths to place footsteps on."""

    @staticmethod
    def straight_line(length=10.0, z_slope=0.0):
        """Generates waypoints for a straight line along the X axis.

        Returns:
            waypoints: np.ndarray of shape [N, 3] (x, y, z)
        """
        num_points = 100
        x = np.linspace(0, length, num_points)
        y = np.zeros_like(x)
        z = x * z_slope
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def circle(radius=3.0, angle_degrees=360.0):
        """Generates waypoints for a circular path starting at (0,0) tangent to X axis.

        Returns:
            waypoints: np.ndarray of shape [N, 3] (x, y, z)
        """
        num_points = 200
        angles = np.radians(np.linspace(0, angle_degrees, num_points))
        # Center the circle so that spawn is at (0,0) and the path goes tangent to the X axis
        # Parametric circle: x = R*sin(a), y = R*(1 - cos(a))
        x = radius * np.sin(angles)
        y = radius * (1.0 - np.cos(angles))
        z = np.zeros_like(angles)
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def sine_wave(length=10.0, amplitude=1.0, wavelength=5.0):
        """Generates waypoints for a sine wave path along the X axis.

        Returns:
            waypoints: np.ndarray of shape [N, 3] (x, y, z)
        """
        num_points = 200
        x = np.linspace(0, length, num_points)
        y = amplitude * np.sin(2.0 * np.pi * x / wavelength)
        z = np.zeros_like(x)
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def custom_waypoints(waypoints_list):
        """Interpolates between custom user waypoints to create a smooth path using PchipInterpolator if available.

        Args:
            waypoints_list: List or array of shape [K, 3] (x, y, z)
        Returns:
            waypoints: np.ndarray of shape [N, 3] (x, y, z)
        """
        pts = np.array(waypoints_list)
        # Calculate cumulative distance along waypoints
        diffs = np.diff(pts, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dists = np.concatenate(([0.0], np.cumsum(dists)))
        
        # Interp along total distance
        total_dist = cum_dists[-1]
        eval_dists = np.linspace(0, total_dist, max(200, int(total_dist * 20)))
        
        try:
            from scipy.interpolate import PchipInterpolator
            x_spline = PchipInterpolator(cum_dists, pts[:, 0])
            y_spline = PchipInterpolator(cum_dists, pts[:, 1])
            z_spline = PchipInterpolator(cum_dists, pts[:, 2])
            x = x_spline(eval_dists)
            y = y_spline(eval_dists)
            z = z_spline(eval_dists)
        except ImportError:
            x = np.interp(eval_dists, cum_dists, pts[:, 0])
            y = np.interp(eval_dists, cum_dists, pts[:, 1])
            z = np.interp(eval_dists, cum_dists, pts[:, 2])

        return np.stack([x, y, z], axis=1)

    @staticmethod
    def random_spline(
        num_control_points=5,
        segment_dist_range=(4.0, 5.0),
        max_heading_offset_deg=20.0,
        seed=None,
    ):
        """Generates waypoints for a smooth random spline path.

        Starting from (0,0,0) facing +X (heading 0), each successive control point is placed
        at a distance in `segment_dist_range` (default 4-5m) and heading offset in `+-max_heading_offset_deg` (default +-20 deg).
        A smooth spline is then interpolated through these control points.

        Args:
            num_control_points: Number of random waypoints to chain (default 5).
            segment_dist_range: (min_dist, max_dist) between waypoints in meters (default 4.0-5.0m).
            max_heading_offset_deg: Max heading perturbation angle per waypoint in degrees (default 20.0°).
            seed: Optional random seed for reproducible random paths.

        Returns:
            waypoints: np.ndarray of shape [N, 3] (x, y, z)
        """
        if seed is not None:
            np.random.seed(seed)

        pts = [[0.0, 0.0, 0.0]]
        curr_heading = 0.0
        curr_x, curr_y = 0.0, 0.0

        for _ in range(num_control_points):
            dist = np.random.uniform(segment_dist_range[0], segment_dist_range[1])
            d_heading = np.radians(np.random.uniform(-max_heading_offset_deg, max_heading_offset_deg))
            curr_heading += d_heading
            curr_x += dist * np.cos(curr_heading)
            curr_y += dist * np.sin(curr_heading)
            pts.append([curr_x, curr_y, 0.0])

        return PathGenerator.custom_waypoints(pts)


def generate_footsteps_along_path(waypoints, step_length=0.28, step_width=0.12, initial_side="left"):
    """Places alternating left/right footsteps offset perpendicular to a 3D path.

    Args:
        waypoints: np.ndarray of shape [N, 3] (x, y, z) representing the path centerline.
        step_length: Target forward distance along path between footsteps (meters).
        step_width: Stance width offset perpendicular to the path (meters).
        initial_side: "left" or "right" to specify which foot takes the first step.

    Returns:
        steps: np.ndarray of shape [M, 5] (x, y, z, theta, foot_side)
               foot_side: 0 = Left, 1 = Right
    """
    # Calculate step distances along path
    diffs = np.diff(waypoints, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dists = np.concatenate(([0.0], np.cumsum(dists)))
    total_path_length = cum_dists[-1]

    # Sample step progress locations along the centerline
    target_dists = np.arange(step_length, total_path_length, step_length)
    if len(target_dists) == 0:
        raise ValueError("Path is shorter than a single step length.")

    steps = []
    side = 0 if initial_side == "left" else 1

    for d in target_dists:
        # Find position on centerline
        x_center = np.interp(d, cum_dists, waypoints[:, 0])
        y_center = np.interp(d, cum_dists, waypoints[:, 1])
        z_center = np.interp(d, cum_dists, waypoints[:, 2])

        # Get local tangent vector by taking finite differences
        eps = 0.05
        d_next = min(d + eps, total_path_length)
        d_prev = max(d - eps, 0.0)
        
        x_next = np.interp(d_next, cum_dists, waypoints[:, 0])
        y_next = np.interp(d_next, cum_dists, waypoints[:, 1])
        x_prev = np.interp(d_prev, cum_dists, waypoints[:, 0])
        y_prev = np.interp(d_prev, cum_dists, waypoints[:, 1])

        dx = x_next - x_prev
        dy = y_next - y_prev
        heading = math.atan2(dy, dx)

        # Perpendicular vector pointing "left" relative to heading direction: (-sin, cos)
        perp_x = -math.sin(heading)
        perp_y = math.cos(heading)

        # Offset left steps positive, right steps negative
        sign = 1.0 if side == 0 else -1.0
        x_step = x_center + sign * (step_width / 2.0) * perp_x
        y_step = y_center + sign * (step_width / 2.0) * perp_y
        z_step = z_center

        steps.append([x_step, y_step, z_step, heading, side])
        
        # Alternate foot side
        side = 1 - side

    return np.array(steps)


def save_footstep_plan(steps, filename):
    """Saves the step sequence to a text file.

    Format:
        x,y,z,theta,side
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    with open(filename, "w") as f:
        f.write("x,y,z,theta,side\n")
        for step in steps:
            f.write(f"{step[0]:.6f},{step[1]:.6f},{step[2]:.6f},{step[3]:.6f},{int(step[4])}\n")
    print(f"[INFO] Successfully saved {len(steps)} footsteps to {filename}")


def plot_footsteps(waypoints, steps):
    """Visualizes the centerline path and placing footsteps using Matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib is not installed. Skipping visualization.")
        return

    plt.figure(figsize=(10, 8))
    # Plot centerline path
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'k--', alpha=0.5, label="Path Centerline")

    # Plot left steps (blue) and right steps (red)
    left_mask = steps[:, 4] == 0
    right_mask = steps[:, 4] == 1

    plt.scatter(steps[left_mask, 0], steps[left_mask, 1], c='blue', marker='o', s=100, label="Left Target Footsteps")
    plt.scatter(steps[right_mask, 0], steps[right_mask, 1], c='red', marker='o', s=100, label="Right Target Footsteps")

    # Add orientation arrows
    for step in steps:
        x, y, _, theta, side = step
        color = 'blue' if side == 0 else 'red'
        arrow_len = 0.1
        plt.arrow(x, y, arrow_len * math.cos(theta), arrow_len * math.sin(theta),
                  head_width=0.04, head_length=0.04, fc=color, ec=color)

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.title("Generated Alternating Footsteps along Custom Path")
    plt.legend()
    
    # Save a copy as visualization artifact
    plt.savefig("footstep_plan_visualisation.png")
    print("[INFO] Saved footstep visualization to footstep_plan_visualisation.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate footprint patterns for Angad.")
    parser.add_argument("--type", type=str, default="sine", choices=["straight", "circle", "sine", "custom", "rand_spline"],
                        help="Type of reference path to generate.")
    parser.add_argument("--step_len", type=float, default=0.28, help="Target forward step length (m).")
    parser.add_argument("--step_wid", type=float, default=0.12, help="Stance width between feet (m).")
    parser.add_argument("--out", type=str, default="plans/sine_plan.txt", help="Output path for target file.")
    parser.add_argument("--plot", action="store_true", help="Plot and visualize footsteps.")
    args = parser.parse_args()

    # Generate waypoints
    if args.type == "straight":
        waypoints = PathGenerator.straight_line(length=8.0)
    elif args.type == "circle":
        waypoints = PathGenerator.circle(radius=2.5, angle_degrees=270.0)
    elif args.type == "sine":
        waypoints = PathGenerator.sine_wave(length=8.0, amplitude=0.4, wavelength=4.0)
    elif args.type == "rand_spline":
        waypoints = PathGenerator.random_spline(
            num_control_points=5,
            segment_dist_range=(4.0, 5.0),
            max_heading_offset_deg=20.0,
        )
    elif args.type == "custom":
        # S-shape custom curve example
        waypoints = PathGenerator.custom_waypoints([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 1.5, 0.0],
            [6.0, 1.5, 0.0],
            [8.0, 0.0, 0.0]
        ])

    steps = generate_footsteps_along_path(waypoints, step_length=args.step_len, step_width=args.step_wid)
    save_footstep_plan(steps, args.out)

    if args.plot:
        plot_footsteps(waypoints, steps)
