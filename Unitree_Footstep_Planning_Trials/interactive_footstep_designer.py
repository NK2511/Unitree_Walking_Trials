"""Interactive Footstep Designer for Angad Bipedal Robot."""

import math, os
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from scipy.interpolate import PchipInterpolator
except ImportError as e:
    print(f"[ERROR] {e}"); exit(1)


class FootstepDesigner:

    def __init__(self):
        self.step_width    = 0.20
        self.step_length   = 0.28
        self.output_filename = "plans/custom_plan.txt"

        self.waypoints = np.array([
            [0.0, 0.0], [1.5, 0.2], [3.0, 1.0],
            [4.5, 1.2], [6.0, 0.5], [7.5, 0.0],
        ])
        self.dragged_point_idx = None
        self.steps_data    = None
        self.spline_points = None

        self.fig, self.ax = plt.subplots(figsize=(11, 7.5))
        self.fig.canvas.manager.set_window_title("Angad Footstep Path Designer (PCHP)")
        plt.subplots_adjust(bottom=0.25, left=0.1)
        self.ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Angad Footstep Planner\nDrag black circles to shape path.")

        self.line_spline,  = self.ax.plot([], [], "g-", lw=2,  label="Pelvis Path")
        self.left_scatter  = self.ax.scatter([], [], c="blue", s=80, label="Left")
        self.right_scatter = self.ax.scatter([], [], c="red",  s=80, label="Right")
        self.waypoint_scatter = self.ax.scatter(
            self.waypoints[:,0], self.waypoints[:,1],
            c="black", s=120, zorder=5, label="Control Points")
        self.arrows = []

        ax_w = plt.axes([0.15, 0.12, 0.65, 0.03])
        self.slider_width = Slider(ax_w, "Stance Width (m)", 0.10, 0.40, valinit=self.step_width, valfmt="%.2f")
        ax_l = plt.axes([0.15, 0.07, 0.65, 0.03])
        self.slider_length = Slider(ax_l, "Step Length (m)", 0.15, 0.50, valinit=self.step_length, valfmt="%.2f")

        self.btn_save  = Button(plt.axes([0.08, 0.015, 0.22, 0.04]), "Save Plan",         color="lightblue", hovercolor="skyblue")
        self.btn_sim   = Button(plt.axes([0.33, 0.015, 0.22, 0.04]), "Simulate (Mink)",  color="#c7f9cc",   hovercolor="#80ed99")
        self.btn_reset = Button(plt.axes([0.58, 0.015, 0.15, 0.04]), "Reset Path",        color="lightgray", hovercolor="darkgray")

        self.slider_width.on_changed(self.update_parameters)
        self.slider_length.on_changed(self.update_parameters)
        self.btn_save.on_clicked(self.save_plan)
        self.btn_sim.on_clicked(self.start_simulation)
        self.btn_reset.on_clicked(self.reset_path)
        self.fig.canvas.mpl_connect("button_press_event",   self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event",  self.on_motion)

        self.generate_and_render()
        self.ax.legend(loc="upper left")

    # ── Step generation ───────────────────────────────────────────────────────

    def generate_and_render(self):
        pts  = self.waypoints
        diff = np.diff(pts, axis=0)
        cum  = np.concatenate(([0.], np.cumsum(np.sqrt(np.sum(diff**2, axis=1)))))
        tot  = cum[-1]
        sx   = PchipInterpolator(cum, pts[:,0])
        sy   = PchipInterpolator(cum, pts[:,1])
        eps  = 0.01

        h0 = math.atan2(sy(eps)-sy(0.), sx(eps)-sx(0.))
        px0, py0 = -math.sin(h0), math.cos(h0)

        s_ev = np.linspace(0, tot, 300)
        self.spline_points = np.stack([sx(s_ev), sy(s_ev)], axis=1)

        steps = [
            [+(self.step_width/2)*px0, +(self.step_width/2)*py0, 0., h0, 0],
            [-(self.step_width/2)*px0, -(self.step_width/2)*py0, 0., h0, 1],
        ]
        side = 0
        for d in np.arange(self.step_length, tot, self.step_length):
            xc, yc = sx(d), sy(d)
            h = math.atan2(sy(min(d+eps,tot))-sy(max(d-eps,0.)),
                           sx(min(d+eps,tot))-sx(max(d-eps,0.)))
            px, py = -math.sin(h), math.cos(h)
            sgn = 1. if side == 0 else -1.
            steps.append([xc + sgn*(self.step_width/2)*px,
                          yc + sgn*(self.step_width/2)*py, 0., h, side])
            side = 1 - side

        xe, ye = sx(tot), sy(tot)
        he  = math.atan2(sy(tot)-sy(tot-eps), sx(tot)-sx(tot-eps))
        pxe, pye = -math.sin(he), math.cos(he)
        if side == 0:
            steps.append([xe+(self.step_width/2)*pxe, ye+(self.step_width/2)*pye, 0., he, 0])
            steps.append([xe-(self.step_width/2)*pxe, ye-(self.step_width/2)*pye, 0., he, 1])
        else:
            steps.append([xe-(self.step_width/2)*pxe, ye-(self.step_width/2)*pye, 0., he, 1])
            steps.append([xe+(self.step_width/2)*pxe, ye+(self.step_width/2)*pye, 0., he, 0])

        self.steps_data = np.array(steps)

        self.line_spline.set_data(self.spline_points[:,0], self.spline_points[:,1])
        lm = self.steps_data[:,4] == 0; rm = self.steps_data[:,4] == 1
        self.left_scatter.set_offsets(self.steps_data[lm,:2])
        self.right_scatter.set_offsets(self.steps_data[rm,:2])

        for a in self.arrows: a.remove()
        self.arrows.clear()
        for step in self.steps_data:
            x, y, _, th, s = step
            c = 'blue' if s == 0 else 'red'
            self.arrows.append(self.ax.annotate(
                "", xy=(x+0.08*math.cos(th), y+0.08*math.sin(th)), xytext=(x,y),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.5)))

        mn = np.min(np.vstack([self.waypoints, self.spline_points]), axis=0) - 0.5
        mx = np.max(np.vstack([self.waypoints, self.spline_points]), axis=0) + 0.5
        self.ax.set_xlim(mn[0], mx[0]); self.ax.set_ylim(mn[1], mx[1])
        self.fig.canvas.draw_idle()

    # ── GUI handlers ──────────────────────────────────────────────────────────

    def update_parameters(self, _):
        self.step_width  = self.slider_width.val
        self.step_length = self.slider_length.val
        self.generate_and_render()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        d = np.hypot(self.waypoints[:,0]-event.xdata, self.waypoints[:,1]-event.ydata)
        i = np.argmin(d)
        if d[i] < 0.25: self.dragged_point_idx = i

    def on_motion(self, event):
        if self.dragged_point_idx is None or event.inaxes != self.ax: return
        nx = event.xdata if self.dragged_point_idx != 0 else 0.
        self.waypoints[self.dragged_point_idx] = [nx, event.ydata]
        self.waypoint_scatter.set_offsets(self.waypoints)
        self.generate_and_render()

    def on_release(self, _): self.dragged_point_idx = None

    def reset_path(self, _):
        self.waypoints = np.array([
            [0.,0.],[1.5,0.],[3.,0.],[4.5,0.],[6.,0.],[7.5,0.]])
        self.waypoint_scatter.set_offsets(self.waypoints)
        self.slider_width.reset(); self.slider_length.reset()
        self.generate_and_render()

    def save_plan(self, _):
        if self.steps_data is None: return
        os.makedirs(os.path.dirname(self.output_filename) or ".", exist_ok=True)
        with open(self.output_filename, "w") as f:
            f.write("x,y,z,theta,side\n")
            for s in self.steps_data:
                f.write(f"{s[0]:.6f},{s[1]:.6f},{s[2]:.6f},{s[3]:.6f},{int(s[4])}\n")
        self.btn_save.label.set_text("SAVED!")
        self.btn_save.color = "lightgreen"
        self.fig.canvas.draw_idle()
        print(f"[INFO] Saved {len(self.steps_data)} steps → {self.output_filename}")

    def start_simulation(self, _):
        import threading
        self.btn_sim.label.set_text("SIMULATING...")
        self.btn_sim.color = "#ffd166"
        self.fig.canvas.draw_idle()
        threading.Thread(target=self.run_mink_simulation, daemon=True).start()

    # ── Kinematic simulation ──────────────────────────────────────────────────

    def run_mink_simulation(self):
        """
        Architecture (exactly as described):

        1. Reset to Keyframe 1 (Stable_Stance) ONCE.
        2. For each step k in order (2 .. N-1):
             a. swing_side = steps[k][4]  (the foot that moves)
             b. swing_from = cur[swing_side]  (captured once at step entry)
             c. Foot target traces a CIRCULAR ARC (semicircle) from swing_from
                to steps[k]: XY moves linearly, Z follows sin(pi*tau)*arc_radius,
                where arc_radius = half the XY chord length (true semicircle).
             d. Stance foot = frozen at cur[1-swing_side].
             e. Pelvis (base) target = XY midpoint of swing foot's CURRENT arc
                position and stance foot. Pelvis arrives at final midpoint
                at the SAME time (same tau) as foot lands.
        3. IK runs at 500 Hz; viewer renders at ~60 Hz (rate-limited) to
           prevent visual jitter from flooding the renderer.
        4. orientation_cost=0 for feet — only position is tracked, avoiding
           orientation-fighting jitter from the QP solver.
        """
        import time
        import numpy as np
        import mujoco, mujoco.viewer, mink

        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        XML = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/scene.xml"))
        if not os.path.exists(XML):
            XML = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/g1.xml"))

        steps = self.steps_data
        N     = len(steps) if steps is not None else 0

        if not os.path.exists(XML):
            print(f"[ERROR] XML not found: {XML}"); self._reset_btn(); return
        if N < 4:
            print("[ERROR] Need ≥4 steps."); self._reset_btn(); return

        # ── Load model, SINGLE reset to Keyframe 0 ───────────────────────────
        model = mujoco.MjModel.from_xml_path(XML)
        data  = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)   # Stable_Stance
        mujoco.mj_kinematics(model, data)
        pelvis_z = data.qpos[2]
        print(f"[Sim] Keyframe 1 → base Z = {pelvis_z:.4f} m")

        # Read body IDs
        bid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
        bid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
        bid_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        R_l0  = data.xmat[bid_l].reshape(3,3).copy()
        R_r0  = data.xmat[bid_r].reshape(3,3).copy()
        R_b0  = data.xmat[bid_b].reshape(3,3).copy()
        foot_z_init = 0.5 * (data.xpos[bid_l][2] + data.xpos[bid_r][2])
        print(f"[Sim] Keyframe 1 → foot Z offset = {foot_z_init:.4f} m")

        # ── Mink configuration seeded from KF1 ───────────────────────────────
        cfg = mink.Configuration(model)
        cfg.update(data.qpos)

        # Feet: track both position and orientation. We set orientation cost to 5.0
        # and base orientation cost to 1.5 to keep things well-behaved.
        ft_l = mink.FrameTask("left_foot_link",  "body", position_cost=10., orientation_cost=5.0, lm_damping=1e-3)
        ft_r = mink.FrameTask("right_foot_link", "body", position_cost=10., orientation_cost=5.0, lm_damping=1e-3)
        ft_b = mink.FrameTask("base",            "body", position_cost=2.0, orientation_cost=1.5, lm_damping=1e-3)
        tasks  = [ft_l, ft_r, ft_b]
        limits = [mink.ConfigurationLimit(model)]

        # ── Helpers ───────────────────────────────────────────────────────────
        def Ryaw(yaw):
            c, s = math.cos(yaw), math.sin(yaw)
            return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

        def foot_target(x, y, z, yaw, side):
            """SE3 for foot: position and path-aligned flat orientation."""
            T = np.eye(4); T[:3,3] = [x, y, z + foot_z_init]
            R_init = R_l0 if side == 0 else R_r0
            T[:3,:3] = Ryaw(yaw) @ R_init
            return mink.SE3.from_matrix(T)

        def base_target(x, y, z_val, yaw):
            T = np.eye(4); T[:3,3] = [x, y, z_val]
            T[:3,:3] = Ryaw(yaw) @ R_b0
            return mink.SE3.from_matrix(T)

        def ease(tau):
            return 0.5*(1. - math.cos(math.pi*float(np.clip(tau,0.,1.))))

        def circular_arc(start, end, tau):
            """
            Traces a semicircular arc from start→end in 3D.
            XY: linear interpolation (eased).
            Z:  sin(pi*tau) * R  where R = half the XY chord → true semicircle.
            """
            s  = ease(tau)
            tx = start[0] + s * (end[0] - start[0])
            ty = start[1] + s * (end[1] - start[1])
            d_xy = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            R    = min(d_xy / 2., 0.10)            # semicircle radius, capped at 10 cm
            tz   = start[2] + math.sin(math.pi*tau) * R
            return tx, ty, tz

        # ── Foot state ────────────────────────────────────────────────────────
        # cur[side] = [x,y,z,yaw] — where that foot currently IS (last landed)
        cur = {
            0: np.array(steps[0][:4], dtype=float),   # initial left
            1: np.array(steps[1][:4], dtype=float),   # initial right
        }

        # ── Timeline: one entry per swing step ───────────────────────────────
        T_HOLD = 0.6   # hold at start/end
        T_STEP = 0.8   # time per swing step
        IK_DT  = 0.002
        RENDER_HZ = 60.
        SOLVER = "daqp"

        timeline = []
        t = T_HOLD
        for k in range(2, N):
            timeline.append((t, t + T_STEP, k))
            t += T_STEP
        total_time = t + T_HOLD

        print(f"[Sim] {N-2} swing steps, total {total_time:.1f} s")
        for i,(ts,te,k) in enumerate(timeline):
            print(f"  [{i}] {'L' if steps[k][4]==0 else 'R'}  "
                  f"t=[{ts:.2f}–{te:.2f}]  "
                  f"→ ({steps[k][0]:.3f}, {steps[k][1]:.3f})")

        # ── Simulation loop ───────────────────────────────────────────────────
        tl_idx     = 0
        swing_from = {}          # side → captured start position for current swing
        render_acc = 0.          # accumulates IK time between renders

        # Prime the IK: set initial foot targets at their current positions
        ft_l.set_target(foot_target(cur[0][0], cur[0][1], cur[0][2], cur[0][3], 0))
        ft_r.set_target(foot_target(cur[1][0], cur[1][1], cur[1][2], cur[1][3], 1))
        
        px_init   = 0.5*(cur[0][0]+cur[1][0])
        py_init   = 0.5*(cur[0][1]+cur[1][1])
        pyaw_init = 0.5*(cur[0][3]+cur[1][3])
        ft_b.set_target(base_target(px_init, py_init, pelvis_z, pyaw_init))

        history_t = []
        history_pelvis_z = []
        history_lknee = []
        history_rknee = []
        breakpoints = []

        paused = True
        def key_callback(keycode):
            nonlocal paused
            if keycode == 72:  # GLFW_KEY_H is 72
                if paused:
                    paused = False
                    print("[Sim] Resuming simulation for 1 step... Press 'H' again to pause.")
                else:
                    paused = True
                    print("[Sim] Paused.")

        try:
            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                viewer.cam.azimuth = 140.; viewer.cam.elevation = -25.; viewer.cam.distance = 3.5
                
                print("\n[Sim] Paused at stance. Press 'H' to start the walking sequence step-by-step!")

                sim_t = 0.
                while viewer.is_running() and sim_t < total_time:
                    t0 = time.perf_counter()

                    if paused:
                        # Keep both feet flat during pause/standing stance
                        ft_l.set_orientation_cost(5.0)
                        ft_r.set_orientation_cost(5.0)
                        # Keep solving IK at frozen targets to prevent sagging/drifting
                        mujoco.mj_kinematics(model, cfg.data)
                        vel = mink.solve_ik(cfg, tasks, IK_DT, SOLVER, limits=limits)
                        cfg.integrate_inplace(vel, IK_DT)

                        render_acc += IK_DT
                        if render_acc >= 1./RENDER_HZ:
                            data.qpos[:] = cfg.q[:]
                            data.qvel[:] = 0.
                            mujoco.mj_kinematics(model, data)
                            viewer.sync()
                            render_acc = 0.

                        # Maintain real-time speed in paused state
                        elapsed = time.perf_counter() - t0
                        if elapsed < IK_DT:
                            time.sleep(IK_DT - elapsed)
                        continue

                    # ── Complete any finished steps ───────────────────────────
                    while tl_idx < len(timeline):
                        ts, te, k = timeline[tl_idx]
                        if sim_t >= te:
                            sd = int(steps[k][4])
                            cur[sd] = np.array(steps[k][:4], dtype=float)
                            swing_from.pop(sd, None)
                            tl_idx += 1
                            paused = True  # Pause on step completion!
                            breakpoints.append(sim_t)
                            print(f"[t={sim_t:.3f}] Completed step {k}. Paused. Press 'H' to take the next step.")
                        else:
                            break

                    # ── Find active swing ─────────────────────────────────────
                    swing_side = -1; tau = 0.; k_act = None
                    if tl_idx < len(timeline):
                        ts, te, k_act = timeline[tl_idx]
                        if sim_t >= ts:
                            swing_side = int(steps[k_act][4])
                            tau = (sim_t - ts) / (te - ts)
                            # Capture swing start exactly once at step entry
                            if swing_side not in swing_from:
                                swing_from[swing_side] = cur[swing_side].copy()
                                print(f"[t={sim_t:.3f}] SWING "
                                      f"{'L' if swing_side==0 else 'R'}  "
                                      f"({swing_from[swing_side][0]:.3f},"
                                      f"{swing_from[swing_side][1]:.3f}) → "
                                      f"({steps[k_act][0]:.3f},{steps[k_act][1]:.3f})")

                    # ── Foot targets ──────────────────────────────────────────
                    foot_xy  = {}
                    foot_yaw = {}

                    for side in (0, 1):
                        if side == swing_side and k_act is not None:
                            sf = swing_from[side]
                            # Circular arc trajectory (semicircle)
                            tx, ty, tz = circular_arc(sf, steps[k_act], tau)
                            tyaw = sf[3] + ease(tau)*(steps[k_act][3]-sf[3])
                        else:
                            tx, ty, tz, tyaw = cur[side]   # stance: frozen

                        task = ft_l if side == 0 else ft_r
                        task.set_target(foot_target(tx, ty, tz, tyaw, side))
                        foot_xy[side]  = (tx, ty)
                        foot_yaw[side] = tyaw

                    # ── Pelvis = midpoint of both foot XY targets ─────────────
                    # Both foot and pelvis arrive at final positions at the same tau.
                    px   = 0.5*(foot_xy[0][0] + foot_xy[1][0])
                    py   = 0.5*(foot_xy[0][1] + foot_xy[1][1])
                    pyaw = 0.5*(foot_yaw[0]   + foot_yaw[1])
                    
                    # Torso/Pelvis target remains at a constant height, moving in a perfectly flat, horizontal line in 3D
                    ft_b.set_target(base_target(px, py, pelvis_z, pyaw))

                    # ── Dynamic orientation cost based on swing/stance ────────
                    ft_l.set_orientation_cost(0.0 if swing_side == 0 else 5.0)
                    ft_r.set_orientation_cost(0.0 if swing_side == 1 else 5.0)

                    # ── IK solve (500 Hz) ─────────────────────────────────────
                    mujoco.mj_kinematics(model, cfg.data)
                    vel = mink.solve_ik(cfg, tasks, IK_DT, SOLVER, limits=limits)
                    cfg.integrate_inplace(vel, IK_DT)

                    # ── Render at 60 Hz only (prevents renderer flooding) ─────
                    render_acc += IK_DT
                    if render_acc >= 1./RENDER_HZ:
                        data.qpos[:] = cfg.q[:]
                        data.qvel[:] = 0.
                        mujoco.mj_kinematics(model, data)
                        
                        # Print diagnostics at 10 Hz
                        if int(sim_t * 1000) % 100 == 0:
                            p_pos = data.xpos[bid_b]
                            lf_pos = data.xpos[bid_l]
                            rf_pos = data.xpos[bid_r]
                            dist_l = np.linalg.norm(p_pos - lf_pos)
                            dist_r = np.linalg.norm(p_pos - rf_pos)
                            
                            left_lbl = "SWING" if swing_side == 0 else "STANCE" if swing_side == 1 else "HOLD"
                            right_lbl = "SWING" if swing_side == 1 else "STANCE" if swing_side == 0 else "HOLD"
                            
                            lknee = data.qpos[7+9]
                            rknee = data.qpos[7+3]
                            lankle_p = data.qpos[7+10]
                            rankle_p = data.qpos[7+4]
                            
                            d_xy = math.sqrt((foot_xy[0][0]-foot_xy[1][0])**2 + (foot_xy[0][1]-foot_xy[1][1])**2)
                            stride_offset = max(0.0, d_xy - self.step_width)
                            
                            print(f"t={sim_t:.2f} | Pelvis Z={p_pos[2]:.4f} | Stride={d_xy:.3f} | Offset={stride_offset:.3f}")
                            print(f"  L-Foot ({left_lbl}): Knee={lknee:.3f}, AnkleP={lankle_p:.3f}, Dist={dist_l:.4f}")
                            print(f"  R-Foot ({right_lbl}): Knee={rknee:.3f}, AnkleP={rankle_p:.3f}, Dist={dist_r:.4f}")

                        # Collect data for plotting
                        history_t.append(sim_t)
                        history_pelvis_z.append(data.xpos[bid_b][2])
                        history_lknee.append(data.qpos[7+9])
                        history_rknee.append(data.qpos[7+3])

                        viewer.sync()
                        render_acc = 0.

                    sim_t += IK_DT
                    elapsed = time.perf_counter() - t0
                    if IK_DT - elapsed > 0:
                        time.sleep(IK_DT - elapsed)

        except Exception as ex:
            import traceback; print(f"[ERROR] {ex}"); traceback.print_exc()

        # Plot the collected data
        if len(history_t) > 0:
            print("\n[Sim] Generating diagnostics plot...")
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
            
            ax1.plot(history_t, history_pelvis_z, label='Pelvis Z', color='blue', lw=2)
            ax1.set_ylabel('Pelvis Z (m)')
            ax1.set_title('Pelvis Height and Knee Angles')
            ax1.grid(True)
            
            ax2.plot(history_t, history_lknee, label='Left Knee Pitch', color='green', lw=1.5)
            ax2.plot(history_t, history_rknee, label='Right Knee Pitch', color='orange', lw=1.5)
            ax2.set_xlabel('Simulation Time (s)')
            ax2.set_ylabel('Knee Angle (rad)')
            ax2.grid(True)
            
            # Add vertical lines for breakpoints
            for bp in breakpoints:
                ax1.axvline(x=bp, color='red', linestyle='--', alpha=0.7, label='Step Completed' if bp == breakpoints[0] else "")
                ax2.axvline(x=bp, color='red', linestyle='--', alpha=0.7)
            
            ax1.legend()
            ax2.legend()
            plt.tight_layout()
            
            # Save the figure to a file instead of displaying it interactively (to avoid thread issues)
            plot_path = os.path.join(os.path.dirname(__file__), "gait_diagnostics.png")
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"[Sim] Saved diagnostics plot to: {os.path.abspath(plot_path)}")

        self._reset_btn()

    def _reset_btn(self):
        self.btn_sim.label.set_text("Simulate (Mink)")
        self.btn_sim.color = "#c7f9cc"
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    designer = FootstepDesigner()
    plt.show()
