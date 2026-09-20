"""
Angad Create Gait
==================
Phase 1: MuJoCo viewer - pose right foot, press S to save waypoints, G to open GUI.
Phase 2: Tkinter GUI - edit edge speeds, adjust phase, play/stop/save.
"""
import sys, os
# Auto-relaunch inside the correct virtual environment if not already using it
VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mjlab_env/bin/python"))
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    print(f"🔄 Auto-switching to mjlab_env Python interpreter...")
    os.execl(VENV_PYTHON, VENV_PYTHON, *sys.argv)

import csv, time, threading, json, re, argparse
import numpy as np
import mujoco, mujoco.viewer, mink
from scipy.interpolate import PchipInterpolator
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

XML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../xmls/Angad/Angad_Primitive.xml"))
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "csvs/angad_walking_reference.csv"))
KEYFRAME_INDEX = 1
N_FRAMES       = 120
INCR           = 0.005

XML_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/scene.xml"))
if not os.path.exists(XML_PATH):
    XML_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../xmls/unitree_g1/g1.xml"))

OUT_CSV  = os.path.abspath(os.path.join(CURRENT_DIR, "csvs/unitree_g1_single_speed_gait.csv"))
KEYFRAME_INDEX = 0

JMAP = {
    "left_hip_pitch_joint":    1.0,
    "left_hip_roll_joint":     1.0,
    "left_hip_yaw_joint":      1.0,
    "left_knee_joint":         1.0,
    "left_ankle_pitch_joint":  1.0,
    "left_ankle_roll_joint":   1.0,
    "right_hip_pitch_joint":   1.0,
    "right_hip_roll_joint":    1.0,
    "right_hip_yaw_joint":     1.0,
    "right_knee_joint":        1.0,
    "right_ankle_pitch_joint": 1.0,
    "right_ankle_roll_joint":  1.0,
}

# ── Global state ──────────────────────────────────────────────────────────────
model = data = configuration = None
rfoot_task = lfoot_task = base_task = posture_task = None
mocap_rf_id = None
waypoints    = []   # list of {"pos": (x,y,z), "weight": float}
selected_idx = 0
stance_qpos  = rf_stance_world = lf_stance_world = None
rf_site_id   = lf_site_id = None
open_gui_flag = False
args = None


def print_status():
    if not waypoints:
        print("\n[No waypoints]"); return
    total = sum(w["weight"] for w in waypoints)
    print(f"\n[Waypoints: {len(waypoints)}]")
    for i, wp in enumerate(waypoints):
        x,y,z = wp["pos"]; w = wp["weight"]
        sel = "==>" if i==selected_idx else "   "
        print(f" {sel} {i+1}: X={x:.3f} Z={z:.3f} | w={w:.2f} ({w/total*100:.0f}%)")

def key_callback(keycode):
    global waypoints, selected_idx, open_gui_flag
    if   keycode == 262: data.mocap_pos[mocap_rf_id][0] += INCR
    elif keycode == 263: data.mocap_pos[mocap_rf_id][0] -= INCR
    elif keycode == 265: data.mocap_pos[mocap_rf_id][1] -= INCR
    elif keycode == 264: data.mocap_pos[mocap_rf_id][1] += INCR
    elif keycode == 266: data.mocap_pos[mocap_rf_id][2] += INCR
    elif keycode == 267: data.mocap_pos[mocap_rf_id][2] -= INCR
    elif keycode in (ord('S'), ord('s')):
        mujoco.mj_kinematics(model, data)
        pos = data.site_xpos[rf_site_id].copy()
        waypoints.append({"pos": tuple(pos), "weight": 1.0})
        selected_idx = len(waypoints)-1
        print(f"[SAVED WP {len(waypoints)}]  X={pos[0]:.4f}  Z={pos[2]:.4f}")
        print_status()
    elif keycode in (ord('D'), ord('d')):
        if waypoints: waypoints.pop(); selected_idx = max(0, len(waypoints)-1)
        print_status()
    elif keycode == 258:  # TAB
        if waypoints: selected_idx=(selected_idx+1)%len(waypoints); print_status()
    elif keycode in (ord('='), ord('+'), 61):
        if waypoints: waypoints[selected_idx]["weight"] = round(waypoints[selected_idx]["weight"]+0.1,2); print_status()
    elif keycode in (ord('-'), 45):
        if waypoints: waypoints[selected_idx]["weight"] = max(0.1, round(waypoints[selected_idx]["weight"]-0.1,2)); print_status()
    elif keycode in (ord('R'), ord('r')):
        waypoints.clear(); selected_idx=0; print("[RESET]")
    elif keycode in (ord('G'), ord('g')):
        if len(waypoints) < 2: print("[Need >= 2 waypoints]"); return
        open_gui_flag = True

# ── Spline helpers ────────────────────────────────────────────────────────────
def build_spline_spatial(wp_list):
    """Spatial spline with UNIFORM t — shape never changes with speed.
    Auto-snaps first and last waypoints to ground Z.
    Inserts stance (0,0) between last WP and first WP so the return
    segment always passes through the origin and stays near Z=0.
    Returns n+1 segment spline (n user edges + 1 return-through-stance edge).
    """
    n   = len(wp_list)
    pts = np.array([w["pos"] for w in wp_list], dtype=float)

    # Snap first and last waypoints to ground Z
    gnd = rf_stance_world[2]
    pts[0, 2]  = gnd
    pts[-1, 2] = gnd

    # Close loop through stance:  WP1 → ... → WP_n → Stance → WP1
    # This gives n+2 control points → n+1 segments
    pc  = np.vstack([pts, rf_stance_world, pts[0]])
    t_u = np.linspace(0, 1, len(pc))   # uniform over n+1 segments

    return (PchipInterpolator(t_u, pc[:,0]),
            PchipInterpolator(t_u, pc[:,1]),
            PchipInterpolator(t_u, pc[:,2]),
            t_u)


def cycloidal_warp(tau):
    return tau - np.sin(2*np.pi*tau) / (2*np.pi)

def temporal_t_map(speeds, stance_speed, n_frames):
    swing_speed = speeds[0]
    n = len(speeds)  # number of user waypoints

    # Swing phase has n - 1 segments
    # Stance phase has 2 segments (WPn -> Stance -> WP1)
    w_swing = (n - 1) / max(0.1, swing_speed)
    w_stance = 2.0 / max(0.1, stance_speed)

    cum_frac = [0.0, w_swing / (w_swing + w_stance), 1.0]
    t_bounds = [0.0, (n - 1) / (n + 1), 1.0]

    # one global cycloid over the whole stride — zero vel/accel only at the
    # stance crossing, smooth single hump through every waypoint in between
    tau = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    p   = cycloidal_warp(tau)

    return np.interp(p, cum_frac, t_bounds)


def solve_ik_traj(speeds, stance_speed, phase_frac, hip_sway=0.03):
    cs_x, cs_y, cs_z, _ = build_spline_spatial(waypoints)
    t  = temporal_t_map(speeds, stance_speed, N_FRAMES)          # non-uniform, respects speeds
    tx, ty, tz = cs_x(t), cs_y(t), cs_z(t)
    tz = np.maximum(tz, rf_stance_world[2])

    d_tmp = mujoco.MjData(model); d_tmp.qpos[:]=stance_qpos
    mujoco.mj_forward(model,d_tmp)
    rf_rot = d_tmp.site_xmat[rf_site_id].reshape(3,3).copy()
    lf_rot = d_tmp.site_xmat[lf_site_id].reshape(3,3).copy()
    lf_st  = lf_stance_world.copy()
    pb_id  = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"base")
    T_pel  = np.eye(4); T_pel[:3,:3]=d_tmp.xmat[pb_id].reshape(3,3); T_pel[:3,3]=d_tmp.xpos[pb_id]
    lf_dx = tx - rf_stance_world[0]; lf_dz = tz - rf_stance_world[2]
    half = int(round(phase_frac*N_FRAMES))

    cfg = mink.Configuration(model); cfg.update(stance_qpos)
    rf_t = mink.FrameTask("right_foot_site","site",position_cost=100.,orientation_cost=50.,lm_damping=1e-2)
    lf_t = mink.FrameTask("left_foot_site","site", position_cost=100.,orientation_cost=50.,lm_damping=1e-2)
    pel_t = mink.FrameTask("base","body",position_cost=2000.,orientation_cost=2000.,lm_damping=1e-2)
    pel_t.set_target(mink.SE3.from_matrix(T_pel))
    post_t = mink.PostureTask(model,cost=1e-4); post_t.set_target(stance_qpos)
    tasks=[rf_t,lf_t,pel_t,post_t]; limits=[mink.ConfigurationLimit(model)]
    DT,MAX_IT,TOL = 0.002,100,8e-4
    results=[]
    for idx in range(N_FRAMES):
        T_rf=np.eye(4); T_rf[:3,:3]=rf_rot; T_rf[:3,3]=[tx[idx],ty[idx],tz[idx]]
        rf_t.set_target(mink.SE3.from_matrix(T_rf))
        j=(idx+half)%N_FRAMES
        T_lf=np.eye(4); T_lf[:3,:3]=lf_rot; T_lf[:3,3]=[lf_st[0]+lf_dx[j],lf_st[1],lf_st[2]+lf_dz[j]]
        lf_t.set_target(mink.SE3.from_matrix(T_lf))

        # Apply side-to-side hip oscillation (sway) towards stance foot (Y-axis)
        T_pel_curr = T_pel.copy()
        T_pel_curr[1, 3] += hip_sway * np.sin(2.0 * np.pi * t[idx])
        pel_t.set_target(mink.SE3.from_matrix(T_pel_curr))

        for _ in range(MAX_IT):
            vel=mink.solve_ik(cfg,tasks,DT,"daqp",limits=limits)
            cfg.integrate_inplace(vel,DT); mujoco.mj_kinematics(model,cfg.data)
            er=np.linalg.norm(cfg.data.site_xpos[rf_site_id]-T_rf[:3,3])
            el=np.linalg.norm(cfg.data.site_xpos[lf_site_id]-T_lf[:3,3])
            if er<TOL and el<TOL: break
        results.append(cfg.q.copy())
        if idx%100==0: print(f"  IK [{idx}/{N_FRAMES}]  R={er*1e3:.1f}mm L={el*1e3:.1f}mm")
    return results

def save_csv(traj, save_path=None):
    if save_path is None:
        save_path = OUT_CSV
    cols=list(JMAP.keys())
    with open(save_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(cols)
        for qpos in traj:
            row=[]
            for jn in JMAP.keys():
                jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,jn)
                row.append(qpos[model.jnt_qposadr[jid]])
            w.writerow(row)
    print(f"[SAVED] {len(traj)} frames → {save_path}")

# ── GUI ───────────────────────────────────────────────────────────────────────
def speed_to_hex(speed):
    t = min(1.0, max(0.0,(speed-0.1)/4.9))
    r=int(34+221*t); g=int(197-197*t); b=int(94-94*t)
    return f"#{r:02x}{g:02x}{b:02x}"


class GaitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Angad Gait Editor")
        self.configure(bg="#0f172a")
        self.resizable(True,True)

        self.swing_speed_var = tk.DoubleVar(value=1.0)
        self.stance_speed_var = tk.DoubleVar(value=1.0)
        self.phase_var   = tk.DoubleVar(value=50.0)
        self.hip_sway_var = tk.DoubleVar(value=0.03)
        self.pb_speed    = tk.StringVar(value="1.0")
        
        default_filename = "angad_walking_reference.csv"
        if args and args.out:
            default_filename = args.out
        self.filename_var = tk.StringVar(value=default_filename)

        self._traj       = None
        self._viewer_running = False
        self._viewer_thread  = None

        self._build()
        self._redraw()

    def _build(self):
        self.L = tk.Frame(self,bg="#0f172a"); self.L.pack(side="left",fill="both",expand=True,padx=(14,4),pady=14)
        self.R = tk.Frame(self,bg="#0f172a"); self.R.pack(side="right",fill="y",padx=(4,14),pady=14)

        # ── Graph ──
        tk.Label(self.L,text="Foot Trajectory  (X=forward, Z=up, origin=stance)",
                 bg="#0f172a",fg="#38bdf8",font=("Segoe UI",11,"bold")).pack(anchor="w")
        self.fig = Figure(figsize=(6.5,5.5),facecolor="#1e293b")
        self.ax  = self.fig.add_subplot(111,facecolor="#1e293b")
        self.cvs = FigureCanvasTkAgg(self.fig,master=self.L)
        self.cvs.get_tk_widget().pack(fill="both",expand=True,pady=(6,0))

        # Drag state
        self._drag_idx = None
        self.cvs.mpl_connect('button_press_event',   self._on_press)
        self.cvs.mpl_connect('motion_notify_event',  self._on_motion)
        self.cvs.mpl_connect('button_release_event', self._on_release)

        # ── Sliders Frame (remakeable) ──
        self.sliders_container = tk.Frame(self.R,bg="#0f172a")
        self.sliders_container.pack(fill="x",pady=(0,8))
        self._build_sliders()

        # ── Phase ──
        tk.Label(self.R,text="Left Leg Phase (% of cycle)",bg="#0f172a",fg="#38bdf8",
                 font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(14,2))
        tk.Scale(self.R,variable=self.phase_var,from_=0,to=100,resolution=1,
                 orient="horizontal",length=240,bg="#0f172a",fg="#f8fafc",
                 highlightthickness=0,troughcolor="#334155",
                 activebackground="#38bdf8",
                 command=lambda _: setattr(self,'_traj',None)).pack(anchor="w")
        tk.Label(self.R,text="Default 50% = 180° out of phase",bg="#0f172a",fg="#64748b",
                 font=("Segoe UI",8,"italic")).pack(anchor="w")

        # ── Playback speed ──
        spf = tk.Frame(self.R,bg="#0f172a"); spf.pack(anchor="w",pady=(14,6))
        tk.Label(spf,text="Speed:",bg="#0f172a",fg="#f8fafc",font=("Segoe UI",9)).pack(side="left")
        for s in ["0.5","1.0","2.0","3.0","4.0","5.0"]:
            tk.Radiobutton(spf,text=f"{s}×",variable=self.pb_speed,value=s,
                           bg="#0f172a",fg="#f8fafc",selectcolor="#1e3a5f",
                           activebackground="#0f172a",font=("Segoe UI",9)).pack(side="left",padx=2)

        # ── Clipboard Copy/Load ──
        cpl_frame = tk.Frame(self.R,bg="#0f172a")
        cpl_frame.pack(anchor="w",pady=(10,4))
        tk.Button(cpl_frame,text="📋 Copy WPs",command=self._copy_waypoints,
                  bg="#475569",fg="white",font=("Segoe UI",9,"bold"),
                  pady=4,padx=8,borderwidth=0,cursor="hand2").pack(side="left",padx=(0,6))
        tk.Button(cpl_frame,text="📥 Load WPs",command=self._load_waypoints,
                  bg="#475569",fg="white",font=("Segoe UI",9,"bold"),
                  pady=4,padx=8,borderwidth=0,cursor="hand2").pack(side="left")

        # ── Auto-tune Cadence ──
        tune_frame = tk.Frame(self.R,bg="#0f172a")
        tune_frame.pack(anchor="w",pady=(0,10))
        tk.Button(tune_frame,text="⚖️ Match Swing/Stance Time",command=self._symmetrize_speeds,
                  bg="#4f46e5",fg="white",font=("Segoe UI",9,"bold"),
                  pady=5,padx=10,borderwidth=0,cursor="hand2").pack(side="left")

        # ── Filename Entry ──
        fn_frame = tk.Frame(self.R, bg="#0f172a")
        fn_frame.pack(anchor="w", pady=(10, 4))
        tk.Label(fn_frame, text="Save Filename:", bg="#0f172a", fg="#38bdf8",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Entry(fn_frame, textvariable=self.filename_var, width=28,
                 bg="#1e293b", fg="#f8fafc", borderwidth=0, insertbackground="white",
                 font=("Segoe UI", 9)).pack(fill="x", pady=2)

        # ── Action buttons ──
        bf = tk.Frame(self.R,bg="#0f172a"); bf.pack(anchor="w",pady=(6,0))
        tk.Button(bf,text="▶  PLAY",command=self._play,
                  bg="#10b981",fg="white",font=("Segoe UI",11,"bold"),
                  pady=10,padx=16,borderwidth=0,cursor="hand2").pack(side="left",padx=(0,6))
        tk.Button(bf,text="■",command=self._stop,
                  bg="#ef4444",fg="white",font=("Segoe UI",14,"bold"),
                  width=3,borderwidth=0,cursor="hand2").pack(side="left",padx=(0,6))
        tk.Button(bf,text="💾  SAVE CSV",command=self._save,
                  bg="#3b82f6",fg="white",font=("Segoe UI",11,"bold"),
                  pady=10,padx=16,borderwidth=0,cursor="hand2").pack(side="left")

    def _build_sliders(self):
        for widget in self.sliders_container.winfo_children():
            widget.destroy()

        tk.Label(self.sliders_container, text="Gait Phase Speeds", bg="#0f172a", fg="#38bdf8",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.ind_rects = []
        
        # ── Swing Speed Slider ──
        fr_swing = tk.Frame(self.sliders_container, bg="#0f172a"); fr_swing.pack(fill="x", pady=2)
        c_swing = tk.Canvas(fr_swing, width=18, height=18, bg="#0f172a", highlightthickness=0)
        c_swing.pack(side="left", padx=(0, 4))
        rid_swing = c_swing.create_rectangle(1, 1, 17, 17, fill="#22c55e", outline="")
        self.ind_rects.append((c_swing, rid_swing, self.swing_speed_var))
        
        tk.Label(fr_swing, text="Swing Speed", bg="#0f172a", fg="#f8fafc", font=("Segoe UI", 8, "bold"),
                 width=11, anchor="w").pack(side="left")
                 
        def _cb_swing(*a):
            self._traj = None
            self._update_indicators()
            self._redraw()
            
        tk.Scale(fr_swing, variable=self.swing_speed_var, from_=0.1, to=5.0, resolution=0.1,
                 orient="horizontal", length=160, bg="#0f172a", fg="#f8fafc",
                 highlightthickness=0, troughcolor="#334155", activebackground="#38bdf8",
                 command=_cb_swing).pack(side="left")

        # ── Stance Speed Slider ──
        fr_stance = tk.Frame(self.sliders_container, bg="#0f172a"); fr_stance.pack(fill="x", pady=2)
        c_stance = tk.Canvas(fr_stance, width=18, height=18, bg="#0f172a", highlightthickness=0)
        c_stance.pack(side="left", padx=(0, 4))
        rid_stance = c_stance.create_rectangle(1, 1, 17, 17, fill="#22c55e", outline="")
        self.ind_rects.append((c_stance, rid_stance, self.stance_speed_var))
        
        tk.Label(fr_stance, text="Stance Speed", bg="#0f172a", fg="#f8fafc", font=("Segoe UI", 8, "bold"),
                 width=11, anchor="w").pack(side="left")
                 
        def _cb_stance(*a):
            self._traj = None
            self._update_indicators()
            self._redraw()
            
        tk.Scale(fr_stance, variable=self.stance_speed_var, from_=0.1, to=5.0, resolution=0.1,
                 orient="horizontal", length=160, bg="#0f172a", fg="#f8fafc",
                 highlightthickness=0, troughcolor="#334155", activebackground="#38bdf8",
                 command=_cb_stance).pack(side="left")
        
        # ── Hip Sway Slider ──
        fr_sway = tk.Frame(self.sliders_container, bg="#0f172a"); fr_sway.pack(fill="x", pady=2)
        c_sway = tk.Canvas(fr_sway, width=18, height=18, bg="#0f172a", highlightthickness=0)
        c_sway.pack(side="left", padx=(0, 4))
        c_sway.create_rectangle(1, 1, 17, 17, fill="#38bdf8", outline="")
        
        tk.Label(fr_sway, text="Hip Sway (m)", bg="#0f172a", fg="#f8fafc", font=("Segoe UI", 8, "bold"),
                 width=11, anchor="w").pack(side="left")
                 
        def _cb_sway(*a):
            self._traj = None
            self._redraw()
            
        tk.Scale(fr_sway, variable=self.hip_sway_var, from_=0.0, to=0.08, resolution=0.005,
                 orient="horizontal", length=160, bg="#0f172a", fg="#f8fafc",
                 highlightthickness=0, troughcolor="#334155", activebackground="#38bdf8",
                 command=_cb_sway).pack(side="left")

        self._update_indicators()

    def _update_indicators(self):
        for c, rid, var in self.ind_rects:
            c.itemconfig(rid, fill=speed_to_hex(var.get()))

    # ── Drag handlers ──────────────────────────────────────────────────
    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        ox,oz = rf_stance_world[0], rf_stance_world[2]
        for i, wp in enumerate(waypoints):
            wx = wp["pos"][0]-ox; wz = wp["pos"][2]-oz
            if ((event.xdata-wx)**2 + (event.ydata-wz)**2)**0.5 < 0.012:
                self._drag_idx = i; return

    def _on_motion(self, event):
        if self._drag_idx is None or event.inaxes != self.ax or event.xdata is None: return
        ox,oz = rf_stance_world[0], rf_stance_world[2]
        old = waypoints[self._drag_idx]["pos"]
        n = len(waypoints)
        new_x = event.xdata + ox
        new_z = event.ydata + oz
        # First and last WP must stay on ground
        if self._drag_idx == 0 or self._drag_idx == n-1:
            new_z = oz
        else:
            new_z = max(oz, new_z)
        waypoints[self._drag_idx]["pos"] = (new_x, old[1], new_z)
        self._traj = None
        self._redraw()

    def _on_release(self, event):
        self._drag_idx = None

    def _redraw(self):
        self._update_indicators()
        ax = self.ax; ax.clear()
        ax.set_facecolor("#1e293b")
        for sp in ax.spines.values(): sp.set_color("#334155")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True,color="#1e3a5f",linestyle="--",alpha=0.6)
        ax.axhline(0,color="#475569",lw=1.5); ax.axvline(0,color="#475569",lw=1.5)

        # Spatial spline — UNIFORM t (n+1 segments: n user + 1 return through stance)
        cs_x, _, cs_z, t_u = build_spline_spatial(waypoints)
        ox, oz = rf_stance_world[0], rf_stance_world[2]
        n = len(waypoints)   # t_u has n+2 points → n+1 segments
        
        swing_color = speed_to_hex(self.swing_speed_var.get())
        stance_color = speed_to_hex(self.stance_speed_var.get())

        # Swing phase: WP1 to WPn (solid line)
        for i in range(n - 1):
            ts = np.linspace(t_u[i], t_u[i+1], 80)
            ax.plot(cs_x(ts)-ox, cs_z(ts)-oz, color=swing_color, lw=2.5)

        # Stance phase: WPn -> Stance -> WP1 (dashed line with stance speed color)
        for i in range(n - 1, n + 1):
            ts = np.linspace(t_u[i], t_u[i+1], 80)
            label = 'stance/return' if i == n - 1 else None
            ax.plot(cs_x(ts)-ox, cs_z(ts)-oz,
                    color=stance_color, lw=2.5, linestyle='--', label=label)

        for i, wp in enumerate(waypoints):
            wx=wp["pos"][0]-ox; wz=wp["pos"][2]-oz
            ax.scatter([wx],[wz],s=110,color="#f59e0b",zorder=5,
                       edgecolors="white",linewidths=1.5,picker=5)
            ax.annotate(f" WP{i+1}",(wx,wz),color="#f8fafc",fontsize=9,fontweight="bold",zorder=6)

        ax.scatter([0],[0],s=130,color="#06b6d4",zorder=6,marker="*",edgecolors="white",lw=2)
        ax.annotate(" Stance",(0,0),color="#06b6d4",fontsize=9,fontweight="bold")
        ax.set_xlabel("X  forward  (m)",color="#94a3b8")
        ax.set_ylabel("Z  up  (m)",color="#94a3b8")
        ax.set_title("Foot Trajectory Spline  (drag waypoints to reshape)",
                     color="#38bdf8",fontsize=10,fontweight="bold")
        ax.legend(fontsize=7,facecolor='#1e293b',labelcolor='#94a3b8',
                  loc='lower right',framealpha=0.7)

        # Force 1:1 aspect ratio and fixed limits (no autoscaling)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.25, 0.35)
        ax.set_ylim(-0.05, 0.20)
        self.cvs.draw()

    def _copy_waypoints(self):
        data_to_copy = []
        for wp in waypoints:
            x, y, z = wp["pos"]
            data_to_copy.append([float(x), float(y), float(z), float(wp["weight"])])
        
        json_str = json.dumps(data_to_copy, indent=2)
        self.clipboard_clear()
        self.clipboard_append(json_str)
        self.update()
        messagebox.showinfo("Copied", "Waypoints JSON copied to clipboard!")

    def _load_waypoints(self):
        popup = tk.Toplevel(self)
        popup.title("Load Waypoints")
        popup.geometry("450x350")
        popup.configure(bg="#0f172a")
        popup.transient(self)
        popup.grab_set()

        tk.Label(popup, text="Paste Waypoints JSON here:", bg="#0f172a", fg="#38bdf8",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        txt = tk.Text(popup, bg="#1e293b", fg="#f8fafc", insertbackground="white",
                      font=("Courier New", 9), borderwidth=0, highlightthickness=1,
                      highlightcolor="#38bdf8", highlightbackground="#334155")
        txt.pack(fill="both", expand=True, padx=10, pady=5)

        # Explicitly support keyboard paste behavior (Ctrl+V / Ctrl+v)
        def paste(event):
            try:
                txt.insert(tk.INSERT, popup.clipboard_get())
            except tk.TclError:
                pass
            return "break"
        txt.bind("<Control-v>", paste)
        txt.bind("<Control-V>", paste)
        txt.focus_set()

        # Auto-fill textbox with clipboard if it contains JSON
        try:
            cb_content = self.clipboard_get()
            parsed_test = json.loads(cb_content)
            if isinstance(parsed_test, list):
                txt.insert("1.0", cb_content)
        except Exception:
            pass

        def _apply():
            global waypoints
            content = txt.get("1.0", "end-1c").strip()
            if not content:
                messagebox.showwarning("Error", "Input is empty!")
                return
            try:
                parsed = json.loads(content)
                if not isinstance(parsed, list):
                    raise ValueError("Must be a JSON list.")
                
                new_wps = []
                for idx, item in enumerate(parsed):
                    if not isinstance(item, list) or len(item) != 4:
                        raise ValueError(f"Waypoint at index {idx} must be a list of 4 elements: [x, y, z, weight]")
                    new_wps.append({
                        "pos": (float(item[0]), float(item[1]), float(item[2])),
                        "weight": float(item[3])
                    })
                
                if len(new_wps) < 2:
                    raise ValueError("Need at least 2 waypoints.")
                
                waypoints = new_wps
                self._traj = None
                self._build_sliders()
                self._redraw()
                popup.destroy()
                messagebox.showinfo("Loaded", f"Successfully loaded {len(waypoints)} waypoints!")
            except Exception as e:
                messagebox.showerror("Error Parsing JSON", f"Failed to parse waypoints:\n{str(e)}")

        btn_fr = tk.Frame(popup, bg="#0f172a")
        btn_fr.pack(fill="x", pady=10, padx=10)
        tk.Button(btn_fr, text="Cancel", command=popup.destroy,
                  bg="#ef4444", fg="white", font=("Segoe UI", 9, "bold"),
                  pady=6, padx=12, borderwidth=0, cursor="hand2").pack(side="right", padx=(6, 0))
        tk.Button(btn_fr, text="Apply", command=_apply,
                  bg="#10b981", fg="white", font=("Segoe UI", 9, "bold"),
                  pady=6, padx=12, borderwidth=0, cursor="hand2").pack(side="right")

    def _get_traj(self):
        if self._traj is None:
            n = len(waypoints)
            speeds = [self.swing_speed_var.get()] * n
            stance_speed = self.stance_speed_var.get()
            phase = self.phase_var.get()/100.0
            hip_sway = self.hip_sway_var.get()
            print("[Solving IK…]")
            self._traj = solve_ik_traj(speeds, stance_speed, phase, hip_sway)
        return self._traj

    def _play(self):
        if self._viewer_running: return
        traj = self._get_traj()
        self._viewer_running = True
        def _run():
            m2 = mujoco.MjModel.from_xml_path(XML_PATH)
            d2 = mujoco.MjData(m2)
            frame_f = 0.0
            with mujoco.viewer.launch_passive(m2,d2) as v:
                while v.is_running() and self._viewer_running:
                    idx = int(frame_f) % len(traj)
                    d2.qpos[:] = traj[idx]
                    mujoco.mj_forward(m2, d2)
                    v.sync()
                    mult = float(self.pb_speed.get())  # read live so changing radio works
                    frame_f += mult
                    time.sleep(0.01)   # fixed ~100 Hz; speed changes frames-per-tick
            self._viewer_running = False
        self._viewer_thread = threading.Thread(target=_run, daemon=True)
        self._viewer_thread.start()

    def _stop(self):
        self._viewer_running = False
        if self._viewer_thread:
            self._viewer_thread.join(timeout=2.0)

    def _save(self):
        traj = self._get_traj()
        fname = self.filename_var.get().strip()
        if not fname:
            fname = "angad_walking_reference.csv"
        if not fname.endswith(".csv"):
            fname += ".csv"

        if os.path.isabs(fname):
            save_path = fname
        else:
            save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "csvs", fname))

        save_csv(traj, save_path)
        messagebox.showinfo("Saved", f"Saved {len(traj)} frames to:\n{save_path}")

    def _symmetrize_speeds(self):
        n = len(waypoints)
        stance_val = self.stance_speed_var.get()
        # S_swing = (n - 1) / 2.0 * S_stance for time symmetry
        new_swing = min(5.0, max(0.1, ((n - 1) / 2.0) * stance_val))
        self.swing_speed_var.set(round(new_swing, 2))
        self._traj = None
        self._update_indicators()
        self._redraw()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global model, data, configuration
    global rfoot_task, lfoot_task, base_task, posture_task
    global mocap_rf_id, rf_site_id, lf_site_id
    global stance_qpos, rf_stance_world, lf_stance_world
    global open_gui_flag
    global args

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=str, help="Relative waypoints (x,z) (x,z) ...")
    parser.add_argument("--out", type=str, help="Default name/path of output CSV file")
    args, _ = parser.parse_known_args()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    model.opt.gravity[:]=0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    mujoco.mj_resetDataKeyframe(model,data,KEYFRAME_INDEX)
    mujoco.mj_forward(model,data)
    stance_qpos = data.qpos.copy()

    rf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    if rf_site_id == -1:
        rf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_site")
    lf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    if lf_site_id == -1:
        lf_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_site")
    rf_stance_world = data.site_xpos[rf_site_id].copy()
    lf_stance_world = data.site_xpos[lf_site_id].copy()
    print(f"[Stance] RF={rf_stance_world}  LF={lf_stance_world}")

    global waypoints
    if args.points:
        pairs = re.findall(r'\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)', args.points)
        if len(pairs) >= 2:
            waypoints = []
            for rx_str, rz_str in pairs:
                rx, rz = float(rx_str), float(rz_str)
                waypoints.append({
                    "pos": (rf_stance_world[0] + rx, rf_stance_world[1], rf_stance_world[2] + rz),
                    "weight": 1.0
                })
            open_gui_flag = True
            print(f"[INFO] Automatically loaded {len(waypoints)} points from command line: {args.points}")
        else:
            print(f"❌ Error: Could not parse at least 2 points from: {args.points}")
            sys.exit(1)

    if not waypoints:
        waypoints.append({"pos": (rf_stance_world[0] - 0.11, rf_stance_world[1], rf_stance_world[2]), "weight": 1.0})
        waypoints.append({"pos": (rf_stance_world[0] + 0.08, rf_stance_world[1], rf_stance_world[2] + 0.065), "weight": 1.0})
        waypoints.append({"pos": (rf_stance_world[0] + 0.21, rf_stance_world[1], rf_stance_world[2]), "weight": 1.0})

    rf_bid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"right_foot_target")
    rf_site_name = "right_foot" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot") != -1 else "right_foot_site"
    lf_site_name = "left_foot" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot") != -1 else "left_foot_site"
    base_site_name = "imu_in_pelvis" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis") != -1 else "base_site"

    if rf_bid<0:
        print("[WARN] right_foot_target mocap body not found"); mocap_rf_id=-1
    else:
        mocap_rf_id=model.body_mocapid[rf_bid]
        mink.move_mocap_to_frame(model,data,"right_foot_target",rf_site_name,"site")

    configuration = mink.Configuration(model); configuration.update(data.qpos)
    posture_task  = mink.PostureTask(model,cost=1e-2)
    posture_task.set_target_from_configuration(configuration)

    base_task  = mink.FrameTask(base_site_name,"site",position_cost=10.,orientation_cost=50.,lm_damping=1.)
    rfoot_task = mink.FrameTask(rf_site_name,"site",position_cost=100.,orientation_cost=50.,lm_damping=1.)
    lfoot_task = mink.FrameTask(lf_site_name,"site",position_cost=100.,orientation_cost=50.,lm_damping=1.)

    bsid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"base_site")
    Tb=np.eye(4); Tb[:3,:3]=data.site_xmat[bsid].reshape(3,3); Tb[:3,3]=data.site_xpos[bsid]
    base_task.set_target(mink.SE3.from_matrix(Tb))
    Tlf=np.eye(4); Tlf[:3,:3]=data.site_xmat[lf_site_id].reshape(3,3); Tlf[:3,3]=lf_stance_world
    lfoot_task.set_target(mink.SE3.from_matrix(Tlf))

    tasks_live=[base_task,rfoot_task,lfoot_task,posture_task]
    limits=[mink.ConfigurationLimit(model)]; ik_dt=1./60.

    if not open_gui_flag:
        print("\n=== Angad Create Gait — Posing Mode ===")
        print("Arrow keys / PgUp-PgDn : Move right foot")
        print("S  : Save waypoint     D : Delete last     R : Reset")
        print("TAB: Select WP    +/- : Adjust transition speed weight")
        print("G  : Open Gait Editor GUI")
        print("========================================\n")

        with mujoco.viewer.launch_passive(model,data,key_callback=key_callback,
                                          show_left_ui=True,show_right_ui=True) as viewer:
            while viewer.is_running() and not open_gui_flag:
                step_start=time.time()
                if mocap_rf_id>=0:
                    rfoot_task.set_target(mink.SE3.from_mocap_id(data,mocap_rf_id))
                vel=mink.solve_ik(configuration,tasks_live,ik_dt,"daqp",limits=limits)
                configuration.integrate_inplace(vel,ik_dt)
                data.qpos[:]=configuration.q; mujoco.mj_forward(model,data)

                viewer.user_scn.ngeom=0
                for i,wp in enumerate(waypoints):
                    wx,wy,wz=wp["pos"]
                    if viewer.user_scn.ngeom<len(viewer.user_scn.geoms):
                        g=viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        rgba=np.array([1.,0.5,0.,0.9]) if i==selected_idx else np.array([0.2,1.,0.2,0.9])
                        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([0.015,0,0]),np.array([wx,wy,wz]),
                                            np.eye(3).flatten(),rgba)
                        viewer.user_scn.ngeom+=1
                viewer.sync()
                el=time.time()-step_start
                if el<ik_dt: time.sleep(ik_dt-el)

    if open_gui_flag and len(waypoints)>=2:
        print("\n[Opening Gait Editor GUI…]")
        GaitGUI().mainloop()


if __name__=="__main__":
    main()
