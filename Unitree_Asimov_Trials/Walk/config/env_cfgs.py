"""Angad bipedal robot velocity tracking environment configurations."""

from pathlib import Path
from unitree_constants import (
    UNITREE_ACTION_SCALE,
    get_unitree_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationTermCfg, RewardTermCfg, CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mdp import (
    self_collision_cost,
    imitation_joint_pos,
    imitation_dynamic_gait,
    alternating_feet_contact,
    imitation_joint_pos_speed_adaptive,
    alternating_feet_contact_speed_adaptive,
    gait_clock_speed_adaptive,
    commands_vel,
)
from mjlab.tasks.velocity.mdp import gait_clock
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# ==========================================================
# MASTER TOGGLE: True = Speed-Adaptive Stride Scaling | False = Static Legacy Rules
# ==========================================================

def unitree_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 rough terrain velocity tracking configuration."""
    cfg = make_velocity_env_cfg()
    cfg.scene.num_envs = 4096 if not play else 1

    cfg.sim.nconmax = 150
    cfg.sim.njmax = 600

    robot_cfg = get_unitree_robot_cfg()
    has_torso = any("torso" in name for a in robot_cfg.articulation.actuators for name in a.joint_names_expr)
    if play:
        from mjlab.utils.spec_config import CollisionCfg
        robot_cfg.collisions = (
            CollisionCfg(
                geom_names_expr=(r".*_foot_(primitive.*|cap_.*)", r".*_foot_link_geom.*", "floor"),
                contype={r".*_foot_(primitive.*|cap_.*)": 0, r".*_foot_link_geom.*": 0, "floor": 1},
                conaffinity={r".*_foot_(primitive.*|cap_.*)": 1, r".*_foot_link_geom.*": 1, "floor": 1},
                condim={r".*_foot_(primitive.*|cap_.*)": 3, r".*_foot_link_geom.*": 3, "floor": 3},
                priority={r".*_foot_(primitive.*|cap_.*)": 1, r".*_foot_link_geom.*": 1, "floor": 0},
                friction={r".*_foot_(primitive.*|cap_.*)": (2.5,), r".*_foot_link_geom.*": (2.5,), "floor": None},
            ),
        )
    cfg.scene.entities = {"robot": robot_cfg}

    # Angad feet sites
    site_names = ("left_foot_site", "right_foot_site")
    geom_names = (
        r".*left_foot_(primitive|link_geom|cap_.*).*",
        r".*right_foot_(primitive|link_geom|cap_.*).*",
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain") if not play else None,
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ANGAD_ACTION_SCALE

    cfg.viewer.body_name = "base"

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 0.9  # Angad is taller than Asimov

    twist_cmd.ranges.lin_vel_x = (-0.8, 2.5)  # Expanded to leverage the full dynamic gait range!
    twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)
    twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

    # Remove base_lin_vel - not available on real robot IMU
    del cfg.observations["policy"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["base_lin_vel"]



    use_dynamic_gait = True  # Toggle this to True to train with the dynamic .npz anchors
    use_imitation = True     # Enabled for Method 7 (Dynamic Stride COT imitation)
    
    if use_dynamic_gait:
        imitation_func = imitation_dynamic_gait
        
        # 1. Relax the pose penalty so the robot isn't penalized for long strides
        cfg.rewards["pose"].params["std_running"] = {
            r".*hip_pitch$": 1.5,   # Increased from 0.8
            r".*hip_roll$": 0.4,
            r".*hip_yaw$": 0.3,
            r".*knee_pitch$": 1.5,  # Increased from 0.8
            r".*ankle_pitch$": 0.3,
            r".*ankle_roll$": 0.2,
        }
        if has_torso:
            cfg.rewards["pose"].params["std_running"][r".*torso_yaw$"] = 0.35
        
        # 2. Add the dynamic velocity curriculum
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    # Level 1: Baby steps (0 to 400 iterations)
                    {"step": 0, "lin_vel_x": (0.0, 0.5), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.1, 0.1)},
                    # Level 2: Jogging (400 to 800 iterations)
                    {"step": 9600, "lin_vel_x": (-0.2, 1.2), "lin_vel_y": (-0.2, 0.2), "ang_vel_z": (-0.3, 0.3)},
                    # Level 3: Fast running (800 to 1200 iterations)
                    {"step": 19200, "lin_vel_x": (-0.5, 2.0), "lin_vel_y": (-0.4, 0.4), "ang_vel_z": (-0.5, 0.5)},
                    # Level 4: Full Spectrum Sprinting! (1200+ iterations)
                    {"step": 28800, "lin_vel_x": (-0.8, 2.5), "lin_vel_y": (-0.6, 0.6), "ang_vel_z": (-0.6, 0.6)},
                ],
            },
        )
        
        # 3. 3D feet proximity penalty to prevent legs from hitting/crossing each other
        # from mdp import feet_proximity
        # cfg.rewards["feet_proximity"] = RewardTermCfg(
        #     func=feet_proximity,
        #     weight=-200.0,  # Increased weight to provide a strong gradient
        #     params={
        #         "min_distance": 0.15,  # 15 cm safety threshold
        #         "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        #     },
        # )
    else:
        imitation_func = imitation_joint_pos
        # Restore default pose penalty for static gait
        cfg.rewards["pose"].params["std_running"] = {
            r".*hip_pitch$": 0.8,
            r".*hip_roll$": 0.35,
            r".*hip_yaw$": 0.3,
            r".*knee_pitch$": 0.8,
            r".*ankle_pitch$": 0.25,
            r".*ankle_roll$": 0.15,
        }
        if has_torso:
            cfg.rewards["pose"].params["std_running"][r".*torso_yaw$"] = 0.35
        # Ensure legacy method uses static velocities without any curriculum overriding it
        if cfg.curriculum is not None:
            cfg.curriculum.pop("command_vel", None)

    gait_clock_func = gait_clock
    alternating_feet_func = alternating_feet_contact

    # Add gait clock observation for phase-aware locomotion
    gait_clock_cfg = ObservationTermCfg(
        func=gait_clock_func,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "gait_frequency": 1.25,
        },
    )
    cfg.observations["policy"].terms["gait_clock"] = gait_clock_cfg
    cfg.observations["critic"].terms["gait_clock"] = gait_clock_cfg

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_pitch$": 0.5,
        r".*hip_roll$": 0.25,
        r".*hip_yaw$": 0.2,
        r".*knee_pitch$": 0.5,
        r".*ankle_pitch$": 0.2,
        r".*ankle_roll$": 0.12,
    }
    if has_torso:
        cfg.rewards["pose"].params["std_walking"][r".*torso_yaw$"] = 0.35

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Increase body angular velocity penalty
    cfg.rewards["body_ang_vel"].weight = -0.08
    cfg.rewards["angular_momentum"].weight = -0.03
    cfg.rewards["air_time"].weight = 0.5
    
    # Method 7: Standard penalties
    cfg.rewards["soft_landing"].weight = -1e-5
    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )



    # Imitation reward for walking reference motion (weight set to 0.0 if disabled)
    cfg.rewards["imitation"] = RewardTermCfg(
        func=imitation_func,
        weight=1.5 if use_imitation else 0.0,
        params={
            "data_path": str(ANGAD_WALKING_REFERENCE),
            "dynamic_npz_path": str(Path(__file__).resolve().parents[2] / "Gaits" / "csvs" / "gait_cot_dynamic_no_sway.npz"),
            "gait_frequency": 1.25,
            "std": 0.5,
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )

    # Alternating feet contact reward for proper bipedal gait
    cfg.rewards["alternating_feet"] = RewardTermCfg(
        func=alternating_feet_func,
        weight=0.5,
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "command_threshold": 0.1,
            "gait_frequency": 1.25,
        },
    )

    if play:
        cfg.scene.terrain = None
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        if cfg.curriculum is not None:
            cfg.curriculum.pop("terrain_levels", None)
            cfg.curriculum.pop("command_vel", None)  # Let joystick use full range

    return cfg


def angad_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Angad flat terrain velocity tracking configuration."""
    cfg = angad_rough_env_cfg(play=play)

    # Switch to flat terrain.
    if not play:
        assert cfg.scene.terrain is not None
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None

        # Disable terrain curriculum.
        if cfg.curriculum is not None and "terrain_levels" in cfg.curriculum:
            del cfg.curriculum["terrain_levels"]

    if play:
        commands = cfg.commands
        assert commands is not None
        twist_cmd = commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg


def angad_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Angad balance-only config."""
    cfg = angad_flat_env_cfg(play=play)

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0

    # Remove velocity curriculum so it doesn't override the forced 0.0 ranges
    if cfg.curriculum is not None:
        cfg.curriculum.pop("command_vel", None)

    return cfg
