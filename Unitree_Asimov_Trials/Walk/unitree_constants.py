"""Unitree G1 Robot Constants & Configuration for Asimov Velocity / COT / Froude Trials.

Defines actuator models, PD gains (KP & KD), joint position limits, and entity specifications
for the Unitree G1 Humanoid Robot.
"""

from pathlib import Path
import mujoco
from typing import Dict

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

# Resolves local Unitree G1 scene XML
UNITREE_G1_XML = Path(__file__).resolve().parents[2] / "xmls" / "unitree_g1" / "scene.xml"
if not UNITREE_G1_XML.exists():
    UNITREE_G1_XML = Path(__file__).resolve().parents[2] / "xmls" / "unitree_g1" / "g1.xml"

UNITREE_ACTION_SCALE = 0.25

UNITREE_G1_GAINS: Dict[str, Dict[str, float]] = {
    "hip_pitch":    {"kp": 100.0, "kd": 2.5, "action_scale": 0.25},
    "hip_roll":     {"kp": 100.0, "kd": 2.5, "action_scale": 0.25},
    "hip_yaw":      {"kp": 100.0, "kd": 2.5, "action_scale": 0.25},
    "knee":         {"kp": 150.0, "kd": 4.0, "action_scale": 0.25},
    "ankle_pitch":  {"kp": 40.0,  "kd": 1.0, "action_scale": 0.25},
    "ankle_roll":   {"kp": 40.0,  "kd": 1.0, "action_scale": 0.25},
}

##
# Actuator Configuration for Unitree G1
##
UNITREE_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=("left_hip_pitch_joint", "right_hip_pitch_joint"),
    stiffness=100.0,
    damping=2.5,
    effort_limit=88.0,
)

UNITREE_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=("left_hip_roll_joint", "right_hip_roll_joint"),
    stiffness=100.0,
    damping=2.5,
    effort_limit=139.0,
)

UNITREE_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    joint_names_expr=("left_hip_yaw_joint", "right_hip_yaw_joint"),
    stiffness=100.0,
    damping=2.5,
    effort_limit=88.0,
)

UNITREE_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    joint_names_expr=("left_knee_joint", "right_knee_joint"),
    stiffness=150.0,
    damping=4.0,
    effort_limit=139.0,
)

UNITREE_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    stiffness=40.0,
    damping=1.0,
    effort_limit=50.0,
)

UNITREE_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=("left_ankle_roll_joint", "right_ankle_roll_joint"),
    stiffness=40.0,
    damping=1.0,
    effort_limit=50.0,
)

# Standing Keyframe Joint Angles
UNITREE_G1_STANDING_JOINTS = {
    "left_hip_pitch_joint":    -0.20,
    "left_hip_roll_joint":      0.00,
    "left_hip_yaw_joint":       0.00,
    "left_knee_joint":          0.42,
    "left_ankle_pitch_joint":  -0.22,
    "left_ankle_roll_joint":    0.00,
    "right_hip_pitch_joint":   -0.20,
    "right_hip_roll_joint":     0.00,
    "right_hip_yaw_joint":      0.00,
    "right_knee_joint":         0.42,
    "right_ankle_pitch_joint": -0.22,
    "right_ankle_roll_joint":   0.00,
}


def get_unitree_spec() -> mujoco.MjSpec:
    """Load Unitree G1 MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(UNITREE_G1_XML))


UNITREE_G1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        UNITREE_ACTUATOR_HIP_PITCH,
        UNITREE_ACTUATOR_HIP_ROLL,
        UNITREE_ACTUATOR_HIP_YAW,
        UNITREE_ACTUATOR_KNEE,
        UNITREE_ACTUATOR_ANKLE_PITCH,
        UNITREE_ACTUATOR_ANKLE_ROLL,
    ),
)


def get_unitree_robot_cfg() -> EntityCfg:
    """Return EntityCfg for Unitree G1."""
    return EntityCfg(
        spec_fn=get_unitree_spec,
        articulation=UNITREE_G1_ARTICULATION,
    )
