"""Tethered Hexapod robot constants."""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.asset_zoo.robots.hexapod.hexapod_constants import (
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  FULL_COLLISION,
  INIT_JOINT_POS,
  JOINT_AXIS_SIGNS,
  JOINT_NAMES,
  LEG_NAMES,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia

HEXAPOD_TETHERED_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "hexapod_tethered"
  / "xmls"
  / "hexapod_tethered.xml"
)
assert HEXAPOD_TETHERED_XML.exists()
HEXAPOD_TETHERED_SLOPE_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "hexapod_tethered"
  / "xmls"
  / "hexapod_tethered_slope.xml"
)
assert HEXAPOD_TETHERED_SLOPE_XML.exists()
HEXAPOD_TETHERED_ASSET_DIR = HEXAPOD_TETHERED_XML.parent / "assets"

SLOPE_ANGLE_DEG = 22.0
SLOPE_ANGLE_RAD = math.radians(SLOPE_ANGLE_DEG)
SLOPE_PITCH_RAD = -SLOPE_ANGLE_RAD
SLOPE_ROOT_QUAT = (
  math.cos(SLOPE_ANGLE_RAD / 2.0),
  0.0,
  -math.sin(SLOPE_ANGLE_RAD / 2.0),
  0.0,
)

ARM_BODY_NAME = "ARM"
ARM_JOINT_NAME = "ARM_JOINT"
TETHER_TENDON_NAME = "cable"
TETHER_ANCHOR_SITE_NAME = "anchor"
TETHER_END_SITE_NAME = "tether_7"
TETHER_FORCE_SENSOR_NAME = "cable_tendonactuatorfrc"


def _load_spec(xml_path: Path) -> mujoco.MjSpec:
  """Load one tethered robot variant with exportable mesh buffers."""
  spec = mujoco.MjSpec.from_file(str(xml_path))
  for asset_path in sorted(HEXAPOD_TETHERED_ASSET_DIR.glob("*.STL")):
    spec.assets[f"assets/{asset_path.name}"] = asset_path.read_bytes()
  return spec


def get_spec() -> mujoco.MjSpec:
  """Load a fresh flat-ground tethered robot specification."""
  return _load_spec(HEXAPOD_TETHERED_XML)


def get_slope_spec() -> mujoco.MjSpec:
  """Load the 22-degree slope robot with its independent cable force limit."""
  spec = _load_spec(HEXAPOD_TETHERED_SLOPE_XML)
  spec.body("trunk").quat[:] = SLOPE_ROOT_QUAT
  return spec


##
# Independent leg actuator and PD config.
##

# Keep these values local to the tethered robot. They intentionally start with
# the same numbers as the independent Hexapod but can now be tuned separately.
ROTOR_INERTIA = 0.00015
HAA_GEAR_RATIO = 18
HFE_GEAR_RATIO = 18
KFE_GEAR_RATIO = 18

HAA_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HAA_GEAR_RATIO),
  velocity_limit=16.44,
  effort_limit=30.0,
)
HFE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HFE_GEAR_RATIO),
  velocity_limit=16.44,
  effort_limit=30.0,
)
KFE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KFE_GEAR_RATIO),
  velocity_limit=16.44,
  effort_limit=30.0,
)

STIFFNESS_HAA = 100.0
DAMPING_HAA = 3.0
STIFFNESS_HFE = 100.0
DAMPING_HFE = 3.0
STIFFNESS_KFE = 100.0
DAMPING_KFE = 3.0

HEXAPOD_TETHERED_HAA_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_HAA",),
  stiffness=STIFFNESS_HAA,
  damping=DAMPING_HAA,
  effort_limit=HAA_ACTUATOR.effort_limit,
  armature=HAA_ACTUATOR.reflected_inertia,
)
HEXAPOD_TETHERED_HFE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_HFE",),
  stiffness=STIFFNESS_HFE,
  damping=DAMPING_HFE,
  effort_limit=HFE_ACTUATOR.effort_limit,
  armature=HFE_ACTUATOR.reflected_inertia,
)
HEXAPOD_TETHERED_KFE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_KFE",),
  stiffness=STIFFNESS_KFE,
  damping=DAMPING_KFE,
  effort_limit=KFE_ACTUATOR.effort_limit,
  armature=KFE_ACTUATOR.reflected_inertia,
)


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.313),
  joint_pos={**INIT_JOINT_POS, ARM_JOINT_NAME: 0.0},
  joint_vel={r".*": 0.0},
)

SLOPE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.313),
  rot=SLOPE_ROOT_QUAT,
  joint_pos={**INIT_JOINT_POS, ARM_JOINT_NAME: 0.0},
  joint_vel={r".*": 0.0},
)

# The XML-defined cable actuator stays in XML so its -200 N/control gain remains
# the single source of truth.
HEXAPOD_TETHERED_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    HEXAPOD_TETHERED_HAA_ACTUATOR_CFG,
    HEXAPOD_TETHERED_HFE_ACTUATOR_CFG,
    HEXAPOD_TETHERED_KFE_ACTUATOR_CFG,
    XmlActuatorCfg(
      target_names_expr=(TETHER_TENDON_NAME,),
      transmission_type=TransmissionType.TENDON,
      command_field="effort",
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_hexapod_tethered_robot_cfg() -> EntityCfg:
  """Get a fresh tethered Hexapod robot configuration."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=HEXAPOD_TETHERED_ARTICULATION,
  )


def get_hexapod_tethered_slope_robot_cfg() -> EntityCfg:
  """Get the independently loaded tethered robot for the fixed slope task."""
  return EntityCfg(
    init_state=SLOPE_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_slope_spec,
    articulation=HEXAPOD_TETHERED_ARTICULATION,
  )


HEXAPOD_TETHERED_ACTION_SCALE = {
  r".*_HAA": 0.25,
  r".*_HFE": 0.25,
  r".*_KFE": 0.25,
}

__all__ = [
  "ARM_BODY_NAME",
  "ARM_JOINT_NAME",
  "FOOT_GEOM_NAMES",
  "FOOT_SITE_NAMES",
  "HEXAPOD_TETHERED_ACTION_SCALE",
  "HEXAPOD_TETHERED_ARTICULATION",
  "HEXAPOD_TETHERED_SLOPE_XML",
  "HEXAPOD_TETHERED_XML",
  "INIT_JOINT_POS",
  "JOINT_AXIS_SIGNS",
  "JOINT_NAMES",
  "LEG_NAMES",
  "SLOPE_ANGLE_DEG",
  "SLOPE_ANGLE_RAD",
  "SLOPE_PITCH_RAD",
  "SLOPE_ROOT_QUAT",
  "TETHER_ANCHOR_SITE_NAME",
  "TETHER_END_SITE_NAME",
  "TETHER_FORCE_SENSOR_NAME",
  "TETHER_TENDON_NAME",
  "get_hexapod_tethered_robot_cfg",
  "get_hexapod_tethered_slope_robot_cfg",
]
