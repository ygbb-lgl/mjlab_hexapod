"""Tethered hexapod constants and native mjlab entity configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

HEXAPOD_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "hexapod_tethered" / "xmls" / "hexapod.xml"
)
assert HEXAPOD_XML.exists()
HEXAPOD_ASSET_DIR = HEXAPOD_XML.parent / "assets"

LEG_NAMES = ("RF", "RM", "RB", "LF", "LM", "LB")
JOINT_NAMES = tuple(
  f"{leg}_{joint}" for leg in LEG_NAMES for joint in ("HAA", "HFE", "KFE")
)
FOOT_SITE_NAMES = LEG_NAMES
FOOT_GEOM_NAMES = tuple(f"{leg}_foot_collision" for leg in LEG_NAMES)

# The SAFE source URDF declares every revolute joint around local +Z, while the
# migrated Playground MJCF uses mirrored axes.  Multiplying an MJCF coordinate by
# this vector produces the coordinate used by the SAFE controller.  The vector is
# deliberately stored in Playground policy order, not Isaac Gym's alphabetic order.
SAFE_JOINT_SIGNS = (
  1.0,
  -1.0,
  -1.0,  # RF
  1.0,
  -1.0,
  -1.0,  # RM
  1.0,
  -1.0,
  1.0,  # RB
  -1.0,
  1.0,
  1.0,  # LF
  -1.0,
  1.0,
  1.0,  # LM
  -1.0,
  1.0,
  -1.0,  # LB
)


def get_spec() -> mujoco.MjSpec:
  """Load a fresh robot specification with exportable mesh buffers."""
  spec = mujoco.MjSpec.from_file(str(HEXAPOD_XML))
  for asset_path in sorted(HEXAPOD_ASSET_DIR.glob("*.STL")):
    spec.assets[f"assets/{asset_path.name}"] = asset_path.read_bytes()
  return spec


##
# Actuator config.
##

# These values match the position loop used by the physical leg drives. Defining
# them here, instead of in MJCF, lets mjlab own actuator construction and ensures
# the exported model uses the same native MuJoCo PD implementation as training.
LEG_STIFFNESS = 100.0
LEG_DAMPING = 1.2
# Use the calibrated leg model's 50 Nm saturation. The separate 90 Nm limit in
# the hardware deployment configuration belongs to the tether spool controller.
LEG_EFFORT_LIMIT = 50.0
LEG_ARMATURE = 0.005

HEXAPOD_LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_(HAA|HFE|KFE)",),
  stiffness=LEG_STIFFNESS,
  damping=LEG_DAMPING,
  effort_limit=LEG_EFFORT_LIMIT,
  armature=LEG_ARMATURE,
)

##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  # At zero joint angles the foot centers are 0.2897 m below the trunk and
  # the collision spheres have a 0.023 m radius. This height starts the feet
  # tangent to the floor instead of inheriting the old model's 12.7 mm overlap.
  pos=(0.0, 0.0, 0.313),
  joint_pos={r".*_(HAA|HFE|KFE)": 0.0},
  joint_vel={r".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^(RF|RM|RB|LF|LM|LB)_foot_collision$"
_collision_regex = r".*_collision\d*"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex,),
  contype=1,
  conaffinity=1,
  solref=(0.01, 1.0),
  condim={_foot_regex: 6, _collision_regex: 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (1.0, 5e-3, 5e-4)},
)

##
# Final config.
##

HEXAPOD_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(HEXAPOD_LEG_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.8,
)


def get_hexapod_robot_cfg() -> EntityCfg:
  """Return a fresh tethered hexapod entity configuration."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=HEXAPOD_ARTICULATION,
  )


# Keep the same torque-aware scaling convention as the other mjlab locomotion
# assets. At a unit action, the policy requests one quarter of the maximum
# torque-equivalent position error.
HEXAPOD_ACTION_SCALE = {
  expr: 0.25 * LEG_EFFORT_LIMIT / LEG_STIFFNESS
  for expr in HEXAPOD_LEG_ACTUATOR_CFG.target_names_expr
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity import Entity

  robot = Entity(get_hexapod_robot_cfg())
  viewer.launch(robot.spec.compile())
