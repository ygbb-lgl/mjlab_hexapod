"""Tethered hexapod velocity environment configurations."""

import math

from mjlab.asset_zoo.robots import (
  HEXAPOD_ACTION_SCALE,
  get_hexapod_robot_cfg,
)
from mjlab.asset_zoo.robots.hexapod_tethered.hexapod_constants import (
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  JOINT_NAMES,
  LEG_NAMES,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def hexapod_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the tethered hexapod rough-terrain velocity task."""
  cfg = make_velocity_env_cfg()

  cfg.sim.nconmax = 100
  cfg.sim.njmax = 2000
  cfg.sim.contact_sensor_maxmatch = 1000
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"

  cfg.scene.entities = {"robot": get_hexapod_robot_cfg()}

  # Keep the policy joint blocks in the deployed MuJoCo Playground order,
  # independent of future changes to MJCF traversal order.
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=JOINT_NAMES, preserve_order=True
    )
    terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=JOINT_NAMES, preserve_order=True
    )

  # The calibrated initial state already places the six collision spheres at the
  # surface. Keep only a small reset-height perturbation so the 40 kg platform is
  # not dropped by up to 5 cm at the beginning of every rollout.
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.01)

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "trunk"
    elif sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=FOOT_GEOM_NAMES,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  leg_geom_names = tuple(
    f"{leg}_{link}_collision" for leg in LEG_NAMES for link in ("HIP", "THIGH", "SHANK")
  )
  leg_ground_cfg = ContactSensorCfg(
    name="leg_ground_touch",
    primary=ContactMatch(
      mode="geom",
      pattern=leg_geom_names,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      pattern=("trunk_collision1", "trunk_collision2"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    leg_ground_cfg,
    trunk_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = JOINT_NAMES
  joint_pos_action.preserve_order = True
  joint_pos_action.scale = HEXAPOD_ACTION_SCALE

  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 2.5
  cfg.viewer.elevation = -12.0

  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*_HAA": 0.05,
    r".*_(HFE|KFE)": 0.05,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_HAA": 0.2,
    r".*_(HFE|KFE)": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_HAA": 0.3,
    r".*_(HFE|KFE)": 0.4,
  }
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["leg_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.2,
    params={"sensor_name": leg_ground_cfg.name},
  )

  # Terrain-relative upright tracking is more reliable than a world-up angle on
  # slopes. A trunk impact is still always terminal, which prevents the policy
  # from learning to drag the chassis after a fall.
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["trunk_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": trunk_ground_cfg.name},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def hexapod_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the tethered hexapod flat-terrain velocity task."""
  cfg = hexapod_rough_env_cfg(play=play)

  cfg.sim.njmax = 500
  cfg.sim.nconmax = 80
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 50

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  remove_sensors = {"terrain_scan", "self_collision", "leg_ground_touch"}
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name not in remove_sensors
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  cfg.rewards.pop("self_collisions", None)
  cfg.rewards.pop("leg_collision", None)

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
