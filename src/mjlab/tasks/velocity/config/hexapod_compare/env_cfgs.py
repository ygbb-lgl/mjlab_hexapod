"""MuJoCo Playground-aligned Hexapod Compare velocity tasks."""

from __future__ import annotations

import math

from mjlab.asset_zoo.robots.hexapod_compare.hexapod_compare_constants import (
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  HEXAPOD_COMPARE_ACTION_SCALE,
  JOINT_NAMES,
  LEG_NAMES,
  get_hexapod_compare_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import rewards

PLAYGROUND_TRACKING_SIGMA = 0.25
PLAYGROUND_AIR_TIME_TARGET = 0.35
PLAYGROUND_MAX_CONTACT_FORCE_FLAT = 200.0
PLAYGROUND_MAX_CONTACT_FORCE_ROUGH = 300.0


def _playground_rewards(*, rough: bool) -> dict[str, RewardTermCfg]:
  joints = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
  feet = SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True)
  shanks = SceneEntityCfg(
    "robot",
    body_names=tuple(f"{leg}_SHANK" for leg in LEG_NAMES),
    preserve_order=True,
  )
  max_force = (
    PLAYGROUND_MAX_CONTACT_FORCE_ROUGH if rough else PLAYGROUND_MAX_CONTACT_FORCE_FLAT
  )
  terms = {
    "tracking_lin_vel": RewardTermCfg(
      func=rewards.tracking_lin_vel,
      weight=5.0,
      params={
        "command_name": "twist",
        "tracking_sigma": PLAYGROUND_TRACKING_SIGMA,
      },
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=rewards.tracking_ang_vel,
      weight=3.0,
      params={
        "command_name": "twist",
        "tracking_sigma": PLAYGROUND_TRACKING_SIGMA,
      },
    ),
    "lin_vel_z": RewardTermCfg(func=rewards.lin_vel_z, weight=-3.0),
    "ang_vel_xy": RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.05),
    "orientation": RewardTermCfg(func=rewards.orientation, weight=-7.0),
    "stand_still": RewardTermCfg(
      func=rewards.stand_still,
      weight=-1.0,
      params={"command_name": "twist", "asset_cfg": joints},
    ),
    "termination": RewardTermCfg(func=rewards.termination, weight=-1.5),
    "pose": RewardTermCfg(
      func=rewards.pose,
      weight=1.0,
      params={
        "command_name": "twist",
        "joint_std": (0.08, 0.05, 0.05),
        "asset_cfg": joints,
      },
    ),
    "torques": RewardTermCfg(
      func=rewards.torques,
      weight=-0.005,
      params={"asset_cfg": joints},
    ),
    "action_rate": RewardTermCfg(func=rewards.action_rate, weight=-0.01),
    "dof_vel": RewardTermCfg(
      func=rewards.dof_vel,
      weight=-1.0e-5,
      params={"asset_cfg": joints},
    ),
    "dof_acc": RewardTermCfg(
      func=rewards.dof_acc,
      weight=-1.0e-7,
      params={"asset_cfg": joints},
    ),
    "energy": RewardTermCfg(
      func=rewards.energy,
      weight=-1.0e-4,
      params={"asset_cfg": joints},
    ),
    "feet_slip": RewardTermCfg(
      func=rewards.feet_slip,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact", "foot_cfg": feet},
    ),
    "feet_contact_forces": RewardTermCfg(
      func=rewards.feet_contact_forces,
      weight=-0.003,
      params={
        "sensor_name": "feet_ground_contact",
        "max_contact_force": max_force,
      },
    ),
  }
  if rough:
    terms["feet_air_time"] = RewardTermCfg(
      func=rewards.feet_air_time,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "target": PLAYGROUND_AIR_TIME_TARGET,
      },
    )
  terms.update(
    {
      "base_height": RewardTermCfg(
        func=rewards.base_height,
        weight=-5.0,
        params={
          "target": 0.3,
          "terrain_sensor_name": "base_height_scan" if rough else None,
        },
      ),
      "dof_pos_limits": RewardTermCfg(
        func=rewards.dof_pos_limits,
        weight=-1.0,
        params={"asset_cfg": joints},
      ),
      "tripod_contact_phase": RewardTermCfg(
        func=rewards.tripod_contact_phase,
        weight=-3.0,
        params={
          "sensor_name": "feet_ground_contact",
          "command_name": "twist",
          "air_time_target": PLAYGROUND_AIR_TIME_TARGET,
        },
      ),
      "tripod_joint_pos_sync": RewardTermCfg(
        func=rewards.tripod_joint_pos_sync,
        weight=-0.8,
        params={
          "command_name": "twist",
          "position_normalizer": 0.25,
          "asset_cfg": joints,
        },
      ),
      "tripod_joint_vel_sync": RewardTermCfg(
        func=rewards.tripod_joint_vel_sync,
        weight=-0.15,
        params={
          "command_name": "twist",
          "velocity_normalizer": 5.0,
          "asset_cfg": joints,
        },
      ),
      "tripod_joint_antiphase": RewardTermCfg(
        func=rewards.tripod_joint_antiphase,
        weight=-0.8,
        params={
          "command_name": "twist",
          "position_normalizer": 0.25,
          "velocity_normalizer": 5.0,
          "asset_cfg": joints,
        },
      ),
      "contact_force_balance": RewardTermCfg(
        func=rewards.contact_force_balance,
        weight=0.0 if rough else -1.0,
        params={
          "sensor_name": "feet_ground_contact",
          "max_contact_force": max_force,
        },
      ),
      "shank_vertical": RewardTermCfg(
        func=rewards.shank_vertical,
        weight=-3.0,
        params={
          "sensor_name": "feet_ground_contact",
          "shank_cfg": shanks,
          "foot_cfg": feet,
        },
      ),
    }
  )
  return terms


def hexapod_compare_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the rough Hexapod Compare velocity task."""
  cfg = make_velocity_env_cfg()

  cfg.sim.nconmax = 100
  cfg.sim.njmax = 2000
  cfg.sim.contact_sensor_maxmatch = 1000
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"

  cfg.scene.entities = {"robot": get_hexapod_compare_robot_cfg()}

  # Preserve the deployed actor ABI while retaining mjlab's native observation
  # terms, corruption levels, and privileged critic structure.
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=JOINT_NAMES, preserve_order=True
    )
    terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=JOINT_NAMES, preserve_order=True
    )
  del cfg.observations["actor"].terms["height_scan"]

  # Keep the calibrated mjlab reset and domain-randomization setup unchanged.
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
    # Match Go1 soft_landing: aggregate every foot contact into one global
    # net-force vector and consume ContactSensor.data.force directly.
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
  base_height_cfg = TerrainHeightSensorCfg(
    name="base_height_scan",
    frame=(ObjRef(type="body", name="trunk", entity="robot"),),
    ray_alignment="world",
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    reduction="mean",
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    leg_ground_cfg,
    trunk_ground_cfg,
    base_height_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = JOINT_NAMES
  joint_pos_action.preserve_order = True
  joint_pos_action.scale = HEXAPOD_COMPARE_ACTION_SCALE

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

  # This is the only behavior-level substitution: formulas and weights come
  # from the corresponding MuJoCo Playground hexapod task.
  cfg.rewards = _playground_rewards(rough=True)

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


def hexapod_compare_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat Hexapod Compare velocity task."""
  cfg = hexapod_compare_rough_env_cfg(play=play)

  cfg.sim.njmax = 500
  cfg.sim.nconmax = 80
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 50

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "leg_ground_touch",
    "base_height_scan",
  }
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name not in remove_sensors
  )
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards = _playground_rewards(rough=False)
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
