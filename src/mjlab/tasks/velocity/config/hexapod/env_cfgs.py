"""Flat-ground velocity environment for the crab-style hexapod."""

from __future__ import annotations

import math

from mjlab.asset_zoo.robots.hexapod.hexapod_constants import (
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  HEXAPOD_ACTION_SCALE,
  JOINT_NAMES,
  LEG_NAMES,
  get_hexapod_robot_cfg,
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
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import rewards
from .rl_cfg import DEFAULT_ROLLOUT_STEPS

GAIT_CYCLE_TIME = 0.6
FOOT_CLEARANCE_HEIGHT = 0.1
OBSERVATION_NOISE_LEVEL = 1.5
OBSERVATION_NOISE_SCALES = {
  "base_lin_vel": 0.1,
  "base_ang_vel": 0.2,
  "projected_gravity": 0.1,
  "joint_pos": 0.05,
  "joint_vel": 1.5,
}

# Go1's foot rewards sum over four feet. Scaling by 4/6 keeps their total
# contribution comparable when the same kernels operate on six feet.
FOUR_TO_SIX_FOOT_SCALE = 4.0 / 6.0

# The landing kernel consumes raw Newtons. Scale Go1's coefficient by the
# measured model-mass ratio (12.743448 / 39.940777) rather than penalizing this
# 39.94 kg robot roughly three times more for the same normalized impact.
SOFT_LANDING_WEIGHT = -1.0e-5 * 12.743448 / 39.9407774168088

_ILLEGAL_CONTACT_GEOMS = (
  "trunk_collision1",
  "trunk_collision2",
  *(
    f"{leg}_{link}_collision" for leg in LEG_NAMES for link in ("HIP", "THIGH", "SHANK")
  ),
)


def _configure_ordered_joint_io(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep policy and action tensors in the XML's explicit 18-joint order."""
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=JOINT_NAMES, preserve_order=True
    )
    terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=JOINT_NAMES, preserve_order=True
    )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = JOINT_NAMES
  action.preserve_order = True
  action.scale = HEXAPOD_ACTION_SCALE


def _configure_observation_noise(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply the Playground-style global level to per-term noise scales."""
  if OBSERVATION_NOISE_LEVEL < 0.0:
    raise ValueError("OBSERVATION_NOISE_LEVEL must be non-negative")

  terms = cfg.observations["actor"].terms
  for name, scale in OBSERVATION_NOISE_SCALES.items():
    amplitude = OBSERVATION_NOISE_LEVEL * scale
    terms[name].noise = (
      None if amplitude == 0.0 else Unoise(n_min=-amplitude, n_max=amplitude)
    )


def _configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Adapt Go1's reward scales to six feet and add two-phase tripod terms."""
  joint_cfg = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
  foot_cfg = SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True)

  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

  cfg.rewards["upright"].weight = 1.0
  cfg.rewards["upright"].params["std"] = math.sqrt(0.2)
  cfg.rewards["upright"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("trunk",)
  )
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["pose"].params.update(
    {
      "asset_cfg": joint_cfg,
      "std_standing": {
        r".*_HAA": 0.05,
        r".*_(HFE|KFE)": 0.07,
      },
      "std_walking": {
        r".*_HAA": 0.20,
        r".*_(HFE|KFE)": 0.25,
      },
      "std_running": {
        r".*_HAA": 0.25,
        r".*_(HFE|KFE)": 0.30,
      },
      "walking_threshold": 0.05,
      "running_threshold": 0.6,
    }
  )

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["body_ang_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("trunk",)
  )
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["dof_pos_limits"].weight = -1.0
  # Go1 sums this term over 12 actions. Normalize its scale for 18 joints so
  # early exploration is not penalized 1.5 times more just for having six legs.
  cfg.rewards["action_rate_l2"].weight = -0.1 * 12.0 / 18.0

  # A generic air-time bonus cannot tell stance feet from swing feet. Keeping
  # it active can reward lifting a fourth foot, directly opposing tripod gait.
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["foot_clearance"].weight = -2.0 * FOUR_TO_SIX_FOOT_SCALE
  cfg.rewards["foot_clearance"].params.update(
    {
      "target_height": FOOT_CLEARANCE_HEIGHT,
      "command_threshold": 0.05,
      "asset_cfg": foot_cfg,
    }
  )
  cfg.rewards["foot_swing_height"].weight = -0.25 * FOUR_TO_SIX_FOOT_SCALE
  cfg.rewards["foot_swing_height"].params.update(
    {
      "target_height": FOOT_CLEARANCE_HEIGHT,
      "command_threshold": 0.05,
    }
  )
  cfg.rewards["foot_slip"].weight = -0.1 * FOUR_TO_SIX_FOOT_SCALE
  cfg.rewards["foot_slip"].params.update(
    {
      "command_threshold": 0.05,
      "asset_cfg": foot_cfg,
    }
  )
  cfg.rewards["soft_landing"].weight = SOFT_LANDING_WEIGHT
  cfg.rewards["soft_landing"].params["command_threshold"] = 0.05

  gait_params = {
    "sensor_name": "feet_ground_contact",
    "command_name": "twist",
    "command_threshold": 0.05,
    # A tiny grazing contact is not a support foot. At 39.94 kg, a nominal
    # tripod foot carries about 130 N, so 5 N safely separates the two.
    "contact_force_threshold": 5.0,
  }
  # 因为rewards是python字典，不需要提前声明，可以直接创建，如果有就相当于覆值了
  cfg.rewards["tripod_contact_pattern"] = RewardTermCfg(
    func=rewards.TripodContactPattern,
    weight=1.0,
    params={
      **gait_params,
      "cycle_time": GAIT_CYCLE_TIME,
      "transition_fraction": 0.08,
    },
  )
  cfg.rewards["tripod_support_count"] = RewardTermCfg(
    func=rewards.tripod_support_count_cost,
    weight=-0.25,
    params=gait_params,
  )


def hexapod_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat-ground velocity task for the six-legged robot."""
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_hexapod_robot_cfg()}

  cfg.sim.njmax = 500
  cfg.sim.nconmax = 80
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.mujoco.impratio = 10.0
  cfg.sim.mujoco.cone = "elliptic"

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.035, num_samples=4)
      sensor.debug_vis = False

  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )

  feet_ground = ContactSensorCfg(
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
  illegal_ground = ContactSensorCfg(
    name="illegal_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=_ILLEGAL_CONTACT_GEOMS,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, illegal_ground)

  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms.pop("height_scan", None)
  _configure_ordered_joint_io(cfg)
  _configure_observation_noise(cfg)

  # MuJoCo q coordinates already follow each XML axis, including axis="0 0 -1".
  # No extra sign flip belongs here; the explicit order prevents silent remapping.

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
      "ranges": (1.0e-4, 2.0e-2),
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
      "ranges": (1.0e-5, 5.0e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
      "operation": "add",
      "ranges": {
        0: (-0.15, 0.15),
        1: (-0.1, 0.1),
        2: (-0.05, 0.05),
      },
    },
  )
  cfg.events["body_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(r".*",)),
      "operation": "scale",
      "distribution": "uniform",
      "ranges": (0.9, 1.1),
      "shared_random": False,
    },
  )
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.dof_armature,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=JOINT_NAMES, preserve_order=True
      ),
      "operation": "scale",
      "distribution": "uniform",
      "ranges": (1.0, 1.05),
      "shared_random": False,
    },
  )
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.01)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)

  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.3, 0.3),
      "lin_vel_y": (-0.2, 0.2),
      "ang_vel_z": (-0.25, 0.25),
    },
    {
      "step": 3000 * DEFAULT_ROLLOUT_STEPS,
      "lin_vel_x": (-0.5, 0.5),
      "lin_vel_y": (-0.3, 0.3),
      "ang_vel_z": (-0.4, 0.4),
    },
    {
      "step": 7000 * DEFAULT_ROLLOUT_STEPS,
      "lin_vel_x": (-0.8, 0.8),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.8, 0.8),
    },
  ]

  _configure_rewards(cfg)

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(60.0)},
  )
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": illegal_ground.name,
      "force_threshold": 20.0,
    },
  )

  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -15.0
  cfg.viewer.azimuth = 135.0

  if play:
    cfg.episode_length_s = int(1.0e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    for sensor in cfg.scene.sensors or ():
      if sensor.name == "foot_height_scan":
        assert isinstance(sensor, TerrainHeightSensorCfg)
        sensor.debug_vis = True

    twist.ranges.lin_vel_x = (-0.8, 0.8)
    twist.ranges.lin_vel_y = (-0.5, 0.5)
    twist.ranges.ang_vel_z = (-0.8, 0.8)

  return cfg
