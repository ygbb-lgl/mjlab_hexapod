"""Flat-ground velocity task for coordinated leg and tether control."""

from __future__ import annotations

import math

import mujoco

from mjlab.asset_zoo.robots.hexapod_tethered.hexapod_tethered_constants import (
  ARM_BODY_NAME,
  ARM_JOINT_NAME,
  FOOT_SITE_NAMES,
  HEXAPOD_TETHERED_ACTION_SCALE,
  JOINT_NAMES,
  SLOPE_ANGLE_DEG,
  SLOPE_ANGLE_RAD,
  SLOPE_ROOT_QUAT,
  TETHER_ANCHOR_SITE_NAME,
  TETHER_END_SITE_NAME,
  TETHER_FORCE_SENSOR_NAME,
  TETHER_TENDON_NAME,
  get_hexapod_tethered_robot_cfg,
  get_hexapod_tethered_slope_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonEffortActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.hexapod.env_cfgs import hexapod_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import events, observations, rewards, terminations
from .rl_cfg import SLOPE_DEFAULT_ROLLOUT_STEPS

TETHER_ACTION_NAME = "cable"
TETHER_ACTION_MIN = 0.0
TETHER_ACTION_MAX = 2.0
TETHER_FORCE_PER_ACTION = -200.0

TETHER_SLOPE_START = 0.30
TETHER_SLOPE_FULL = 0.35
TETHER_TENSION_MAX = 300.0
TETHER_SLACK_RATIO = 0.15

SLOPE_TETHER_STEEP_THRESHOLD = 0.17
SLOPE_TETHER_SLACK_THRESHOLD = 15.0
SLOPE_TETHER_MAX_SAFE_FORCE = 250.0
SLOPE_TETHER_COMMAND_THRESHOLD = 0.05
SLOPE_PUSH_START_ITERATION = 1000

OBSERVATION_NOISE_LEVEL = 1.5
OBSERVATION_NOISE_SCALES = {
  "base_lin_vel": 0.1,
  "base_ang_vel": 0.2,
  "projected_gravity": 0.05,
  "joint_pos": 0.05,
  "joint_vel": 1.5,
  "cable_tension": 0.05,
}
TETHER_DIRECTION_NOISE_SCALES = {
  "yaw": 0.01,
  "pitch": 0.01,
}


def _scaled_uniform_noise(scale: float) -> Unoise | None:
  """Apply the task-wide observation-noise level to a scalar scale."""
  amplitude = OBSERVATION_NOISE_LEVEL * scale
  return None if amplitude == 0.0 else Unoise(n_min=-amplitude, n_max=amplitude)


def _scaled_tether_direction_noise() -> observations.TetherDirectionNoiseCfg | None:
  """Scale the deployable yaw and pitch sensor noise together."""
  if OBSERVATION_NOISE_LEVEL == 0.0:
    return None
  yaw_amplitude = OBSERVATION_NOISE_LEVEL * TETHER_DIRECTION_NOISE_SCALES["yaw"]
  pitch_amplitude = OBSERVATION_NOISE_LEVEL * TETHER_DIRECTION_NOISE_SCALES["pitch"]
  return observations.TetherDirectionNoiseCfg(
    yaw_min=-yaw_amplitude,
    yaw_max=yaw_amplitude,
    pitch_min=-pitch_amplitude,
    pitch_max=pitch_amplitude,
  )


def _configure_policy_io(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """Explicitly define the 71-D actor and 110-D asymmetric critic."""
  if OBSERVATION_NOISE_LEVEL < 0.0:
    raise ValueError("OBSERVATION_NOISE_LEVEL must be non-negative")

  joint_action = cfg.actions["joint_pos"]
  assert isinstance(joint_action, JointPositionActionCfg)
  joint_action.actuator_names = JOINT_NAMES
  joint_action.scale = HEXAPOD_TETHERED_ACTION_SCALE.copy()
  joint_action.use_default_offset = True
  joint_action.preserve_order = True

  cfg.actions[TETHER_ACTION_NAME] = TendonEffortActionCfg(
    entity_name="robot",
    actuator_names=(TETHER_TENDON_NAME,),
    scale=1.0,
    clip={TETHER_TENDON_NAME: (TETHER_ACTION_MIN, TETHER_ACTION_MAX)},
    preserve_order=True,
  )

  leg_cfg = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
  arm_cfg = SceneEntityCfg("robot", joint_names=(ARM_JOINT_NAME,))
  arm_tether_cfg = SceneEntityCfg(
    "robot",
    joint_names=(ARM_JOINT_NAME,),
    body_names=(ARM_BODY_NAME,),
    site_names=(TETHER_ANCHOR_SITE_NAME, TETHER_END_SITE_NAME),
    preserve_order=True,
  )
  tendon_cfg = SceneEntityCfg("robot", tendon_names=(TETHER_TENDON_NAME,))

  # Flat terrain intentionally has no height_scan term. Dictionary insertion
  # order is the policy ABI, so keep this list synchronized with deployment.
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=_scaled_uniform_noise(OBSERVATION_NOISE_SCALES["base_lin_vel"]),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=_scaled_uniform_noise(OBSERVATION_NOISE_SCALES["base_ang_vel"]),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=_scaled_uniform_noise(OBSERVATION_NOISE_SCALES["projected_gravity"]),
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": leg_cfg, "biased": True},
      noise=_scaled_uniform_noise(OBSERVATION_NOISE_SCALES["joint_pos"]),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": leg_cfg},
      noise=_scaled_uniform_noise(OBSERVATION_NOISE_SCALES["joint_vel"]),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "cable_tension": ObservationTermCfg(
      func=observations.cable_tension,
      params={"sensor_name": f"robot/{TETHER_FORCE_SENSOR_NAME}"},
      noise=_scaled_uniform_noise(OBSERVATION_NOISE_SCALES["cable_tension"]),
      clip=(0.0, float("inf")),
    ),
    "tether_direction_body": ObservationTermCfg(
      func=observations.tether_direction_from_yaw_pitch,
      params={"asset_cfg": arm_tether_cfg, "biased": True},
      noise=_scaled_tether_direction_noise(),
    ),
  }

  # The critic repeats the actor layout with clean signals, then appends
  # contact and tether state that is available only during training.
  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": leg_cfg},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": leg_cfg},
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "cable_tension": ObservationTermCfg(
      func=observations.cable_tension,
      params={"sensor_name": f"robot/{TETHER_FORCE_SENSOR_NAME}"},
    ),
    "tether_direction_body": ObservationTermCfg(
      func=observations.tether_direction_from_yaw_pitch,
      params={"asset_cfg": arm_tether_cfg},
    ),
    "foot_height": ObservationTermCfg(
      func=velocity_mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=velocity_mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=velocity_mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=velocity_mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "arm_joint_pos": ObservationTermCfg(
      func=observations.joint_position,
      params={"asset_cfg": arm_cfg},
    ),
    "arm_joint_vel": ObservationTermCfg(
      func=observations.joint_velocity,
      params={"asset_cfg": arm_cfg},
    ),
    "tendon_length": ObservationTermCfg(
      func=observations.tendon_length,
      params={"asset_cfg": tendon_cfg},
    ),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def _configure_tether_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Preserve Hexapod locomotion terms and append Playground tether terms."""
  leg_cfg = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
  trunk_tether_cfg = SceneEntityCfg(
    "robot",
    body_names=("trunk",),
    site_names=(TETHER_ANCHOR_SITE_NAME, TETHER_END_SITE_NAME),
    preserve_order=True,
  )
  common_params = {
    "sensor_name": f"robot/{TETHER_FORCE_SENSOR_NAME}",
    "tension_max": TETHER_TENSION_MAX,
    "slope_start": TETHER_SLOPE_START,
    "slope_full": TETHER_SLOPE_FULL,
    "asset_cfg": trunk_tether_cfg,
  }

  # The independent Hexapod's action-rate and limit costs remain leg-only.
  # Otherwise the new cable action would silently alter an existing locomotion
  # reward and would also be penalized a second time below.
  cfg.rewards["action_rate_l2"].func = rewards.action_rate_l2_for_term
  cfg.rewards["action_rate_l2"].params = {"action_name": "joint_pos"}
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = leg_cfg

  cfg.rewards["tether_flat_penalty"] = RewardTermCfg(
    func=rewards.tether_flat_penalty,
    weight=-3.0,
    params=common_params,
  )
  cfg.rewards["tether_uphill_reward"] = RewardTermCfg(
    func=rewards.tether_uphill_reward,
    weight=1.0,
    params=common_params,
  )
  cfg.rewards["tether_pitch_slack_penalty"] = RewardTermCfg(
    func=rewards.tether_pitch_slack_penalty,
    weight=-0.02,
    params={**common_params, "slack_ratio": TETHER_SLACK_RATIO},
  )
  cfg.rewards["tether_action_rate"] = RewardTermCfg(
    func=rewards.tether_action_rate,
    weight=-0.05,
    params={"action_name": TETHER_ACTION_NAME},
  )


def _configure_slope_tether_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace flat shaping with MJLab-calibrated fixed-slope tether rewards."""
  sensor_name = f"robot/{TETHER_FORCE_SENSOR_NAME}"
  tether_cfg = SceneEntityCfg(
    "robot",
    body_names=("trunk",),
    site_names=(TETHER_ANCHOR_SITE_NAME, TETHER_END_SITE_NAME),
    preserve_order=True,
  )
  steep_params = {
    "sensor_name": sensor_name,
    "slope_angle_deg": SLOPE_ANGLE_DEG,
    "steep_threshold": SLOPE_TETHER_STEEP_THRESHOLD,
    "command_name": "twist",
    "command_threshold": SLOPE_TETHER_COMMAND_THRESHOLD,
    "asset_cfg": tether_cfg,
  }

  # The fixed slope owns its tension shaping. Reusing the flat tilt proxy here
  # duplicates engagement incentives and can misclassify recovery poses as flat.
  cfg.rewards["tether_flat_penalty"].weight = 0.0
  cfg.rewards["tether_uphill_reward"].weight = 0.0
  cfg.rewards["tether_pitch_slack_penalty"].weight = 0.0
  cfg.rewards["tether_action_rate"].weight = -0.1

  cfg.rewards["tether_work_assist"] = RewardTermCfg(
    func=rewards.tether_work_assist,
    weight=0.5,
    params={
      "sensor_name": sensor_name,
      "command_name": "twist",
      "asset_cfg": tether_cfg,
      "command_threshold": SLOPE_TETHER_COMMAND_THRESHOLD,
    },
  )
  cfg.rewards["tether_eng_steep"] = RewardTermCfg(
    func=rewards.tether_eng_steep,
    weight=0.5,
    params=steep_params,
  )
  cfg.rewards["tether_slack_penalty"] = RewardTermCfg(
    func=rewards.tether_slack_penalty,
    weight=-1.0,
    params={
      **steep_params,
      "slack_threshold": SLOPE_TETHER_SLACK_THRESHOLD,
    },
  )
  cfg.rewards["tether_overload"] = RewardTermCfg(
    func=rewards.tether_overload,
    weight=-2.0,
    params={
      "sensor_name": sensor_name,
      "max_safe_force": SLOPE_TETHER_MAX_SAFE_FORCE,
    },
  )


def _configure_slope_surface_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Override only slope terms that need terrain-normal geometry."""
  foot_cfg = SceneEntityCfg(
    "robot",
    site_names=FOOT_SITE_NAMES,
    preserve_order=True,
  )

  cfg.rewards["foot_clearance"].func = rewards.slope_foot_clearance
  cfg.rewards["foot_clearance"].params["asset_cfg"] = foot_cfg

  cfg.rewards["foot_swing_height"].func = rewards.SlopeFootSwingHeight
  cfg.rewards["foot_swing_height"].params["asset_cfg"] = foot_cfg

  cfg.rewards["foot_slip"].func = rewards.slope_foot_slip
  cfg.rewards["foot_slip"].params.update(
    {
      "height_sensor_name": "foot_height_scan",
      "asset_cfg": foot_cfg,
    }
  )


def _tilt_terrain_to_fixed_slope(spec: mujoco.MjSpec) -> None:
  """Rotate the standard plane so uphill is world +X at 22 degrees."""
  spec.body("terrain").quat[:] = SLOPE_ROOT_QUAT


def hexapod_tethered_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat joystick task with one coordinated tether action."""
  cfg = hexapod_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_hexapod_tethered_robot_cfg()}

  _configure_policy_io(cfg, play=play)
  _configure_tether_rewards(cfg)

  cfg.events["tether_anchor"] = EventTermCfg(
    mode="reset",
    func=envs_mdp.dr.site_pos,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(TETHER_ANCHOR_SITE_NAME,)),
      "operation": "abs",
      "ranges": {
        0: (-10.0, 10.0),
        1: (-5.0, 5.0),
        2: (0.1, 0.5),
      },
    },
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "illegal_ground_contact":
      assert isinstance(sensor, ContactSensorCfg)
      pattern = sensor.primary.pattern
      pattern = (pattern,) if isinstance(pattern, str) else pattern
      sensor.primary.pattern = (*pattern, "ARM_collision")
      break

  return cfg


def hexapod_tethered_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the fixed 22-degree slope task with coordinated tether control."""
  cfg = hexapod_tethered_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_hexapod_tethered_slope_robot_cfg()}
  cfg.scene.spec_fn = _tilt_terrain_to_fixed_slope

  # Keep the flat task's scales while expressing slope-sensitive geometry in
  # the local terrain frame. The flat task keeps its generic reward functions.
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("foot_height_scan",)
  _configure_slope_surface_rewards(cfg)
  _configure_slope_tether_rewards(cfg)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  if not play:
    initial_lin_vel_x = (0.05, 0.3)
    initial_lin_vel_y = (-0.05, 0.05)
    initial_ang_vel_z = (-0.1, 0.1)
    slope_velocity_stages = [
      {
        "step": 0,
        "lin_vel_x": initial_lin_vel_x,
        "lin_vel_y": initial_lin_vel_y,
        "ang_vel_z": initial_ang_vel_z,
      },
      {
        "step": 3000 * SLOPE_DEFAULT_ROLLOUT_STEPS,
        "lin_vel_x": (0.0, 0.5),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (-0.25, 0.25),
      },
      {
        "step": 6000 * SLOPE_DEFAULT_ROLLOUT_STEPS,
        "lin_vel_x": (-0.8, 0.8),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]
    cfg.curriculum["command_vel"].params["velocity_stages"] = slope_velocity_stages
    twist.ranges.lin_vel_x = initial_lin_vel_x
    twist.ranges.lin_vel_y = initial_lin_vel_y
    twist.ranges.ang_vel_z = initial_ang_vel_z

    push = cfg.events["push_robot"]
    push.func = events.push_by_setting_velocity_after_step
    push.params["start_step"] = SLOPE_PUSH_START_ITERATION * SLOPE_DEFAULT_ROLLOUT_STEPS

  reset_base = cfg.events["reset_base"]
  reset_base.func = events.reset_root_state_on_slope
  # The first command stage is intentionally uphill-biased. Keep the initial
  # heading near world +X so body-frame +X commands also point toward the
  # uphill anchor instead of disabling the tether direction gate in half of
  # the environments through the inherited flat-task yaw range [-pi, pi].
  reset_base.params["pose_range"]["yaw"] = (-0.15, 0.15)
  reset_base.params = {
    **reset_base.params,
    "slope_angle_rad": SLOPE_ANGLE_RAD,
    "asset_cfg": SceneEntityCfg("robot"),
  }
  cfg.events["tether_anchor"] = EventTermCfg(
    mode="reset",
    func=events.randomize_anchor_above_slope,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(TETHER_ANCHOR_SITE_NAME,)),
      "slope_angle_rad": SLOPE_ANGLE_RAD,
      "x_range": (3.0, 13.0),
      "y_range": (-2.0, 2.0),
      # Signed distance along the slope's upward unit normal.
      "normal_height_range": (0.0, 0.3),
    },
  )
  cfg.terminations["trunk_front_past_anchor"] = TerminationTermCfg(
    func=terminations.trunk_front_past_anchor,
    time_out=False,
    params={
      "forward_offset": 0.3,
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("trunk",),
        site_names=(TETHER_ANCHOR_SITE_NAME,),
        preserve_order=True,
      ),
    },
  )
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=terminations.bad_orientation_relative_to_slope,
    params={
      "limit_angle": math.radians(60.0),
      "slope_angle_rad": SLOPE_ANGLE_RAD,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  cfg.viewer.elevation = -20.0
  cfg.viewer.azimuth = 120.0
  return cfg
