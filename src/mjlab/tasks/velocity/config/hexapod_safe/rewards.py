"""Reward functions ported from the tuned EL-4090 source environment."""

from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")


def _command(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  command = env.command_manager.get_command(name)
  if command is None:
    raise ValueError(f"Command {name!r} is not available")
  return command


def tracking_lin_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  tracking_sigma: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  error = torch.sum(
    torch.square(
      _command(env, command_name)[:, :2] - robot.data.root_link_lin_vel_b[:, :2]
    ),
    dim=1,
  )
  return torch.exp(-error / tracking_sigma)


def tracking_ang_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  tracking_sigma: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  error = torch.square(
    _command(env, command_name)[:, 2] - robot.data.root_link_ang_vel_b[:, 2]
  )
  return torch.exp(-error / tracking_sigma)


def lin_vel_z(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return torch.square(robot.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(robot.data.root_link_ang_vel_b[:, :2]), dim=1)


def orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)


def torques(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.square(robot.data.qfrc_actuator[:, asset_cfg.joint_ids]), dim=1
  )


def dof_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


class dof_acc:
  """Finite-difference acceleration used by legged_gym."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _ROBOT_CFG)
    robot: Entity = env.scene[asset_cfg.name]
    self._joint_ids = asset_cfg.joint_ids
    self._previous_velocity = torch.zeros_like(robot.data.joint_vel[:, self._joint_ids])
    self._step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    velocity = robot.data.joint_vel[:, self._joint_ids]
    acceleration = (self._previous_velocity - velocity) / self._step_dt
    self._previous_velocity[:] = velocity
    return torch.sum(torch.square(acceleration), dim=1)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._previous_velocity[env_ids] = 0.0


def base_height(
  env: ManagerBasedRlEnv,
  target: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return torch.square(robot.data.root_link_pos_w[:, 2] - target)


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  foot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Foot slip using the source task's effective runtime contact mask.

  The source reward loader orders terms alphabetically. ``feet_air_time`` thus
  updates ``last_contacts`` immediately before this reward is evaluated, making
  its documented one-step filter equivalent to the current contact mask.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    raise RuntimeError("feet_slip requires contact force data")
  contact = force[:, :, 2] > 1.0
  robot: Entity = env.scene[foot_cfg.name]
  velocity_xy = robot.data.body_link_lin_vel_w[:, foot_cfg.body_ids, :2]
  speed_squared = torch.square(torch.norm(velocity_xy, dim=2))
  return torch.sum(contact * speed_squared, dim=1)


class _SourceContactTiming:
  """Shared EL-4090 contact timers driven by vertical force thresholding."""

  def __init__(self, env: ManagerBasedRlEnv, sensor: ContactSensor):
    shape = (env.num_envs, len(sensor.primary_names))
    self.last_contacts = torch.zeros(shape, device=env.device, dtype=torch.bool)
    self.air_time = torch.zeros(shape, device=env.device)
    self.contact_time = torch.zeros(shape, device=env.device)

  def update(
    self,
    sensor: ContactSensor,
    step_dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    force = sensor.data.force
    if force is None:
      raise RuntimeError("feet_air_time requires contact force data")
    contact = force[:, :, 2] > 1.0
    filtered_contact = contact | self.last_contacts
    self.last_contacts[:] = contact
    first_contact = (self.air_time > 0.0) & filtered_contact
    self.air_time += step_dt
    self.contact_time += step_dt
    landing_air_time = self.air_time.clone()
    self.air_time *= ~filtered_contact
    self.contact_time *= filtered_contact
    return landing_air_time, first_contact

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.last_contacts[env_ids] = False
    self.air_time[env_ids] = 0.0
    self.contact_time[env_ids] = 0.0


_SOURCE_CONTACT_TIMINGS: WeakKeyDictionary[ContactSensor, _SourceContactTiming] = (
  WeakKeyDictionary()
)


def _source_contact_timing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> tuple[ContactSensor, _SourceContactTiming]:
  sensor: ContactSensor = env.scene[sensor_name]
  state = _SOURCE_CONTACT_TIMINGS.get(sensor)
  if state is None:
    state = _SourceContactTiming(env, sensor)
    _SOURCE_CONTACT_TIMINGS[sensor] = state
  return sensor, state


class feet_air_time:
  """Source-exact EL-4090 landing reward and shared gait timer update."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    _, self._timing = _source_contact_timing(env, cfg.params["sensor_name"])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold: float = 0.5,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    landing_air_time, first_contact = self._timing.update(sensor, env.step_dt)
    reward = torch.sum((landing_air_time - threshold) * first_contact, dim=1)
    moving = torch.norm(_command(env, command_name)[:, :2], dim=1) > 0.1
    return reward * moving

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._timing.reset(env_ids)


def collision(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    raise RuntimeError("collision requires contact force data")
  return torch.sum((torch.norm(force, dim=2) > 0.1).float(), dim=1)


def feet_stumble(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    raise RuntimeError("feet_stumble requires contact force data")
  tangential = torch.norm(force[:, :, :2], dim=2)
  normal = torch.abs(force[:, :, 2])
  return torch.any(tangential > 5.0 * normal, dim=1).float()


def action_rate(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
  if action_name not in env.action_manager.active_terms:
    raise ValueError(f"Action term {action_name!r} is not available")
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  position_error = torch.sum(
    torch.abs(
      robot.data.joint_pos[:, asset_cfg.joint_ids]
      - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )
  standing = torch.norm(_command(env, command_name)[:, :2], dim=1) < 0.1
  return position_error * standing


def dof_pos_limits(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  position = robot.data.joint_pos[:, asset_cfg.joint_ids]
  limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  below = -(position - limits[:, :, 0]).clamp(max=0.0)
  above = (position - limits[:, :, 1]).clamp(min=0.0)
  return torch.sum(below + above, dim=1)


def dof_vel_limits(
  env: ManagerBasedRlEnv,
  velocity_limit: float,
  soft_limit: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  excess = (
    torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids])
    - velocity_limit * soft_limit
  )
  return torch.sum(excess.clamp(min=0.0, max=1.0), dim=1)


def torque_limits(
  env: ManagerBasedRlEnv,
  torque_limit: float,
  soft_limit: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  excess = (
    torch.abs(robot.data.qfrc_actuator[:, asset_cfg.joint_ids])
    - torque_limit * soft_limit
  )
  return torch.sum(excess.clamp(min=0.0), dim=1)


def feet_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contact_force: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    raise RuntimeError("feet_contact_forces requires contact force data")
  excess = torch.norm(force, dim=2) - max_contact_force
  return torch.sum(excess.clamp(min=0.0), dim=1)


def shank_vertical(
  env: ManagerBasedRlEnv,
  shank_cfg: SceneEntityCfg,
  foot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[shank_cfg.name]
  shank_pos = robot.data.body_link_pos_w[:, shank_cfg.body_ids]
  foot_pos = robot.data.body_link_pos_w[:, foot_cfg.body_ids]
  return torch.sum(torch.norm(shank_pos[:, :, :2] - foot_pos[:, :, :2], dim=2), dim=1)


class _gait_timing:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor, self._source_timing = _source_contact_timing(env, cfg.params["sensor_name"])
    names = sensor.primary_names
    leg_to_index = {name.split("_")[0]: index for index, name in enumerate(names)}
    self._group_1 = tuple(leg_to_index[name] for name in ("LF", "LB", "RM"))
    self._group_2 = tuple(leg_to_index[name] for name in ("RF", "RB", "LM"))

  @staticmethod
  def _sync(
    air_time: torch.Tensor,
    contact_time: torch.Tensor,
    first: int,
    second: int,
  ) -> torch.Tensor:
    air_error = torch.square(air_time[:, first] - air_time[:, second])
    contact_error = torch.square(contact_time[:, first] - contact_time[:, second])
    return air_error.clamp(max=4.0) + contact_error.clamp(max=4.0)

  @staticmethod
  def _async(
    air_time: torch.Tensor,
    contact_time: torch.Tensor,
    first: int,
    second: int,
  ) -> torch.Tensor:
    first_error = torch.square(air_time[:, first] - contact_time[:, second])
    second_error = torch.square(contact_time[:, first] - air_time[:, second])
    return first_error.clamp(max=4.0) + second_error.clamp(max=4.0)

  @staticmethod
  def _moving(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return (torch.norm(command[:, :2], dim=1) > 0.1) | (torch.abs(command[:, 2]) > 0.1)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids


class feet_sync(_gait_timing):
  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
  ) -> torch.Tensor:
    del sensor_name
    air_time = self._source_timing.air_time
    contact_time = self._source_timing.contact_time
    result = torch.zeros(env.num_envs, device=env.device)
    for group in (self._group_1, self._group_2):
      for first, second in (
        (group[0], group[1]),
        (group[0], group[2]),
        (group[1], group[2]),
      ):
        result += self._sync(air_time, contact_time, first, second)
    return result * self._moving(env, command_name)


class feet_async(_gait_timing):
  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
  ) -> torch.Tensor:
    del sensor_name
    air_time = self._source_timing.air_time
    contact_time = self._source_timing.contact_time
    result = torch.zeros(env.num_envs, device=env.device)
    for first in self._group_1:
      for second in self._group_2:
        result += self._async(air_time, contact_time, first, second)
    return result * self._moving(env, command_name)
