"""Rewards for the MuJoCo Playground-aligned Hexapod Compare task."""

from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_TRIPOD_1 = (3, 5, 1)  # LF, LB, RM in Playground policy order.
_TRIPOD_2 = (0, 2, 4)  # RF, RB, LM in Playground policy order.


def _command(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  command = env.command_manager.get_command(name)
  if command is None:
    raise ValueError(f"Command {name!r} is not available")
  return command


def _robot(env: ManagerBasedRlEnv, cfg: SceneEntityCfg) -> Entity:
  return env.scene[cfg.name]


def tracking_lin_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  tracking_sigma: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  velocity = _robot(env, asset_cfg).data.root_link_lin_vel_b[:, :2]
  error = torch.sum(torch.square(_command(env, command_name)[:, :2] - velocity), dim=1)
  return torch.exp(-error / tracking_sigma)


def tracking_ang_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  tracking_sigma: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  yaw_rate = _robot(env, asset_cfg).data.root_link_ang_vel_b[:, 2]
  error = torch.square(_command(env, command_name)[:, 2] - yaw_rate)
  return torch.exp(-error / tracking_sigma)


def lin_vel_z(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  velocity = _robot(env, asset_cfg).data.root_link_lin_vel_w
  return torch.square(velocity[:, 2])


def ang_vel_xy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  velocity = _robot(env, asset_cfg).data.root_link_ang_vel_w
  return torch.sum(torch.square(velocity[:, :2]), dim=1)


def orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  gravity = _robot(env, asset_cfg).data.projected_gravity_b
  return torch.sum(torch.square(gravity[:, :2]), dim=1)


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot = _robot(env, asset_cfg)
  position = robot.data.joint_pos[:, asset_cfg.joint_ids]
  default = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  command_norm = torch.linalg.vector_norm(_command(env, command_name), dim=1)
  return torch.sum(torch.abs(position - default), dim=1) * (command_norm < 0.1)


def termination(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.reset_terminated.float()


def pose(
  env: ManagerBasedRlEnv,
  command_name: str,
  joint_std: tuple[float, float, float],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot = _robot(env, asset_cfg)
  position = robot.data.joint_pos[:, asset_cfg.joint_ids]
  default = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  std = torch.tensor(joint_std * 6, device=env.device, dtype=position.dtype)
  reward = torch.exp(
    -torch.mean(torch.square((position - default) / (std + 1.0e-6)), dim=1)
  )
  command_norm = torch.linalg.vector_norm(_command(env, command_name), dim=1)
  return reward * (command_norm < 0.1)


def torques(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  torque = _robot(env, asset_cfg).data.qfrc_actuator[:, asset_cfg.joint_ids]
  return torch.linalg.vector_norm(torque, dim=1) + torch.sum(torch.abs(torque), dim=1)


def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(torch.square(delta), dim=1)


def dof_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  velocity = _robot(env, asset_cfg).data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(velocity), dim=1)


def dof_acc(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  acceleration = _robot(env, asset_cfg).data.joint_acc[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(acceleration), dim=1)


def energy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot = _robot(env, asset_cfg)
  velocity = robot.data.joint_vel[:, asset_cfg.joint_ids]
  torque = robot.data.qfrc_actuator[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(velocity) * torch.abs(torque), dim=1)


def _contact(sensor: ContactSensor) -> torch.Tensor:
  found = sensor.data.found
  if found is None:
    raise RuntimeError("The Playground rewards require contact found data")
  return found > 0


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  foot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  velocity = _robot(env, foot_cfg).data.site_lin_vel_w[:, foot_cfg.site_ids, :2]
  speed_squared = torch.sum(torch.square(velocity), dim=2)
  return torch.sum(speed_squared * _contact(sensor), dim=1)


def feet_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contact_force: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    raise RuntimeError("feet_contact_forces requires contact force data")
  # Identical force extraction and L2 magnitude convention to Go1's
  # mjlab.tasks.velocity.mdp.soft_landing reward.
  force_norm = torch.norm(force, dim=-1)
  contact = _contact(sensor)
  excess = (force_norm - max_contact_force).clamp(min=0.0)
  return torch.sum(excess * contact, dim=1)


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  target: float,
) -> torch.Tensor:
  """Retain the zero-weight Complex term for source configuration parity."""
  sensor, state = _contact_timing(env, sensor_name)
  state.advance(env, sensor)
  command_norm = torch.linalg.vector_norm(_command(env, command_name), dim=1)
  reward = torch.sum((state.reward_air_time - target) * state.first_contact, dim=1)
  return reward * (command_norm > 0.15)


def base_height(
  env: ManagerBasedRlEnv,
  target: float,
  terrain_sensor_name: str | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  robot = _robot(env, asset_cfg)
  if terrain_sensor_name is None:
    height = robot.data.root_link_pos_w[:, 2]
  else:
    sensor: TerrainHeightSensor = env.scene[terrain_sensor_name]
    height = sensor.data.heights.squeeze(1)
  return torch.square(height - target)


def dof_pos_limits(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot = _robot(env, asset_cfg)
  position = robot.data.joint_pos[:, asset_cfg.joint_ids]
  limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  below = -(position - limits[:, :, 0]).clamp(max=0.0)
  above = (position - limits[:, :, 1]).clamp(min=0.0)
  return torch.sum(below + above, dim=1)


class _ContactTiming:
  """Control-step contact state matching ``joystick.py::step``."""

  def __init__(self, env: ManagerBasedRlEnv, sensor: ContactSensor) -> None:
    shape = (env.num_envs, len(sensor.primary_names))
    self.last_contact = torch.zeros(shape, device=env.device, dtype=torch.bool)
    self.air_time = torch.zeros(shape, device=env.device)
    self.contact_time = torch.zeros(shape, device=env.device)
    self.reward_air_time = torch.zeros_like(self.air_time)
    self.reward_contact_time = torch.zeros_like(self.contact_time)
    self.first_contact = torch.zeros_like(self.last_contact)
    self._last_step = -1

  def advance(self, env: ManagerBasedRlEnv, sensor: ContactSensor) -> None:
    if self._last_step == env.common_step_counter:
      return
    contact = _contact(sensor)
    filtered = contact | self.last_contact
    self.first_contact = (self.air_time > 0.0) & filtered
    self.air_time += env.step_dt
    self.contact_time += env.step_dt
    self.reward_air_time = self.air_time.clone()
    self.reward_contact_time = self.contact_time.clone()
    self.air_time *= ~filtered
    self.contact_time *= filtered
    self.last_contact = contact.clone()
    self._last_step = env.common_step_counter

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    for value in (
      self.last_contact,
      self.air_time,
      self.contact_time,
      self.reward_air_time,
      self.reward_contact_time,
      self.first_contact,
    ):
      value[env_ids] = 0
    self._last_step = -1


_CONTACT_TIMINGS: WeakKeyDictionary[ContactSensor, _ContactTiming] = WeakKeyDictionary()


def _contact_timing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> tuple[ContactSensor, _ContactTiming]:
  sensor: ContactSensor = env.scene[sensor_name]
  state = _CONTACT_TIMINGS.get(sensor)
  if state is None:
    state = _ContactTiming(env, sensor)
    _CONTACT_TIMINGS[sensor] = state
  return sensor, state


def _tripod_indices(
  sensor: ContactSensor,
  device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  leg_to_index = {
    name.split("_")[0]: index for index, name in enumerate(sensor.primary_names)
  }
  order = ("RF", "RM", "RB", "LF", "LM", "LB")
  policy_to_sensor = [leg_to_index[name] for name in order]
  group_1 = torch.tensor(
    [policy_to_sensor[index] for index in _TRIPOD_1], device=device
  )
  group_2 = torch.tensor(
    [policy_to_sensor[index] for index in _TRIPOD_2], device=device
  )
  return group_1, group_2


class tripod_contact_phase:
  """Alternating-tripod contact and clock cost from MuJoCo Playground."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    sensor, self._timing = _contact_timing(env, cfg.params["sensor_name"])
    self._sensor = sensor
    self._group_1, self._group_2 = _tripod_indices(sensor, env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    air_time_target: float,
  ) -> torch.Tensor:
    del sensor_name
    self._timing.advance(env, self._sensor)
    contact = _contact(self._sensor).float()
    groups = torch.stack([contact[:, self._group_1], contact[:, self._group_2]], dim=1)
    means = groups.mean(dim=2, keepdim=True)
    sync_cost = torch.sum(torch.mean(torch.square(groups - means), dim=2), dim=1)
    anti_phase_cost = torch.square(torch.sum(means, dim=1).squeeze(1) - 1.0)

    scale = max(air_time_target, env.step_dt)
    air = self._timing.reward_air_time
    stance = self._timing.reward_contact_time
    air_groups = (
      torch.stack([air[:, self._group_1], air[:, self._group_2]], dim=1) / scale
    )
    stance_groups = (
      torch.stack([stance[:, self._group_1], stance[:, self._group_2]], dim=1) / scale
    )
    timers = torch.tanh(torch.cat([air_groups, stance_groups], dim=1))
    timer_sync = torch.sum(
      torch.mean(torch.square(timers - timers.mean(dim=2, keepdim=True)), dim=2),
      dim=1,
    )
    air_means = air_groups.mean(dim=2)
    stance_means = stance_groups.mean(dim=2)
    cross_clock = torch.sum(
      torch.square(torch.tanh(air_means - stance_means.flip(dims=(1,)))),
      dim=1,
    )
    cost = sync_cost + anti_phase_cost + 0.5 * timer_sync + 0.5 * cross_clock
    moving = torch.linalg.vector_norm(_command(env, command_name), dim=1) > 0.1
    return cost * moving

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._timing.reset(env_ids)


def _joint_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  robot = _robot(env, asset_cfg)
  position = robot.data.joint_pos[:, asset_cfg.joint_ids].reshape(-1, 6, 3)
  default = robot.data.default_joint_pos[:, asset_cfg.joint_ids].reshape(-1, 6, 3)
  velocity = robot.data.joint_vel[:, asset_cfg.joint_ids].reshape(-1, 6, 3)
  return position, default, velocity


def tripod_joint_pos_sync(
  env: ManagerBasedRlEnv,
  command_name: str,
  position_normalizer: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  position, default, _ = _joint_state(env, asset_cfg)
  delta = (position - default) / (position_normalizer + 1.0e-6)
  weight = torch.tensor((0.5, 1.0, 0.75), device=env.device)
  costs = torch.zeros(env.num_envs, device=env.device)
  for group in (_TRIPOD_1, _TRIPOD_2):
    value = delta[:, group]
    error = value - value.mean(dim=1, keepdim=True)
    costs += torch.mean(torch.square(torch.tanh(error)) * weight, dim=(1, 2))
  moving = torch.linalg.vector_norm(_command(env, command_name), dim=1) > 0.1
  return costs * moving


def tripod_joint_vel_sync(
  env: ManagerBasedRlEnv,
  command_name: str,
  velocity_normalizer: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  _, _, velocity = _joint_state(env, asset_cfg)
  velocity = velocity / (velocity_normalizer + 1.0e-6)
  weight = torch.tensor((0.5, 1.0, 0.75), device=env.device)
  costs = torch.zeros(env.num_envs, device=env.device)
  for group in (_TRIPOD_1, _TRIPOD_2):
    value = velocity[:, group]
    error = value - value.mean(dim=1, keepdim=True)
    costs += torch.mean(torch.square(torch.tanh(error)) * weight, dim=(1, 2))
  moving = torch.linalg.vector_norm(_command(env, command_name), dim=1) > 0.1
  return costs * moving


def tripod_joint_antiphase(
  env: ManagerBasedRlEnv,
  command_name: str,
  position_normalizer: float,
  velocity_normalizer: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  position, default, velocity = _joint_state(env, asset_cfg)
  delta = (position - default) / (position_normalizer + 1.0e-6)
  velocity = velocity / (velocity_normalizer + 1.0e-6)
  weight = torch.tensor((0.5, 1.0, 0.75), device=env.device)
  pos_error = delta[:, _TRIPOD_1].mean(dim=1) + delta[:, _TRIPOD_2].mean(dim=1)
  vel_error = velocity[:, _TRIPOD_1].mean(dim=1) + velocity[:, _TRIPOD_2].mean(dim=1)
  pos_cost = torch.mean(torch.square(torch.tanh(pos_error)) * weight, dim=1)
  vel_cost = torch.mean(torch.square(torch.tanh(vel_error)) * weight, dim=1)
  moving = torch.linalg.vector_norm(_command(env, command_name), dim=1) > 0.1
  return (pos_cost + 0.25 * vel_cost) * moving


def contact_force_balance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contact_force: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    raise RuntimeError("contact_force_balance requires contact force data")
  # netforce is expressed in the global frame, as in Go1 soft_landing.
  normal = force[:, :, 2].clamp(min=0.0)
  mask = (normal > 1.0).float()
  normalized = (normal / max(max_contact_force, 1.0e-6)).clamp(0.0, 3.0)
  group_1, group_2 = _tripod_indices(sensor, env.device)
  cost = torch.zeros(env.num_envs, device=env.device)
  for group in (group_1, group_2):
    group_force = normalized[:, group]
    group_mask = mask[:, group]
    count = group_mask.sum(dim=1)
    mean = (group_force * group_mask).sum(dim=1) / (count + 1.0e-6)
    variance = (torch.square(group_force - mean.unsqueeze(1)) * group_mask).sum(
      dim=1
    ) / (count + 1.0e-6)
    cost += variance * (count > 1.0)
  return cost


def shank_vertical(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  shank_cfg: SceneEntityCfg,
  foot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  robot = _robot(env, shank_cfg)
  shank = robot.data.body_link_pos_w[:, shank_cfg.body_ids, :2]
  foot = robot.data.site_pos_w[:, foot_cfg.site_ids, :2]
  offset = torch.linalg.vector_norm(shank - foot, dim=2)
  return torch.sum(offset * _contact(sensor), dim=1)
