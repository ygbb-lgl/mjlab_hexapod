"""Cable rewards for the tethered Hexapod task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RingPatternCfg, TerrainHeightSensor

from .events import get_slope_angle_rad
from .observations import cable_tension, tether_direction_body

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _project_foot_state_to_surface(
  frame_pos_w: torch.Tensor,
  hit_pos_w: torch.Tensor,
  normal_w: torch.Tensor,
  valid: torch.Tensor,
  foot_vel_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return normal clearance and tangent velocity for each foot."""
  normal_norm = torch.linalg.vector_norm(normal_w, dim=-1, keepdim=True)
  finite = (
    torch.isfinite(frame_pos_w).all(dim=-1)
    & torch.isfinite(hit_pos_w).all(dim=-1)
    & torch.isfinite(normal_w).all(dim=-1)
    & torch.isfinite(foot_vel_w).all(dim=-1)
  )
  reliable = valid & finite & (normal_norm.squeeze(-1) > 1.0e-8)

  unit_normal = normal_w / normal_norm.clamp(min=1.0e-8)
  unit_normal = torch.where(
    (unit_normal[..., 2] < 0.0).unsqueeze(-1),
    -unit_normal,
    unit_normal,
  )
  world_up = torch.zeros_like(unit_normal)
  world_up[..., 2] = 1.0
  unit_normal = torch.where(reliable.unsqueeze(-1), unit_normal, world_up)

  normal_height = torch.sum((frame_pos_w - hit_pos_w) * unit_normal, dim=-1)
  normal_height = torch.where(
    reliable,
    torch.clamp(normal_height, min=0.0),
    torch.zeros_like(normal_height),
  )
  normal_velocity = torch.sum(foot_vel_w * unit_normal, dim=-1, keepdim=True)
  tangent_velocity = foot_vel_w - normal_velocity * unit_normal
  return normal_height, tangent_velocity


def _terrain_aligned_foot_kinematics(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Read center rays and express foot motion relative to the hit surface."""
  sensor = env.scene[height_sensor_name]
  if not isinstance(sensor, TerrainHeightSensor):
    raise TypeError(
      f"{height_sensor_name!r} must be a TerrainHeightSensor, "
      f"got {type(sensor).__name__}"
    )
  pattern = sensor.cfg.pattern
  if not isinstance(pattern, RingPatternCfg) or not pattern.include_center:
    raise ValueError(
      "Slope foot rewards require a ring height pattern whose first ray is "
      "the foot-center ray"
    )

  data = sensor.data
  batch_size = data.frame_pos_w.shape[0]
  num_frames = sensor.num_frames
  num_rays = sensor.num_rays_per_frame
  hit_pos_w = data.hit_pos_w.reshape(batch_size, num_frames, num_rays, 3)[:, :, 0]
  normal_w = data.normals_w.reshape(batch_size, num_frames, num_rays, 3)[:, :, 0]
  valid = data.distances.reshape(batch_size, num_frames, num_rays)[:, :, 0] >= 0.0

  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
  if foot_vel_w.shape[1] != num_frames:
    raise ValueError(
      "Foot site count must match the terrain-height sensor frame count: "
      f"got {foot_vel_w.shape[1]} sites and {num_frames} frames"
    )
  return _project_foot_state_to_surface(
    data.frame_pos_w,
    hit_pos_w,
    normal_w,
    valid,
    foot_vel_w,
  )


def _moving_command_gate(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command {command_name!r} not found."
  command_magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(
    command[:, 2]
  )
  return (command_magnitude > command_threshold).float()


def slope_foot_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize normal-clearance error, weighted by surface-tangent speed."""
  normal_height, tangent_velocity = _terrain_aligned_foot_kinematics(
    env, height_sensor_name, asset_cfg
  )
  tangent_speed = torch.linalg.vector_norm(tangent_velocity, dim=-1)
  cost = torch.sum(torch.abs(normal_height - target_height) * tangent_speed, dim=1)
  return cost * _moving_command_gate(env, command_name, command_threshold)


class SlopeFootSwingHeight:
  """Penalize peak surface-normal swing height when each foot lands."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor = env.scene[cfg.params["height_sensor_name"]]
    if not isinstance(sensor, TerrainHeightSensor):
      raise TypeError(
        "SlopeFootSwingHeight requires a TerrainHeightSensor, "
        f"got {type(sensor).__name__}"
      )
    self.peak_heights = torch.zeros(
      (env.num_envs, sensor.num_frames),
      device=env.device,
      dtype=torch.float32,
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    contact_sensor = env.scene[sensor_name]
    if not isinstance(contact_sensor, ContactSensor):
      raise TypeError(
        f"{sensor_name!r} must be a ContactSensor, got {type(contact_sensor).__name__}"
      )
    normal_height, _ = _terrain_aligned_foot_kinematics(
      env, height_sensor_name, asset_cfg
    )
    assert contact_sensor.data.found is not None
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, normal_height),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    active = _moving_command_gate(env, command_name, command_threshold)
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active

    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    env.extras["log"]["Metrics/peak_height_mean"] = torch.sum(
      peak_heights_at_landing
    ) / torch.clamp(num_landings, min=1)
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.peak_heights[env_ids] = 0.0


def slope_foot_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  height_sensor_name: str,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize contact-foot velocity projected into the terrain tangent plane."""
  contact_sensor = env.scene[sensor_name]
  if not isinstance(contact_sensor, ContactSensor):
    raise TypeError(
      f"{sensor_name!r} must be a ContactSensor, got {type(contact_sensor).__name__}"
    )
  _, tangent_velocity = _terrain_aligned_foot_kinematics(
    env, height_sensor_name, asset_cfg
  )
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  tangent_speed = torch.linalg.vector_norm(tangent_velocity, dim=-1)
  cost = torch.sum(torch.square(tangent_speed) * in_contact, dim=1)
  cost *= _moving_command_gate(env, command_name, command_threshold)

  num_in_contact = torch.sum(in_contact)
  env.extras["log"]["Metrics/slip_velocity_mean"] = torch.sum(
    tangent_speed * in_contact
  ) / torch.clamp(num_in_contact, min=1)
  return cost


def _action_history_for_term(
  env: ManagerBasedRlEnv,
  action_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return current and previous raw actions for one action term."""
  start = 0
  for name, dimension in zip(
    env.action_manager.active_terms,
    env.action_manager.action_term_dim,
    strict=True,
  ):
    end = start + dimension
    if name == action_name:
      return (
        env.action_manager.action[:, start:end],
        env.action_manager.prev_action[:, start:end],
      )
    start = end
  raise ValueError(f"Action term {action_name!r} is not active")


def action_rate_l2_for_term(
  env: ManagerBasedRlEnv,
  action_name: str,
) -> torch.Tensor:
  """Penalize raw action changes for exactly one action term."""
  action, previous = _action_history_for_term(env, action_name)
  return torch.sum(torch.square(action - previous), dim=1)


def tether_action_rate(
  env: ManagerBasedRlEnv,
  action_name: str,
  action_min: float = 0.0,
  action_max: float = 2.0,
) -> torch.Tensor:
  """Penalize changes of only the scalar cable action."""
  action, previous = _action_history_for_term(env, action_name)
  if action.shape[1] != 1:
    raise ValueError(
      f"Tether action term must be scalar, got dimension {action.shape[1]}"
    )
  cable_action = action[:, 0]
  env.extras["log"]["Metrics/cable_action_mean"] = cable_action.mean()
  env.extras["log"]["Metrics/cable_action_std"] = cable_action.std(unbiased=False)
  env.extras["log"]["Metrics/cable_action_at_min_fraction"] = (
    (cable_action <= action_min).float().mean()
  )
  env.extras["log"]["Metrics/cable_action_at_max_fraction"] = (
    (cable_action >= action_max).float().mean()
  )
  return torch.square(action[:, 0] - previous[:, 0])


def _slope_gate(
  gravity_b: torch.Tensor,
  slope_start: float,
  slope_full: float,
) -> torch.Tensor:
  """Reproduce Playground's body-tilt proxy and linear gate."""
  if slope_full <= slope_start:
    raise ValueError("slope_full must be greater than slope_start")
  gravity_b = gravity_b / torch.clamp(
    torch.linalg.vector_norm(gravity_b, dim=1, keepdim=True), min=1.0e-8
  )
  slope_proxy = torch.acos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0))
  return torch.clamp(
    (slope_proxy - slope_start) / (slope_full - slope_start),
    min=0.0,
    max=1.0,
  )


def _normalized_tension(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  tension_max: float,
) -> torch.Tensor:
  if tension_max <= 0.0:
    raise ValueError("tension_max must be positive")
  tension = cable_tension(env, sensor_name).squeeze(1)
  return torch.clamp(tension / tension_max, min=0.0, max=1.0)


def tether_flat_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  tension_max: float,
  slope_start: float,
  slope_full: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize cable tension while the robot is approximately level."""
  asset: Entity = env.scene[asset_cfg.name]
  gate = _slope_gate(asset.data.projected_gravity_b, slope_start, slope_full)
  tension = _normalized_tension(env, sensor_name, tension_max)
  env.extras["log"]["Metrics/cable_tension_mean"] = cable_tension(
    env, sensor_name
  ).mean()
  env.extras["log"]["Metrics/tether_slope_gate_mean"] = gate.mean()
  return (1.0 - gate) * tension


def tether_uphill_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  tension_max: float,
  slope_start: float,
  slope_full: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward tension aligned with Playground's uphill-direction proxy."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity_b = asset.data.projected_gravity_b
  gate = _slope_gate(gravity_b, slope_start, slope_full)

  # Projected gravity points downhill in the body frame, hence its negative
  # horizontal component points uphill.
  gravity_xy = -gravity_b[:, :2]
  gravity_xy = gravity_xy / torch.clamp(
    torch.linalg.vector_norm(gravity_xy, dim=1, keepdim=True), min=1.0e-8
  )
  tether_xy = tether_direction_body(env, asset_cfg)[:, :2]
  tether_xy = tether_xy / torch.clamp(
    torch.linalg.vector_norm(tether_xy, dim=1, keepdim=True), min=1.0e-8
  )
  alignment = torch.clamp(torch.sum(tether_xy * gravity_xy, dim=1), min=0.0)
  tension = _normalized_tension(env, sensor_name, tension_max)
  return gate * tension * alignment


def tether_pitch_slack_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  tension_max: float,
  slope_start: float,
  slope_full: float,
  slack_ratio: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize tension below ``slack_ratio`` when the tilt gate is active."""
  asset: Entity = env.scene[asset_cfg.name]
  gate = _slope_gate(asset.data.projected_gravity_b, slope_start, slope_full)
  tension = _normalized_tension(env, sensor_name, tension_max)
  return gate * torch.clamp(slack_ratio - tension, min=0.0)


def _tether_direction_world(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the world-frame unit vector from ``tether_7`` to the anchor."""
  asset: Entity = env.scene[asset_cfg.name]
  site_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids]
  if site_pos_w.shape[1] != 2:
    raise ValueError("Slope tether rewards require sites (anchor, tether_7)")
  direction = site_pos_w[:, 0] - site_pos_w[:, 1]
  return direction / (torch.linalg.vector_norm(direction, dim=1, keepdim=True) + 1e-6)


def _tether_command_gate(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  command_threshold: float,
) -> torch.Tensor:
  """Return one when the commanded planar velocity points toward the anchor."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command {command_name!r} not found."
  direction_b = tether_direction_body(env, asset_cfg)
  toward_anchor_speed = torch.sum(command[:, :2] * direction_b[:, :2], dim=1)
  return (toward_anchor_speed > command_threshold).float()


def tether_work_assist(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  command_threshold: float,
) -> torch.Tensor:
  """Reward positive tether power while motion toward the anchor is commanded."""
  asset: Entity = env.scene[asset_cfg.name]
  tension = cable_tension(env, sensor_name).squeeze(1)
  direction_w = _tether_direction_world(env, asset_cfg)
  velocity_projection = torch.sum(asset.data.root_link_lin_vel_w * direction_w, dim=1)
  wants_assist = _tether_command_gate(
    env,
    command_name,
    asset_cfg,
    command_threshold,
  )
  power = tension * velocity_projection
  return wants_assist * torch.clamp(power, min=0.0) * 0.01


def _reward_slope_angles(
  env: ManagerBasedRlEnv,
  slope_angle_deg: float,
  slope_event_name: str | None,
) -> torch.Tensor:
  """Resolve either the fixed angle or a randomized per-environment buffer."""
  if slope_event_name is not None:
    return get_slope_angle_rad(env, slope_event_name)
  return torch.tensor(
    math.radians(slope_angle_deg),
    device=getattr(env, "device", None),
    dtype=torch.float32,
  )


def tether_eng_steep(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  slope_angle_deg: float,
  steep_threshold: float,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
  slope_event_name: str | None = None,
) -> torch.Tensor:
  """Reward slope tension only when motion toward the anchor is commanded."""
  slope_angles = _reward_slope_angles(
    env,
    slope_angle_deg,
    slope_event_name,
  )
  slope_factor = torch.clamp(torch.rad2deg(slope_angles) / 60.0, 0.0, 1.0)
  target_tension = 30.0 + (180.0 - 30.0) * slope_factor
  tension = cable_tension(env, sensor_name).squeeze(1)
  engagement = torch.clamp(tension / (target_tension + 1e-6), 0.0, 1.0)
  is_steep = (slope_angles > steep_threshold).float()
  command_gate = _tether_command_gate(
    env,
    command_name,
    asset_cfg,
    command_threshold,
  )
  env.extras["log"]["Metrics/cable_tension_mean"] = tension.mean()
  env.extras["log"]["Metrics/cable_tension_p95"] = torch.quantile(tension, 0.95)
  env.extras["log"]["Metrics/tether_target_tension_mean"] = target_tension.mean()
  env.extras["log"]["Metrics/tether_assist_command_fraction"] = command_gate.mean()
  if slope_event_name is not None:
    slope_angles_deg = torch.rad2deg(slope_angles)
    env.extras["log"]["Metrics/slope_angle_deg_mean"] = slope_angles_deg.mean()
    env.extras["log"]["Metrics/slope_angle_deg_min"] = slope_angles_deg.min()
    env.extras["log"]["Metrics/slope_angle_deg_max"] = slope_angles_deg.max()
  return is_steep * command_gate * engagement


def tether_slack_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  slope_angle_deg: float,
  steep_threshold: float,
  slack_threshold: float,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
  slope_event_name: str | None = None,
) -> torch.Tensor:
  """Penalize slack only while slope assistance is requested."""
  tension = cable_tension(env, sensor_name).squeeze(1)
  slope_angles = _reward_slope_angles(
    env,
    slope_angle_deg,
    slope_event_name,
  )
  is_steep = (slope_angles > steep_threshold).float()
  is_slack = (tension < slack_threshold).float()
  command_gate = _tether_command_gate(
    env,
    command_name,
    asset_cfg,
    command_threshold,
  )
  env.extras["log"]["Metrics/cable_slack_fraction"] = is_slack.mean()
  return is_steep * command_gate * is_slack


def tether_overload(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_safe_force: float,
) -> torch.Tensor:
  """Penalize cable tension above Playground's safe-force threshold."""
  tension = cable_tension(env, sensor_name).squeeze(1)
  env.extras["log"]["Metrics/cable_overload_fraction"] = (
    (tension > max_safe_force).float().mean()
  )
  return torch.clamp(tension - max_safe_force, min=0.0) / 100.0
