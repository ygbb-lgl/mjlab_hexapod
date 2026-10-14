"""Cable rewards for the tethered Hexapod task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import cable_tension, tether_direction_body

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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
) -> torch.Tensor:
  """Penalize changes of only the scalar cable action."""
  action, previous = _action_history_for_term(env, action_name)
  if action.shape[1] != 1:
    raise ValueError(
      f"Tether action term must be scalar, got dimension {action.shape[1]}"
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

  gravity_xy = gravity_b[:, :2]
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
