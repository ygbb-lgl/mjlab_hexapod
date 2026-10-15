"""Termination terms for the fixed-slope tethered Hexapod task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def bad_orientation_relative_to_slope(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  slope_angle_rad: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Terminate when trunk up deviates too far from the fixed slope normal."""
  if not 0.0 < limit_angle <= math.pi:
    raise ValueError("limit_angle must be in (0, pi]")
  asset: Entity = env.scene[asset_cfg.name]
  body_up = torch.zeros(
    (env.num_envs, 3),
    device=env.device,
    dtype=asset.data.root_link_quat_w.dtype,
  )
  body_up[:, 2] = 1.0
  body_up_w = quat_apply(asset.data.root_link_quat_w, body_up)
  slope_normal = body_up.new_tensor(
    (-math.sin(slope_angle_rad), 0.0, math.cos(slope_angle_rad))
  )
  alignment = torch.sum(body_up_w * slope_normal, dim=1)
  tilt = torch.acos(torch.clamp(alignment, -1.0, 1.0))
  return tilt > limit_angle


def trunk_front_past_anchor(
  env: ManagerBasedRlEnv,
  forward_offset: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Terminate when the trunk front passes the anchor along world +X."""
  if forward_offset < 0.0:
    raise ValueError("forward_offset must be non-negative")
  if not isinstance(asset_cfg.body_ids, list) or len(asset_cfg.body_ids) != 1:
    raise ValueError("trunk_front_past_anchor requires exactly one trunk body")
  if not isinstance(asset_cfg.site_ids, list) or len(asset_cfg.site_ids) != 1:
    raise ValueError("trunk_front_past_anchor requires exactly one anchor site")

  asset: Entity = env.scene[asset_cfg.name]
  trunk_x = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 0].squeeze(1)
  anchor_x = asset.data.site_pos_w[:, asset_cfg.site_ids, 0].squeeze(1)
  return trunk_x + forward_offset > anchor_x
