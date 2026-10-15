"""Reset events for the fixed-slope tethered Hexapod task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def push_by_setting_velocity_after_step(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  start_step: int,
  velocity_range: dict[str, tuple[float, float]],
) -> None:
  """Delay velocity-push disturbances until the slope warm-up is complete."""
  if start_step < 0:
    raise ValueError("start_step must be non-negative")
  if env.common_step_counter < start_step:
    return
  mdp.push_by_setting_velocity(
    env,
    env_ids,
    velocity_range=velocity_range,
  )


def reset_root_state_on_slope(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None,
  slope_angle_rad: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Apply the usual root reset and preserve height normal to the slope."""
  env_ids = mdp.resolve_env_ids(env, env_ids)
  mdp.reset_root_state_uniform(
    env,
    env_ids,
    pose_range=pose_range,
    velocity_range=velocity_range,
    asset_cfg=asset_cfg,
  )

  asset: Entity = env.scene[asset_cfg.name]
  qpos = asset.data.data.qpos
  qpos_ids = asset.indexing.free_joint_q_adr
  root_pose = qpos[env_ids[:, None], qpos_ids].clone()
  root_pose[:, 2] += root_pose[:, 0] * math.tan(slope_angle_rad)
  asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)


@requires_model_fields("site_pos")
def randomize_anchor_above_slope(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  slope_angle_rad: float,
  x_range: tuple[float, float],
  y_range: tuple[float, float],
  normal_height_range: tuple[float, float],
) -> None:
  """Place each anchor above its environment's copy of the fixed slope."""
  if normal_height_range[0] < 0.0:
    raise ValueError("normal_height_range must not extend below the slope")
  env_ids = mdp.resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  if not isinstance(asset_cfg.site_ids, list) or len(asset_cfg.site_ids) != 1:
    raise ValueError("randomize_anchor_above_slope requires exactly one anchor site")

  count = len(env_ids)
  x_local = torch.empty(count, device=env.device).uniform_(*x_range)
  y_local = torch.empty(count, device=env.device).uniform_(*y_range)
  height_normal = torch.empty(count, device=env.device).uniform_(*normal_height_range)
  origins = env.scene.env_origins[env_ids]
  anchor_pos = torch.empty((count, 3), device=env.device)
  anchor_pos[:, 0] = origins[:, 0] + x_local
  anchor_pos[:, 1] = origins[:, 1] + y_local
  anchor_pos[:, 2] = anchor_pos[:, 0] * math.tan(
    slope_angle_rad
  ) + height_normal / math.cos(slope_angle_rad)

  site_ids = asset.indexing.site_ids[asset_cfg.site_ids].to(dtype=env_ids.dtype)
  env_grid, site_grid = torch.meshgrid(env_ids, site_ids, indexing="ij")
  env.sim.model.site_pos[env_grid, site_grid] = anchor_pos[:, None, :]
