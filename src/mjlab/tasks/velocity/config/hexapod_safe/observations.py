"""Observation terms for the tuned EL-4090 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.hexapod_tethered.hexapod_constants import (
  SAFE_JOINT_SIGNS,
)
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")


def signed_joint_pos_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Joint-position error in source coordinates and Playground order."""
  robot: Entity = env.scene[asset_cfg.name]
  signs = torch.tensor(
    SAFE_JOINT_SIGNS, device=env.device, dtype=robot.data.joint_pos.dtype
  )
  joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
  default_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  return (joint_pos - default_pos) * signs


def signed_joint_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Joint velocity in source coordinates and Playground order."""
  robot: Entity = env.scene[asset_cfg.name]
  signs = torch.tensor(
    SAFE_JOINT_SIGNS, device=env.device, dtype=robot.data.joint_vel.dtype
  )
  return robot.data.joint_vel[:, asset_cfg.joint_ids] * signs
