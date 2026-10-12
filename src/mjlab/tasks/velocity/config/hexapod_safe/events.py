"""Domain-randomization events used by the SAFE task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")


def set_planar_push_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  max_velocity: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> None:
  """Overwrite planar root velocity exactly as legged_gym's push event does."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  robot: Entity = env.scene[asset_cfg.name]
  velocity = robot.data.root_link_vel_w[env_ids].clone()
  velocity[:, :2].uniform_(-max_velocity, max_velocity)
  robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
