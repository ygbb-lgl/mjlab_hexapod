"""Velocity command sampling matching the EL-4090 task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class El4090VelocityCommand(UniformVelocityCommand):
  """Zero sampled planar commands whose norm is at most 0.2 m/s."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    cfg = cast(El4090VelocityCommandCfg, self.cfg)
    planar = self.vel_command_b[env_ids, :2]
    moving = torch.norm(planar, dim=1) > cfg.planar_deadzone
    self.vel_command_b[env_ids, :2] *= moving.unsqueeze(1)

  def get_export_metadata(self) -> dict[str, float]:
    """Return command preprocessing needed by standalone policy runners."""
    cfg = cast(El4090VelocityCommandCfg, self.cfg)
    return {"planar_command_deadzone": cfg.planar_deadzone}


@dataclass(kw_only=True)
class El4090VelocityCommandCfg(UniformVelocityCommandCfg):
  planar_deadzone: float = 0.2

  def build(self, env: ManagerBasedRlEnv) -> El4090VelocityCommand:
    return El4090VelocityCommand(self, env)
