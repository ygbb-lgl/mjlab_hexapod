"""Hexapod-specific observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def tripod_phase_fraction(
  env: ManagerBasedRlEnv,
  cycle_time: float,
) -> torch.Tensor:
  """Return the normalized two-phase gait clock in ``[0, 1)``."""
  if cycle_time <= 0.0:
    raise ValueError(f"cycle_time must be positive, got {cycle_time}")
  elapsed = env.episode_length_buf * env.step_dt
  return torch.remainder(elapsed, cycle_time) / cycle_time
