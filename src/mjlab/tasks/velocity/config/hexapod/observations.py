"""Hexapod-specific observations."""

from __future__ import annotations

import math
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


def tripod_phase(
  env: ManagerBasedRlEnv,
  cycle_time: float,
) -> torch.Tensor:
  """Return sine/cosine features for the alternating-tripod clock."""
  angle = 2.0 * math.pi * tripod_phase_fraction(env, cycle_time)
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
