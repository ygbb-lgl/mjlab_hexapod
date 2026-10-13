"""Tripod-gait rewards for the crab-style hexapod."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensor

from .observations import tripod_phase_fraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


TRIPOD_A = ("RF", "RB", "LM")
TRIPOD_B = ("RM", "LF", "LB")


def _moving_command(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  command_magnitude = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return command_magnitude > command_threshold


def _load_bearing_contact(
  sensor: ContactSensor,
  contact_force_threshold: float,
) -> torch.Tensor:
  """Return contacts carrying more than a negligible grazing force."""
  if contact_force_threshold < 0.0:
    raise ValueError(
      f"contact_force_threshold must be non-negative, got {contact_force_threshold}"
    )
  assert sensor.data.found is not None
  assert sensor.data.force is not None
  force = torch.linalg.vector_norm(sensor.data.force, dim=-1)
  return (sensor.data.found > 0) & (force >= contact_force_threshold)


class TripodContactPattern:
  """Reward the two alternating three-foot support triangles.

  Phase A supports ``RF + RB + LM`` and phase B supports ``RM + LF + LB``.
  Around each phase switch, either complete tripod is accepted for a short
  transition window; the separate support-count cost still favors three feet.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    transition_fraction = cfg.params["transition_fraction"]
    if not 0.0 <= transition_fraction < 0.5:
      raise ValueError(
        f"transition_fraction must be in [0, 0.5), got {transition_fraction}"
      )
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    name_to_index = {
      name.removesuffix("_foot_collision"): index
      for index, name in enumerate(sensor.primary_names)
    }
    expected = set(TRIPOD_A + TRIPOD_B)
    if set(name_to_index) != expected:
      raise ValueError(
        "Tripod reward requires exactly RF/RM/RB/LF/LM/LB foot contacts; "
        f"got {tuple(sensor.primary_names)}"
      )
    self._tripod_a = tuple(name_to_index[name] for name in TRIPOD_A)
    self._tripod_b = tuple(name_to_index[name] for name in TRIPOD_B)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    cycle_time: float,
    command_threshold: float,
    transition_fraction: float,
    contact_force_threshold: float,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    contact = _load_bearing_contact(sensor, contact_force_threshold)

    phase = tripod_phase_fraction(env, cycle_time)
    a_stance = phase < 0.5
    target_a = torch.zeros_like(contact)
    target_a[:, self._tripod_a] = True
    target_a[:, self._tripod_b] = False
    target_b = ~target_a

    target = torch.where(a_stance.unsqueeze(1), target_a, target_b)
    alternate = torch.where(a_stance.unsqueeze(1), target_b, target_a)
    score = (contact == target).float().mean(dim=1)
    alternate_score = (contact == alternate).float().mean(dim=1)

    half_window = 0.5 * transition_fraction
    switch_distance = torch.minimum(
      torch.minimum(phase, 1.0 - phase),
      torch.abs(phase - 0.5),
    )
    in_transition = switch_distance < half_window
    score = torch.where(in_transition, torch.maximum(score, alternate_score), score)

    moving = _moving_command(env, command_name, command_threshold)
    standing_score = contact.float().mean(dim=1)
    reward = torch.where(moving, score, standing_score)

    env.extras["log"]["Metrics/support_feet_mean"] = contact.float().sum(dim=1).mean()
    env.extras["log"]["Metrics/tripod_match_mean"] = score.mean()
    return reward


def tripod_support_count_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float,
  contact_force_threshold: float,
) -> torch.Tensor:
  """Penalize deviation from three support feet, or six while standing."""
  sensor: ContactSensor = env.scene[sensor_name]
  contact = _load_bearing_contact(sensor, contact_force_threshold)
  contact_count = contact.float().sum(dim=1)
  moving = _moving_command(env, command_name, command_threshold)
  target_count = torch.where(
    moving,
    torch.full_like(contact_count, 3.0),
    torch.full_like(contact_count, 6.0),
  )
  return torch.square((contact_count - target_count) / 3.0)
