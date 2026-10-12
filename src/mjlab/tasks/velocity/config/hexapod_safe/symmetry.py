"""Left-right symmetry augmentation for the tuned 66-D policy."""

from __future__ import annotations

import torch
from tensordict import TensorDictBase
from tensordict import cat as tensordict_cat

OBS_DIM = 66
_MIRROR_JOINTS = (9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8)


def _mirror_vector(observation: torch.Tensor) -> torch.Tensor:
  if observation.shape[-1] != OBS_DIM:
    raise ValueError(
      f"EL-4090 symmetry expects {OBS_DIM} observations, got {observation.shape[-1]}"
    )
  mirrored = observation.clone()
  permutation = torch.tensor(_MIRROR_JOINTS, device=observation.device)

  mirrored[:, 1] = -observation[:, 1]
  mirrored[:, 3] = -observation[:, 3]
  mirrored[:, 5] = -observation[:, 5]
  mirrored[:, 7] = -observation[:, 7]
  for start in (9, 27, 45):
    mirrored[:, start : start + 18] = observation[:, start : start + 18][:, permutation]
  mirrored[:, 64] = -observation[:, 64]
  mirrored[:, 65] = -observation[:, 65]
  return mirrored


@torch.no_grad()
def hexapod_left_right_symmetry(
  obs: TensorDictBase | None = None,
  actions: torch.Tensor | None = None,
  env=None,
) -> tuple[TensorDictBase | None, torch.Tensor | None]:
  """Append a left-right mirrored copy for RSL-RL augmentation and loss."""
  del env
  augmented_obs: TensorDictBase | None = None
  if obs is not None:
    mirrored_obs = obs.clone()
    for key, value in obs.items():
      if isinstance(value, torch.Tensor) and value.shape[-1] == OBS_DIM:
        mirrored_obs[key] = _mirror_vector(value)
    augmented_obs = tensordict_cat([obs, mirrored_obs], dim=0)

  augmented_actions = None
  if actions is not None:
    mirrored_actions = actions[:, _MIRROR_JOINTS]
    augmented_actions = torch.cat([actions, mirrored_actions], dim=0)

  return augmented_obs, augmented_actions
