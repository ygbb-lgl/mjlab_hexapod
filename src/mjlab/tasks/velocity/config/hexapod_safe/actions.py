"""Joint-axis conversion for the tuned EL-4090 task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.asset_zoo.robots.hexapod_tethered.hexapod_constants import (
  JOINT_NAMES,
  SAFE_JOINT_SIGNS,
)
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class SignedJointPositionActionCfg(JointPositionActionCfg):
  """Position action with the source URDF-to-MJCF axis conversion."""

  policy_action_clip: float = 100.0

  def build(self, env: ManagerBasedRlEnv) -> SignedJointPositionAction:
    return SignedJointPositionAction(self, env)


class SignedJointPositionAction(JointPositionAction):
  """Apply ordinary PPO actions after converting source joint coordinates.

  This is only a coordinate adapter. It does not filter, project, or otherwise
  modify the policy output with a safety algorithm.
  """

  def __init__(self, cfg: SignedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if tuple(self.target_names) != JOINT_NAMES:
      raise ValueError(
        "EL-4090 action order must match MuJoCo Playground. "
        f"Expected {JOINT_NAMES}, got {tuple(self.target_names)}"
      )
    if self.action_dim != len(SAFE_JOINT_SIGNS):
      raise ValueError(
        f"Expected {len(SAFE_JOINT_SIGNS)} actions, got {self.action_dim}"
      )
    self._joint_signs = torch.tensor(
      SAFE_JOINT_SIGNS,
      device=self.device,
      dtype=self._raw_actions.dtype,
    ).unsqueeze(0)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions.to(self.device)
    self._processed_actions = (
      self._raw_actions * self._scale * self._joint_signs + self._offset
    )
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )

  def get_export_metadata(self) -> dict[str, list[float] | str | float]:
    """Return coordinate conversion data for standalone policy runners."""
    cfg = cast(SignedJointPositionActionCfg, self.cfg)
    return {
      "hexapod_action_mode": "signed_axes",
      "hexapod_joint_signs": list(SAFE_JOINT_SIGNS),
      "policy_action_clip": cfg.policy_action_clip,
    }
