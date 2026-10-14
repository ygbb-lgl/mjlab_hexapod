"""Tether observations for the tethered Hexapod task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import NoiseCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def cable_tension(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Return non-negative cable tension in Newtons.

  MuJoCo reports a negative tendon actuator force for this pulling actuator,
  because the XML gain is ``-200``.
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return torch.clamp(-sensor.data, min=0.0)


def joint_position(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  biased: bool = False,
) -> torch.Tensor:
  """Return absolute positions for the selected joints."""
  asset: Entity = env.scene[asset_cfg.name]
  position = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return position[:, asset_cfg.joint_ids]


def joint_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return velocities for the selected joints."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


def tendon_length(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the selected spatial tendon lengths."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.tendon_len[:, asset_cfg.tendon_ids]


def tendon_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the selected spatial tendon velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.tendon_vel[:, asset_cfg.tendon_ids]


def tether_direction_body(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the unit vector from ``tether_7`` to anchor in a body frame.

  ``asset_cfg.site_names`` must resolve to ``(anchor, tether_7)`` and
  ``asset_cfg.body_names`` must resolve to the desired reference body.
  """
  asset: Entity = env.scene[asset_cfg.name]
  site_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids]
  if site_pos_w.shape[1] != 2:
    raise ValueError(
      "tether_direction_body requires exactly two sites: anchor and tether_7"
    )
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
  if body_quat_w.shape[1] != 1:
    raise ValueError("tether_direction_body requires exactly one reference body")

  direction_w = site_pos_w[:, 0] - site_pos_w[:, 1]
  direction_w = direction_w / torch.clamp(
    torch.linalg.vector_norm(direction_w, dim=1, keepdim=True), min=1.0e-8
  )
  direction_b = quat_apply_inverse(body_quat_w[:, 0], direction_w)
  return direction_b / torch.clamp(
    torch.linalg.vector_norm(direction_b, dim=1, keepdim=True), min=1.0e-8
  )


def tether_angle_zx(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return tether pitch as ``atan2(v_z, v_x)`` in the ARM frame."""
  direction = tether_direction_body(env, asset_cfg)
  return torch.atan2(direction[:, 2], direction[:, 0]).unsqueeze(1)


def tether_direction_from_yaw_pitch_angles(
  yaw: torch.Tensor,
  pitch: torch.Tensor,
) -> torch.Tensor:
  """Reconstruct the trunk-frame tether direction from measured angles.

  ``yaw`` is the ``ARM_JOINT`` position. Its XML axis is ``(0, 0, -1)``, so
  positive yaw rotates the horizontal direction from trunk ``+x`` toward
  trunk ``-y``. ``pitch`` is positive from the horizontal plane toward
  trunk ``+z``.
  """
  horizontal = torch.cos(pitch)
  return torch.stack(
    (
      horizontal * torch.cos(yaw),
      -horizontal * torch.sin(yaw),
      torch.sin(pitch),
    ),
    dim=-1,
  )


def tether_direction_from_yaw_pitch(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  biased: bool = False,
) -> torch.Tensor:
  """Return the trunk-frame direction reconstructed from deployable sensors.

  The passive ARM joint supplies tether yaw. The tether direction measured in
  the ARM frame supplies pitch through ``atan2(v_z, v_x)``. This intentionally
  uses the same two angles available on hardware instead of simulator-only
  world-site positions.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  if not isinstance(joint_ids, list) or len(joint_ids) != 1:
    raise ValueError("tether_direction_from_yaw_pitch requires exactly one yaw joint")

  position = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  yaw = position[:, joint_ids[0]]
  pitch = tether_angle_zx(env, asset_cfg).squeeze(1)
  return tether_direction_from_yaw_pitch_angles(yaw, pitch)


@dataclass(kw_only=True)
class TetherDirectionNoiseCfg(NoiseCfg):
  """Apply yaw/pitch sensor noise while preserving a unit direction vector."""

  yaw_min: float = -0.01
  yaw_max: float = 0.01
  pitch_min: float = -0.005
  pitch_max: float = 0.005

  def __post_init__(self) -> None:
    if self.yaw_min >= self.yaw_max:
      raise ValueError("yaw_min must be less than yaw_max")
    if self.pitch_min >= self.pitch_max:
      raise ValueError("pitch_min must be less than pitch_max")
    if self.operation != "add":
      raise ValueError("TetherDirectionNoiseCfg only supports additive angle noise")

  def apply(self, data: torch.Tensor) -> torch.Tensor:
    """Perturb the two measured angles and reconstruct the direction."""
    if data.shape[-1] != 3:
      raise ValueError("TetherDirectionNoiseCfg expects a 3-D direction vector")

    yaw = torch.atan2(-data[..., 1], data[..., 0])
    pitch = torch.atan2(
      data[..., 2],
      torch.linalg.vector_norm(data[..., :2], dim=-1),
    )
    yaw_noise = torch.rand_like(yaw) * (self.yaw_max - self.yaw_min) + self.yaw_min
    pitch_noise = (
      torch.rand_like(pitch) * (self.pitch_max - self.pitch_min) + self.pitch_min
    )
    return tether_direction_from_yaw_pitch_angles(
      yaw + yaw_noise,
      pitch + pitch_noise,
    )
