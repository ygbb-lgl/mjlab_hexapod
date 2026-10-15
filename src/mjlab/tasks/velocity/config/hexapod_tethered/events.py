"""Reset events for the fixed and randomized-slope tethered Hexapod tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class SlopeCurriculumStage(TypedDict):
  """One global-step stage for randomized slope angles."""

  step: int
  angle_range_deg: tuple[float, float]


def _validate_slope_stages(stages: list[SlopeCurriculumStage]) -> None:
  if not stages:
    raise ValueError("slope_stages must contain at least one stage")
  if stages[0]["step"] != 0:
    raise ValueError("the first slope stage must start at step 0")

  previous_step = -1
  for index, stage in enumerate(stages):
    step = stage["step"]
    lower, upper = stage["angle_range_deg"]
    if step < previous_step:
      raise ValueError(
        "slope stages must be in nondecreasing step order, but "
        f"stage {index} starts at {step} after {previous_step}"
      )
    if not 0.0 <= lower <= upper < 90.0:
      raise ValueError(
        "slope angle ranges must satisfy 0 <= lower <= upper < 90 degrees"
      )
    previous_step = step


class RandomizeSlopeAngle:
  """Sample one slope angle per environment and pose the mocap terrain plane.

  Positive angles make world ``+X`` uphill. The plane is rotated by the
  corresponding negative pitch, so its upward normal is
  ``(-sin(angle), 0, cos(angle))``. Stage thresholds use
  :attr:`ManagerBasedRlEnv.common_step_counter`.

  The mixture scene marks the terrain body as mocap. Writing its per-environment
  pose lets the regular forward-kinematics path update collision, ray casting,
  and visualization from the same source of truth. Each plane is positioned at
  its environment origin, so it rotates about the local spawn point instead of
  the global origin.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    stages: list[SlopeCurriculumStage] = cfg.params["slope_stages"]
    _validate_slope_stages(stages)
    self._stages = stages
    initial_angle = math.radians(stages[0]["angle_range_deg"][0])
    self.slope_angle_rad = torch.full(
      (env.num_envs,),
      initial_angle,
      device=env.device,
      dtype=torch.float32,
    )

    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    terrain: Entity = env.scene[asset_cfg.name]
    geom_ids = terrain.indexing.geom_ids[asset_cfg.geom_ids]
    if geom_ids.numel() != 1:
      raise ValueError("RandomizeSlopeAngle requires exactly one terrain geom")
    terrain_geom_id = int(geom_ids.item())
    terrain_body_id = int(env.sim.mj_model.geom_bodyid[terrain_geom_id])
    self._terrain_mocap_id = int(env.sim.mj_model.body_mocapid[terrain_body_id])
    if self._terrain_mocap_id < 0:
      raise ValueError(
        "RandomizeSlopeAngle requires the terrain body to be marked as mocap"
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    slope_stages: list[SlopeCurriculumStage],
    asset_cfg: SceneEntityCfg,
  ) -> None:
    del slope_stages  # Validated and cached when the event manager is built.
    del asset_cfg  # Resolved and validated when the event manager is built.
    env_ids = mdp.resolve_env_ids(env, env_ids)

    active_stage = self._stages[0]
    for stage in self._stages[1:]:
      if env.common_step_counter < stage["step"]:
        break
      active_stage = stage

    lower_deg, upper_deg = active_stage["angle_range_deg"]
    sampled_deg = torch.empty(
      len(env_ids), device=env.device, dtype=torch.float32
    ).uniform_(lower_deg, upper_deg)
    sampled_rad = torch.deg2rad(sampled_deg)
    self.slope_angle_rad[env_ids] = sampled_rad

    zeros = torch.zeros_like(sampled_rad)
    slope_quat = quat_from_euler_xyz(zeros, -sampled_rad, zeros)
    env.sim.data.mocap_pos[env_ids, self._terrain_mocap_id] = env.scene.env_origins[
      env_ids
    ]
    env.sim.data.mocap_quat[env_ids, self._terrain_mocap_id] = slope_quat

    all_angles_deg = torch.rad2deg(self.slope_angle_rad)
    env.extras["log"]["Curriculum/slope_angle_deg_mean"] = all_angles_deg.mean()
    env.extras["log"]["Curriculum/slope_angle_deg_min"] = all_angles_deg.min()
    env.extras["log"]["Curriculum/slope_angle_deg_max"] = all_angles_deg.max()
    env.extras["log"]["Curriculum/slope_range_deg_min"] = lower_deg
    env.extras["log"]["Curriculum/slope_range_deg_max"] = upper_deg


def get_slope_angle_rad(
  env: ManagerBasedRlEnv,
  event_name: str,
) -> torch.Tensor:
  """Return the per-environment angle buffer owned by a slope reset event."""
  term = env.event_manager.get_term_cfg(event_name).func
  if not isinstance(term, RandomizeSlopeAngle):
    raise TypeError(
      f"Event {event_name!r} must use RandomizeSlopeAngle, got {type(term).__name__}"
    )
  return term.slope_angle_rad


def _slope_angles(
  env: ManagerBasedRlEnv,
  slope_angle_rad: float,
  slope_event_name: str | None,
) -> torch.Tensor:
  if slope_event_name is not None:
    return get_slope_angle_rad(env, slope_event_name)
  return torch.full(
    (env.num_envs,),
    slope_angle_rad,
    device=env.device,
    dtype=torch.float32,
  )


def _slope_frame_axes(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Return unit uphill tangents and upward normals for slope angles."""
  zeros = torch.zeros_like(angles)
  tangent = torch.stack((torch.cos(angles), zeros, torch.sin(angles)), dim=-1)
  normal = torch.stack((-torch.sin(angles), zeros, torch.cos(angles)), dim=-1)
  return tangent, normal


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
  slope_event_name: str | None = None,
) -> None:
  """Apply the fixed-slope root reset around the global terrain origin."""
  if slope_event_name is not None:
    reset_root_state_on_slope_mixture(
      env,
      env_ids,
      pose_range=pose_range,
      velocity_range=velocity_range,
      slope_event_name=slope_event_name,
      asset_cfg=asset_cfg,
    )
    return

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
  angles = torch.full_like(root_pose[:, 0], slope_angle_rad)
  root_pose[:, 2] += root_pose[:, 0] * torch.tan(angles)
  asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)


def reset_root_state_on_slope_mixture(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None,
  slope_event_name: str,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Reset the root in coordinates attached to each randomized slope.

  The usual ``pose_range`` samples are reinterpreted as uphill-tangent ``x``,
  lateral ``y``, and terrain-normal ``z`` offsets. The entity's default
  position is expressed in the same frame. This keeps the initial clearance
  and displacement physically constant as the sampled angle changes.
  """
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
  origins = env.scene.env_origins[env_ids].to(root_pose)

  # reset_root_state_uniform produced origin + default position + sampled
  # position. Reinterpret that relative vector as (tangent, lateral, normal)
  # coordinates instead of leaving it in the world frame.
  slope_coordinates = root_pose[:, :3] - origins
  angles = get_slope_angle_rad(env, slope_event_name)[env_ids].to(root_pose)
  tangent, normal = _slope_frame_axes(angles)
  root_pose[:, :3] = (
    origins + slope_coordinates[:, 0:1] * tangent + slope_coordinates[:, 2:3] * normal
  )
  root_pose[:, 1] += slope_coordinates[:, 1]

  default_quat = asset.data.default_root_state[env_ids, 3:7].to(root_pose)
  orientation_delta = quat_mul(
    quat_inv(default_quat.contiguous()),
    root_pose[:, 3:7].contiguous(),
  )
  zeros = torch.zeros_like(angles)
  slope_quat = quat_from_euler_xyz(zeros, -angles, zeros)
  root_pose[:, 3:7] = quat_mul(slope_quat, orientation_delta)
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
  slope_event_name: str | None = None,
) -> None:
  """Place each anchor at a signed-normal height above the current slope."""
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
  angles = _slope_angles(env, slope_angle_rad, slope_event_name)[env_ids].to(origins)
  anchor_pos = torch.empty((count, 3), device=env.device)
  anchor_pos[:, 0] = origins[:, 0] + x_local
  anchor_pos[:, 1] = origins[:, 1] + y_local
  if slope_event_name is None:
    # The fixed-slope scene owns one global plane rotated about world origin.
    anchor_pos[:, 2] = anchor_pos[:, 0] * torch.tan(angles) + height_normal / torch.cos(
      angles
    )
  else:
    # Mixture ranges use slope-frame coordinates: x is uphill distance and z is
    # signed distance along the upward normal. This keeps cable geometry stable
    # when the curriculum expands to steeper angles.
    tangent, normal = _slope_frame_axes(angles)
    anchor_pos = origins + x_local[:, None] * tangent + height_normal[:, None] * normal
    anchor_pos[:, 1] += y_local

  site_ids = asset.indexing.site_ids[asset_cfg.site_ids].to(dtype=env_ids.dtype)
  env_grid, site_grid = torch.meshgrid(env_ids, site_ids, indexing="ij")
  env.sim.model.site_pos[env_grid, site_grid] = anchor_pos[:, None, :]
