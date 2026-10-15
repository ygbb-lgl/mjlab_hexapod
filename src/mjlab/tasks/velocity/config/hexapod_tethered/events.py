"""Reset events for the fixed and randomized-slope tethered Hexapod tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, TypedDict

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_inv, quat_mul

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
  """Sample one slope angle per environment and rotate the infinite plane.

  Positive angles make world ``+X`` uphill. The plane is rotated by the
  corresponding negative pitch, so its upward normal is
  ``(-sin(angle), 0, cos(angle))``. Stage thresholds use
  :attr:`ManagerBasedRlEnv.common_step_counter`.
  """

  # Class-based manager terms cannot use ``requires_model_fields`` without
  # erasing their class type. EventManager reads the same marker from instances.
  model_fields: ClassVar[tuple[str, ...]] = ("geom_quat",)

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

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    slope_stages: list[SlopeCurriculumStage],
    asset_cfg: SceneEntityCfg,
  ) -> None:
    del slope_stages  # Validated and cached when the event manager is built.
    env_ids = mdp.resolve_env_ids(env, env_ids).to(dtype=torch.int)

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

    terrain: Entity = env.scene[asset_cfg.name]
    geom_ids = terrain.indexing.geom_ids[asset_cfg.geom_ids]
    if geom_ids.numel() != 1:
      raise ValueError("RandomizeSlopeAngle requires exactly one terrain geom")
    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    zeros = torch.zeros_like(sampled_rad)
    slope_quat = quat_from_euler_xyz(zeros, -sampled_rad, zeros)
    env.sim.model.geom_quat[env_grid, geom_grid] = slope_quat[:, None, :]

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
  """Apply the usual root reset and align it with the current slope."""
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
  angles = _slope_angles(env, slope_angle_rad, slope_event_name)[env_ids].to(root_pose)

  if slope_event_name is None:
    root_pose[:, 2] += root_pose[:, 0] * torch.tan(angles)
  else:
    default_root_state = asset.data.default_root_state
    origins_z = env.scene.env_origins[env_ids, 2].to(root_pose)
    default_root_z = default_root_state[env_ids, 2].to(root_pose)
    normal_z_perturb = root_pose[:, 2] - origins_z - default_root_z
    reference_normal_height = default_root_z * math.cos(slope_angle_rad)
    root_pose[:, 2] = root_pose[:, 0] * torch.tan(angles) + (
      reference_normal_height + normal_z_perturb
    ) / torch.cos(angles)

    default_quat = default_root_state[env_ids, 3:7].to(root_pose).contiguous()
    orientation_delta = quat_mul(quat_inv(default_quat), root_pose[:, 3:7].contiguous())
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
  anchor_pos[:, 2] = anchor_pos[:, 0] * torch.tan(angles) + height_normal / torch.cos(
    angles
  )

  site_ids = asset.indexing.site_ids[asset_cfg.site_ids].to(dtype=env_ids.dtype)
  env_grid, site_grid = torch.meshgrid(env_ids, site_ids, indexing="ij")
  env.sim.model.site_pos[env_grid, site_grid] = anchor_pos[:, None, :]
