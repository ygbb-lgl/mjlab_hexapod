"""Flat, fixed-slope, and slope-mixture tether-control tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  hexapod_tethered_flat_env_cfg,
  hexapod_tethered_slope_env_cfg,
  hexapod_tethered_slope_mixture_env_cfg,
)
from .rl_cfg import (
  hexapod_tethered_ppo_runner_cfg,
  hexapod_tethered_slope_mixture_ppo_runner_cfg,
  hexapod_tethered_slope_ppo_runner_cfg,
)
from .runner import (
  TetheredSlopeMixtureOnPolicyRunner,
  TetheredVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Hexapod-Tethered",
  env_cfg=hexapod_tethered_flat_env_cfg(),
  play_env_cfg=hexapod_tethered_flat_env_cfg(play=True),
  rl_cfg=hexapod_tethered_ppo_runner_cfg(),
  runner_cls=TetheredVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Slope-Hexapod-Tethered",
  env_cfg=hexapod_tethered_slope_env_cfg(),
  play_env_cfg=hexapod_tethered_slope_env_cfg(play=True),
  rl_cfg=hexapod_tethered_slope_ppo_runner_cfg(),
  runner_cls=TetheredVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Slope-Mixture-Hexapod-Tethered",
  env_cfg=hexapod_tethered_slope_mixture_env_cfg(),
  play_env_cfg=hexapod_tethered_slope_mixture_env_cfg(play=True),
  rl_cfg=hexapod_tethered_slope_mixture_ppo_runner_cfg(),
  runner_cls=TetheredSlopeMixtureOnPolicyRunner,
)
