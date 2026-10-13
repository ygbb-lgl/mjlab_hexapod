"""Hexapod Compare velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import hexapod_compare_flat_env_cfg, hexapod_compare_rough_env_cfg
from .rl_cfg import hexapod_compare_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Hexapod-Compare",
  env_cfg=hexapod_compare_rough_env_cfg(),
  play_env_cfg=hexapod_compare_rough_env_cfg(play=True),
  rl_cfg=hexapod_compare_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Hexapod-Compare",
  env_cfg=hexapod_compare_flat_env_cfg(),
  play_env_cfg=hexapod_compare_flat_env_cfg(play=True),
  rl_cfg=hexapod_compare_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
