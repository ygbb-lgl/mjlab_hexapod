"""Flat-terrain velocity task for the crab-style hexapod."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import hexapod_flat_env_cfg
from .rl_cfg import hexapod_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Hexapod",
  env_cfg=hexapod_flat_env_cfg(),
  play_env_cfg=hexapod_flat_env_cfg(play=True),
  rl_cfg=hexapod_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
