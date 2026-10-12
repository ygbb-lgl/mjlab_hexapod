"""Tuned EL-4090 hexapod velocity task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import hexapod_safe_env_cfg
from .rl_cfg import hexapod_safe_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Hexapod-Safe",
  env_cfg=hexapod_safe_env_cfg(),
  play_env_cfg=hexapod_safe_env_cfg(play=True),
  rl_cfg=hexapod_safe_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
