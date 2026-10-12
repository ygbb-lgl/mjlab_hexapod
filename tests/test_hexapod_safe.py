"""Configuration tests for the tuned EL-4090 task."""

import torch
from tensordict import TensorDict

from mjlab.asset_zoo.robots.hexapod_tethered.hexapod_constants import JOINT_NAMES
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.hexapod_safe.actions import (
  SignedJointPositionActionCfg,
)
from mjlab.tasks.velocity.config.hexapod_safe.env_cfgs import (
  EL4090_ACTION_SCALE,
  EL4090_REWARD_TORQUE_LIMIT,
)
from mjlab.tasks.velocity.config.hexapod_safe.symmetry import (
  hexapod_left_right_symmetry,
)

_TASK_ID = "Mjlab-Velocity-Flat-Hexapod-Safe"


def test_safe_task_preserves_playground_io_layout() -> None:
  cfg = load_env_cfg(_TASK_ID)
  expected_terms = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
  )
  assert tuple(cfg.observations["actor"].terms) == expected_terms
  assert tuple(cfg.observations["critic"].terms) == expected_terms
  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, SignedJointPositionActionCfg)
  assert action_cfg.scale == {".*": EL4090_ACTION_SCALE}
  assert tuple(action_cfg.actuator_names) == JOINT_NAMES
  assert action_cfg.preserve_order
  joint_cfg = cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"]
  assert tuple(joint_cfg.joint_names) == JOINT_NAMES
  assert joint_cfg.preserve_order


def test_safe_reward_weights_match_legged_gym() -> None:
  cfg = load_env_cfg(_TASK_ID)
  assert {name: term.weight for name, term in cfg.rewards.items()} == {
    "tracking_lin_vel": 6.0,
    "tracking_ang_vel": 5.5,
    "lin_vel_z": -5.0,
    "ang_vel_xy": -0.5,
    "orientation": -5.0,
    "torques": -0.0001,
    "dof_vel": -0.0001,
    "dof_acc": -1.0e-6,
    "base_height": -50.0,
    "feet_slip": -0.1,
    "feet_air_time": 3.0,
    "collision": -1.0,
    "feet_stumble": -1.0,
    "action_rate": -0.01,
    "stand_still": -5.0,
    "dof_pos_limits": -0.1,
    "dof_vel_limits": -1.0,
    "torque_limits": -0.01,
    "feet_contact_forces": -0.03,
    "shank_vertical": -1.0,
    "feet_async": -2.0,
    "feet_sync": -2.0,
  }
  assert cfg.rewards["base_height"].params["target"] == 0.45
  assert cfg.rewards["feet_contact_forces"].params["max_contact_force"] == 250.0
  assert (
    cfg.rewards["torque_limits"].params["torque_limit"] == EL4090_REWARD_TORQUE_LIMIT
  )


def test_safe_ppo_and_symmetry_settings_match_source() -> None:
  cfg = load_rl_cfg(_TASK_ID)
  assert isinstance(cfg, RslRlOnPolicyRunnerCfg)
  assert cfg.actor.hidden_dims == (512, 256, 128)
  assert cfg.actor.distribution_cfg is not None
  assert cfg.actor.distribution_cfg["init_std"] == 0.3
  assert cfg.num_steps_per_env == 24
  assert cfg.max_iterations == 1500
  assert cfg.clip_actions == 100.0
  assert cfg.algorithm.symmetry_cfg is not None
  assert cfg.algorithm.symmetry_cfg["mirror_loss_coeff"] == 0.6
  assert cfg.obs_groups == {"actor": ("actor",), "critic": ("actor",)}


def test_safe_symmetry_uses_playground_blocks() -> None:
  vector = torch.arange(66, dtype=torch.float32).unsqueeze(0)
  observations = TensorDict(
    {"actor": vector.clone(), "critic": vector.clone()}, batch_size=[1]
  )
  actions = torch.arange(18, dtype=torch.float32).unsqueeze(0)
  mirrored_obs, mirrored_actions = hexapod_left_right_symmetry(observations, actions)
  assert mirrored_obs is not None
  assert mirrored_actions is not None
  assert mirrored_obs.batch_size == torch.Size([2])
  assert torch.equal(mirrored_actions[1, :9], actions[0, 9:])
  assert torch.equal(mirrored_actions[1, 9:], actions[0, :9])
  actor_obs = mirrored_obs["actor"]
  assert isinstance(actor_obs, torch.Tensor)
  assert torch.equal(actor_obs[1, 9:18], vector[0, 18:27])
  assert torch.equal(actor_obs[1, 18:27], vector[0, 9:18])
