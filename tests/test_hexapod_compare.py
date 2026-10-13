"""Configuration checks for the Hexapod Compare velocity tasks."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.asset_zoo.robots.hexapod_compare.hexapod_compare_constants import (
  HEXAPOD_COMPARE_ACTION_SCALE,
  JOINT_NAMES,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.velocity.config.hexapod_compare import rewards
from mjlab.tasks.velocity.config.hexapod_compare.env_cfgs import (
  hexapod_compare_flat_env_cfg,
  hexapod_compare_rough_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg

_POLICY_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
)

_FLAT_REWARD_WEIGHTS = {
  "tracking_lin_vel": 5.0,
  "tracking_ang_vel": 3.0,
  "lin_vel_z": -3.0,
  "ang_vel_xy": -0.05,
  "orientation": -7.0,
  "stand_still": -1.0,
  "termination": -1.5,
  "pose": 1.0,
  "torques": -0.005,
  "action_rate": -0.01,
  "dof_vel": -1.0e-5,
  "dof_acc": -1.0e-7,
  "energy": -1.0e-4,
  "feet_slip": -0.1,
  "feet_contact_forces": -0.003,
  "base_height": -5.0,
  "dof_pos_limits": -1.0,
  "tripod_contact_phase": -3.0,
  "tripod_joint_pos_sync": -0.8,
  "tripod_joint_vel_sync": -0.15,
  "tripod_joint_antiphase": -0.8,
  "contact_force_balance": -1.0,
  "shank_vertical": -3.0,
}


def test_playground_policy_abi_and_noise() -> None:
  cfg = hexapod_compare_flat_env_cfg()
  assert tuple(cfg.observations["actor"].terms) == _POLICY_TERMS
  assert tuple(cfg.observations["critic"].terms)[:7] == _POLICY_TERMS
  assert "foot_contact_forces" in cfg.observations["critic"].terms

  expected_noise = {
    "base_lin_vel": (-0.5, 0.5),
    "base_ang_vel": (-0.2, 0.2),
    "projected_gravity": (-0.05, 0.05),
    "joint_pos": (-0.01, 0.01),
    "joint_vel": (-1.5, 1.5),
  }
  for name, (noise_min, noise_max) in expected_noise.items():
    noise = cfg.observations["actor"].terms[name].noise
    assert isinstance(noise, UniformNoiseCfg)
    assert noise.n_min == noise_min
    assert noise.n_max == noise_max

  joint_cfg = cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"]
  assert tuple(joint_cfg.joint_names) == JOINT_NAMES
  assert joint_cfg.preserve_order


def test_action_order_preserves_mjlab_control_and_command_configuration() -> None:
  cfg = hexapod_compare_flat_env_cfg()
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert tuple(action.actuator_names) == JOINT_NAMES
  assert action.preserve_order
  assert action.scale == HEXAPOD_COMPARE_ACTION_SCALE

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.resampling_time_range == (3.0, 8.0)

  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert cfg.episode_length_s == 20.0


def test_flat_rewards_match_playground_joystick() -> None:
  cfg = hexapod_compare_flat_env_cfg()
  assert {name: term.weight for name, term in cfg.rewards.items()} == (
    _FLAT_REWARD_WEIGHTS
  )
  force_term = cfg.rewards["feet_contact_forces"]
  assert force_term.params["max_contact_force"] == 200.0


def test_rough_rewards_match_playground_complex() -> None:
  cfg = hexapod_compare_rough_env_cfg()
  expected = {**_FLAT_REWARD_WEIGHTS, "feet_air_time": 0.0}
  expected["contact_force_balance"] = 0.0
  assert {name: term.weight for name, term in cfg.rewards.items()} == expected
  force_term = cfg.rewards["feet_contact_forces"]
  assert force_term.params["max_contact_force"] == 300.0
  assert cfg.rewards["base_height"].params["terrain_sensor_name"] == (
    "base_height_scan"
  )


def test_foot_force_sensor_matches_go1_soft_landing_path() -> None:
  for cfg in (hexapod_compare_flat_env_cfg(), hexapod_compare_rough_env_cfg()):
    sensor = next(
      sensor
      for sensor in cfg.scene.sensors or ()
      if sensor.name == "feet_ground_contact"
    )
    assert isinstance(sensor, ContactSensorCfg)
    assert sensor.reduce == "netforce"
    assert sensor.fields == ("found", "force")
    assert sensor.num_slots == 1
    assert sensor.track_air_time

  force = torch.tensor([[[3.0, 4.0, 0.0]]])
  sensor = SimpleNamespace(data=SimpleNamespace(force=force, found=torch.ones(1, 1)))
  env = cast(Any, SimpleNamespace(scene={"feet": sensor}))
  cost = rewards.feet_contact_forces(
    env,
    sensor_name="feet",
    max_contact_force=4.0,
  )
  torch.testing.assert_close(cost, torch.tensor([1.0]))
