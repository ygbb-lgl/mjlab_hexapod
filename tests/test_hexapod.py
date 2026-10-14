"""Configuration and tripod-gait tests for the crab-style hexapod."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.asset_zoo.robots.hexapod.hexapod_constants import (
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  HEXAPOD_ACTION_SCALE,
  JOINT_NAMES,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensorCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.tasks.velocity.config.hexapod import env_cfgs as hexapod_env_cfgs
from mjlab.tasks.velocity.config.hexapod import rewards
from mjlab.tasks.velocity.config.hexapod.env_cfgs import (
  FOOT_CLEARANCE_HEIGHT,
  GAIT_CYCLE_TIME,
  OBSERVATION_NOISE_LEVEL,
  OBSERVATION_NOISE_SCALES,
  SOFT_LANDING_WEIGHT,
  hexapod_flat_env_cfg,
)
from mjlab.tasks.velocity.config.hexapod.rl_cfg import hexapod_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg


def test_flat_configuration_uses_ordered_six_leg_interface() -> None:
  cfg = hexapod_flat_env_cfg()

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.terrain.terrain_generator is None

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert tuple(action.actuator_names) == JOINT_NAMES
  assert action.preserve_order
  assert action.scale == HEXAPOD_ACTION_SCALE
  assert action.clip is None

  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert "height_scan" not in terms
    assert "gait_phase" not in terms
    joint_cfg = terms["joint_pos"].params["asset_cfg"]
    assert tuple(joint_cfg.joint_names) == JOINT_NAMES
    assert joint_cfg.preserve_order

  assert tuple(cfg.observations["actor"].terms) == (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
  )

  for name, scale in OBSERVATION_NOISE_SCALES.items():
    noise = cfg.observations["actor"].terms[name].noise
    assert isinstance(noise, UniformNoiseCfg)
    assert noise.n_min == pytest.approx(-OBSERVATION_NOISE_LEVEL * scale)
    assert noise.n_max == pytest.approx(OBSERVATION_NOISE_LEVEL * scale)


def test_flat_configuration_wires_six_foot_sensors() -> None:
  cfg = hexapod_flat_env_cfg()
  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}

  feet = sensors["feet_ground_contact"]
  assert isinstance(feet, ContactSensorCfg)
  assert feet.primary.pattern == FOOT_GEOM_NAMES
  assert feet.reduce == "netforce"
  assert feet.track_air_time

  height = sensors["foot_height_scan"]
  assert isinstance(height, TerrainHeightSensorCfg)
  assert not height.debug_vis
  assert isinstance(height.frame, tuple)
  frames = cast(tuple[ObjRef, ...], height.frame)
  assert tuple(frame.name for frame in frames) == FOOT_SITE_NAMES
  assert "illegal_ground_contact" in sensors
  assert "terrain_scan" not in sensors

  play_cfg = hexapod_flat_env_cfg(play=True)
  play_sensors = {sensor.name: sensor for sensor in play_cfg.scene.sensors or ()}
  play_height = play_sensors["foot_height_scan"]
  assert isinstance(play_height, TerrainHeightSensorCfg)
  assert play_height.debug_vis


def test_zero_observation_noise_level_disables_noise(monkeypatch) -> None:
  monkeypatch.setattr(hexapod_env_cfgs, "OBSERVATION_NOISE_LEVEL", 0.0)
  cfg = hexapod_env_cfgs.hexapod_flat_env_cfg()

  for name in OBSERVATION_NOISE_SCALES:
    assert cfg.observations["actor"].terms[name].noise is None


def test_rewards_and_commands_are_scaled_for_the_xml_model() -> None:
  cfg = hexapod_flat_env_cfg()
  weights = {name: term.weight for name, term in cfg.rewards.items()}

  assert weights["track_linear_velocity"] == 2.0
  assert weights["track_angular_velocity"] == 2.0
  assert weights["air_time"] == 0.0
  assert weights["action_rate_l2"] == pytest.approx(-0.1 * 12.0 / 18.0)
  assert weights["foot_clearance"] == pytest.approx(-2.0 * 4.0 / 6.0)
  assert weights["foot_swing_height"] == pytest.approx(-0.25 * 4.0 / 6.0)
  assert weights["foot_slip"] == pytest.approx(-0.1 * 4.0 / 6.0)
  assert weights["soft_landing"] == pytest.approx(SOFT_LANDING_WEIGHT)
  assert weights["tripod_contact_pattern"] == 1.0
  assert weights["tripod_support_count"] == -0.25
  assert cfg.rewards["tripod_contact_pattern"].params["contact_force_threshold"] == 5.0
  assert cfg.rewards["foot_clearance"].params["target_height"] == (
    FOOT_CLEARANCE_HEIGHT
  )

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (-0.3, 0.3)
  assert command.ranges.lin_vel_y == (-0.2, 0.2)
  assert command.ranges.ang_vel_z == (-0.25, 0.25)


def _fake_gait_env(
  found: torch.Tensor,
  command: torch.Tensor,
  episode_steps: torch.Tensor,
) -> Any:
  force = torch.zeros((*found.shape, 3), dtype=torch.float32)
  force[..., 2] = found.float() * 100.0
  sensor = SimpleNamespace(
    primary_names=[
      f"{leg}_foot_collision" for leg in ("RF", "RM", "RB", "LF", "LM", "LB")
    ],
    data=SimpleNamespace(found=found, force=force),
  )
  command_manager = SimpleNamespace(get_command=lambda _: command)
  return cast(
    Any,
    SimpleNamespace(
      scene={"feet": sensor},
      command_manager=command_manager,
      episode_length_buf=episode_steps,
      step_dt=0.02,
      extras={"log": {}},
    ),
  )


def test_tripod_reward_matches_both_support_phases() -> None:
  # With a 0.6 s period and 0.02 s policy step, steps 5 and 20 are safely
  # inside phase A and phase B rather than in the transition window.
  found = torch.tensor(
    [
      [1, 0, 1, 0, 1, 0],
      [0, 1, 0, 1, 0, 1],
    ]
  )
  command = torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
  env = _fake_gait_env(found, command, torch.tensor([5, 20]))
  cfg = RewardTermCfg(
    func=rewards.TripodContactPattern,
    weight=1.0,
    params={
      "sensor_name": "feet",
      "command_name": "twist",
      "cycle_time": GAIT_CYCLE_TIME,
      "command_threshold": 0.05,
      "transition_fraction": 0.08,
      "contact_force_threshold": 5.0,
    },
  )
  term = rewards.TripodContactPattern(cfg, env)

  score = term(env, **cfg.params)
  count_cost = rewards.tripod_support_count_cost(
    env,
    sensor_name="feet",
    command_name="twist",
    command_threshold=0.05,
    contact_force_threshold=5.0,
  )

  torch.testing.assert_close(score, torch.ones(2))
  torch.testing.assert_close(count_cost, torch.zeros(2))


def test_tripod_reward_requires_all_feet_while_standing() -> None:
  found = torch.ones((1, 6), dtype=torch.int64)
  env = _fake_gait_env(found, torch.zeros((1, 3)), torch.tensor([10]))
  cfg = RewardTermCfg(
    func=rewards.TripodContactPattern,
    weight=1.0,
    params={
      "sensor_name": "feet",
      "command_name": "twist",
      "cycle_time": GAIT_CYCLE_TIME,
      "command_threshold": 0.05,
      "transition_fraction": 0.08,
      "contact_force_threshold": 5.0,
    },
  )
  term = rewards.TripodContactPattern(cfg, env)

  torch.testing.assert_close(term(env, **cfg.params), torch.ones(1))
  torch.testing.assert_close(
    rewards.tripod_support_count_cost(
      env,
      sensor_name="feet",
      command_name="twist",
      command_threshold=0.05,
      contact_force_threshold=5.0,
    ),
    torch.zeros(1),
  )


def test_tripod_reward_ignores_grazing_contacts() -> None:
  found = torch.ones((1, 6), dtype=torch.int64)
  command = torch.tensor([[0.2, 0.0, 0.0]])
  env = _fake_gait_env(found, command, torch.tensor([5]))
  env.scene["feet"].data.force.fill_(1.0)
  cost = rewards.tripod_support_count_cost(
    env,
    sensor_name="feet",
    command_name="twist",
    command_threshold=0.05,
    contact_force_threshold=5.0,
  )
  torch.testing.assert_close(cost, torch.ones(1))


def test_runner_leaves_policy_actions_unclipped_and_saves_every_500_iterations() -> (
  None
):
  cfg = hexapod_ppo_runner_cfg()
  assert cfg.clip_actions is None
  assert cfg.save_interval == 500
  assert cfg.experiment_name == "hexapod_velocity"
