"""Asset and task tests for the tethered Hexapod."""

import math
from collections.abc import Iterator
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.hexapod import hexapod_constants
from mjlab.asset_zoo.robots.hexapod_tethered import hexapod_tethered_constants
from mjlab.asset_zoo.robots.hexapod_tethered.hexapod_tethered_constants import (
  ARM_JOINT_NAME,
  JOINT_NAMES,
  SLOPE_ANGLE_RAD,
  SLOPE_ROOT_QUAT,
  TETHER_ANCHOR_SITE_NAME,
  TETHER_FORCE_SENSOR_NAME,
  TETHER_TENDON_NAME,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.envs.mdp.actions import (
  JointPositionAction,
  JointPositionActionCfg,
  TendonEffortActionCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor, ContactSensorCfg, TerrainHeightSensor
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.hexapod.env_cfgs import hexapod_flat_env_cfg
from mjlab.tasks.velocity.config.hexapod_tethered import env_cfgs as tethered_env_cfgs
from mjlab.tasks.velocity.config.hexapod_tethered import (
  events,
  observations,
  rewards,
  terminations,
)
from mjlab.tasks.velocity.config.hexapod_tethered.env_cfgs import (
  OBSERVATION_NOISE_LEVEL,
  OBSERVATION_NOISE_SCALES,
  SLOPE_MIXTURE_EVENT_NAME,
  TETHER_ACTION_MAX,
  TETHER_ACTION_MIN,
  TETHER_DIRECTION_NOISE_SCALES,
  TETHER_FORCE_PER_ACTION,
  hexapod_tethered_flat_env_cfg,
  hexapod_tethered_slope_env_cfg,
  hexapod_tethered_slope_mixture_env_cfg,
)
from mjlab.tasks.velocity.config.hexapod_tethered.rl_cfg import (
  HexapodTetheredSlopePpoRunnerCfg,
  hexapod_tethered_ppo_runner_cfg,
  hexapod_tethered_slope_ppo_runner_cfg,
)
from mjlab.tasks.velocity.config.hexapod_tethered.runner import (
  PRETRAINED_CABLE_ACTION_MEAN,
  PRETRAINED_CABLE_ACTION_STD,
  PRETRAINED_TENSION_NORMALIZER_STD,
  TetheredVelocityOnPolicyRunner,
  get_tethered_policy_metadata,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
)
from mjlab.utils.noise import UniformNoiseCfg


@pytest.fixture(scope="module")
def tethered_entity() -> Entity:
  return Entity(hexapod_tethered_constants.get_hexapod_tethered_robot_cfg())


@pytest.fixture(scope="module")
def tethered_model(tethered_entity: Entity) -> mujoco.MjModel:
  return tethered_entity.spec.compile()


@pytest.fixture(scope="module")
def tethered_slope_entity() -> Entity:
  return Entity(hexapod_tethered_constants.get_hexapod_tethered_slope_robot_cfg())


@pytest.fixture(scope="module")
def tethered_slope_model(tethered_slope_entity: Entity) -> mujoco.MjModel:
  return tethered_slope_entity.spec.compile()


@pytest.fixture(scope="module")
def tethered_slope_mixture_entity() -> Entity:
  cfg = hexapod_tethered_constants.get_hexapod_tethered_slope_mixture_robot_cfg()
  return Entity(cfg)


@pytest.fixture(scope="module")
def tethered_slope_mixture_model(
  tethered_slope_mixture_entity: Entity,
) -> mujoco.MjModel:
  return tethered_slope_mixture_entity.spec.compile()


@pytest.fixture(scope="module")
def tethered_env() -> Iterator[ManagerBasedRlEnv]:
  cfg = hexapod_tethered_flat_env_cfg()
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device="cpu")
  yield env
  env.close()


@pytest.fixture(scope="module")
def tethered_slope_env() -> Iterator[ManagerBasedRlEnv]:
  cfg = hexapod_tethered_slope_env_cfg()
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device="cpu")
  yield env
  env.close()


@pytest.fixture(scope="module")
def tethered_slope_mixture_env() -> Iterator[ManagerBasedRlEnv]:
  cfg = hexapod_tethered_slope_mixture_env_cfg()
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg, device="cpu")
  yield env
  env.close()


def test_tethered_asset_preserves_leg_model_and_adds_passive_arm(
  tethered_entity: Entity,
  tethered_model: mujoco.MjModel,
) -> None:
  assert tethered_entity.num_joints == 19
  assert tuple(tethered_entity.joint_names) == (*JOINT_NAMES, ARM_JOINT_NAME)
  assert tethered_entity.num_actuators == 19
  assert tuple(tethered_entity.tendon_names) == (TETHER_TENDON_NAME,)
  np.testing.assert_allclose(tethered_model.site("anchor").pos, (5.0, 0.0, 2.0))

  arm = tethered_model.joint(ARM_JOINT_NAME)
  np.testing.assert_allclose(arm.axis, (0.0, 0.0, -1.0))
  assert arm.damping[0] == pytest.approx(1.0)
  assert arm.frictionloss[0] == pytest.approx(0.5)
  assert arm.armature[0] == pytest.approx(0.005)
  assert tethered_model.body_subtreemass[tethered_model.body("trunk").id] == (
    pytest.approx(40.5787774168)
  )

  flat_actuators = hexapod_tethered_constants.HEXAPOD_TETHERED_FLAT_ARTICULATION
  slope_actuators = hexapod_tethered_constants.HEXAPOD_TETHERED_SLOPE_ARTICULATION
  base_actuators = hexapod_constants.HEXAPOD_ARTICULATION
  assert hexapod_tethered_constants.HEXAPOD_TETHERED_ARTICULATION is flat_actuators
  assert flat_actuators is not slope_actuators
  assert flat_actuators is not base_actuators
  for flat_cfg, slope_cfg, base_cfg in zip(
    flat_actuators.actuators[:3],
    slope_actuators.actuators[:3],
    base_actuators.actuators,
    strict=True,
  ):
    assert isinstance(flat_cfg, BuiltinPositionActuatorCfg)
    assert isinstance(slope_cfg, BuiltinPositionActuatorCfg)
    assert isinstance(base_cfg, BuiltinPositionActuatorCfg)
    assert flat_cfg is not slope_cfg
    assert flat_cfg is not base_cfg
    assert flat_cfg.stiffness == slope_cfg.stiffness == base_cfg.stiffness
    assert flat_cfg.damping == slope_cfg.damping == base_cfg.damping
    assert flat_cfg.effort_limit == pytest.approx(30.0)
    assert slope_cfg.effort_limit == pytest.approx(60.0)

  for joint_name in JOINT_NAMES:
    actuator = tethered_model.actuator(joint_name)
    assert actuator.gainprm[0] == pytest.approx(100.0)
    assert actuator.biasprm[1] == pytest.approx(-100.0)
    assert actuator.biasprm[2] == pytest.approx(-3.0)
    np.testing.assert_allclose(actuator.forcerange, (-30.0, 30.0))

    joint = tethered_model.joint(joint_name)
    assert joint.armature[0] == pytest.approx(0.0486)

  arm_joint_id = tethered_model.joint(ARM_JOINT_NAME).id
  joint_actuator_ids = np.flatnonzero(
    tethered_model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
  )
  assert arm_joint_id not in tethered_model.actuator_trnid[joint_actuator_ids, 0]


def test_slope_leg_actuators_use_60_nm_limit(
  tethered_slope_model: mujoco.MjModel,
) -> None:
  for joint_name in JOINT_NAMES:
    actuator = tethered_slope_model.actuator(joint_name)
    assert actuator.gainprm[0] == pytest.approx(100.0)
    assert actuator.biasprm[1] == pytest.approx(-100.0)
    assert actuator.biasprm[2] == pytest.approx(-3.0)
    np.testing.assert_allclose(actuator.forcerange, (-60.0, 60.0))

    joint = tethered_slope_model.joint(joint_name)
    assert joint.armature[0] == pytest.approx(0.0486)


def test_slope_mixture_asset_has_neutral_root_and_slope_force_limits(
  tethered_slope_mixture_model: mujoco.MjModel,
) -> None:
  np.testing.assert_allclose(
    tethered_slope_mixture_model.qpos0[:7],
    (0.0, 0.0, 0.313, 1.0, 0.0, 0.0, 0.0),
  )
  cable = tethered_slope_mixture_model.actuator(TETHER_TENDON_NAME)
  np.testing.assert_allclose(cable.forcerange, (-400.0, 0.0))
  for joint_name in JOINT_NAMES:
    actuator = tethered_slope_mixture_model.actuator(joint_name)
    np.testing.assert_allclose(actuator.forcerange, (-60.0, 60.0))


def test_tasks_select_their_own_leg_actuator_variant() -> None:
  flat_cfg = hexapod_tethered_flat_env_cfg()
  slope_cfg = hexapod_tethered_slope_env_cfg()
  mixture_cfg = hexapod_tethered_slope_mixture_env_cfg()

  assert (
    flat_cfg.scene.entities["robot"].articulation
    is hexapod_tethered_constants.HEXAPOD_TETHERED_FLAT_ARTICULATION
  )
  for cfg in (slope_cfg, mixture_cfg):
    assert (
      cfg.scene.entities["robot"].articulation
      is hexapod_tethered_constants.HEXAPOD_TETHERED_SLOPE_ARTICULATION
    )


def test_flat_cable_actuator_is_effectively_force_free(
  tethered_model: mujoco.MjModel,
) -> None:
  cable = tethered_model.actuator(TETHER_TENDON_NAME)
  np.testing.assert_allclose(cable.ctrlrange, (TETHER_ACTION_MIN, TETHER_ACTION_MAX))
  np.testing.assert_allclose(cable.forcerange, (-0.0001, 0.0))
  assert cable.gainprm[0] == pytest.approx(TETHER_FORCE_PER_ACTION)

  data = mujoco.MjData(tethered_model)
  mujoco.mj_resetDataKeyframe(tethered_model, data, 0)
  data.ctrl[cable.id] = 1.0
  mujoco.mj_forward(tethered_model, data)
  assert data.actuator_force[cable.id] == pytest.approx(-0.0001)


def test_slope_cable_actuator_maps_control_to_negative_tendon_force(
  tethered_slope_model: mujoco.MjModel,
) -> None:
  cable = tethered_slope_model.actuator(TETHER_TENDON_NAME)
  np.testing.assert_allclose(cable.ctrlrange, (TETHER_ACTION_MIN, TETHER_ACTION_MAX))
  np.testing.assert_allclose(cable.forcerange, (-400.0, 0.0))
  assert cable.gainprm[0] == pytest.approx(TETHER_FORCE_PER_ACTION)

  data = mujoco.MjData(tethered_slope_model)
  mujoco.mj_resetDataKeyframe(tethered_slope_model, data, 0)
  data.ctrl[cable.id] = 1.0
  mujoco.mj_forward(tethered_slope_model, data)
  assert data.actuator_force[cable.id] == pytest.approx(-200.0)

  sensor = tethered_slope_model.sensor(TETHER_FORCE_SENSOR_NAME)
  assert data.sensordata[sensor.adr[0]] == pytest.approx(-200.0)

  data.ctrl[cable.id] = 3.0
  mujoco.mj_forward(tethered_slope_model, data)
  assert data.actuator_force[cable.id] == pytest.approx(-400.0)
  assert data.sensordata[sensor.adr[0]] == pytest.approx(-400.0)


def test_slope_asset_starts_parallel_to_22_degree_plane(
  tethered_slope_model: mujoco.MjModel,
) -> None:
  data = mujoco.MjData(tethered_slope_model)
  mujoco.mj_resetDataKeyframe(tethered_slope_model, data, 0)
  mujoco.mj_forward(tethered_slope_model, data)

  np.testing.assert_allclose(data.qpos[3:7], SLOPE_ROOT_QUAT, atol=1.0e-10)
  foot_pos = np.stack(
    [
      data.site(tethered_slope_model.site(name).id).xpos
      for name in ("RF", "RM", "RB", "LF", "LM", "LB")
    ]
  )
  clearance = foot_pos[:, 2] - foot_pos[:, 0] * np.tan(SLOPE_ANGLE_RAD)
  np.testing.assert_allclose(clearance, 5.46e-4, atol=1.0e-5)


def test_task_keeps_hexapod_rewards_and_appends_tether_terms() -> None:
  base_cfg = hexapod_flat_env_cfg()
  cfg = hexapod_tethered_flat_env_cfg()

  for name, term in base_cfg.rewards.items():
    assert cfg.rewards[name].weight == pytest.approx(term.weight), name

  assert cfg.rewards["action_rate_l2"].func is rewards.action_rate_l2_for_term
  assert cfg.rewards["action_rate_l2"].params == {"action_name": "joint_pos"}
  assert (
    tuple(cfg.rewards["dof_pos_limits"].params["asset_cfg"].joint_names) == JOINT_NAMES
  )
  assert cfg.rewards["tether_flat_penalty"].weight == -3.0
  assert cfg.rewards["tether_uphill_reward"].weight == 1.0
  assert cfg.rewards["tether_pitch_slack_penalty"].weight == -0.02
  assert cfg.rewards["tether_action_rate"].weight == -0.05

  flat_params = cfg.rewards["tether_flat_penalty"].params
  slack_params = cfg.rewards["tether_pitch_slack_penalty"].params
  assert flat_params["tension_max"] == pytest.approx(300.0)
  assert flat_params["slope_start"] == pytest.approx(0.30)
  assert flat_params["slope_full"] == pytest.approx(0.35)
  assert slack_params["slack_ratio"] == pytest.approx(0.15)


def test_slope_task_keeps_flat_interface_and_adds_playground_rewards() -> None:
  flat_cfg = hexapod_tethered_flat_env_cfg()
  slope_cfg = hexapod_tethered_slope_env_cfg()

  slope_owned_terms = {
    "tether_flat_penalty",
    "tether_uphill_reward",
    "tether_pitch_slack_penalty",
    "tether_action_rate",
  }
  for name, term in flat_cfg.rewards.items():
    if name in slope_owned_terms:
      continue
    assert slope_cfg.rewards[name].weight == pytest.approx(term.weight), name

  expected_weights = {
    "tether_work_assist": 0.5,
    "tether_eng_steep": 0.5,
    "tether_slack_penalty": -1.0,
    "tether_overload": -2.0,
  }
  for name, weight in expected_weights.items():
    assert slope_cfg.rewards[name].weight == pytest.approx(weight)
  assert slope_cfg.rewards["tether_flat_penalty"].weight == 0.0
  assert slope_cfg.rewards["tether_uphill_reward"].weight == 0.0
  assert slope_cfg.rewards["tether_pitch_slack_penalty"].weight == 0.0
  assert slope_cfg.rewards["tether_action_rate"].weight == pytest.approx(-0.1)

  assert tuple(slope_cfg.actions) == tuple(flat_cfg.actions)
  assert tuple(slope_cfg.observations["actor"].terms) == tuple(
    flat_cfg.observations["actor"].terms
  )
  assert tuple(slope_cfg.observations["critic"].terms) == tuple(
    flat_cfg.observations["critic"].terms
  )
  assert slope_cfg.rewards["upright"].params["terrain_sensor_names"] == (
    "foot_height_scan",
  )
  assert flat_cfg.rewards["foot_clearance"].func is velocity_mdp.feet_clearance
  assert flat_cfg.rewards["foot_swing_height"].func is velocity_mdp.feet_swing_height
  assert flat_cfg.rewards["foot_slip"].func is velocity_mdp.feet_slip
  assert slope_cfg.rewards["foot_clearance"].func is rewards.slope_foot_clearance
  assert slope_cfg.rewards["foot_swing_height"].func is rewards.SlopeFootSwingHeight
  assert slope_cfg.rewards["foot_slip"].func is rewards.slope_foot_slip
  assert (
    slope_cfg.rewards["foot_clearance"].params["height_sensor_name"]
    == "foot_height_scan"
  )
  assert (
    slope_cfg.rewards["foot_swing_height"].params["height_sensor_name"]
    == "foot_height_scan"
  )
  assert (
    slope_cfg.rewards["foot_slip"].params["height_sensor_name"] == "foot_height_scan"
  )

  reset_base = slope_cfg.events["reset_base"]
  assert reset_base.func is events.reset_root_state_on_slope
  assert reset_base.params["slope_angle_rad"] == pytest.approx(SLOPE_ANGLE_RAD)
  assert reset_base.params["pose_range"]["yaw"] == (-0.15, 0.15)
  anchor = slope_cfg.events["tether_anchor"]
  assert anchor.func is events.randomize_anchor_above_slope
  assert anchor.params["x_range"] == (3.0, 13.0)
  assert anchor.params["y_range"] == (-2.0, 2.0)
  assert anchor.params["normal_height_range"] == (0.0, 0.3)
  assert "trunk_front_past_anchor" not in flat_cfg.terminations
  anchor_termination = slope_cfg.terminations["trunk_front_past_anchor"]
  assert anchor_termination.func is terminations.trunk_front_past_anchor
  assert anchor_termination.time_out is False
  assert anchor_termination.params["forward_offset"] == pytest.approx(0.3)
  termination_asset_cfg = anchor_termination.params["asset_cfg"]
  assert tuple(termination_asset_cfg.body_names) == ("trunk",)
  assert tuple(termination_asset_cfg.site_names) == (TETHER_ANCHOR_SITE_NAME,)
  fell_over = slope_cfg.terminations["fell_over"]
  assert fell_over.func is terminations.bad_orientation_relative_to_slope
  assert fell_over.params["slope_angle_rad"] == pytest.approx(SLOPE_ANGLE_RAD)

  stages = slope_cfg.curriculum["command_vel"].params["velocity_stages"]
  assert stages == [
    {
      "step": 0,
      "lin_vel_x": (-0.3, 0.3),
      "lin_vel_y": (-0.2, 0.2),
      "ang_vel_z": (-0.25, 0.25),
    },
    {
      "step": 3000 * 24,
      "lin_vel_x": (-0.5, 0.5),
      "lin_vel_y": (-0.3, 0.3),
      "ang_vel_z": (-0.4, 0.4),
    },
    {
      "step": 7000 * 24,
      "lin_vel_x": (-0.8, 0.8),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.8, 0.8),
    },
  ]
  assert stages == flat_cfg.curriculum["command_vel"].params["velocity_stages"]
  slope_twist = slope_cfg.commands["twist"]
  flat_twist = flat_cfg.commands["twist"]
  assert isinstance(slope_twist, UniformVelocityCommandCfg)
  assert isinstance(flat_twist, UniformVelocityCommandCfg)
  assert slope_twist.ranges.lin_vel_x == flat_twist.ranges.lin_vel_x
  assert slope_twist.ranges.lin_vel_y == flat_twist.ranges.lin_vel_y
  assert slope_twist.ranges.ang_vel_z == flat_twist.ranges.ang_vel_z
  push = slope_cfg.events["push_robot"]
  assert push.func is events.push_by_setting_velocity_after_step
  assert push.params["start_step"] == 1000 * 24


def test_slope_mixture_inherits_slope_task_and_adds_nested_curricula() -> None:
  slope_cfg = hexapod_tethered_slope_env_cfg()
  mixture_cfg = hexapod_tethered_slope_mixture_env_cfg()

  assert mixture_cfg.scene.spec_fn is tethered_env_cfgs._make_terrain_mocap
  mixture_robot = mixture_cfg.scene.entities["robot"]
  slope_robot = slope_cfg.scene.entities["robot"]
  assert mixture_robot.spec_fn is hexapod_tethered_constants.get_slope_mixture_spec
  assert mixture_robot.spec_fn is not slope_robot.spec_fn
  assert mixture_robot.init_state.rot == (1.0, 0.0, 0.0, 0.0)
  assert slope_robot.init_state.rot == SLOPE_ROOT_QUAT
  assert tuple(mixture_cfg.actions) == tuple(slope_cfg.actions)
  assert tuple(mixture_cfg.observations["actor"].terms) == tuple(
    slope_cfg.observations["actor"].terms
  )
  assert tuple(mixture_cfg.observations["critic"].terms) == tuple(
    slope_cfg.observations["critic"].terms
  )
  assert tuple(mixture_cfg.rewards) == tuple(slope_cfg.rewards)
  for name, term in slope_cfg.rewards.items():
    assert mixture_cfg.rewards[name].weight == pytest.approx(term.weight), name

  assert next(iter(mixture_cfg.events)) == SLOPE_MIXTURE_EVENT_NAME
  slope_event = mixture_cfg.events[SLOPE_MIXTURE_EVENT_NAME]
  assert slope_event.func is events.RandomizeSlopeAngle
  assert slope_event.params["asset_cfg"].name == "terrain"
  assert slope_event.params["asset_cfg"].geom_names == ("terrain",)
  assert slope_event.params["slope_stages"] == [
    {"step": 0, "angle_range_deg": (20.0, 30.0)},
    {"step": 6000 * 24, "angle_range_deg": (20.0, 40.0)},
    {"step": 12000 * 24, "angle_range_deg": (20.0, 50.0)},
    {"step": 18000 * 24, "angle_range_deg": (20.0, 60.0)},
  ]

  for event_name in ("reset_base", "tether_anchor"):
    assert (
      mixture_cfg.events[event_name].params["slope_event_name"]
      == SLOPE_MIXTURE_EVENT_NAME
    )
  mixture_reset = mixture_cfg.events["reset_base"]
  assert mixture_reset.func is events.reset_root_state_on_slope_mixture
  assert "slope_angle_rad" not in mixture_reset.params
  assert (
    mixture_cfg.terminations["fell_over"].params["slope_event_name"]
    == SLOPE_MIXTURE_EVENT_NAME
  )
  for reward_name in ("tether_eng_steep", "tether_slack_penalty"):
    assert (
      mixture_cfg.rewards[reward_name].params["slope_event_name"]
      == SLOPE_MIXTURE_EVENT_NAME
    )

  stages = mixture_cfg.curriculum["command_vel"].params["velocity_stages"]
  assert [stage["step"] for stage in stages] == [
    iteration * 24
    for iteration in (
      0,
      2000,
      4000,
      6000,
      8000,
      10000,
      12000,
      14000,
      16000,
      18000,
      20000,
      22000,
    )
  ]
  expected_velocity_ranges = [
    ((-0.3, 0.3), (-0.2, 0.2), (-0.25, 0.25)),
    ((-0.5, 0.5), (-0.3, 0.3), (-0.4, 0.4)),
    ((-0.8, 0.8), (-0.5, 0.5), (-0.8, 0.8)),
  ] * 4
  assert [
    (stage["lin_vel_x"], stage["lin_vel_y"], stage["ang_vel_z"]) for stage in stages
  ] == expected_velocity_ranges
  assert stages[-1] == {
    "step": 22000 * 24,
    "lin_vel_x": (-0.8, 0.8),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_z": (-0.8, 0.8),
  }

  play_cfg = hexapod_tethered_slope_mixture_env_cfg(play=True)
  assert play_cfg.curriculum == {}
  assert play_cfg.events[SLOPE_MIXTURE_EVENT_NAME].params["slope_stages"] == [
    {"step": 0, "angle_range_deg": (20.0, 60.0)}
  ]


def test_slope_orientation_termination_uses_surface_normal() -> None:
  slope_quat = torch.tensor(SLOPE_ROOT_QUAT).repeat(3, 1)
  delta = torch.tensor([0.0, math.radians(59.9), math.radians(60.1)])
  zero = torch.zeros_like(delta)
  root_quat = quat_mul(slope_quat, quat_from_euler_xyz(delta, zero, zero))
  asset = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=root_quat))
  env = cast(
    Any,
    SimpleNamespace(
      scene={"robot": asset},
      num_envs=3,
      device=torch.device("cpu"),
    ),
  )
  asset_cfg = cast(Any, SimpleNamespace(name="robot"))

  terminated = terminations.bad_orientation_relative_to_slope(
    env,
    limit_angle=math.radians(60.0),
    slope_angle_rad=SLOPE_ANGLE_RAD,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(terminated, torch.tensor([False, False, True]))


def test_trunk_front_past_anchor_uses_strict_world_x_threshold() -> None:
  asset = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor(
        [
          [[0.70, 0.0, 0.0]],
          [[0.71, 0.0, 0.0]],
          [[4.70, 0.0, 0.0]],
        ]
      ),
      site_pos_w=torch.tensor(
        [
          [[1.0, 0.0, 0.0]],
          [[1.0, 0.0, 0.0]],
          [[5.0, 0.0, 0.0]],
        ]
      ),
    )
  )
  env = cast(Any, SimpleNamespace(scene={"robot": asset}))
  asset_cfg = cast(
    Any,
    SimpleNamespace(name="robot", body_ids=[0], site_ids=[0]),
  )

  terminated = terminations.trunk_front_past_anchor(
    env,
    forward_offset=0.3,
    asset_cfg=asset_cfg,
  )

  # Equality is not enough: the requested condition is strictly "exceeds".
  torch.testing.assert_close(
    terminated,
    torch.tensor([False, True, False]),
  )


def test_anchor_normal_height_range_rejects_below_slope_values() -> None:
  with pytest.raises(ValueError, match="must not extend below the slope"):
    events.randomize_anchor_above_slope(
      cast(Any, None),
      env_ids=None,
      asset_cfg=cast(Any, None),
      slope_angle_rad=SLOPE_ANGLE_RAD,
      x_range=(3.0, 13.0),
      y_range=(-2.0, 2.0),
      normal_height_range=(-0.01, 0.3),
    )


def test_slope_foot_projection_uses_normal_height_and_tangent_velocity() -> None:
  slope_angle = torch.tensor(SLOPE_ANGLE_RAD)
  normal = torch.stack(
    (-torch.sin(slope_angle), torch.tensor(0.0), torch.cos(slope_angle))
  )
  uphill = torch.stack(
    (torch.cos(slope_angle), torch.tensor(0.0), torch.sin(slope_angle))
  )
  normal_w = normal.reshape(1, 1, 3).expand(2, 1, 3)
  hit_pos_w = torch.zeros((2, 1, 3))
  frame_pos_w = hit_pos_w + 0.1 * normal_w
  foot_vel_w = torch.stack((uphill, normal)).reshape(2, 1, 3)

  normal_height, tangent_velocity = rewards._project_foot_state_to_surface(
    frame_pos_w,
    hit_pos_w,
    normal_w,
    torch.ones((2, 1), dtype=torch.bool),
    foot_vel_w,
  )

  torch.testing.assert_close(normal_height, torch.full((2, 1), 0.1))
  torch.testing.assert_close(tangent_velocity[0, 0], uphill)
  torch.testing.assert_close(
    tangent_velocity[1, 0],
    torch.zeros(3),
    atol=1.0e-6,
    rtol=0.0,
  )


def test_slope_tether_rewards_use_direction_gate_and_playground_kernels(
  monkeypatch,
) -> None:
  tension = torch.tensor([[100.0], [42.5]])
  monkeypatch.setattr(rewards, "cable_tension", lambda _env, _name: tension)

  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor(
        [
          [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
          [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
      ),
      root_link_lin_vel_w=torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
      body_link_quat_w=torch.tensor(
        [
          [[1.0, 0.0, 0.0, 0.0]],
          [[1.0, 0.0, 0.0, 0.0]],
        ]
      ),
    )
  )
  command = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
  env = cast(
    Any,
    SimpleNamespace(
      scene={"robot": asset},
      command_manager=SimpleNamespace(get_command=lambda _name: command),
      extras={"log": {}},
    ),
  )
  asset_cfg = cast(
    Any,
    SimpleNamespace(name="robot", site_ids=[0, 1], body_ids=[0]),
  )

  work = rewards.tether_work_assist(env, "sensor", "twist", asset_cfg, 0.05)
  torch.testing.assert_close(work, torch.tensor([0.5, 0.0]), atol=1.0e-6, rtol=0.0)

  engagement = rewards.tether_eng_steep(
    env,
    "sensor",
    22.0,
    0.17,
    "twist",
    0.05,
    asset_cfg,
  )
  torch.testing.assert_close(
    engagement,
    torch.tensor([1.0, 0.0]),
    atol=1.0e-6,
    rtol=0.0,
  )

  slack_tension = torch.tensor([[14.0], [15.0]])
  monkeypatch.setattr(rewards, "cable_tension", lambda _env, _name: slack_tension)
  slack = rewards.tether_slack_penalty(
    env,
    "sensor",
    22.0,
    0.17,
    15.0,
    "twist",
    0.05,
    asset_cfg,
  )
  torch.testing.assert_close(slack, torch.tensor([1.0, 0.0]))

  overload_tension = torch.tensor([[300.0], [250.0]])
  monkeypatch.setattr(rewards, "cable_tension", lambda _env, _name: overload_tension)
  overload = rewards.tether_overload(env, "sensor", 250.0)
  torch.testing.assert_close(overload, torch.tensor([0.5, 0.0]))


def test_tether_reward_slope_gate_matches_playground() -> None:
  angles = torch.tensor([0.0, 0.25, 0.325, 0.4], dtype=torch.float64)
  gravity = torch.stack(
    (torch.sin(angles), torch.zeros_like(angles), -torch.cos(angles)),
    dim=1,
  )

  gate = rewards._slope_gate(gravity, slope_start=0.30, slope_full=0.35)
  torch.testing.assert_close(
    gate,
    torch.tensor([0.0, 0.0, 0.5, 1.0], dtype=torch.float64),
    atol=1.0e-12,
    rtol=0.0,
  )


def test_tether_direction_reconstructs_yaw_and_pitch_with_xml_axis_sign() -> None:
  yaw = torch.tensor([0.0, torch.pi / 2, -torch.pi / 2])
  pitch = torch.tensor([0.0, 0.0, torch.pi / 2])

  direction = observations.tether_direction_from_yaw_pitch_angles(yaw, pitch)
  torch.testing.assert_close(
    direction,
    torch.tensor(
      [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
      ]
    ),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    torch.linalg.vector_norm(direction, dim=1),
    torch.ones(3),
  )

  noisy_direction = observations.TetherDirectionNoiseCfg().apply(direction)
  torch.testing.assert_close(
    torch.linalg.vector_norm(noisy_direction, dim=1),
    torch.ones(3),
    atol=1.0e-6,
    rtol=0.0,
  )


def test_task_declares_sensor_policy_interface() -> None:
  cfg = hexapod_tethered_flat_env_cfg()
  base_cfg = hexapod_flat_env_cfg()

  assert tuple(cfg.actions) == ("joint_pos", "cable")
  joint_action = cfg.actions["joint_pos"]
  assert isinstance(joint_action, JointPositionActionCfg)
  assert joint_action.scale == (
    hexapod_tethered_constants.HEXAPOD_TETHERED_ACTION_SCALE
  )
  cable_action = cfg.actions["cable"]
  assert isinstance(cable_action, TendonEffortActionCfg)
  assert cable_action.scale == 1.0
  assert cable_action.clip == {"cable": (0.0, 2.0)}
  assert "encoder_bias" in cfg.events

  actor_terms = cfg.observations["actor"].terms
  assert tuple(actor_terms) == (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "cable_tension",
    "tether_direction_body",
  )
  assert actor_terms["actions"].func is mdp.last_action
  assert actor_terms["joint_pos"].params["biased"] is True
  gravity_noise = actor_terms["projected_gravity"].noise
  assert isinstance(gravity_noise, UniformNoiseCfg)
  gravity_amplitude = (
    OBSERVATION_NOISE_LEVEL * OBSERVATION_NOISE_SCALES["projected_gravity"]
  )
  assert gravity_noise.n_min == pytest.approx(-gravity_amplitude)
  assert gravity_noise.n_max == pytest.approx(gravity_amplitude)

  for name, scale in OBSERVATION_NOISE_SCALES.items():
    noise = actor_terms[name].noise
    assert isinstance(noise, UniformNoiseCfg)
    assert noise.n_min == pytest.approx(-OBSERVATION_NOISE_LEVEL * scale)
    assert noise.n_max == pytest.approx(OBSERVATION_NOISE_LEVEL * scale)

  direction_noise = actor_terms["tether_direction_body"].noise
  assert isinstance(direction_noise, observations.TetherDirectionNoiseCfg)
  yaw_amplitude = OBSERVATION_NOISE_LEVEL * TETHER_DIRECTION_NOISE_SCALES["yaw"]
  pitch_amplitude = OBSERVATION_NOISE_LEVEL * TETHER_DIRECTION_NOISE_SCALES["pitch"]
  assert direction_noise.yaw_min == pytest.approx(-yaw_amplitude)
  assert direction_noise.yaw_max == pytest.approx(yaw_amplitude)
  assert direction_noise.pitch_min == pytest.approx(-pitch_amplitude)
  assert direction_noise.pitch_max == pytest.approx(pitch_amplitude)

  assert cfg.observations["actor"].enable_corruption is True
  assert cfg.observations["critic"].enable_corruption is False

  assert tuple(cfg.observations["critic"].terms) == (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "cable_tension",
    "tether_direction_body",
    "foot_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
    "arm_joint_pos",
    "arm_joint_vel",
    "tendon_length",
  )

  anchor_event = cfg.events["tether_anchor"]
  assert isinstance(anchor_event, EventTermCfg)
  assert anchor_event.mode == "reset"
  assert anchor_event.func is mdp.dr.site_pos
  assert anchor_event.params["operation"] == "abs"
  assert anchor_event.params["ranges"] == {
    0: (-10.0, 10.0),
    1: (-5.0, 5.0),
    2: (0.1, 0.5),
  }
  assert anchor_event.params["asset_cfg"].site_names == (TETHER_ANCHOR_SITE_NAME,)
  assert cfg.sim.mujoco.timestep == pytest.approx(base_cfg.sim.mujoco.timestep)
  assert cfg.sim.mujoco.timestep == pytest.approx(0.005)
  assert cfg.decimation == base_cfg.decimation == 4

  illegal_sensor = next(
    sensor
    for sensor in cfg.scene.sensors or ()
    if sensor.name == "illegal_ground_contact"
  )
  assert isinstance(illegal_sensor, ContactSensorCfg)
  assert "ARM_collision" in illegal_sensor.primary.pattern


def test_zero_tethered_observation_noise_level_disables_noise(monkeypatch) -> None:
  monkeypatch.setattr(tethered_env_cfgs, "OBSERVATION_NOISE_LEVEL", 0.0)
  cfg = tethered_env_cfgs.hexapod_tethered_flat_env_cfg()
  actor_terms = cfg.observations["actor"].terms

  for name in OBSERVATION_NOISE_SCALES:
    assert actor_terms[name].noise is None
  assert actor_terms["tether_direction_body"].noise is None


def test_tethered_play_mode_disables_observation_corruption() -> None:
  cfg = hexapod_tethered_flat_env_cfg(play=True)

  assert cfg.observations["actor"].enable_corruption is False


def test_runtime_policy_dimensions_and_tension_feedback(
  tethered_env: ManagerBasedRlEnv,
) -> None:
  assert tethered_env.action_manager.action_term_dim == [18, 1]
  joint_action = tethered_env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  assert tuple(joint_action.target_names) == JOINT_NAMES
  assert tethered_env.observation_manager.group_obs_dim == {
    "actor": (71,),
    "critic": (110,),
  }

  observations_by_group, _ = tethered_env.reset()
  actor_observations = observations_by_group["actor"]
  critic_observations = observations_by_group["critic"]
  assert isinstance(actor_observations, torch.Tensor)
  assert isinstance(critic_observations, torch.Tensor)
  assert actor_observations.shape == (2, 71)
  assert critic_observations.shape == (2, 110)
  torch.testing.assert_close(
    torch.linalg.vector_norm(actor_observations[:, 68:71], dim=1),
    torch.ones(2),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    torch.linalg.vector_norm(critic_observations[:, 68:71], dim=1),
    torch.ones(2),
    atol=1.0e-6,
    rtol=0.0,
  )

  action = torch.zeros((2, 19))
  action[:, 18] = 1.0
  next_observations, *_ = tethered_env.step(action)
  next_actor = next_observations["actor"]
  assert isinstance(next_actor, torch.Tensor)
  torch.testing.assert_close(next_actor[:, 45:64], action)
  tension = observations.cable_tension(tethered_env, "robot/cable_tendonactuatorfrc")
  torch.testing.assert_close(tension, torch.full_like(tension, 0.0001))

  following_observations, *_ = tethered_env.step(torch.zeros((2, 19)))
  following_actor = following_observations["actor"]
  assert isinstance(following_actor, torch.Tensor)
  torch.testing.assert_close(following_actor[:, 45:64], torch.zeros((2, 19)))


def test_runtime_slope_keeps_policy_interface_and_produces_tension(
  tethered_slope_env: ManagerBasedRlEnv,
) -> None:
  assert tethered_slope_env.action_manager.action_term_dim == [18, 1]
  assert tethered_slope_env.observation_manager.group_obs_dim == {
    "actor": (71,),
    "critic": (110,),
  }

  observations_by_group, _ = tethered_slope_env.reset()
  actor_observations = observations_by_group["actor"]
  critic_observations = observations_by_group["critic"]
  assert isinstance(actor_observations, torch.Tensor)
  assert isinstance(critic_observations, torch.Tensor)
  assert actor_observations.shape == (2, 71)
  assert critic_observations.shape == (2, 110)

  robot: Entity = tethered_slope_env.scene["robot"]
  anchor_id = robot.site_names.index(TETHER_ANCHOR_SITE_NAME)
  anchor_pos_w = robot.data.site_pos_w[:, anchor_id]
  slope_angle = torch.tensor(SLOPE_ANGLE_RAD, device=tethered_slope_env.device)
  anchor_normal_height = (
    -torch.sin(slope_angle) * anchor_pos_w[:, 0]
    + torch.cos(slope_angle) * anchor_pos_w[:, 2]
  )
  assert torch.all(anchor_normal_height >= -1.0e-5)
  assert torch.all(anchor_normal_height <= 0.3 + 1.0e-5)
  assert "trunk_front_past_anchor" in (
    tethered_slope_env.termination_manager.active_terms
  )

  action = torch.zeros((2, 19), device=tethered_slope_env.device)
  action[:, 18] = 1.0
  tethered_slope_env.step(action)
  tension = observations.cable_tension(
    tethered_slope_env, "robot/cable_tendonactuatorfrc"
  )
  torch.testing.assert_close(tension, torch.full_like(tension, 200.0))


def test_runtime_slope_mixture_uses_per_environment_surface_geometry(
  tethered_slope_mixture_env: ManagerBasedRlEnv,
) -> None:
  env = tethered_slope_mixture_env
  observations_by_group, _ = env.reset(seed=123)
  actor_observations = observations_by_group["actor"]
  critic_observations = observations_by_group["critic"]
  assert isinstance(actor_observations, torch.Tensor)
  assert isinstance(critic_observations, torch.Tensor)
  assert actor_observations.shape == (4, 71)
  assert critic_observations.shape == (4, 110)
  assert env.action_manager.action_term_dim == [18, 1]

  slope_term = env.event_manager.get_term_cfg(SLOPE_MIXTURE_EVENT_NAME).func
  assert isinstance(slope_term, events.RandomizeSlopeAngle)
  angles = slope_term.slope_angle_rad
  assert torch.all(angles >= math.radians(20.0))
  assert torch.all(angles <= math.radians(30.0))
  assert torch.unique(angles).numel() > 1

  zeros = torch.zeros_like(angles)
  expected_terrain_quat = quat_from_euler_xyz(zeros, -angles, zeros)
  expected_terrain_xmat = matrix_from_quat(expected_terrain_quat)
  terrain: Entity = env.scene["terrain"]
  terrain_geom_id = int(terrain.indexing.geom_ids[0].item())
  terrain_body_id = env.sim.mj_model.body("terrain").id
  terrain_mocap_id = int(env.sim.mj_model.body_mocapid[terrain_body_id])
  assert terrain_mocap_id >= 0
  assert "geom_quat" not in env.sim.expanded_fields
  torch.testing.assert_close(
    env.sim.data.mocap_pos[:, terrain_mocap_id],
    env.scene.env_origins,
  )
  torch.testing.assert_close(
    env.sim.data.mocap_quat[:, terrain_mocap_id],
    expected_terrain_quat,
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    env.sim.data.geom_xpos[:, terrain_geom_id],
    env.scene.env_origins,
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    env.sim.data.geom_xmat[:, terrain_geom_id],
    expected_terrain_xmat,
    atol=1.0e-6,
    rtol=0.0,
  )

  # The mocap terrain's data transform must drive raycasts and survive the same
  # regular forward-kinematics pass used by simulation steps.
  env.sim.forward()
  torch.testing.assert_close(
    env.sim.data.geom_xpos[:, terrain_geom_id],
    env.scene.env_origins,
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    env.sim.data.geom_xmat[:, terrain_geom_id],
    expected_terrain_xmat,
    atol=1.0e-6,
    rtol=0.0,
  )

  slope_normal = torch.stack(
    (-torch.sin(angles), zeros, torch.cos(angles)),
    dim=1,
  )
  slope_tangent = torch.stack(
    (torch.cos(angles), zeros, torch.sin(angles)),
    dim=1,
  )
  torch.testing.assert_close(
    env.sim.data.geom_xmat[:, terrain_geom_id, :, 2],
    slope_normal,
    atol=1.0e-6,
    rtol=0.0,
  )

  foot_height_scan = env.scene["foot_height_scan"]
  assert isinstance(foot_height_scan, TerrainHeightSensor)
  scan_data = foot_height_scan.data
  valid_hits = scan_data.distances >= 0.0
  assert torch.all(valid_hits.any(dim=1))
  expected_ray_normals = slope_normal[:, None, :].expand_as(scan_data.normals_w)
  torch.testing.assert_close(
    scan_data.normals_w[valid_hits],
    expected_ray_normals[valid_hits],
    atol=1.0e-5,
    rtol=0.0,
  )
  ray_origins = env.scene.env_origins[:, None, :]
  signed_hit_heights = torch.sum(
    (scan_data.hit_pos_w - ray_origins) * expected_ray_normals,
    dim=-1,
  )
  torch.testing.assert_close(
    signed_hit_heights[valid_hits],
    torch.zeros_like(signed_hit_heights[valid_hits]),
    atol=1.0e-5,
    rtol=0.0,
  )

  robot: Entity = env.scene["robot"]
  body_up = torch.zeros_like(slope_normal)
  body_up[:, 2] = 1.0
  torch.testing.assert_close(
    quat_apply(robot.data.root_link_quat_w, body_up),
    slope_normal,
    atol=1.0e-5,
    rtol=0.0,
  )

  root_relative = robot.data.root_link_pos_w - env.scene.env_origins
  root_slope_x = torch.sum(root_relative * slope_tangent, dim=1)
  root_slope_y = root_relative[:, 1]
  root_normal_height = torch.sum(root_relative * slope_normal, dim=1)
  assert torch.all(root_slope_x >= -0.5 - 1.0e-5)
  assert torch.all(root_slope_x <= 0.5 + 1.0e-5)
  assert torch.all(root_slope_y >= -0.5 - 1.0e-5)
  assert torch.all(root_slope_y <= 0.5 + 1.0e-5)
  assert torch.all(root_normal_height >= 0.313 - 1.0e-5)
  assert torch.all(root_normal_height <= 0.323 + 1.0e-5)

  anchor_id = robot.site_names.index(TETHER_ANCHOR_SITE_NAME)
  anchor_relative = robot.data.site_pos_w[:, anchor_id] - env.scene.env_origins
  anchor_slope_x = torch.sum(anchor_relative * slope_tangent, dim=1)
  anchor_slope_y = anchor_relative[:, 1]
  anchor_normal_height = torch.sum(anchor_relative * slope_normal, dim=1)
  assert torch.all(anchor_slope_x >= 3.0 - 1.0e-5)
  assert torch.all(anchor_slope_x <= 13.0 + 1.0e-5)
  assert torch.all(anchor_slope_y >= -2.0 - 1.0e-5)
  assert torch.all(anchor_slope_y <= 2.0 + 1.0e-5)
  assert torch.all(anchor_normal_height >= -1.0e-5)
  assert torch.all(anchor_normal_height <= 0.3 + 1.0e-5)

  foot_geom_ids = torch.tensor(
    [
      robot.geom_names.index(name)
      for name in hexapod_tethered_constants.FOOT_GEOM_NAMES
    ],
    device=env.device,
  )
  foot_centers = robot.data.geom_pose_w[:, foot_geom_ids, :3]
  foot_global_ids = robot.indexing.geom_ids[foot_geom_ids].cpu().numpy()
  foot_radii = torch.as_tensor(
    env.sim.mj_model.geom_size[foot_global_ids, 0],
    device=env.device,
    dtype=foot_centers.dtype,
  )
  foot_bottom_height = (
    torch.sum(
      (foot_centers - env.scene.env_origins[:, None]) * slope_normal[:, None],
      dim=-1,
    )
    - foot_radii
  )
  assert torch.all(foot_bottom_height >= -1.0e-4)
  assert torch.all(foot_bottom_height <= 0.0101)

  # Native/offscreen viewers copy mocap state into a host MjData and run
  # mj_forward. Verify that path renders the same per-environment plane pose.
  host_data = mujoco.MjData(env.sim.mj_model)
  env_index = 0
  host_data.qpos[:] = env.sim.data.qpos[env_index].cpu().numpy()
  host_data.qvel[:] = env.sim.data.qvel[env_index].cpu().numpy()
  host_data.mocap_pos[:] = env.sim.data.mocap_pos[env_index].cpu().numpy()
  host_data.mocap_quat[:] = env.sim.data.mocap_quat[env_index].cpu().numpy()
  mujoco.mj_forward(env.sim.mj_model, host_data)
  np.testing.assert_allclose(
    host_data.geom_xpos[terrain_geom_id],
    env.scene.env_origins[env_index].cpu().numpy(),
    atol=1.0e-6,
  )
  np.testing.assert_allclose(
    host_data.geom_xmat[terrain_geom_id].reshape(3, 3),
    expected_terrain_xmat[env_index].cpu().numpy(),
    atol=1.0e-6,
  )

  # The randomized normal offset may start the feet up to 10 mm above the
  # surface. Over a short settling window every world must still collide with
  # its posed plane, rather than falling through it.
  contact_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  feet_ground_contact = env.scene["feet_ground_contact"]
  assert isinstance(feet_ground_contact, ContactSensor)
  for _ in range(10):
    env.step(torch.zeros((env.num_envs, 19), device=env.device))
    assert feet_ground_contact.data.found is not None
    contact_seen |= (feet_ground_contact.data.found > 0).any(dim=1)
  assert torch.all(contact_seen)


def test_tethered_pretrained_load_keeps_legs_and_resets_cable(
  tethered_slope_env: ManagerBasedRlEnv,
  tmp_path,
) -> None:
  wrapped_env = RslRlVecEnvWrapper(tethered_slope_env)
  source_cfg = hexapod_tethered_ppo_runner_cfg()
  source_cfg.upload_model = False
  source = TetheredVelocityOnPolicyRunner(
    wrapped_env,
    asdict(source_cfg),
    log_dir=str(tmp_path / "source"),
    device="cpu",
  )
  source_actor = source.alg.get_policy()
  source_actor_state = source_actor.state_dict()
  source_actor_state["mlp.0.weight"][:, 63].fill_(0.3)
  source_actor_state["mlp.0.weight"][:, 67].fill_(0.4)
  source_actor_state["mlp.6.weight"][18].fill_(0.7)
  source_actor_state["mlp.6.bias"][18] = 0.8
  source_actor_state["distribution.std_param"].fill_(0.9)
  source_actor_state["obs_normalizer._mean"][:, 63] = 3.0
  source_actor_state["obs_normalizer._mean"][:, 67] = 5.0
  source_actor_state["obs_normalizer._var"][:, 63] = 0.01
  source_actor_state["obs_normalizer._var"][:, 67] = 0.01
  source_actor_state["obs_normalizer._std"][:, 63] = 0.1
  source_actor_state["obs_normalizer._std"][:, 67] = 0.1
  source_actor_state["obs_normalizer.count"].fill_(1234)
  source_actor.load_state_dict(source_actor_state)

  checkpoint = source.alg.save()
  checkpoint["iter"] = 4321
  checkpoint["infos"] = {"env_state": {"common_step_counter": 99999}}
  checkpoint_path = tmp_path / "flat_tethered.pt"
  torch.save(checkpoint, checkpoint_path)

  target_cfg = hexapod_tethered_slope_ppo_runner_cfg()
  target_cfg.upload_model = False
  target = TetheredVelocityOnPolicyRunner(
    wrapped_env,
    asdict(target_cfg),
    log_dir=str(tmp_path / "target"),
    device="cpu",
  )
  critic_mlp_before = {
    key: value.clone() for key, value in target.alg.critic.mlp.state_dict().items()
  }
  target.current_learning_iteration = 17
  wrapped_env.unwrapped.common_step_counter = 42

  target.load_pretrained(str(checkpoint_path), map_location="cpu")

  actor_state = target.alg.get_policy().state_dict()
  source_actor_state = source_actor.state_dict()
  torch.testing.assert_close(
    actor_state["mlp.6.weight"][:18],
    source_actor_state["mlp.6.weight"][:18],
  )
  torch.testing.assert_close(actor_state["mlp.6.weight"][18], torch.zeros(128))
  assert actor_state["mlp.6.bias"][18].item() == pytest.approx(
    PRETRAINED_CABLE_ACTION_MEAN
  )
  torch.testing.assert_close(
    actor_state["distribution.std_param"][:18],
    source_actor_state["distribution.std_param"][:18],
  )
  assert actor_state["distribution.std_param"][18].item() == pytest.approx(
    PRETRAINED_CABLE_ACTION_STD
  )
  torch.testing.assert_close(actor_state["mlp.0.weight"][:, 63], torch.zeros(512))
  torch.testing.assert_close(actor_state["mlp.0.weight"][:, 67], torch.zeros(512))
  assert actor_state["obs_normalizer._mean"][0, 63].item() == 0.0
  assert actor_state["obs_normalizer._std"][0, 63].item() == 1.0
  assert actor_state["obs_normalizer._mean"][0, 67].item() == 0.0
  assert actor_state["obs_normalizer._std"][0, 67].item() == pytest.approx(
    PRETRAINED_TENSION_NORMALIZER_STD
  )
  assert actor_state["obs_normalizer.count"].item() == 1234

  for key, value in critic_mlp_before.items():
    torch.testing.assert_close(target.alg.critic.mlp.state_dict()[key], value)
  critic_normalizer_state = target.alg.critic.obs_normalizer.state_dict()
  assert critic_normalizer_state["_std"][0, 67].item() == pytest.approx(
    PRETRAINED_TENSION_NORMALIZER_STD
  )
  assert target.alg.optimizer.state == {}
  assert target.current_learning_iteration == 0
  assert wrapped_env.unwrapped.common_step_counter == 0


def test_tethered_runner_has_independent_ppo_configuration() -> None:
  cfg = hexapod_tethered_ppo_runner_cfg()

  assert cfg.experiment_name == "hexapod_tethered_velocity"
  assert cfg.save_interval == 500
  assert cfg.num_steps_per_env == 24
  assert cfg.max_iterations == 10_000
  assert cfg.clip_actions is None
  assert cfg.actor.hidden_dims == (512, 256, 128)
  assert cfg.actor.activation == "elu"
  assert cfg.actor.obs_normalization is True
  assert cfg.actor.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }
  assert cfg.critic.hidden_dims == (512, 256, 128)
  assert cfg.critic.activation == "elu"
  assert cfg.critic.obs_normalization is True
  assert cfg.algorithm.value_loss_coef == pytest.approx(1.0)
  assert cfg.algorithm.use_clipped_value_loss is True
  assert cfg.algorithm.clip_param == pytest.approx(0.2)
  assert cfg.algorithm.entropy_coef == pytest.approx(0.01)
  assert cfg.algorithm.num_learning_epochs == 5
  assert cfg.algorithm.num_mini_batches == 4
  assert cfg.algorithm.learning_rate == pytest.approx(1.0e-3)
  assert cfg.algorithm.schedule == "adaptive"
  assert cfg.algorithm.gamma == pytest.approx(0.99)
  assert cfg.algorithm.lam == pytest.approx(0.95)
  assert cfg.algorithm.desired_kl == pytest.approx(0.01)
  assert cfg.algorithm.max_grad_norm == pytest.approx(1.0)


def test_slope_runner_owns_independent_ppo_configuration() -> None:
  flat_cfg = hexapod_tethered_ppo_runner_cfg()
  slope_cfg = hexapod_tethered_slope_ppo_runner_cfg()

  assert isinstance(slope_cfg, HexapodTetheredSlopePpoRunnerCfg)
  assert type(slope_cfg).__bases__ == (type(flat_cfg),)
  assert slope_cfg.experiment_name == "hexapod_tethered_slope_velocity"
  assert slope_cfg.save_interval == flat_cfg.save_interval == 500
  assert slope_cfg.num_steps_per_env == flat_cfg.num_steps_per_env == 24
  assert slope_cfg.max_iterations == flat_cfg.max_iterations == 10_000
  assert slope_cfg.actor is not flat_cfg.actor
  assert slope_cfg.critic is not flat_cfg.critic
  assert slope_cfg.algorithm is not flat_cfg.algorithm
  assert slope_cfg.algorithm.learning_rate == pytest.approx(3.0e-4)
  assert flat_cfg.algorithm.learning_rate == pytest.approx(1.0e-3)


def test_tethered_export_metadata_describes_mixed_policy_actions(
  tethered_env: ManagerBasedRlEnv,
) -> None:
  metadata = get_tethered_policy_metadata(tethered_env, "test/run")

  assert metadata["joint_names"] == [*JOINT_NAMES, ARM_JOINT_NAME]
  assert metadata["controlled_joint_names"] == list(JOINT_NAMES)
  assert metadata["actor_observation_term_dims"] == [
    3,
    3,
    3,
    18,
    18,
    19,
    3,
    1,
    3,
  ]
  assert metadata["physics_dt"] == pytest.approx(0.005)
  assert metadata["control_dt"] == pytest.approx(0.02)
  assert metadata["policy_action_names"] == [*JOINT_NAMES, "cable"]
  assert metadata["policy_action_types"] == [
    *("joint_position" for _ in JOINT_NAMES),
    "tendon_effort",
  ]
  assert metadata["policy_action_term_dims"] == [18, 1]
  policy_action_scale = metadata["policy_action_scale"]
  policy_action_offset = metadata["policy_action_offset"]
  assert isinstance(policy_action_scale, list)
  assert isinstance(policy_action_offset, list)
  assert len(policy_action_scale) == 19
  assert len(policy_action_offset) == 19
  assert metadata["action_observation_source"] == "last_action"
  assert metadata["action_observation_extra_delay_steps"] == 0.0
  assert metadata["tether_action_index"] == 18.0
  assert metadata["tether_action_clip"] == [0.0, 2.0]
  assert metadata["tether_force_per_action"] == -200.0
  assert metadata["tether_tendon_name"] == "cable"
  assert metadata["tether_force_sensor_name"] == TETHER_FORCE_SENSOR_NAME
  assert metadata["tether_arm_joint_name"] == ARM_JOINT_NAME
  assert metadata["tether_anchor_site_name"] == "anchor"
  assert metadata["tether_end_site_name"] == "tether_7"
  assert metadata["tether_direction_frame"] == "trunk"
  assert metadata["tether_direction_convention"] == "tether_7_to_anchor"
  assert metadata["tether_direction_components"] == ["x", "y", "z"]
  assert metadata["tether_yaw_axis_sign"] == -1.0
