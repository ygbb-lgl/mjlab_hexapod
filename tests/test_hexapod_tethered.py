"""Asset and task tests for the tethered Hexapod."""

from collections.abc import Iterator

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
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.velocity.config.hexapod.env_cfgs import hexapod_flat_env_cfg
from mjlab.tasks.velocity.config.hexapod_tethered import env_cfgs as tethered_env_cfgs
from mjlab.tasks.velocity.config.hexapod_tethered import observations, rewards
from mjlab.tasks.velocity.config.hexapod_tethered.env_cfgs import (
  OBSERVATION_NOISE_LEVEL,
  OBSERVATION_NOISE_SCALES,
  TETHER_ACTION_MAX,
  TETHER_ACTION_MIN,
  TETHER_DIRECTION_NOISE_SCALES,
  TETHER_FORCE_PER_ACTION,
  hexapod_tethered_flat_env_cfg,
)
from mjlab.tasks.velocity.config.hexapod_tethered.rl_cfg import (
  hexapod_tethered_ppo_runner_cfg,
)
from mjlab.tasks.velocity.config.hexapod_tethered.runner import (
  get_tethered_policy_metadata,
)
from mjlab.utils.noise import UniformNoiseCfg


@pytest.fixture(scope="module")
def tethered_entity() -> Entity:
  return Entity(hexapod_tethered_constants.get_hexapod_tethered_robot_cfg())


@pytest.fixture(scope="module")
def tethered_model(tethered_entity: Entity) -> mujoco.MjModel:
  return tethered_entity.spec.compile()


@pytest.fixture(scope="module")
def tethered_env() -> Iterator[ManagerBasedRlEnv]:
  cfg = hexapod_tethered_flat_env_cfg()
  cfg.scene.num_envs = 2
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

  tethered_actuators = hexapod_tethered_constants.HEXAPOD_TETHERED_ARTICULATION
  base_actuators = hexapod_constants.HEXAPOD_ARTICULATION
  assert tethered_actuators is not base_actuators
  for tethered_cfg, base_cfg in zip(
    tethered_actuators.actuators[:3],
    base_actuators.actuators,
    strict=True,
  ):
    assert isinstance(tethered_cfg, BuiltinPositionActuatorCfg)
    assert isinstance(base_cfg, BuiltinPositionActuatorCfg)
    assert tethered_cfg is not base_cfg
    assert tethered_cfg.stiffness == base_cfg.stiffness
    assert tethered_cfg.damping == base_cfg.damping

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


def test_cable_actuator_maps_control_to_negative_tendon_force(
  tethered_model: mujoco.MjModel,
) -> None:
  cable = tethered_model.actuator(TETHER_TENDON_NAME)
  np.testing.assert_allclose(cable.ctrlrange, (TETHER_ACTION_MIN, TETHER_ACTION_MAX))
  np.testing.assert_allclose(cable.forcerange, (-400.0, 0.0))
  assert cable.gainprm[0] == pytest.approx(TETHER_FORCE_PER_ACTION)

  data = mujoco.MjData(tethered_model)
  mujoco.mj_resetDataKeyframe(tethered_model, data, 0)
  data.ctrl[cable.id] = 1.0
  mujoco.mj_forward(tethered_model, data)
  assert data.actuator_force[cable.id] == pytest.approx(-200.0)

  sensor = tethered_model.sensor(TETHER_FORCE_SENSOR_NAME)
  assert data.sensordata[sensor.adr[0]] == pytest.approx(-200.0)

  data.ctrl[cable.id] = 3.0
  mujoco.mj_forward(tethered_model, data)
  assert data.actuator_force[cable.id] == pytest.approx(-400.0)
  assert data.sensordata[sensor.adr[0]] == pytest.approx(-400.0)


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
  torch.testing.assert_close(tension, torch.full_like(tension, 200.0))

  following_observations, *_ = tethered_env.step(torch.zeros((2, 19)))
  following_actor = following_observations["actor"]
  assert isinstance(following_actor, torch.Tensor)
  torch.testing.assert_close(following_actor[:, 45:64], torch.zeros((2, 19)))


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
