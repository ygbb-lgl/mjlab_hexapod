"""Tests specific to velocity tasks."""

import pytest

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  GO1_ACTION_SCALE,
  HEXAPOD_ACTION_SCALE,
)
from mjlab.asset_zoo.robots.hexapod.hexapod_constants import (
  JOINT_NAMES,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


@pytest.fixture(scope="module")
def g1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all G1 velocity task IDs."""
  return [t for t in velocity_task_ids if "G1" in t]


@pytest.fixture(scope="module")
def go1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all Go1 velocity task IDs."""
  return [t for t in velocity_task_ids if "Go1" in t]


@pytest.fixture(scope="module")
def hexapod_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all hexapod velocity task IDs."""
  return [t for t in velocity_task_ids if "Hexapod" in t]


@pytest.fixture(scope="module")
def rough_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all rough terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Rough" in t]


@pytest.fixture(scope="module")
def flat_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all flat terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Flat" in t]


def test_velocity_tasks_have_twist_command(velocity_task_ids: list[str]) -> None:
  """All velocity tasks should have a velocity command."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "twist" in cfg.commands, f"Task {task_id} missing 'twist' command"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg), (
      f"Task {task_id} twist command is not UniformVelocityCommandCfg"
    )


def test_g1_velocity_has_required_sensors(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should have feet/ground and self collision sensors."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_go1_velocity_has_required_sensors(go1_velocity_task_ids: list[str]) -> None:
  """Go1 velocity tasks should have feet/ground and collision sensors."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    if "Rough" in task_id:
      for name in (
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
      ):
        assert name in sensor_names, f"Task {task_id} missing {name} sensor"


def test_hexapod_velocity_has_required_sensors(
  hexapod_velocity_task_ids: list[str],
) -> None:
  """Hexapod tasks should expose six-foot and illegal-contact sensing."""
  for task_id in hexapod_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    assert cfg.scene.sensors is not None
    sensor_names = {sensor.name for sensor in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names
    assert "illegal_ground_contact" in sensor_names


def test_flat_velocity_tasks_have_plane_terrain(
  flat_velocity_task_ids: list[str],
) -> None:
  """Flat velocity tasks should have terrain_type='plane' and no terrain_generator."""
  for task_id in flat_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "plane", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, expected 'plane'"
    )
    assert cfg.scene.terrain.terrain_generator is None, (
      f"Task {task_id} has terrain_generator, expected None for flat terrain"
    )


def test_rough_velocity_tasks_have_generator_terrain(
  rough_velocity_task_ids: list[str],
) -> None:
  """Rough velocity tasks should have generator terrain."""
  for task_id in rough_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "generator", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, "
      "expected 'generator'"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator, expected one for rough terrain"
    )


def test_rough_velocity_training_has_curriculum_enabled() -> None:
  """Rough velocity training tasks should have terrain curriculum enabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_rough_velocity_play_has_curriculum_disabled() -> None:
  """Rough velocity play tasks should have terrain curriculum disabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is False, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected False"
    )


def test_g1_velocity_has_correct_action_scale(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should use G1_ACTION_SCALE."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_go1_velocity_has_correct_action_scale(
  go1_velocity_task_ids: list[str],
) -> None:
  """Go1 velocity tasks should use GO1_ACTION_SCALE."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == GO1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected GO1_ACTION_SCALE"
    )


def test_hexapod_velocity_has_correct_action_scale(
  hexapod_velocity_task_ids: list[str],
) -> None:
  """Hexapod tasks should use HEXAPOD_ACTION_SCALE."""
  for task_id in hexapod_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    assert joint_pos_action.scale == HEXAPOD_ACTION_SCALE


def test_hexapod_policy_io_uses_playground_order(
  hexapod_velocity_task_ids: list[str],
) -> None:
  """All hexapod policies keep the deployed Playground vector ordering."""
  core_terms = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
  )
  for task_id in hexapod_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    actor_terms = cfg.observations["actor"].terms
    assert tuple(actor_terms)[:7] == core_terms
    joint_cfg = actor_terms["joint_pos"].params["asset_cfg"]
    assert tuple(joint_cfg.joint_names) == JOINT_NAMES
    assert joint_cfg.preserve_order
    action_cfg = cfg.actions["joint_pos"]
    assert isinstance(action_cfg, JointPositionActionCfg)
    assert tuple(action_cfg.actuator_names) == JOINT_NAMES
    assert action_cfg.preserve_order


def test_hexapod_reset_uses_small_drop_height(
  hexapod_velocity_task_ids: list[str],
) -> None:
  """The heavy hexapod should not be dropped 5 cm on every reset."""
  for task_id in hexapod_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    pose_range = cfg.events["reset_base"].params["pose_range"]
    assert pose_range["z"] == (0.0, 0.01)
