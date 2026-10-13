"""Tests for the Hexapod Compare asset."""

import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.hexapod_compare import hexapod_compare_constants
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def hexapod_entity() -> Entity:
  return Entity(hexapod_compare_constants.get_hexapod_compare_robot_cfg())


@pytest.fixture(scope="module")
def hexapod_model(hexapod_entity: Entity) -> mujoco.MjModel:
  return hexapod_entity.spec.compile()


def test_hexapod_entity_creation(hexapod_entity: Entity) -> None:
  assert hexapod_entity.num_actuators == 18
  assert hexapod_entity.num_joints == 18
  assert hexapod_entity.is_actuated
  assert not hexapod_entity.is_fixed_base
  assert tuple(hexapod_entity.joint_names) == hexapod_compare_constants.JOINT_NAMES


def test_safe_joint_signs_cover_playground_order() -> None:
  assert len(hexapod_compare_constants.SAFE_JOINT_SIGNS) == len(
    hexapod_compare_constants.JOINT_NAMES
  )
  negative_joints = {
    name
    for name, sign in zip(
      hexapod_compare_constants.JOINT_NAMES,
      hexapod_compare_constants.SAFE_JOINT_SIGNS,
      strict=True,
    )
    if sign < 0
  }
  assert negative_joints == {
    "RF_HFE",
    "RF_KFE",
    "RM_HFE",
    "RM_KFE",
    "RB_HFE",
    "LF_HAA",
    "LM_HAA",
    "LB_HAA",
    "LB_KFE",
  }


def test_actuator_parameters(hexapod_model: mujoco.MjModel) -> None:
  cfg = hexapod_compare_constants.HEXAPOD_COMPARE_LEG_ACTUATOR_CFG
  effort_limit = cfg.effort_limit
  assert effort_limit is not None
  for actuator_id in range(hexapod_model.nu):
    actuator = hexapod_model.actuator(actuator_id)
    assert actuator.gainprm[0] == cfg.stiffness
    assert actuator.biasprm[1] == -cfg.stiffness
    assert actuator.biasprm[2] == -cfg.damping
    assert actuator.forcerange[0] == -effort_limit
    assert actuator.forcerange[1] == effort_limit


def test_keyframe_joint_positions(
  hexapod_entity: Entity,
  hexapod_model: mujoco.MjModel,
) -> None:
  key = hexapod_model.key("init_state")
  expected_joint_pos = hexapod_compare_constants.INIT_STATE.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, hexapod_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    hexapod_entity.joint_names,
    expected_values,
    strict=True,
  ):
    qpos_idx = hexapod_model.joint(joint_name).qposadr[0]
    np.testing.assert_allclose(key.qpos[qpos_idx], expected_value)


def test_keyframe_starts_without_floor_penetration(
  hexapod_model: mujoco.MjModel,
) -> None:
  data = mujoco.MjData(hexapod_model)
  mujoco.mj_resetDataKeyframe(hexapod_model, data, 0)
  mujoco.mj_forward(hexapod_model, data)
  for site_name in hexapod_compare_constants.FOOT_SITE_NAMES:
    assert data.site(site_name).xpos[2] >= 0.023


def test_collision_configuration(hexapod_model: mujoco.MjModel) -> None:
  collision_geoms = [
    hexapod_model.geom(i)
    for i in range(hexapod_model.ngeom)
    if "_collision" in hexapod_model.geom(i).name
  ]
  assert len(collision_geoms) == 26
  for geom in collision_geoms:
    assert geom.contype == 1
    assert geom.conaffinity == 1
    if re.match(r"^(RF|RM|RB|LF|LM|LB)_foot_collision$", geom.name):
      assert geom.condim == 6
      assert geom.priority == 1
      assert geom.friction[0] == 1.0
    else:
      assert geom.condim == 1


def test_builtin_sensor_names(hexapod_model: mujoco.MjModel) -> None:
  sensor_names = {
    hexapod_model.sensor(sensor_id).name for sensor_id in range(hexapod_model.nsensor)
  }
  assert sensor_names == {
    "imu_ang_vel",
    "imu_lin_vel",
    "imu_lin_acc",
    "imu_upvector",
    "root_angmom",
  }


def test_exported_scene_loads_with_mesh_assets(tmp_path: Path) -> None:
  scene = Scene(
    SceneCfg(
      entities={"robot": hexapod_compare_constants.get_hexapod_compare_robot_cfg()}
    ),
    device="cpu",
  )
  output_dir = tmp_path / "hexapod_scene"
  scene.write(output_dir)

  asset_files = tuple((output_dir / "assets").rglob("*.STL"))
  assert len(asset_files) == 25

  model = mujoco.MjModel.from_xml_path(str(output_dir / "scene.xml"))
  assert model.nu == 18
  assert model.key(0).name == "init_state"
