"""Model-level checks for the crab-style hexapod asset."""

import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.hexapod import hexapod_constants
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def hexapod_entity() -> Entity:
  return Entity(hexapod_constants.get_hexapod_robot_cfg())


@pytest.fixture(scope="module")
def hexapod_model(hexapod_entity: Entity) -> mujoco.MjModel:
  return hexapod_entity.spec.compile()


def test_joint_order_and_xml_axis_signs(
  hexapod_entity: Entity,
  hexapod_model: mujoco.MjModel,
) -> None:
  assert hexapod_entity.num_joints == 18
  assert hexapod_entity.num_actuators == 18
  assert tuple(hexapod_entity.joint_names) == hexapod_constants.JOINT_NAMES

  for name, expected_sign in hexapod_constants.JOINT_AXIS_SIGNS.items():
    np.testing.assert_allclose(
      hexapod_model.joint(name).axis,
      (0.0, 0.0, float(expected_sign)),
    )


def test_actuator_pd_and_soft_joint_limits(hexapod_model: mujoco.MjModel) -> None:
  assert isinstance(hexapod_constants.STIFFNESS_HAA, float)
  assert isinstance(hexapod_constants.STIFFNESS_HFE, float)
  assert isinstance(hexapod_constants.STIFFNESS_KFE, float)
  for actuator_id in range(hexapod_model.nu):
    actuator = hexapod_model.actuator(actuator_id)
    assert actuator.gainprm[0] == 100.0
    assert actuator.biasprm[1] == -100.0
    assert actuator.biasprm[2] == -3.0
    np.testing.assert_allclose(actuator.forcerange, (-30.0, 30.0))

  for name in hexapod_constants.JOINT_NAMES:
    expected = (-0.8, 0.8) if name.endswith("HAA") else (-0.6, 0.6)
    np.testing.assert_allclose(hexapod_model.joint(name).range, expected)

  assert hexapod_constants.HEXAPOD_ACTION_SCALE == {
    r".*_HAA": pytest.approx(0.25),
    r".*_HFE": pytest.approx(0.25),
    r".*_KFE": pytest.approx(0.25),
  }


def test_initial_pose_places_all_feet_on_the_floor(
  hexapod_entity: Entity,
  hexapod_model: mujoco.MjModel,
) -> None:
  data = mujoco.MjData(hexapod_model)
  mujoco.mj_resetDataKeyframe(hexapod_model, data, 0)
  mujoco.mj_forward(hexapod_model, data)

  heights = np.array(
    [data.site(name).xpos[2] for name in hexapod_constants.FOOT_SITE_NAMES]
  )
  np.testing.assert_allclose(heights, 0.023, atol=5.0e-4)
  assert hexapod_model.body_subtreemass[0] == pytest.approx(39.9407774)

  expected = resolve_expr(
    hexapod_constants.INIT_JOINT_POS,
    hexapod_entity.joint_names,
    0.0,
  )
  actual = [
    data.qpos[hexapod_model.joint(name).qposadr[0]]
    for name in hexapod_constants.JOINT_NAMES
  ]
  np.testing.assert_allclose(actual, expected)


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


def test_exported_scene_loads_with_mesh_assets(tmp_path: Path) -> None:
  scene = Scene(
    SceneCfg(entities={"robot": hexapod_constants.get_hexapod_robot_cfg()}),
    device="cpu",
  )
  output_dir = tmp_path / "hexapod_scene"
  scene.write(output_dir)

  assert len(tuple((output_dir / "assets").rglob("*.STL"))) == 25
  model = mujoco.MjModel.from_xml_path(str(output_dir / "scene.xml"))
  assert model.nu == 18
  assert model.key(0).name == "init_state"
