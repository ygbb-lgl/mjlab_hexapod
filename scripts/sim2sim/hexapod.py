"""Run a mjlab hexapod ONNX policy in native MuJoCo.

This is a deliberate simulator boundary: training uses MuJoCo Warp through mjlab,
while this program loads an exported scene into the native MuJoCo CPU runtime.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as ort
import tyro

import mjlab


@dataclass(frozen=True)
class Sim2SimCfg:
  policy: str
  """ONNX policy exported by the mjlab velocity runner."""

  model: str
  """Flat-terrain scene.xml produced by ``uv run export-scene``."""

  command_x: float = 0.5
  command_y: float = 0.0
  command_yaw: float = 0.0
  duration: float = 60.0
  control_dt: float = 0.02
  sim_dt: float = 0.005
  action_clip: float | None = None
  """Override normalized action clipping (policy metadata selects the default)."""
  headless: bool = False
  realtime: bool = True


def _split_metadata(value: str) -> list[str]:
  return [] if not value else value.split(",")


def _parse_numeric_entries(value: str) -> list[float | np.ndarray]:
  entries: list[float | np.ndarray] = []
  for item in _split_metadata(value):
    if ";" in item:
      entries.append(np.asarray([float(v) for v in item.split(";")]))
    else:
      entries.append(float(item))
  return entries


def _resolve_name(
  model: mujoco.MjModel,
  obj_type: mujoco.mjtObj,
  name: str,
) -> int:
  for candidate in (name, f"robot/{name}"):
    obj_id = mujoco.mj_name2id(model, obj_type, candidate)
    if obj_id >= 0:
      return obj_id
  raise ValueError(f"Could not find {name!r} or 'robot/{name}' in the model")


def _sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
  sensor_id = _resolve_name(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
  address = model.sensor_adr[sensor_id]
  dimension = model.sensor_dim[sensor_id]
  return np.asarray(data.sensordata[address : address + dimension]).copy()


class HexapodPolicy:
  """ONNX policy plus the mjlab metadata needed to reproduce actor inputs."""

  _SUPPORTED_OBSERVATIONS = {
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
  }

  def __init__(self, policy_path: Path, model: mujoco.MjModel):
    self.session = ort.InferenceSession(
      str(policy_path),
      providers=["CPUExecutionProvider"],
    )
    if len(self.session.get_inputs()) != 1:
      raise ValueError("The sim2sim runner expects an ONNX policy with one input")
    self.input_name = self.session.get_inputs()[0].name

    metadata = self.session.get_modelmeta().custom_metadata_map
    required = {
      "joint_names",
      "default_joint_pos",
      "action_scale",
      "observation_names",
      "observation_terms_scale",
      "observation_terms_clip",
    }
    missing = required - metadata.keys()
    if missing:
      raise ValueError(
        "Policy is missing mjlab ONNX metadata: " + ", ".join(sorted(missing))
      )

    self.joint_names = _split_metadata(metadata["joint_names"])
    self.default_joint_pos = np.asarray(
      [float(v) for v in _split_metadata(metadata["default_joint_pos"])],
      dtype=np.float32,
    )
    self.action_scale = np.asarray(
      [float(v) for v in _split_metadata(metadata["action_scale"])],
      dtype=np.float32,
    )
    self.observation_names = _split_metadata(metadata["observation_names"])
    unsupported = set(self.observation_names) - self._SUPPORTED_OBSERVATIONS
    if unsupported:
      raise ValueError(
        "This native MuJoCo runner supports the flat task only; unsupported "
        f"observation terms: {sorted(unsupported)}"
      )

    self.observation_scales = _parse_numeric_entries(
      metadata["observation_terms_scale"]
    )
    self.observation_clips = _parse_numeric_entries(metadata["observation_terms_clip"])
    if not (
      len(self.observation_names)
      == len(self.observation_scales)
      == len(self.observation_clips)
    ):
      raise ValueError("Inconsistent observation metadata lengths")

    if len(self.joint_names) != len(self.default_joint_pos):
      raise ValueError("Joint names and default positions have different lengths")
    if len(self.joint_names) != len(self.action_scale):
      raise ValueError("Joint names and action scales have different lengths")

    self.joint_ids = np.asarray(
      [
        _resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in self.joint_names
      ],
      dtype=np.int32,
    )
    self.qpos_ids = model.jnt_qposadr[self.joint_ids]
    self.qvel_ids = model.jnt_dofadr[self.joint_ids]
    self.ctrl_ids = self._find_ctrl_ids(model)
    self.root_body_id = _resolve_name(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)
    self.joint_signs = np.ones(len(self.joint_names), dtype=np.float32)
    self.default_action_clip = float(metadata.get("policy_action_clip", "inf"))
    self.planar_command_deadzone = float(metadata.get("planar_command_deadzone", "0.0"))
    if metadata.get("hexapod_action_mode") == "signed_axes":
      if "hexapod_joint_signs" not in metadata:
        raise ValueError("Signed-axis policy is missing joint-sign metadata")
      self.joint_signs = np.asarray(
        [float(v) for v in _split_metadata(metadata["hexapod_joint_signs"])],
        dtype=np.float32,
      )
      if self.joint_signs.shape != self.default_joint_pos.shape:
        raise ValueError("Joint signs do not match the exported joint count")

  def _find_ctrl_ids(self, model: mujoco.MjModel) -> np.ndarray:
    ctrl_ids: list[int] = []
    for joint_id, joint_name in zip(
      self.joint_ids,
      self.joint_names,
      strict=True,
    ):
      matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
      if len(matches) != 1:
        raise ValueError(
          f"Expected one actuator for {joint_name}, found {len(matches)}"
        )
      ctrl_ids.append(int(matches[0]))
    return np.asarray(ctrl_ids, dtype=np.int32)

  def _observation_terms(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    command: np.ndarray,
  ) -> dict[str, np.ndarray]:
    body_rotation = data.xmat[self.root_body_id].reshape(3, 3)
    projected_gravity = body_rotation.T @ np.asarray([0.0, 0.0, -1.0])
    return {
      "base_lin_vel": _sensor_data(model, data, "imu_lin_vel"),
      "base_ang_vel": _sensor_data(model, data, "imu_ang_vel"),
      "projected_gravity": projected_gravity,
      "joint_pos": (data.qpos[self.qpos_ids] - self.default_joint_pos)
      * self.joint_signs,
      "joint_vel": data.qvel[self.qvel_ids] * self.joint_signs,
      "actions": self.last_action,
      "command": command,
    }

  def act(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    command: np.ndarray,
    action_clip: float,
  ) -> np.ndarray:
    policy_command = command.copy()
    if np.linalg.norm(policy_command[:2]) <= self.planar_command_deadzone:
      policy_command[:2] = 0.0
    terms = self._observation_terms(model, data, policy_command)
    processed: list[np.ndarray] = []
    for index, name in enumerate(self.observation_names):
      value = np.asarray(terms[name], dtype=np.float32)
      clip = self.observation_clips[index]
      if isinstance(clip, np.ndarray):
        value = np.clip(value, clip[0], clip[1])
      else:
        raise ValueError(f"Observation clip for {name!r} must have two values")
      value = value * self.observation_scales[index]
      processed.append(value.reshape(-1))

    observation = np.concatenate(processed).astype(np.float32, copy=False)
    action = self.session.run(None, {self.input_name: observation[None]})[0]
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action = np.clip(action, -action_clip, action_clip)
    if action.shape != self.action_scale.shape:
      raise ValueError(
        f"Policy returned {action.shape}, expected {self.action_scale.shape}"
      )
    self.last_action = action
    return self.default_joint_pos + self.action_scale * action * self.joint_signs


def _reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
  else:
    mujoco.mj_resetData(model, data)
  mujoco.mj_forward(model, data)


def _run_loop(
  cfg: Sim2SimCfg,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  policy: HexapodPolicy,
  viewer: Any | None,
) -> None:
  command = np.asarray(
    [cfg.command_x, cfg.command_y, cfg.command_yaw],
    dtype=np.float32,
  )
  decimation = round(cfg.control_dt / cfg.sim_dt)
  if decimation < 1 or not np.isclose(decimation * cfg.sim_dt, cfg.control_dt):
    raise ValueError("control_dt must be an integer multiple of sim_dt")

  start_sim_time = data.time
  start_wall_time = time.perf_counter()
  while data.time - start_sim_time < cfg.duration:
    if viewer is not None and not viewer.is_running():
      break

    action_clip = (
      policy.default_action_clip if cfg.action_clip is None else cfg.action_clip
    )
    targets = policy.act(model, data, command, action_clip)
    data.ctrl[policy.ctrl_ids] = targets
    for _ in range(decimation):
      mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
      raise RuntimeError("MuJoCo state became non-finite")

    if viewer is not None:
      viewer.sync()
    if cfg.realtime:
      target_wall_time = start_wall_time + data.time - start_sim_time
      time.sleep(max(0.0, target_wall_time - time.perf_counter()))


def run(cfg: Sim2SimCfg) -> None:
  """Load the exported artifacts and execute the policy."""
  if cfg.sim_dt <= 0.0 or cfg.control_dt <= 0.0:
    raise ValueError("sim_dt and control_dt must be positive")
  if cfg.duration <= 0.0:
    raise ValueError("duration must be positive")
  if cfg.action_clip is not None and cfg.action_clip <= 0.0:
    raise ValueError("action_clip must be positive")

  policy_path = Path(cfg.policy).expanduser().resolve()
  model_path = Path(cfg.model).expanduser().resolve()
  if not policy_path.is_file():
    raise FileNotFoundError(f"Policy not found: {policy_path}")
  if not model_path.is_file():
    raise FileNotFoundError(f"Model not found: {model_path}")

  model = mujoco.MjModel.from_xml_path(str(model_path))
  model.opt.timestep = cfg.sim_dt
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  model.opt.impratio = 10.0
  model.opt.iterations = 10
  model.opt.ls_iterations = 20
  model.opt.ccd_iterations = 50
  data = mujoco.MjData(model)
  _reset(model, data)
  policy = HexapodPolicy(policy_path, model)

  print(
    f"Loaded {len(policy.joint_names)} joints, "
    f"command=({cfg.command_x}, {cfg.command_y}, {cfg.command_yaw})"
  )
  if cfg.headless:
    _run_loop(cfg, model, data, policy, None)
  else:
    with mujoco.viewer.launch_passive(model, data) as viewer:
      _run_loop(cfg, model, data, policy, viewer)


def main() -> None:
  run(tyro.cli(Sim2SimCfg, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
