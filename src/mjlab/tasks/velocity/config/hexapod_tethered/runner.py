"""Policy export support for the tethered Hexapod task."""

import math

import torch

from mjlab.asset_zoo.robots.hexapod_tethered.hexapod_tethered_constants import (
  ARM_JOINT_NAME,
  TETHER_ANCHOR_SITE_NAME,
  TETHER_END_SITE_NAME,
  TETHER_FORCE_SENSOR_NAME,
  TETHER_TENDON_NAME,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  TETHER_ACTION_MAX,
  TETHER_ACTION_MIN,
  TETHER_ACTION_NAME,
  TETHER_FORCE_PER_ACTION,
)

PRETRAINED_CABLE_ACTION_MEAN = 0.1
PRETRAINED_CABLE_ACTION_STD = 0.2
PRETRAINED_TENSION_NORMALIZER_STD = 200.0
SLOPE_MIXTURE_CHECKPOINT_KEY = "slope_mixture_curriculum"
SLOPE_MIXTURE_CHECKPOINT_VERSION = 1


def get_tethered_policy_metadata(
  env: ManagerBasedRlEnv,
  run_name: str,
) -> dict[str, list | str | float]:
  """Describe the mixed 18-joint plus one-tendon policy interface."""
  metadata = get_base_metadata(env, run_name)
  robot: Entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)

  target_ids = joint_action.target_ids
  controlled_default_pos = robot.data.default_joint_pos[0, target_ids].cpu().tolist()
  joint_action_scale = metadata["action_scale"]
  assert isinstance(joint_action_scale, list)

  # Keep get_base_metadata's standard all-joint fields intact. These explicit
  # fields describe the heterogeneous policy output without pretending that
  # the passive ARM joint is actuated or that the cable is a joint.
  metadata["controlled_joint_names"] = list(joint_action.target_names)
  metadata["controlled_joint_default_pos"] = controlled_default_pos
  metadata["actor_observation_term_dims"] = [
    math.prod(shape) for shape in env.observation_manager.group_obs_term_dim["actor"]
  ]
  physics_dt = env.cfg.sim.mujoco.timestep
  metadata["physics_dt"] = physics_dt
  metadata["control_dt"] = physics_dt * env.cfg.decimation
  metadata["policy_action_names"] = [*joint_action.target_names, TETHER_ACTION_NAME]
  metadata["policy_action_types"] = [
    *("joint_position" for _ in joint_action.target_names),
    "tendon_effort",
  ]
  metadata["policy_action_term_dims"] = env.action_manager.action_term_dim
  metadata["policy_action_scale"] = [*joint_action_scale, 1.0]
  metadata["policy_action_offset"] = [*controlled_default_pos, 0.0]
  metadata["action_observation_source"] = "last_action"
  metadata["action_observation_extra_delay_steps"] = 0.0
  metadata["tether_action_name"] = TETHER_ACTION_NAME
  metadata["tether_action_index"] = float(joint_action.action_dim)
  metadata["tether_action_clip"] = [TETHER_ACTION_MIN, TETHER_ACTION_MAX]
  metadata["tether_force_per_action"] = TETHER_FORCE_PER_ACTION
  metadata["tether_tendon_name"] = TETHER_TENDON_NAME
  metadata["tether_force_sensor_name"] = TETHER_FORCE_SENSOR_NAME
  metadata["tether_arm_joint_name"] = ARM_JOINT_NAME
  metadata["tether_anchor_site_name"] = TETHER_ANCHOR_SITE_NAME
  metadata["tether_end_site_name"] = TETHER_END_SITE_NAME
  metadata["tether_direction_frame"] = "trunk"
  metadata["tether_direction_convention"] = "tether_7_to_anchor"
  metadata["tether_direction_components"] = ["x", "y", "z"]
  metadata["tether_yaw_axis_sign"] = -1.0
  return metadata


class TetheredVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """Export leg metadata separately from the passive ARM and cable action."""

  def load_pretrained(
    self,
    path: str,
    *,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Transfer a flat tethered actor while resetting slope-only cable state."""
    infos = super().load_pretrained(
      path,
      strict=strict,
      map_location=map_location,
    )
    tension_index = self._observation_term_start("critic", "cable_tension")
    normalizer = self.alg.critic.obs_normalizer
    normalizer_state = normalizer.state_dict()
    values = {
      "_mean": 0.0,
      "_var": PRETRAINED_TENSION_NORMALIZER_STD**2,
      "_std": PRETRAINED_TENSION_NORMALIZER_STD,
    }
    if all(key in normalizer_state for key in values):
      for key, value in values.items():
        normalizer_state[key][:, tension_index] = value
      normalizer.load_state_dict(normalizer_state)
    return infos

  def _prepare_pretrained_actor_state(
    self, actor_state_dict: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Keep the learned gait while making cable exploration safe on the slope."""
    prepared = {key: value.clone() for key, value in actor_state_dict.items()}
    reference = self.alg.get_policy().state_dict()
    action_dim = sum(self.env.unwrapped.action_manager.action_term_dim)
    obs_shape = self.env.unwrapped.observation_manager.group_obs_dim["actor"]
    assert isinstance(obs_shape, tuple)
    obs_dim = math.prod(obs_shape)

    output_weight_key = self._find_linear_weight(
      reference,
      output_size=action_dim,
    )
    input_weight_key = self._find_linear_weight(
      reference,
      input_size=obs_dim,
    )
    output_bias_key = output_weight_key.removesuffix("weight") + "bias"
    required = (output_weight_key, output_bias_key, input_weight_key)
    if any(key not in prepared for key in required):
      raise ValueError(
        "The pretrained checkpoint is not a compatible tethered Hexapod actor; "
        "expected the same 71-observation, 19-action network"
      )
    for key in required:
      if prepared[key].shape != reference[key].shape:
        raise ValueError(
          "The pretrained checkpoint is not a compatible tethered Hexapod actor; "
          f"{key} has shape {tuple(prepared[key].shape)}, expected "
          f"{tuple(reference[key].shape)}"
        )

    cable_index = self._action_term_start(TETHER_ACTION_NAME)
    prepared[output_weight_key][cable_index].zero_()
    prepared[output_bias_key][cable_index] = PRETRAINED_CABLE_ACTION_MEAN

    last_cable_action_index = (
      self._observation_term_start("actor", "actions") + cable_index
    )
    tension_index = self._observation_term_start("actor", "cable_tension")
    prepared[input_weight_key][:, last_cable_action_index].zero_()
    prepared[input_weight_key][:, tension_index].zero_()

    if "distribution.std_param" in prepared:
      prepared["distribution.std_param"][cable_index] = PRETRAINED_CABLE_ACTION_STD
    elif "distribution.log_std_param" in prepared:
      prepared["distribution.log_std_param"][cable_index] = math.log(
        PRETRAINED_CABLE_ACTION_STD
      )
    else:
      raise ValueError("Pretrained tethered actor has no Gaussian std parameter")

    self._reset_actor_normalizer_dimension(
      prepared,
      last_cable_action_index,
      std=1.0,
    )
    self._reset_actor_normalizer_dimension(
      prepared,
      tension_index,
      std=PRETRAINED_TENSION_NORMALIZER_STD,
    )
    return prepared

  @staticmethod
  def _find_linear_weight(
    state_dict: dict[str, torch.Tensor],
    *,
    input_size: int | None = None,
    output_size: int | None = None,
  ) -> str:
    matches = []
    for key, value in state_dict.items():
      if not key.startswith("mlp.") or not key.endswith(".weight"):
        continue
      if value.ndim != 2:
        continue
      if input_size is not None and value.shape[1] != input_size:
        continue
      if output_size is not None and value.shape[0] != output_size:
        continue
      matches.append(key)
    if len(matches) != 1:
      raise ValueError(
        "Could not uniquely identify the pretrained actor linear layer: "
        f"found {matches}"
      )
    return matches[0]

  def _action_term_start(self, term_name: str) -> int:
    start = 0
    manager = self.env.unwrapped.action_manager
    for name, dimension in zip(
      manager.active_terms,
      manager.action_term_dim,
      strict=True,
    ):
      if name == term_name:
        if dimension != 1:
          raise ValueError(f"Action term {term_name!r} must be scalar")
        return start
      start += dimension
    raise ValueError(f"Action term {term_name!r} is not active")

  def _observation_term_start(self, group_name: str, term_name: str) -> int:
    manager = self.env.unwrapped.observation_manager
    start = 0
    for name, shape in zip(
      manager.active_terms[group_name],
      manager.group_obs_term_dim[group_name],
      strict=True,
    ):
      if name == term_name:
        return start
      start += math.prod(shape)
    raise ValueError(f"Observation term {term_name!r} is not active in {group_name!r}")

  @staticmethod
  def _reset_actor_normalizer_dimension(
    state_dict: dict[str, torch.Tensor],
    index: int,
    *,
    std: float,
  ) -> None:
    values = {
      "obs_normalizer._mean": 0.0,
      "obs_normalizer._var": std**2,
      "obs_normalizer._std": std,
    }
    for key, value in values.items():
      if key not in state_dict:
        raise ValueError(f"Pretrained tethered actor is missing {key}")
      state_dict[key][:, index] = value

  def get_policy_metadata(self, run_name: str) -> dict[str, list | str | float]:
    metadata = get_tethered_policy_metadata(self.env.unwrapped, run_name)
    clip_actions = self.env.clip_actions
    metadata["policy_action_clip"] = (
      float("inf") if clip_actions is None else float(clip_actions)
    )
    return metadata


class TetheredSlopeMixtureOnPolicyRunner(TetheredVelocityOnPolicyRunner):
  """Resume fixed-slope PPO state with a fresh mixture curriculum clock."""

  def save(self, path: str, infos: dict | None = None) -> None:
    """Mark checkpoints so later mixture resumes retain curriculum progress."""
    mixture_state = {"version": SLOPE_MIXTURE_CHECKPOINT_VERSION}
    infos = {**(infos or {}), SLOPE_MIXTURE_CHECKPOINT_KEY: mixture_state}
    super().save(path, infos)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load PPO state and translate fixed-slope progress into mixture progress.

    A fixed-slope checkpoint has no mixture marker. Its actor, critic, optimizer,
    and PPO iteration are restored normally, but its environment step counter is
    reset so angle and command curricula start at mixture stage zero. A checkpoint
    saved by this runner carries the marker and therefore restores that counter.
    """
    infos = super().load(
      path,
      load_cfg=load_cfg,
      strict=strict,
      map_location=map_location,
    )
    if load_cfg is None:
      is_mixture_checkpoint = (
        isinstance(infos, dict) and SLOPE_MIXTURE_CHECKPOINT_KEY in infos
      )
      if not is_mixture_checkpoint:
        self.env.unwrapped.common_step_counter = 0
        self._restore_configured_learning_rate()
      else:
        # RslRlVecEnvWrapper performs its first reset before this checkpoint is
        # loaded. Resample once after restoring the mixture clock so the first
        # rollout uses the checkpoint's active slope/command curriculum stage.
        self.env.reset()
    return infos

  def _restore_configured_learning_rate(self) -> None:
    """Keep optimizer moments while applying the mixture task's initial LR."""
    learning_rate = float(self.cfg["algorithm"]["learning_rate"])
    self.alg.learning_rate = learning_rate
    for param_group in self.alg.optimizer.param_groups:
      param_group["lr"] = learning_rate
