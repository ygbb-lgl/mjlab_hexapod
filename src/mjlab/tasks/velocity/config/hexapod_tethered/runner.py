"""Policy export support for the tethered Hexapod task."""

import math

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

  def get_policy_metadata(self, run_name: str) -> dict[str, list | str | float]:
    metadata = get_tethered_policy_metadata(self.env.unwrapped, run_name)
    clip_actions = self.env.clip_actions
    metadata["policy_action_clip"] = (
      float("inf") if clip_actions is None else float(clip_actions)
    )
    return metadata
