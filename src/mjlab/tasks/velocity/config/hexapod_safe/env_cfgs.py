"""Flat-ground mjlab port of the tuned EL-4090 task parameters and rewards."""

from __future__ import annotations

import math

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.hexapod_compare.hexapod_compare_constants import (
  FOOT_GEOM_NAMES,
  FULL_COLLISION,
  INIT_STATE,
  JOINT_NAMES,
  LEG_ARMATURE,
  LEG_DAMPING,
  LEG_EFFORT_LIMIT,
  get_spec,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import observations, rewards
from .actions import SignedJointPositionActionCfg
from .events import set_planar_push_velocity

EL4090_ACTION_SCALE = 0.25
EL4090_STIFFNESS = 120.0
EL4090_REWARD_TORQUE_LIMIT = 80.0


def get_el4090_hexapod_robot_cfg() -> EntityCfg:
  """Use the migrated hexapod model with the EL-4090 controller parameters."""
  actuator = BuiltinPositionActuatorCfg(
    target_names_expr=(r".*_(HAA|HFE|KFE)",),
    stiffness=EL4090_STIFFNESS,
    damping=LEG_DAMPING,
    effort_limit=LEG_EFFORT_LIMIT,
    armature=LEG_ARMATURE,
  )
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(actuator,),
      soft_joint_pos_limit_factor=0.9,
    ),
  )


def _el4090_observations() -> dict[str, ObservationGroupCfg]:
  joint_cfg = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
  terms = {
    # Keep this insertion order identical to MuJoCo Playground's 66-D state.
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel,
      noise=Unoise(n_min=-1.2, n_max=1.2),
      clip=(-50.0, 50.0),
      scale=2.0,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=Unoise(n_min=-1.2, n_max=1.2),
      clip=(-400.0, 400.0),
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.75, n_max=0.75),
      clip=(-100.0, 100.0),
      scale=1.0,
    ),
    "joint_pos": ObservationTermCfg(
      func=observations.signed_joint_pos_rel,
      params={"asset_cfg": joint_cfg},
      noise=Unoise(n_min=-0.075, n_max=0.075),
      clip=(-100.0, 100.0),
      scale=1.0,
    ),
    "joint_vel": ObservationTermCfg(
      func=observations.signed_joint_vel,
      params={"asset_cfg": joint_cfg},
      noise=Unoise(n_min=-2.25, n_max=2.25),
      clip=(-2000.0, 2000.0),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(
      func=envs_mdp.last_action,
      clip=(-100.0, 100.0),
      scale=1.0,
    ),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
      clip=(-100.0, 100.0),
      scale=(2.0, 2.0, 0.25),
    ),
  }
  return {
    "actor": ObservationGroupCfg(
      terms=terms,
      concatenate_terms=True,
      enable_corruption=True,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=terms,
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="sanitize",
    ),
  }


def _el4090_rewards() -> dict[str, RewardTermCfg]:
  joints = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
  feet = SceneEntityCfg(
    "robot",
    body_names=tuple(f"{leg}_FOOT" for leg in ("RF", "RM", "RB", "LF", "LM", "LB")),
    preserve_order=True,
  )
  shanks = SceneEntityCfg(
    "robot",
    body_names=tuple(f"{leg}_SHANK" for leg in ("RF", "RM", "RB", "LF", "LM", "LB")),
    preserve_order=True,
  )
  gait_params = {
    "sensor_name": "feet_ground_contact",
    "command_name": "twist",
  }
  return {
    "tracking_lin_vel": RewardTermCfg(
      func=rewards.tracking_lin_vel,
      weight=6.0,
      params={"command_name": "twist", "tracking_sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=rewards.tracking_ang_vel,
      weight=5.5,
      params={"command_name": "twist", "tracking_sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(func=rewards.lin_vel_z, weight=-5.0),
    "ang_vel_xy": RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.5),
    "orientation": RewardTermCfg(func=rewards.orientation, weight=-5.0),
    "torques": RewardTermCfg(
      func=rewards.torques,
      weight=-0.0001,
      params={"asset_cfg": joints},
    ),
    "dof_vel": RewardTermCfg(
      func=rewards.dof_vel,
      weight=-0.0001,
      params={"asset_cfg": joints},
    ),
    "dof_acc": RewardTermCfg(
      func=rewards.dof_acc,
      weight=-1.0e-6,
      params={"asset_cfg": joints},
    ),
    "base_height": RewardTermCfg(
      func=rewards.base_height,
      weight=-50.0,
      params={"target": 0.45},
    ),
    "feet_slip": RewardTermCfg(
      func=rewards.feet_slip,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact", "foot_cfg": feet},
    ),
    "feet_air_time": RewardTermCfg(
      func=rewards.feet_air_time,
      weight=3.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "threshold": 0.5,
      },
    ),
    "collision": RewardTermCfg(
      func=rewards.collision,
      weight=-1.0,
      params={"sensor_name": "penalized_contact"},
    ),
    "feet_stumble": RewardTermCfg(
      func=rewards.feet_stumble,
      weight=-1.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "action_rate": RewardTermCfg(
      func=rewards.action_rate,
      weight=-0.01,
      params={"action_name": "joint_pos"},
    ),
    "stand_still": RewardTermCfg(
      func=rewards.stand_still,
      weight=-5.0,
      params={"command_name": "twist", "asset_cfg": joints},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=rewards.dof_pos_limits,
      weight=-0.1,
      params={"asset_cfg": joints},
    ),
    "dof_vel_limits": RewardTermCfg(
      func=rewards.dof_vel_limits,
      weight=-1.0,
      params={
        "velocity_limit": 15.8,
        "soft_limit": 0.9,
        "asset_cfg": joints,
      },
    ),
    "torque_limits": RewardTermCfg(
      func=rewards.torque_limits,
      weight=-0.01,
      params={
        # Keep the tuned threshold from the source URDF. The migrated
        # robot itself retains its calibrated 50 Nm actuator saturation.
        "torque_limit": EL4090_REWARD_TORQUE_LIMIT,
        "soft_limit": 1.0,
        "asset_cfg": joints,
      },
    ),
    "feet_contact_forces": RewardTermCfg(
      func=rewards.feet_contact_forces,
      weight=-0.03,
      params={
        "sensor_name": "feet_ground_contact",
        "max_contact_force": 250.0,
      },
    ),
    "shank_vertical": RewardTermCfg(
      func=rewards.shank_vertical,
      weight=-1.0,
      params={"shank_cfg": shanks, "foot_cfg": feet},
    ),
    "feet_async": RewardTermCfg(
      func=rewards.feet_async,
      weight=-2.0,
      params=gait_params,
    ),
    "feet_sync": RewardTermCfg(
      func=rewards.feet_sync,
      weight=-2.0,
      params=gait_params,
    ),
  }


def hexapod_safe_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the tuned EL-4090 flat task using the migrated hexapod MJCF."""
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_el4090_hexapod_robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  feet_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  penalized_bodies = ("trunk",) + tuple(
    f"{leg}_{link}"
    for leg in ("RF", "RM", "RB", "LF", "LM", "LB")
    for link in ("THIGH", "SHANK")
  )
  penalized_contact = ContactSensorCfg(
    name="penalized_contact",
    primary=ContactMatch(mode="body", pattern=penalized_bodies, entity="robot"),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  trunk_ground = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(mode="body", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_contact, penalized_contact, trunk_ground)

  cfg.observations = _el4090_observations()
  cfg.actions = {
    "joint_pos": SignedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      scale={".*": EL4090_ACTION_SCALE},
      use_default_offset=True,
      preserve_order=True,
    )
  }
  cfg.commands = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 4.0),
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      rel_world_envs=0.0,
      rel_forward_envs=0.0,
      heading_command=False,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 1.5),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.5, 1.5),
      ),
    )
  }
  cfg.events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=JOINT_NAMES, preserve_order=True
        ),
      },
    ),
    "robot_friction": EventTermCfg(
      func=envs_mdp.dr.geom_friction,
      mode="startup",
      params={
        # Isaac Gym assigns the randomized bucket to every rigid shape.
        "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_collision\d*",)),
        "operation": "abs",
        "ranges": (0.3, 1.25),
        "shared_random": True,
      },
    ),
    "base_mass": EventTermCfg(
      func=envs_mdp.dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
        "operation": "add",
        "ranges": (-5.0, 5.0),
      },
    ),
    "push_robot": EventTermCfg(
      func=set_planar_push_velocity,
      mode="interval",
      interval_range_s=(3.0, 3.0),
      params={"max_velocity": 1.0},
    ),
  }
  cfg.rewards = _el4090_rewards()
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # The source terminates when projected_gravity[:, 2] > 0: the body has
    # rotated by more than 90 degrees from upright.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.pi / 2.0},
    ),
  }
  cfg.curriculum = {}
  cfg.metrics = {}
  cfg.decimation = 4
  cfg.episode_length_s = 20.0
  cfg.sim.mujoco.timestep = 0.005
  cfg.sim.nconmax = 100
  cfg.sim.njmax = 1500
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.mujoco.iterations = 10
  cfg.sim.mujoco.ls_iterations = 20
  cfg.sim.mujoco.impratio = 10.0
  cfg.sim.mujoco.cone = "elliptic"
  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 2.5
  cfg.viewer.elevation = -12.0

  if play:
    cfg.episode_length_s = int(1.0e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
