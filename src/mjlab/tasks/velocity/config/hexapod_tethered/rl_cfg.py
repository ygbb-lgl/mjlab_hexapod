"""Independent PPO configuration for tethered Hexapod velocity tracking."""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

DEFAULT_ROLLOUT_STEPS = 24
SLOPE_DEFAULT_ROLLOUT_STEPS = 24


def hexapod_tethered_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the PPO runner configuration for the tethered Hexapod."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="hexapod_tethered_velocity",
    save_interval=500,
    num_steps_per_env=DEFAULT_ROLLOUT_STEPS,
    max_iterations=10_000,
  )


@dataclass
class HexapodTetheredSlopePpoRunnerCfg(RslRlOnPolicyRunnerCfg):
  """PPO settings owned by the fixed-slope task, independent of flat."""

  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    )
  )
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    )
  )
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    )
  )
  experiment_name: str = "hexapod_tethered_slope_velocity"
  save_interval: int = 500
  num_steps_per_env: int = SLOPE_DEFAULT_ROLLOUT_STEPS
  max_iterations: int = 10_000


def hexapod_tethered_slope_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the PPO runner configuration for the fixed-slope task."""
  return HexapodTetheredSlopePpoRunnerCfg()
