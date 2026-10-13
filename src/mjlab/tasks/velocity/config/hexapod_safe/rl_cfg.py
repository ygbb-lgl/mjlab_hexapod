"""RSL-RL configuration matching the tuned EL-4090 PPO setup."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def hexapod_safe_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=1,
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.3,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
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
      symmetry_cfg={
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.6,
        "data_augmentation_func": (
          "mjlab.tasks.velocity.config.hexapod_safe.symmetry:"
          "hexapod_left_right_symmetry"
        ),
      },
    ),
    clip_actions=100.0,
    # The source task has no privileged critic observations. Both networks use
    # the same noisy 66-D actor vector.
    obs_groups={"actor": ("actor",), "critic": ("actor",)},
    experiment_name="el_4090_safe_mjlab",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=1500,
  )
