Hexapod velocity tasks
=======================

The crab-style Hexapod has an independent flat-ground task derived from the
Go1 velocity-task structure and recalibrated from its own MJCF. It uses the
model's 18-joint order, a geometry-calibrated lightly bent default pose, a
0.6-second gait clock, and alternating support triangles ``RF+RB+LM`` and
``RM+LF+LB``. The gait clock is internal to the contact-pattern reward and is
not part of the policy observation.

The Hexapod Compare model and its velocity tasks are implemented with native mjlab
managers and sensors. For the Compare flat and rough tasks, only the reward equations
and reward scales come from the corresponding ``mujoco_playground`` hexapod tasks.
The robot model, action scale, command generator, randomization, termination rules,
PPO configuration,
simulation settings, and reward aggregation remain mjlab-native; in particular, no
Playground aggregate reward clipping is added. Its initial height is
geometry-calibrated so the foot spheres begin tangent to the ground, and reset
height noise is limited to 1 cm to avoid artificial chassis impacts.

Four tasks are registered:

``Mjlab-Velocity-Flat-Hexapod``
   Flat-ground training for the new alternating-tripod gait configuration.

``Mjlab-Velocity-Flat-Hexapod-Compare``
   Flat-ground training and the supported native-MuJoCo sim2sim workflow.

``Mjlab-Velocity-Rough-Hexapod-Compare``
   Generated rough terrain with terrain and command curricula.

``Mjlab-Velocity-Flat-Hexapod-Safe``
   Flat-ground training migrated from the tuned ``el_4090/safe`` legged_gym
   task. It reuses the Hexapod Compare MJCF, the source task's reward
   equations and weights, 0.25 action scale, observation noise and scaling,
   PPO configuration, and symmetry augmentation. Despite the retained source
   name, this is ordinary PPO and contains no safety layer or action projection.

Policy interface
----------------

All four tasks retain the deployed MuJoCo Playground ordering. The 18 joint
entries are ordered by leg as ``RF, RM, RB, LF, LM, LB`` and by joint within
each leg as ``HAA, HFE, KFE``. All tasks use 66 actor entries:

``lin_vel[3], ang_vel[3], gravity[3], joint_pos[18], joint_vel[18],``
``last_action[18], command[3]``.

During training, uniform actor-observation noise follows the Playground-style
``level * scale`` convention. ``OBSERVATION_NOISE_LEVEL`` defaults to ``1.5``;
the scales for linear velocity, angular velocity, gravity, joint position, and
joint velocity are ``0.1``, ``0.2``, ``0.1``, ``0.05``, and ``1.5`` respectively.
Setting the level to ``0.0`` disables these noise terms. The critic and play
configuration remain noise-free.

The Compare actor keeps that Playground order, while the privileged critic remains
the native mjlab velocity-task structure. Foot contact force follows the same path
as the Go1 ``soft_landing`` term: a ``reduce="netforce"`` contact sensor exposes one
world-frame 3-D force per foot, and its reward magnitude is computed directly with
an L2 norm.

The task migrated from ``el_4090/safe`` also has exactly 66 inputs and 18
outputs in the ordering above. Coordinate signs are converted only inside its
action and observation terms. To map an MJCF joint coordinate to the source
URDF's all-positive-axis convention, multiply by:

``(+,-,-), (+,-,-), (+,-,+), (-,+,+), (-,+,+), (-,+,-)``

for ``RF, RM, RB, LF, LM, LB`` respectively. Thus the negated joints are
``RF_HFE``, ``RF_KFE``, ``RM_HFE``, ``RM_KFE``, ``RB_HFE``, ``LF_HAA``,
``LM_HAA``, ``LB_HAA``, and ``LB_KFE``. Do not apply another sign conversion
to the policy tensors.

Train and inspect
-----------------

Start with the flat task and verify that the robot can stand before increasing the
command range or enabling rough terrain:

.. code-block:: bash

   uv run train Mjlab-Velocity-Flat-Hexapod --env.scene.num-envs 1024 \
     --agent.logger wandb --agent.upload-model False \
     --agent.run-name tripod_v1
   uv run play Mjlab-Velocity-Flat-Hexapod \
     --checkpoint-file logs/rsl_rl/hexapod_velocity/<run>/model_<iteration>.pt

Like the Go1 task, the independent Hexapod task leaves policy actions unclipped
and saves a checkpoint every 500 iterations. The command curriculum starts at
``vx in [-0.3, 0.3] m/s``, ``vy in [-0.2, 0.2] m/s``, and
``yaw in [-0.25, 0.25] rad/s``; expand these only after the contact-pattern and
velocity-tracking curves have stabilized.

For the Compare task:

.. code-block:: bash

   uv run train Mjlab-Velocity-Flat-Hexapod-Compare --env.scene.num-envs 4096
   uv run play Mjlab-Velocity-Flat-Hexapod-Compare \
     --checkpoint-file logs/rsl_rl/hexapod_compare_velocity/<run>/model_<iteration>.pt

For the SAFE task:

.. code-block:: bash

   uv run train Mjlab-Velocity-Flat-Hexapod-Safe --env.scene.num-envs 4096
   uv run play Mjlab-Velocity-Flat-Hexapod-Safe \
     --checkpoint-file logs/rsl_rl/el_4090_safe_mjlab/<run>/model_<iteration>.pt

Every saved checkpoint also produces an ONNX policy in the run directory. Its
metadata records joint order, default pose, action scale, and observation layout.
Do not deploy a policy if ONNX export emitted a warning.

Native MuJoCo sim2sim
---------------------

Export the flat scene, then run the ONNX policy outside MuJoCo Warp:

.. code-block:: bash

   uv run export-scene Mjlab-Velocity-Flat-Hexapod-Compare \
     --output-dir export/hexapod_compare_flat
   uv run scripts/sim2sim/hexapod.py \
     --policy logs/rsl_rl/hexapod_compare_velocity/<run>/<run>.onnx \
     --model export/hexapod_compare_flat/scene.xml \
     --command-x 0.5

The runner reconstructs observations from ONNX metadata, leaves policy actions
unclipped unless the policy explicitly declares a clipping range, applies the
exported per-joint action scale and default pose, and steps native CPU MuJoCo at
200 Hz with a 50 Hz policy. It intentionally rejects a rough policy because the
native runner does not reproduce mjlab's terrain height scan.

The same runner recognizes the tuned task's ONNX metadata and reproduces its
joint-axis conversion and action clipping. The source task does not zero small
sampled commands, so this port does not add a command deadzone. It does not run
a safety filter:

.. code-block:: bash

   uv run export-scene Mjlab-Velocity-Flat-Hexapod-Safe \
     --output-dir export/hexapod_safe
   uv run scripts/sim2sim/hexapod.py \
     --policy logs/rsl_rl/el_4090_safe_mjlab/<run>/<run>.onnx \
     --model export/hexapod_safe/scene.xml \
     --command-x 0.5

Before sim2real, confirm the following values against the physical controller:

- joint order is ``RF``, ``RM``, ``RB``, ``LF``, ``LM``, ``LB``, with
  ``HAA``, ``HFE``, ``KFE`` within each leg;
- the controller uses the exported default joint positions and action scale;
- PD gains and torque limits match ``hexapod_compare_constants.py``;
- IMU angular velocity and gravity use the robot body frame; and
- the hardware command loop runs at 50 Hz and matches the exported action-clipping
  metadata; the independent Hexapod task does not clip actions.
