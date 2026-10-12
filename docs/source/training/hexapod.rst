Tethered hexapod
================

The migrated hexapod uses the standard mjlab velocity task instead of the former
``mujoco_playground`` environment. The actor observations, reward functions,
terrain curriculum, randomization events, termination terms, PPO runner, and ONNX
export therefore follow the same manager-based structure as the built-in legged
robots. Its initial height is geometry-calibrated so the foot spheres begin tangent
to the ground; reset height noise is limited to 1 cm to avoid artificial chassis
impacts.

Three tasks are registered:

``Mjlab-Velocity-Flat-Hexapod``
   Flat-ground training and the supported native-MuJoCo sim2sim workflow.

``Mjlab-Velocity-Rough-Hexapod``
   Generated rough terrain with terrain and command curricula.

``Mjlab-Velocity-Flat-Hexapod-Safe``
   Flat-ground training migrated from the tuned ``el_4090/safe`` legged_gym
   task. It uses the current tethered-hexapod MJCF, the source task's reward
   equations and weights, 0.25 action scale, observation noise and scaling,
   PPO configuration, and symmetry augmentation. Despite the retained source
   name, this is ordinary PPO and contains no safety layer or action projection.

Policy interface
----------------

All three tasks retain the deployed MuJoCo Playground ordering. The 18 joint
entries are ordered by leg as ``RF, RM, RB, LF, LM, LB`` and by joint within
each leg as ``HAA, HFE, KFE``. The first 66 actor entries are therefore:

``lin_vel[3], ang_vel[3], gravity[3], joint_pos[18], joint_vel[18],``
``last_action[18], command[3]``.

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

   uv run train Mjlab-Velocity-Flat-Hexapod --num-envs 4096
   uv run play Mjlab-Velocity-Flat-Hexapod \
     --checkpoint-file logs/rsl_rl/hexapod_velocity/<run>/model_<iteration>.pt

For the SAFE task:

.. code-block:: bash

   uv run train Mjlab-Velocity-Flat-Hexapod-Safe --num-envs 4096
   uv run play Mjlab-Velocity-Flat-Hexapod-Safe \
     --checkpoint-file logs/rsl_rl/el_4090_safe_mjlab/<run>/model_<iteration>.pt

Every saved checkpoint also produces an ONNX policy in the run directory. Its
metadata records joint order, default pose, action scale, and observation layout.
Do not deploy a policy if ONNX export emitted a warning.

Native MuJoCo sim2sim
---------------------

Export the flat scene, then run the ONNX policy outside MuJoCo Warp:

.. code-block:: bash

   uv run export-scene Mjlab-Velocity-Flat-Hexapod \
     --output-dir export/hexapod_flat
   uv run scripts/sim2sim/hexapod.py \
     --policy logs/rsl_rl/hexapod_velocity/<run>/<run>.onnx \
     --model export/hexapod_flat/scene.xml \
     --command-x 0.5

The runner reconstructs observations from ONNX metadata, clips policy actions to
the range stored in the policy metadata, applies the exported per-joint action
scale and default pose, and steps native CPU MuJoCo at 200 Hz with a 50 Hz
policy. It intentionally rejects a rough policy because the native runner does
not reproduce mjlab's terrain height scan.

The same runner recognizes the tuned task's ONNX metadata and reproduces its
joint-axis conversion, command deadzone, and action clipping. It does not run a
safety filter:

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
- PD gains and torque limits match ``hexapod_constants.py``;
- IMU angular velocity and gravity use the robot body frame; and
- the hardware command loop runs at 50 Hz and clips normalized actions.
