Hexapod velocity tasks
=======================

The crab-style Hexapod has an independent flat-ground task derived from the
Go1 velocity-task structure and recalibrated from its own MJCF. It uses the
model's 18-joint order, its XML-default joint pose, a
0.6-second gait clock, and alternating support triangles ``RF+RB+LM`` and
``RM+LF+LB``. The gait clock is internal to the contact-pattern reward and is
not part of the policy observation.

One task is registered:

``Mjlab-Velocity-Flat-Hexapod``
   Flat-ground training for the new alternating-tripod gait configuration.

Policy interface
----------------

The task uses an explicit 18-joint ordering. The joint
entries are ordered by leg as ``RF, RM, RB, LF, LM, LB`` and by joint within
each leg as ``HAA, HFE, KFE``. The actor uses 66 entries:

``lin_vel[3], ang_vel[3], gravity[3], joint_pos[18], joint_vel[18],``
``last_action[18], command[3]``.

During training, uniform actor-observation noise follows the Playground-style
``level * scale`` convention. ``OBSERVATION_NOISE_LEVEL`` defaults to ``1.5``;
the scales for linear velocity, angular velocity, gravity, joint position, and
joint velocity are ``0.1``, ``0.2``, ``0.1``, ``0.05``, and ``1.5`` respectively.
Setting the level to ``0.0`` disables these noise terms. The critic and play
configuration remain noise-free.

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

The runner reconstructs observations from ONNX metadata, leaves policy actions
unclipped unless the policy explicitly declares a clipping range, applies the
exported per-joint action scale and default pose, and steps native CPU MuJoCo at
200 Hz with a 50 Hz policy.

Before sim2real, confirm the following values against the physical controller:

- joint order is ``RF``, ``RM``, ``RB``, ``LF``, ``LM``, ``LB``, with
  ``HAA``, ``HFE``, ``KFE`` within each leg;
- the controller uses the exported default joint positions and action scale;
- PD gains and torque limits match ``hexapod_constants.py``;
- IMU angular velocity and gravity use the robot body frame; and
- the hardware command loop runs at 50 Hz and matches the exported action-clipping
  metadata; the independent Hexapod task does not clip actions.
