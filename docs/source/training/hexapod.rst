Hexapod velocity tasks
=======================

The crab-style Hexapod has an independent flat-ground task derived from the
Go1 velocity-task structure and recalibrated from its own MJCF. It uses the
model's 18-joint order, its XML-default joint pose, a
0.6-second gait clock, and alternating support triangles ``RF+RB+LM`` and
``RM+LF+LB``. The gait clock is internal to the contact-pattern reward and is
not part of the policy observation.

Two flat-ground tasks are registered:

``Mjlab-Velocity-Flat-Hexapod``
   Flat-ground training for the new alternating-tripod gait configuration.

``Mjlab-Velocity-Flat-Hexapod-Tethered``
   The same locomotion task and scales, with a passive shoulder, spatial tendon,
   and one additional policy output for coordinated cable tension control.

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

The tethered actor uses a 71-entry sim-to-real observation layout. It has the
same core ordering as the Playground joystick, with ``last_action`` enlarged
to 19 entries, followed by cable tension in Newtons and a three-component unit
tether direction in the trunk frame. The direction is reconstructed from the
two hardware-observable angles: ``ARM_JOINT`` is yaw and
``atan2(tether_z, tether_x)`` in the ARM frame is pitch. Because the XML joint
axis is ``(0, 0, -1)``, the conversion is
``[cos(pitch) cos(yaw), -cos(pitch) sin(yaw), sin(pitch)]``. Angular noise is
applied before reconstruction so the observation remains unit length.

Its action is ``joint_target[18], cable[1]``. The cable action is clipped to
``[0, 2]`` and the XML gain of ``-200`` converts it to ``[0, -400] N``;
negative MuJoCo actuator force is pulling, while the policy observes the
corresponding non-negative tension. The tethered task keeps MJLab's observation
noise and encoder-bias randomization, except that the projected-gravity noise
scale is ``0.05``. It does not globally clip policy outputs; only the cable term
is physically clipped to ``[0, 2]``. The action-history term uses MJLab's
standard ``last_action`` timing. Checkpoints trained with the previous 70-entry
angle layout are not compatible with this 71-entry policy and must be retrained.

The tethered critic has 110 entries. Its first 71 entries use the actor layout
with clean, unbiased state values. It keeps the independent Hexapod's foot
height, air-time, contact, and contact-force privileged observations, then adds
clean ARM position/velocity and tendon length.

The tethered robot defines its HAA/HFE/KFE motor and PD configurations locally
in ``hexapod_tethered_constants.py``. They currently use the same ``Kp=100``
and ``Kd=3`` values as the independent robot, but the configuration objects and
action-scale mapping are separate so either robot can be tuned independently.

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

Train or play the tethered policy with:

.. code-block:: bash

   uv run train Mjlab-Velocity-Flat-Hexapod-Tethered \
     --env.scene.num-envs 1024 --agent.logger wandb \
     --agent.upload-model False --agent.run-name tether_flat_v1
   uv run play Mjlab-Velocity-Flat-Hexapod-Tethered \
     --checkpoint-file \
     logs/rsl_rl/hexapod_tethered_velocity/<run>/model_<iteration>.pt

The tethered task inherits the independent MJLab Hexapod timing: the simulator
runs at 200 Hz (``0.005 s``) and the shared policy at 50 Hz (four physics steps
per policy action). At every reset, the world-frame tether anchor is sampled
uniformly from ``x in [-10, 10] m``, ``y in [-5, 5] m``, and
``z in [0.1, 0.5] m``; the XML position ``(5, 0, 2) m`` is only the model
default before the reset event runs.

On a level-only training task, ``tether_flat_penalty=-3`` intentionally makes
the useful solution keep the cable slack. The uphill and slack terms become
active only when the body-tilt proxy crosses roughly ``0.30 rad``. A policy
that learns sustained adaptive uphill pulling therefore also needs a later
slope/terrain task; this flat migration primarily validates the 19-D control
interface, tension sensing, and leg--tether coordination path.

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

The tethered asset can be exported independently with
``uv run export-scene hexapod_tethered``. Its XML-defined cable actuator is
internally ordered before MJLab's generated leg actuators, so external native
MuJoCo programs must resolve actuator IDs by name instead of copying the
19-entry policy vector directly into ``data.ctrl``. The current
``scripts/sim2sim/hexapod.py`` supports only the independent 18-action task;
a tether-aware deployment adapter must additionally reconstruct cable tension,
the trunk-frame unit tether direction from measured yaw and pitch, and the
19-entry action history, then run the native simulator at ``0.005 s`` with a
``0.02 s`` policy period.

For the independent task, the existing runner reconstructs observations from
ONNX metadata, leaves policy actions unclipped unless the policy explicitly
declares a clipping range, applies the exported per-joint action scale and
default pose, and steps native CPU MuJoCo at 200 Hz with a 50 Hz policy.

Before sim2real, confirm the following values against the physical controller:

- joint order is ``RF``, ``RM``, ``RB``, ``LF``, ``LM``, ``LB``, with
  ``HAA``, ``HFE``, ``KFE`` within each leg;
- the controller uses the exported default joint positions and action scale;
- PD gains and torque limits match ``hexapod_constants.py``;
- IMU angular velocity and gravity use the robot body frame; and
- the hardware command loop runs at 50 Hz and matches the exported action-clipping
  metadata; the independent Hexapod task does not clip actions.
