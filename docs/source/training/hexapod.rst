Hexapod velocity tasks
=======================

The crab-style Hexapod has an independent flat-ground task derived from the
Go1 velocity-task structure and recalibrated from its own MJCF. It uses the
model's 18-joint order, its XML-default joint pose, a
0.6-second gait clock, and alternating support triangles ``RF+RB+LM`` and
``RM+LF+LB``. The gait clock is internal to the contact-pattern reward and is
not part of the policy observation.

Three Hexapod tasks are registered:

``Mjlab-Velocity-Flat-Hexapod``
   Flat-ground training for the new alternating-tripod gait configuration.

``Mjlab-Velocity-Flat-Hexapod-Tethered``
   The same locomotion task and scales, with a passive shoulder, spatial tendon,
   and one additional policy output for coordinated cable tension control.

``Mjlab-Velocity-Slope-Hexapod-Tethered``
   The tethered policy interface on a fixed 22-degree plane, with an active
   0--400 N pulling range and four additional slope-specific tether rewards.

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
joint velocity are ``0.1``, ``0.2``, ``0.05``, ``0.05``, and ``1.5``
respectively.
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
``[0, 2]``. The flat XML limits pulling force to only ``0.0001 N``, so that task
is effectively force-free even though the actuator gain remains ``-200``. The
independent slope XML instead allows ``[-400, 0] N``: the gain maps cable action
``[0, 2]`` to ``[0, -400] N``. Negative MuJoCo actuator force is pulling, while
the policy observes the corresponding non-negative tension. Both tethered tasks
keep MJLab's observation noise and encoder-bias randomization; the
projected-gravity noise scale is ``0.05``. They do not globally clip policy
outputs; only the cable term is physically clipped to ``[0, 2]``. The
action-history term uses MJLab's standard ``last_action`` timing. Checkpoints
trained with the previous 70-entry angle layout are not compatible with this
71-entry policy and must be retrained.

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

Train or play the fixed-slope policy with:

.. code-block:: bash

   uv run train Mjlab-Velocity-Slope-Hexapod-Tethered \
     --env.scene.num-envs 1024 --agent.logger wandb \
     --agent.upload-model False --agent.run-name tether_slope_22deg_v1 \
     --agent.pretrained-checkpoint \
     /absolute/path/to/flat_tethered/model_<iteration>.pt
   uv run play Mjlab-Velocity-Slope-Hexapod-Tethered \
     --checkpoint-file \
     logs/rsl_rl/hexapod_tethered_slope_velocity/<run>/model_<iteration>.pt

``--agent.pretrained-checkpoint`` is transfer learning, not resume. The source
must be a ``Flat-Hexapod-Tethered`` checkpoint with the same 71 observations and
19 actions. It loads the learned actor and observation statistics, keeps all 18
leg outputs, and reinitializes the cable output to a mean action of ``0.1`` and
a standard deviation of ``0.2``. The critic, optimizer, PPO iteration, and
``common_step_counter`` start from zero, so the slope command curriculum starts
at its first stage. Use ``--agent.resume True`` only to continue an interrupted
slope run; the two options are intentionally mutually exclusive.

Both tethered tasks inherit the independent MJLab Hexapod timing: the simulator
runs at 200 Hz (``0.005 s``) and the shared policy at 50 Hz (four physics steps
per policy action). On the flat task, the world-frame tether anchor is sampled
at every reset from ``x in [-10, 10] m``, ``y in [-5, 5] m``, and
``z in [0.1, 0.5] m``. On the slope, its local horizontal coordinates are
sampled from ``x in [3, 13] m`` and ``y in [-2, 2] m``, with a normal distance
of ``[0, 0.3] m`` above the plane. The XML position ``(5, 0, 2) m`` is only the
model default before the reset event runs. This last range is signed distance
along the upward slope normal, rather than world-Z height, so it never samples
below the plane. A slope episode terminates when
``trunk_x + 0.3 > anchor_x`` in world coordinates; equality alone does not
trigger the condition.

On the level task, ``tether_flat_penalty=-3`` intentionally makes the useful
solution keep the cable slack. Its ``0.0001 N`` force cap makes this task useful
for validating the 19-D interface without learning meaningful traction.

The slope task rotates the plane and initial trunk pose by 22 degrees, with
uphill along world ``+X``. Its robot XML includes the same common model and STL
meshes as the flat asset but owns the wider cable force range. It preserves the
flat task's 71-entry actor, 110-entry critic, and 19-entry action interface. Its
PPO class also owns independent actor, critic, algorithm, and runner settings,
although their initial values match the flat configuration.

The slope task separately overrides four terrain-sensitive measurements while
retaining the flat reward weights. ``upright`` uses the terrain normal estimated
by ``foot_height_scan`` instead of world vertical. ``foot_clearance`` and
``foot_swing_height`` use the center ray under each foot and measure the
foot-to-hit displacement along that ray's surface normal. ``foot_slip`` removes
the normal component of the foot velocity and penalizes the remaining velocity
in the surface tangent plane. The flat task continues to use the generic
world-Z height and world-XY slip functions.

The fixed-slope task disables the flat-only tension proxy terms and retains a
``tether_action_rate`` weight of ``-0.1``. It uses the following
Playground-derived kernels with scales recalibrated for MJLab's signed reward
sum:

- ``tether_work_assist=0.5`` rewards positive tether power;
- ``tether_eng_steep=0.5`` rewards reaching the slope-dependent target tension;
- ``tether_slack_penalty=-1.0`` penalizes tension below ``15 N``; and
- ``tether_overload=-2.0`` penalizes tension above ``250 N``.

The first three terms are active only when the planar velocity command points
toward the anchor. Overload protection is always active. This prevents standing
and downhill commands from learning a permanent cable load. Tension mean, P95,
slack fraction, overload fraction, cable action statistics, and action-limit
saturation fractions are logged to W&B under ``Metrics/``.

Like the Go1 task, the independent Hexapod task leaves policy actions unclipped
and saves a checkpoint every 500 iterations. The slope-local command curriculum
uses its own zeroed ``common_step_counter`` after transfer:

- iteration 0: ``vx=[0.05, 0.3]``, ``vy=[-0.05, 0.05]``,
  ``yaw=[-0.1, 0.1]``;
- iteration 3000: ``vx=[0.0, 0.5]``, ``vy=[-0.2, 0.2]``,
  ``yaw=[-0.25, 0.25]``; and
- iteration 6000: ``vx=[-0.8, 0.8]``, ``vy=[-0.5, 0.5]``,
  ``yaw=[-0.8, 0.8]``.

The slope PPO learning rate is ``3e-4``. Random velocity pushes are suppressed
for the first 1000 iterations and then use the standard Hexapod disturbance
range. At reset, initial yaw is limited to ``[-0.15, 0.15] rad`` around world
``+X`` so the first forward-only curriculum stage actually points uphill toward
the anchor. Fall termination compares trunk up with the 22-degree surface
normal, not world vertical.

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
