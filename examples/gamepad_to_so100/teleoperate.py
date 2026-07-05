# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teleoperate an SO100 follower arm in end-effector space with only a gamepad.

No leader arm is needed: gamepad stick/button input is mapped to end-effector deltas,
converted to joint commands through the same kinematics processor steps used by the
keyboard example (``examples/keyboard_to_so100/teleoperate.py``) and the HIL-SERL gym
environment.

The end-effector reference pose is computed from the previously *commanded* joints (via
the IK solution carried across ticks), not the measured ones - see the keyboard example's
docstring for why.

The SO100 has 5 joints, so on top of x/y/z the end-effector has exactly two controllable
orientation degrees of freedom: pitch and roll about the gripper axis. Both are enabled
here via ``use_orientation=True``.

Gamepad mapping (see `GamepadTeleop` / `GamepadControllerHID`):
    left stick          -> x / y translation
    right stick (up/down)   -> z
    right stick (left/right) -> roll about the gripper axis
    D-pad up / down     -> pitch nose up / down
    LT / RT             -> close / open gripper

Orientation and D-pad support are currently only implemented for the Logitech Dual Action
controller's HID report layout on macOS (vendor 0x046d, product 0xc216); other
controllers/platforms fall back to x/y/z + gripper only (see `gamepad_utils.py`).

For recording datasets, replaying episodes, and evaluating policies, see
``examples/keyboard_to_so100/`` - actions are recorded in end-effector space, so those
scripts work identically regardless of which teleop device produced the data (only
``record.py``'s teleop construction would need the swap shown here).
"""

import argparse
import logging
import math
import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    MapDeltaActionToRobotActionStep,
    RobotProcessorPipeline,
    TransitionKey,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Meters moved per control tick at full stick deflection (~0.12 m/s at 30 FPS).
EE_STEP_SIZE_M = 0.004

# Workspace box (meters, in the robot base frame) the end-effector target is clipped to.
# Adjust to your mounting: the SO100 reaches roughly 0.45 m.
EE_BOUNDS = {"min": [-0.45, -0.45, 0.02], "max": [0.45, 0.45, 0.45]}

# Gripper position change per tick for the discrete open/close commands. Depending on how
# your gripper was assembled and calibrated, open/close may be swapped: flip the sign.
GRIPPER_SPEED_FACTOR = 0.05

# Radians rotated per control tick for the D-pad's discrete pitch (~15 deg/s at 30 FPS).
# Roll is continuous (from the right stick), scaled by the same constant at full deflection.
EE_ROT_STEP_RAD = 0.0087

# The gamepad's delta_x/delta_y come from a fixed stick layout that has no idea where you're
# actually standing relative to the robot's base frame. If pushing "forward" doesn't move
# the arm away from you, or left/right feels swapped, rotate here - try 90, 180, or 270.
# (Same convention and same fix as examples/keyboard_to_so100/teleoperate.py.)
OPERATOR_VIEW_ROTATION_DEG = 0.0


def _apply_operator_view_rotation(action: RobotAction, degrees: float) -> RobotAction:
    """Rotate delta_x/delta_y in the ground plane so they match the operator's viewpoint."""
    if not degrees:
        return action
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = action["delta_x"], action["delta_y"]
    action["delta_x"] = dx * cos_t - dy * sin_t
    action["delta_y"] = dx * sin_t + dy * cos_t
    return action


def main(operator_view_rotation_deg: float = OPERATOR_VIEW_ROTATION_DEG):
    # Initialize the robot and teleoperator
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5A460814411", id="my_awesome_follower_arm", use_degrees=True
    )
    teleop_config = GamepadTeleopConfig(use_gripper=True, use_orientation=True)

    # Initialize the robot and teleoperator
    robot = SO100Follower(robot_config)
    teleop_device = GamepadTeleop(teleop_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Carry complementary data (the previous IK solution) across control ticks so
    # EEReferenceAndDelta can reference the previously commanded joints.
    pipeline_memory: dict = {}

    def to_transition_with_memory(action_observation: tuple[RobotAction, RobotObservation]):
        transition = robot_action_observation_to_transition(action_observation)
        transition[TransitionKey.COMPLEMENTARY_DATA] = pipeline_memory
        return transition

    # Build pipeline to convert gamepad deltas to ee pose action to joint action
    gamepad_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            # {delta_x/y/z, delta_pitch, delta_roll, gripper} -> {enabled, target_*, gripper_vel}
            MapDeltaActionToRobotActionStep(rotation_scale=EE_ROT_STEP_RAD),
            # Velocity-style control: per-tick deltas are applied relative to the
            # previously commanded pose (the IK solution from the last tick).
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": EE_STEP_SIZE_M, "y": EE_STEP_SIZE_M, "z": EE_STEP_SIZE_M},
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=False,
                use_ik_solution=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=EE_BOUNDS,
                max_ee_step_m=0.05,
            ),
            # Gamepad gripper commands are discrete: {0=close, 1=stay, 2=open}
            GripperVelocityToJoint(
                speed_factor=GRIPPER_SPEED_FACTOR,
                discrete_gripper=True,
            ),
            # Seed IK from its previous solution (not the measured joints) so the commanded
            # joints are deterministic for a stable target and cannot oscillate with the
            # motors. This step also publishes its solution to the complementary data.
            InverseKinematicsRLStep(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=to_transition_with_memory,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer (optional: the example runs fine without the viz extra)
    visualize = True
    try:
        init_rerun(session_name="gamepad_so100_teleop")
    except (ImportError, RuntimeError) as e:
        logging.warning(f"Rerun visualization unavailable ({e}), running without visualization.")
        visualize = False

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop. Use the gamepad to teleoperate the robot...")
    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = robot.get_observation()

            # Get teleop action
            gamepad_action = _apply_operator_view_rotation(
                teleop_device.get_action(), operator_view_rotation_deg
            )

            # Gamepad deltas -> EE pose -> Joints transition
            joint_action = gamepad_to_robot_joints_processor((gamepad_action, robot_obs))

            # Send action to robot
            _ = robot.send_action(joint_action)

            # Visualize
            if visualize:
                log_rerun_data(observation=gamepad_action, action=joint_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("Teleoperation stopped.")
    finally:
        robot.disconnect()
        teleop_device.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--operator-view-rotation-deg",
        type=float,
        default=OPERATOR_VIEW_ROTATION_DEG,
        help=(
            "Degrees to rotate delta_x/delta_y so they match where you're standing "
            "relative to the robot. If pushing 'forward' doesn't move the arm away from "
            "you, or left/right feels swapped, try 90, 180, or 270."
        ),
    )
    args = parser.parse_args()
    main(operator_view_rotation_deg=args.operator_view_rotation_deg)
