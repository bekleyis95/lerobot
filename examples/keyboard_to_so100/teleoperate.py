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

"""Teleoperate an SO100 follower arm in end-effector space with only a keyboard.

No leader arm is needed: keyboard key presses are mapped to end-effector deltas,
converted to joint commands through the same kinematics processor steps used by
the HIL-SERL gym environment.

The end-effector reference pose is computed from the previously *commanded* joints
(via the IK solution carried across ticks), not the measured ones. Measured joints
sag under gravity and jitter around the command, so feeding them back into the
target makes the end-effector drift and oscillate. Commanding relative to the last
commanded pose also bounds how far the target can lead the robot, so motion
reverses instantly at workspace limits instead of having to unwind an accumulated
overshoot.

The SO100 has 5 joints, so on top of x/y/z the end-effector has exactly two
controllable orientation degrees of freedom: pitch (where the nose points, e.g.
pointing down at a piece on the table) and roll about the gripper axis (which acts
as yaw of the jaws when the nose points down). Both are enabled here via
``use_orientation=True``. True world-yaw is coupled to the base pan and follows
from the commanded x/y position.

Key mapping (see `KeyboardEndEffectorTeleop`):
    arrow keys   -> x / y translation
    shift        -> z down          right shift -> z up
    i / k        -> pitch nose up / down
    j / l        -> roll about the gripper axis
    left ctrl    -> close gripper   right ctrl  -> open gripper
"""

import argparse
import logging
import math
import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import build_ee_delta_to_joints_pipeline
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
from lerobot.types import RobotAction
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Meters moved per control tick per unit keyboard delta (~0.12 m/s at 30 FPS).
EE_STEP_SIZE_M = 0.004

# Workspace box (meters, in the robot base frame) the end-effector target is clipped to.
# Adjust to your mounting: the SO100 reaches roughly 0.45 m.
EE_BOUNDS = {"min": [-0.45, -0.45, 0.02], "max": [0.45, 0.45, 0.45]}

# Gripper position change per tick for the discrete open/close keys. Depending on how
# your gripper was assembled and calibrated, open/close may be swapped: flip the sign.
GRIPPER_SPEED_FACTOR = 0.05

# Radians rotated per control tick per unit keyboard delta (~15 deg/s at 30 FPS).
EE_ROT_STEP_RAD = 0.0087

# The keyboard's delta_x/delta_y come from a fixed key layout that has no idea where you're
# actually standing relative to the robot's base frame. If pushing "forward" doesn't move
# the arm away from you, or left/right feels swapped, rotate here - try 90, 180, or 270.
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
    robot_config = SO100FollowerConfig(port="/dev/tty.usbmodem58FA0919081", id="randy", use_degrees=True)
    teleop_config = KeyboardEndEffectorTeleopConfig(use_gripper=True, use_orientation=True)

    # Initialize the robot and teleoperator
    robot = SO100Follower(robot_config)
    teleop_device = KeyboardEndEffectorTeleop(teleop_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Build pipeline to convert keyboard deltas to ee pose action to joint action.
    # `pipeline_memory` carries the previous IK solution across control ticks so
    # EEReferenceAndDelta can reference the previously commanded joints -- see
    # build_ee_delta_to_joints_pipeline's docstring for why. It isn't read
    # directly here; the pipeline closes over it internally on every call.
    keyboard_to_robot_joints_processor, pipeline_memory = build_ee_delta_to_joints_pipeline(
        kinematics_solver,
        list(robot.bus.motors.keys()),
        ee_step_size_m=EE_STEP_SIZE_M,
        ee_bounds=EE_BOUNDS,
        gripper_speed_factor=GRIPPER_SPEED_FACTOR,
        rotation_scale=EE_ROT_STEP_RAD,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Init rerun viewer (optional: the example runs fine without the viz extra)
    visualize = True
    try:
        init_rerun(session_name="keyboard_so100_teleop")
    except (ImportError, RuntimeError) as e:
        logging.warning(f"Rerun visualization unavailable ({e}), running without visualization.")
        visualize = False

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop. Use the keyboard to teleoperate the robot...")
    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = robot.get_observation()

            # Get teleop action
            keyboard_action = _apply_operator_view_rotation(
                teleop_device.get_action(), operator_view_rotation_deg
            )

            # Keyboard deltas -> EE pose -> Joints transition
            joint_action = keyboard_to_robot_joints_processor((keyboard_action, robot_obs))

            # Send action to robot
            _ = robot.send_action(joint_action)

            # Visualize
            if visualize:
                log_rerun_data(observation=keyboard_action, action=joint_action)

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
