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

"""Scripted, non-interactive showcase of SO100 end-effector control: no keyboard needed.

Demonstrates the same 5-DOF end-effector control as ``teleoperate.py`` (3 translation +
pitch + roll), but driven by a scripted trajectory instead of key presses, so it makes a
good smoke test after calibration or a quick capability demo to record.

The arm first closes its gripper and moves to a fixed HOME_JOINT_ANGLES_DEG pose - reached
out and elevated, well clear of the table/mount - via a direct joint-space interpolation
(no IK needed, since the target is already specified in joint space). Every phase below is
then defined as an offset from that home pose, and every phase starts and ends at zero
offset, so phases chain smoothly:

    1. circle       - TCP traces a small circle, orientation held fixed
    2. nod          - pitch oscillates (nose up/down), TCP position held fixed
    3. twist        - roll oscillates (about the gripper's own axis), TCP position held fixed
    4. figure_eight - TCP traces a figure-eight, orientation held fixed
    5. corkscrew    - combined: TCP bobs up/down while rolling, at once

IK during the phases is seeded from the previous tick's solution (not the measured
joints), the same continuity fix used by ``teleoperate.py``/``record.py``, so the motion
is smooth rather than fighting the motors every tick.
"""

import time

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation

FPS = 30

# Amplitudes are kept well inside the SO100's ~0.4 m reach so the demo is safe to run
# from any reachable starting pose without checking workspace bounds first; the IK task
# is soft, so if a target ever is out of reach the arm eases toward it instead of faulting.
CIRCLE_RADIUS_M = 0.06
FIGURE_EIGHT_HALF_WIDTH_M = 0.06
BOB_HEIGHT_M = 0.04
PITCH_AMPLITUDE_RAD = np.deg2rad(25)
ROLL_AMPLITUDE_RAD = np.deg2rad(25)

# Fixed home pose: reached out and elevated (FK: ~0.30 m forward, ~0.16 m up, nose angled
# down), well clear of the table/mount so the showcase always has room to move. Defined in
# joint space (adjust for your mounting/workspace) so homing is a direct motor command,
# with no IK/reachability question. The gripper closes on the way to this pose.
HOME_JOINT_ANGLES_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -40.0,
    "elbow_flex": 40.0,
    "wrist_flex": 20.0,
    "wrist_roll": 0.0,
}
HOME_GRIPPER_POS = 100.0  # 0=open, 100=closed (see GripperVelocityToJoint)
HOME_DURATION_S = 3.0


def circle(t: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """TCP traces a circle in the local XY plane; orientation unchanged."""
    theta = 2 * np.pi * t / duration
    dp = np.array([CIRCLE_RADIUS_M * (np.cos(theta) - 1.0), CIRCLE_RADIUS_M * np.sin(theta), 0.0])
    return dp, np.zeros(3)


def nod(t: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Pitch oscillates (nose up/down) two full nods; TCP position unchanged."""
    theta = 2 * np.pi * 2 * t / duration
    dr = np.array([0.0, PITCH_AMPLITUDE_RAD * np.sin(theta), 0.0])
    return np.zeros(3), dr


def twist(t: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Roll oscillates about the gripper's own axis two full twists; TCP position unchanged."""
    theta = 2 * np.pi * 2 * t / duration
    dr = np.array([0.0, 0.0, ROLL_AMPLITUDE_RAD * np.sin(theta)])
    return np.zeros(3), dr


def figure_eight(t: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """TCP traces a figure-eight (lemniscate) in the local XZ plane; orientation unchanged."""
    theta = 2 * np.pi * t / duration
    dp = np.array(
        [
            FIGURE_EIGHT_HALF_WIDTH_M * np.sin(theta),
            0.0,
            BOB_HEIGHT_M * np.sin(2 * theta),
        ]
    )
    return dp, np.zeros(3)


def corkscrew(t: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Combined move: TCP bobs up/down while rolling, at the same time."""
    theta = 2 * np.pi * 2 * t / duration
    dp = np.array([0.0, 0.0, BOB_HEIGHT_M * np.sin(theta)])
    dr = np.array([0.0, 0.0, ROLL_AMPLITUDE_RAD * np.sin(theta)])
    return dp, dr


PHASES = [
    ("circle", 6.0, circle),
    ("nod", 6.0, nod),
    ("twist", 6.0, twist),
    ("figure_eight", 6.0, figure_eight),
    ("corkscrew", 6.0, corkscrew),
]


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _go_home(robot, home_deg: dict, gripper_closed_pos: float, duration: float) -> None:
    """Ease every joint and the gripper from wherever they are to `home_deg`/closed, in joint space."""
    obs = robot.get_observation()
    start_deg = {n: obs[f"{n}.pos"] for n in home_deg}
    start_gripper = obs["gripper.pos"]

    n_ticks = max(int(duration * FPS), 1)
    for tick in range(n_ticks):
        t0 = time.perf_counter()
        s = _smoothstep((tick + 1) / n_ticks)
        action = {f"{n}.pos": start_deg[n] + s * (target - start_deg[n]) for n, target in home_deg.items()}
        action["gripper.pos"] = start_gripper + s * (gripper_closed_pos - start_gripper)
        robot.send_action(action)
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


def main():
    # Initialize the robot
    robot_config = SO100FollowerConfig(port="/dev/tty.usbmodem58FA0919081", id="randy", use_degrees=True)
    robot = SO100Follower(robot_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    #   https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    robot.connect()
    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    try:
        # Remember the pose the arm was in before homing: disconnecting disables torque
        # by default (SO100FollowerConfig.disable_torque_on_disconnect), so we return here
        # before disconnecting rather than leaving the arm hanging limp from the elevated,
        # reached-out home pose.
        obs_before = robot.get_observation()
        park_deg = {n: obs_before[f"{n}.pos"] for n in HOME_JOINT_ANGLES_DEG}

        print("Closing gripper and moving to home pose...")
        _go_home(robot, HOME_JOINT_ANGLES_DEG, HOME_GRIPPER_POS, HOME_DURATION_S)

        # The fixed home pose (not wherever the arm happened to start) is the reference:
        # every phase is an offset from this, and every phase returns to zero offset. The
        # gripper target stays the constant HOME_GRIPPER_POS rather than a post-home
        # observation: the servo is still settling into fully closed at that instant, and
        # freezing that lagging reading would hold the gripper just short of closed for
        # the whole rest of the run instead of fully closing it.
        obs = robot.get_observation()
        q_curr = np.array([obs[f"{n}.pos"] for n in robot.bus.motors])
        gripper_pos = HOME_GRIPPER_POS
        t_home = kinematics_solver.forward_kinematics(q_curr)
        p_home, r_home = t_home[:3, 3].copy(), t_home[:3, :3].copy()

        print(f"Home pose: pos={np.round(p_home, 3)}")
        print("Showcasing translation, pitch, and roll control. Press Ctrl+C to stop early.")
        for name, duration, offset_fn in PHASES:
            print(f"Phase: {name} ({duration:.0f}s)")
            n_ticks = int(duration * FPS)
            for tick in range(n_ticks):
                t0 = time.perf_counter()
                t = tick / FPS

                dp, dr = offset_fn(t, duration)
                r_delta = Rotation.from_rotvec(dr).as_matrix()

                target = np.eye(4)
                target[:3, :3] = r_home @ r_delta
                target[:3, 3] = p_home + dp

                q_curr = kinematics_solver.inverse_kinematics(q_curr, target)
                action = {f"{n}.pos": float(q_curr[i]) for i, n in enumerate(robot.bus.motors)}
                action["gripper.pos"] = float(gripper_pos)
                robot.send_action(action)

                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

        print("Returning to a safe resting pose before disconnecting...")
        _go_home(robot, park_deg, HOME_GRIPPER_POS, HOME_DURATION_S)

        print("Showcase complete.")
    except KeyboardInterrupt:
        print("Showcase stopped early.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
