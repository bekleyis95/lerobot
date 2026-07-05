#!/usr/bin/env python

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

import logging
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _hidapi_available, _pygame_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..utils import TeleopEvents

if TYPE_CHECKING or _pygame_available:
    import pygame
else:
    pygame = None  # type: ignore[assignment]

if TYPE_CHECKING or _hidapi_available:
    import hid
else:
    hid = None  # type: ignore[assignment]

# USB product ID for the Logitech Dual Action, which reports a different HID layout than
# the Logitech RumblePad 2 the rest of GamepadControllerHID's parsing assumes.
LOGITECH_DUAL_ACTION_PRODUCT_ID = 0xC216


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def get_rotation_deltas(self):
        """Get the current orientation deltas (droll, dpitch), normalized to [-1, 1].

        Not every controller/mode supports orientation; defaults to no rotation.
        """
        return 0.0, 0.0

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        if not pynput_can_capture():
            logging.warning(
                "Keyboard control is unavailable in this environment. pynput cannot capture keys "
                "on Wayland or headless machines, or on macOS without Accessibility / Input "
                "Monitoring permission. Keyboard motion will be inactive."
            )
            self.running = False
            return

        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = TeleopEvents.FAILURE
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z


class GamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        require_package("pygame", extra="gamepad")
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False

    def start(self):
        """Initialize pygame and the gamepad."""
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info(f"Initialized gamepad: {self.joystick.get_name()}")

        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print("  B/Circle button: Exit")
        print("  Y/Triangle button: End episode with SUCCESS")
        print("  A/Cross button: End episode with FAILURE")
        print("  X/Square button: Rerecord episode")

    def stop(self):
        """Clean up pygame resources."""
        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 3:
                    self.episode_end_status = TeleopEvents.SUCCESS
                # A button (1) for failure
                elif event.button == 1:
                    self.episode_end_status = TeleopEvents.FAILURE
                # X button (0) for rerecord
                elif event.button == 0:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE

                # RB button (6) for closing gripper
                elif event.button == 6:
                    self.close_gripper_command = True

                # LT button (7) for opening gripper
                elif event.button == 7:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [0, 2, 3]:
                    self.episode_end_status = None

                elif event.button == 6:
                    self.close_gripper_command = False

                elif event.button == 7:
                    self.open_gripper_command = False

            # Check for RB button (typically button 5) for intervention flag
            if self.joystick.get_button(5):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        try:
            # Read joystick axes
            # Left stick X and Y (typically axes 0 and 1)
            y_input = self.joystick.get_axis(0)  # Up/Down (often inverted)
            x_input = self.joystick.get_axis(1)  # Left/Right

            # Right stick Y (typically axis 3 or 4)
            z_input = self.joystick.get_axis(3)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Calculate deltas (note: may need to invert axes depending on controller)
            delta_x = -x_input * self.x_step_size  # Forward/backward
            delta_y = -y_input * self.y_step_size  # Left/right
            delta_z = -z_input * self.z_step_size  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
        require_package("hidapi", extra="gamepad", import_name="hid")
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # D-pad hat state (Dual Action only); 8 = neutral, see _parse_dual_action_report.
        self.dpad_hat = 8

        # Button states
        self.buttons = {}

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        logging.error(
            "No gamepad found, check the connection and the product string in HID to add your gamepad"
        )
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            logging.info(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            logging.info(f"Connected to {manufacturer} {product}")

            logging.info("Gamepad controls (HID mode):")
            logging.info("  Left analog stick: Move in X-Y plane")
            logging.info("  Right analog stick: Move in Z axis (vertical)")
            logging.info("  Button 1/B/Circle: Exit")
            logging.info("  Button 2/A/Cross: End episode with SUCCESS")
            logging.info("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            logging.error(f"Error opening gamepad: {e}")
            logging.error("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            if not data or len(data) < 8:
                return

            # Report layout varies by controller model/mode; dispatch on product ID.
            product_id = (self.device_info or {}).get("product_id")
            if product_id == LOGITECH_DUAL_ACTION_PRODUCT_ID:
                self._parse_dual_action_report(data)
            else:
                self._parse_rumblepad2_report(data)

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def _parse_rumblepad2_report(self, data):
        """Byte offsets for the Logitech RumblePad 2."""
        # Normalize joystick values from 0-255 to -1.0-1.0
        self.left_y = (data[1] - 128) / 128.0
        self.left_x = (data[2] - 128) / 128.0
        self.right_x = (data[3] - 128) / 128.0
        self.right_y = (data[4] - 128) / 128.0
        self._apply_deadzone()

        # Parse button states (byte 5 in the Logitech RumblePad 2)
        buttons = data[5]

        # Check if RB is pressed then the intervention flag should be set
        self.intervention_flag = data[6] in [2, 6, 10, 14]

        # Check if RT is pressed
        self.open_gripper_command = data[6] in [8, 10, 12]

        # Check if LT is pressed
        self.close_gripper_command = data[6] in [4, 6, 12]

        # Check if Y/Triangle button (bit 7) is pressed for saving
        # Check if X/Square button (bit 5) is pressed for failure
        # Check if A/Cross button (bit 4) is pressed for rerecording
        if buttons & 1 << 7:
            self.episode_end_status = TeleopEvents.SUCCESS
        elif buttons & 1 << 5:
            self.episode_end_status = TeleopEvents.FAILURE
        elif buttons & 1 << 4:
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE
        else:
            self.episode_end_status = None

    def _parse_dual_action_report(self, data):
        """Byte offsets for the Logitech Dual Action (vendor 0x046d, product 0xc216).

        Reverse-engineered from raw report dumps (a different, simpler layout than the
        RumblePad 2's): sticks are data[0:4] in natural order, and buttons are bit-packed
        into data[4] (D-pad hat in the low nibble, face buttons in the high nibble) and
        data[5] (stick-clicks and triggers) - unlike the RumblePad 2, data[6]/data[7] are
        unused by this controller.

        Right stick X and the D-pad up/down are otherwise-unused inputs on this layout, so
        they double as end-effector roll/pitch for orientation control (see
        `get_rotation_deltas`).
        """
        self.left_x = (data[0] - 128) / 128.0
        self.left_y = (data[1] - 128) / 128.0
        self.right_x = (data[2] - 128) / 128.0
        self.right_y = (data[3] - 128) / 128.0
        self._apply_deadzone()

        face_buttons = data[4]
        other_buttons = data[5]

        # Standard HID hat switch in the low nibble: 0=up, 2=right, 4=down, 6=left, 8=neutral.
        self.dpad_hat = face_buttons & 0x0F

        # LT = bit 2 (close), RT = bit 3 (open); flip if your controller's grip feels backwards.
        self.close_gripper_command = bool(other_buttons & (1 << 2))
        self.open_gripper_command = bool(other_buttons & (1 << 3))

        # No dedicated intervention button identified for this controller; R3 (bit 1) is
        # the closest spare, mirroring the RumblePad 2's use of a shoulder button.
        self.intervention_flag = bool(other_buttons & (1 << 1))

        # Y (bit 7) for saving, X (bit 4) for failure, A (bit 5) for rerecording.
        if face_buttons & (1 << 7):
            self.episode_end_status = TeleopEvents.SUCCESS
        elif face_buttons & (1 << 4):
            self.episode_end_status = TeleopEvents.FAILURE
        elif face_buttons & (1 << 5):
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE
        else:
            self.episode_end_status = None

    def _apply_deadzone(self):
        self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
        self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
        self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
        self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_x * self.x_step_size  # Forward/backward
        delta_y = -self.left_y * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z

    def get_rotation_deltas(self):
        """Get the current orientation deltas: roll from the right stick X, pitch from
        the D-pad up/down (bit-packed hat: 0=up, 4=down, 8=neutral - see
        `_parse_dual_action_report`). Pitch only works on the Dual Action layout; the
        RumblePad 2 path never updates `dpad_hat`, so pitch reads as 0 there.
        """
        droll = self.right_x
        if self.dpad_hat == 0:
            dpitch = 1.0
        elif self.dpad_hat == 4:
            dpitch = -1.0
        else:
            dpitch = 0.0
        return droll, dpitch
