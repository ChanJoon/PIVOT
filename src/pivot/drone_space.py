"""
Drone Action Space for PIVOT

Copied and adapted from see-point-fly_base to remove external dependency.
Handles conversion of camera-relative waypoints to AirSim commands.
"""

import math
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import yaml
import os


@dataclass
class ActionPoint:
    """Represents a relative movement action in 3D space (camera-relative coordinates)"""
    dx: float  # Lateral (camera x-axis: negative=left, positive=right)
    dy: float  # Forward (camera y-axis: positive=forward)
    dz: float  # Vertical (camera z-axis: positive=up)
    action_type: str
    screen_x: float = 0.0  # 2D projected coordinates for visualization
    screen_y: float = 0.0
    done: bool = False  # Flag indicating mission/goal completion

    def __str__(self):
        return f"Action({self.action_type}): Move({self.dx:.1f}, {self.dy:.1f}, {self.dz:.1f})"


class DroneActionSpace:
    """Base class for drone action spaces"""
    def __init__(self, n_samples: int = 8):
        self.n_samples = n_samples
        self.movement_unit = 1.0
        self.camera_fov = 90.0
        self.max_movement = 2.0

    def action_to_commands(self, action: ActionPoint) -> List[Tuple[str, dict]]:
        """Convert a relative movement action into drone commands"""
        raise NotImplementedError("Subclasses must implement action_to_commands")


class AirSimDroneActionSpace(DroneActionSpace):
    """
    AirSim-specific action space

    Converts camera-relative waypoints (dx, dy, dz) into AirSim commands:
    1. Rotate to face the target direction (yaw)
    2. Move forward with vertical component

    IMPORTANT: Expects camera-relative coordinates where:
    - dx: lateral (negative=left, positive=right)
    - dy: forward (positive=forward)
    - dz: vertical (positive=up)

    No camera-to-body transformation is needed - AirSim's body frame
    already aligns with the camera frame for forward-facing cameras.
    """

    def __init__(self, n_samples: int = 8, config_path: str = "config_pivot.yaml"):
        super().__init__(n_samples)

        # Load configuration
        config = self._load_config(config_path)
        self.base_velocity = config.get("base_velocity", 2.0)
        self.base_yaw_rate = config.get("base_yaw_rate", 30.0)
        self.min_command_duration = config.get("min_command_duration", 2.0)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[DroneSpace] Warning: Could not load config from {config_path}: {e}")
        return {}

    def sample_actions(self) -> List[ActionPoint]:
        """Sample possible relative movements from current position (0,0,0)"""
        actions = []

        for _ in range(self.n_samples):
            distance = np.random.uniform(0.5, self.max_movement)
            azimuth = np.random.uniform(-self.camera_fov / 2, self.camera_fov / 2)
            elevation = np.random.uniform(-self.camera_fov / 4, self.camera_fov / 4)

            dx = (
                distance
                * math.cos(math.radians(elevation))
                * math.sin(math.radians(azimuth))
            )
            dy = (
                distance
                * math.cos(math.radians(elevation))
                * math.cos(math.radians(azimuth))
            )
            dz = distance * math.sin(math.radians(elevation))

            action = ActionPoint(dx, dy, dz, "move")
            actions.append(action)

        return actions

    def action_to_commands(self, action: ActionPoint) -> List[Tuple[str, dict]]:
        """
        Convert a relative movement action into AirSim API commands

        Strategy: Rotate first to face target, then move forward

        Args:
            action: ActionPoint with camera-relative coordinates (dx, dy, dz)

        Returns:
            List of (command_type, parameters) tuples
        """
        commands = []

        total_distance = math.sqrt(action.dx**2 + action.dy**2 + action.dz**2)

        if total_distance < 0.01:
            return []

        # Step 1: Calculate yaw angle needed to face the target
        distance_xy = math.sqrt(action.dx**2 + action.dy**2)

        if distance_xy > 0.01:
            # Calculate target angle from dx, dy
            # atan2(dx, dy) gives angle where dx=lateral, dy=forward
            target_angle = math.degrees(math.atan2(action.dx, action.dy))

            # Normalize to -180 to 180 range
            if target_angle > 180:
                target_angle -= 360
            elif target_angle < -180:
                target_angle += 360

            print(f"[DroneSpace] dx={action.dx:.3f}, dy={action.dy:.3f}, target_angle={target_angle:.1f}°")

            # Add rotation command if angle is significant
            if abs(target_angle) > 10:
                commands.append(
                    ("rotate_yaw", {"angle": target_angle, "yaw_rate": self.base_yaw_rate})
                )
                print(f"[DroneSpace] Adding rotation: {target_angle:.1f}°")
            else:
                print(f"[DroneSpace] Target angle {target_angle:.1f}° within threshold, no yaw")

        # Step 2: Move forward after rotation (or if no rotation needed)
        # Use full velocity for forward movement
        vx = self.base_velocity  # Forward velocity (body frame)
        vy = 0  # No sideways movement (rotate first, then move forward)

        # Calculate vertical velocity to maintain correct angle
        if distance_xy > 0.01:
            vz = (-action.dz / distance_xy) * self.base_velocity  # Negative because AirSim z is down
            duration = max(distance_xy / self.base_velocity, self.min_command_duration)
        else:
            vz = 0
            duration = self.min_command_duration

        print(f"[DroneSpace] Movement: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, duration={duration:.2f}s")

        commands.append(
            (
                "move_velocity_body",
                {
                    "vx": vx,
                    "vy": vy,
                    "vz": vz,
                    "duration": duration,
                    "yaw_rate": 0,  # No yaw during movement, already rotated
                },
            )
        )

        return commands
