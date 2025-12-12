"""
Trajectory Executor for PIVOT

Executes selected trajectories using AirSim with camera-relative coordinates.
"""

import airsim
import time
import math
from typing import Dict, Any, Optional

from .trajectory import Trajectory, Waypoint
from .drone_space import AirSimDroneActionSpace, ActionPoint


class TrajectoryExecutor:
    """
    Executes selected PIVOT trajectories using AirSim

    Converts PIVOT Trajectory objects to ActionPoints,
    then uses AirSimDroneActionSpace to convert to AirSim commands.

    IMPORTANT: Waypoints are camera-relative coordinates.
    No coordinate transformation is needed - AirSim's body frame
    aligns with the camera frame for forward-facing cameras.
    """

    def __init__(self, airsim_client, config: Dict[str, Any]):
        """
        Initialize trajectory executor

        Args:
            airsim_client: AirSim MultirotorClient instance
            config: Configuration dictionary
        """
        self.client = airsim_client
        self.config = config

        # Initialize drone action space
        self.drone_space = AirSimDroneActionSpace(
            config_path=config.get('config_path', 'config_pivot.yaml')
        )

        # Depth-based step limiting (paper-style navigation: map chosen direction via depth, cap distance)
        self.enable_depth_step_limit = config.get('enable_depth_step_limit', True)
        self.max_step_distance_m = float(config.get('max_step_distance_m', 1.0))
        self.depth_step_margin_m = float(config.get('depth_step_margin_m', 0.0))
        self.camera_name = config.get('camera_name', '0')
        self.fov_factor_h = math.tan(math.radians(float(config.get('fov_horizontal', 90.0)) / 2))
        self.fov_factor_v = math.tan(math.radians(float(config.get('fov_vertical', 90.0)) / 2))

        # Configuration for yaw handling
        self.include_yaw = config.get('include_yaw_in_candidates', True)
        self.yaw_threshold = config.get('yaw_threshold', 5.0)  # degrees
        self.base_yaw_rate = config.get('base_yaw_rate', 30.0)  # degrees/s

        print(f"[TrajectoryExecutor] Initialized")
        print(f"[TrajectoryExecutor] Yaw handling: {self.include_yaw}")

    def execute_trajectory(self, trajectory: Trajectory) -> Dict[str, Any]:
        """
        Execute the selected trajectory using AirSim

        For each waypoint:
        1. Convert to ActionPoint (camera-relative coordinates)
        2. Use DroneActionSpace to generate AirSim commands
        3. Execute commands

        Args:
            trajectory: Trajectory to execute

        Returns:
            Dictionary with execution results
        """
        print(f"\n[Executor] Executing trajectory {trajectory.id}")
        print(f"[Executor] Waypoints: {trajectory.get_num_waypoints()}")

        execution_log = []
        start_time = time.time()

        # Execute each waypoint
        for i, waypoint in enumerate(trajectory.waypoints):
            print(f"\n[Executor] Waypoint {i + 1}/{len(trajectory.waypoints)}: {waypoint}")

            waypoint_result = self._execute_waypoint(waypoint, i)
            execution_log.append(waypoint_result)

        end_time = time.time()
        duration = end_time - start_time

        result = {
            'success': True,
            'trajectory_id': trajectory.id,
            'num_waypoints': len(trajectory.waypoints),
            'duration_seconds': duration,
            'execution_log': execution_log
        }

        print(f"\n[Executor] ✓ Trajectory executed in {duration:.2f}s")
        return result

    def _execute_waypoint(
        self,
        waypoint: Waypoint,
        index: int
    ) -> Dict[str, Any]:
        """
        Execute a single waypoint

        Args:
            waypoint: Waypoint to execute (camera-relative coordinates)
            index: Waypoint index in trajectory

        Returns:
            Dictionary with waypoint execution results
        """
        waypoint_start = time.time()

        if self.enable_depth_step_limit:
            waypoint = self._limit_waypoint_by_depth(waypoint)

        # Convert waypoint to ActionPoint (camera-relative, no transformation needed!)
        action = ActionPoint(
            dx=waypoint.x,  # Lateral (camera x)
            dy=waypoint.y,  # Forward (camera y)
            dz=waypoint.z,  # Vertical (camera z)
            action_type="move"
        )

        print(f"[Executor] Camera-relative: ({waypoint.x:.2f}, {waypoint.y:.2f}, {waypoint.z:.2f})")

        # Generate AirSim commands using drone action space
        # This will handle rotation + forward movement
        commands = self.drone_space.action_to_commands(action)
        print(f"[Executor] Generated {len(commands)} AirSim commands")

        # Execute each command
        for cmd_idx, (cmd_type, params) in enumerate(commands):
            print(f"[Executor] Command {cmd_idx + 1}/{len(commands)}: {cmd_type}")
            self._execute_command(cmd_type, params)

        waypoint_duration = time.time() - waypoint_start

        return {
            'waypoint_index': index,
            'waypoint': {'x': waypoint.x, 'y': waypoint.y, 'z': waypoint.z},
            'num_commands': len(commands),
            'duration_seconds': waypoint_duration,
            'success': True
        }

    def _limit_waypoint_by_depth(self, waypoint: Waypoint) -> Waypoint:
        """
        Use an AirSim depth image to limit the movement distance along the chosen direction.

        Behavior:
        - Project the candidate direction to a pixel.
        - Read depth at that pixel (meters).
        - Scale the (x,y,z) direction vector to min(depth, max_step_distance_m).
        - If depth is unavailable/invalid, fall back to clamping the original vector magnitude.
        """
        depth_image = self._capture_depth_planar()
        if depth_image is None:
            return self._clamp_waypoint_distance(waypoint, self.max_step_distance_m)

        h, w = depth_image.shape[:2]
        px, py = self._project_3d_to_2d(waypoint, w, h)

        depth_val = float(depth_image[py, px])
        if not math.isfinite(depth_val) or depth_val <= 0.0:
            return self._clamp_waypoint_distance(waypoint, self.max_step_distance_m)

        step = max(0.0, min(depth_val - self.depth_step_margin_m, self.max_step_distance_m))
        return self._scale_waypoint_to_distance(waypoint, step)

    def _capture_depth_planar(self) -> Optional["np.ndarray"]:
        try:
            import numpy as np

            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPlanar, True, False)
            ])
            if not responses:
                return None

            response = responses[0]
            if response.width == 0 or response.height == 0 or not response.image_data_float:
                return None

            depth_array = np.array(response.image_data_float, dtype=np.float32)
            return depth_array.reshape(response.height, response.width)
        except Exception:
            return None

    def _project_3d_to_2d(self, waypoint: Waypoint, image_width: int, image_height: int) -> tuple[int, int]:
        center_x = image_width / 2
        center_y = image_height / 2

        y = max(float(waypoint.y), 0.1)
        x_projected = (float(waypoint.x) / (y * self.fov_factor_h)) * (image_width / 2)
        z_projected = (float(waypoint.z) / (y * self.fov_factor_v)) * (image_height / 2)

        screen_x = int(center_x + x_projected)
        screen_y = int(center_y - z_projected)

        screen_x = max(0, min(image_width - 1, screen_x))
        screen_y = max(0, min(image_height - 1, screen_y))
        return screen_x, screen_y

    @staticmethod
    def _scale_waypoint_to_distance(waypoint: Waypoint, distance_m: float) -> Waypoint:
        direction_norm = math.sqrt(waypoint.x**2 + waypoint.y**2 + waypoint.z**2)
        if direction_norm < 1e-6:
            return Waypoint(x=0.0, y=distance_m, z=0.0)
        scale = distance_m / direction_norm
        return Waypoint(x=waypoint.x * scale, y=waypoint.y * scale, z=waypoint.z * scale)

    @staticmethod
    def _clamp_waypoint_distance(waypoint: Waypoint, max_distance_m: float) -> Waypoint:
        direction_norm = math.sqrt(waypoint.x**2 + waypoint.y**2 + waypoint.z**2)
        if direction_norm <= max_distance_m:
            return waypoint
        return TrajectoryExecutor._scale_waypoint_to_distance(waypoint, max_distance_m)

    def _execute_command(self, cmd_type: str, params: Dict[str, Any]):
        """
        Execute a single AirSim command

        Args:
            cmd_type: Command type ('rotate_yaw' or 'move_velocity_body')
            params: Command parameters
        """
        try:
            if cmd_type == "rotate_yaw":
                # Yaw rotation command
                angle = params["angle"]
                yaw_rate = params.get("yaw_rate", 30.0)
                duration = abs(angle) / yaw_rate
                rate = yaw_rate if angle > 0 else -yaw_rate

                print(f"  → Rotating yaw: {angle:.1f}° at {yaw_rate:.1f}°/s")
                self.client.rotateByYawRateAsync(rate, duration).join()
                self.client.hoverAsync().join()

            elif cmd_type == "move_velocity_body":
                # Velocity-based movement command
                vx = params["vx"]
                vy = params["vy"]
                vz = params["vz"]
                duration = params["duration"]
                yaw_rate = params.get("yaw_rate", 0)

                print(f"  → Moving: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f} for {duration:.2f}s")

                yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)

                self.client.moveByVelocityBodyFrameAsync(
                    vx, vy, vz, duration,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode
                ).join()

                self.client.hoverAsync().join()

            else:
                print(f"  Warning: Unknown command type: {cmd_type}")

        except Exception as e:
            print(f"  Error executing command: {e}")
            # Continue execution despite errors

    def hover(self):
        """Command drone to hover in place"""
        print("[Executor] Hovering...")
        self.client.hoverAsync().join()

    def land(self):
        """Command drone to land"""
        print("[Executor] Landing...")
        self.client.landAsync().join()

    def emergency_stop(self):
        """Emergency stop - hover and land"""
        print("[Executor] EMERGENCY STOP")
        try:
            self.client.hoverAsync().join()
            time.sleep(1)
            self.client.landAsync().join()
        except Exception as e:
            print(f"[Executor] Emergency stop error: {e}")
