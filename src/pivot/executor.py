"""
Trajectory Executor for PIVOT

Executes selected trajectories using AirSim, leveraging
see-point-fly's command conversion utilities.
"""

import airsim
import time
import math
from typing import Dict, Any, Tuple

# Import from see-point-fly
import sys
sys.path.append('/home/as06047/github/see-point-fly/src')
from spf.airsim.drone_space import AirSimDroneActionSpace
from spf.base.drone_space import ActionPoint

from .trajectory import Trajectory, Waypoint


class TrajectoryExecutor:
    """
    Executes selected PIVOT trajectories using AirSim

    Converts PIVOT Trajectory objects to see-point-fly ActionPoints,
    then uses AirSimDroneActionSpace to convert to AirSim commands.
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

        # Initialize drone action space from see-point-fly
        # This handles the conversion from ActionPoints to AirSim commands
        spf_config_path = config.get('spf_config_path', '../see-point-fly/config_airsim.yaml')
        self.drone_space = AirSimDroneActionSpace(config_path=spf_config_path)

        # Configuration for yaw handling
        self.include_yaw = config.get('include_yaw_in_candidates', True)
        self.yaw_threshold = config.get('yaw_threshold', 5.0)  # degrees
        self.base_yaw_rate = config.get('base_yaw_rate', 30.0)  # degrees/s

        print(f"[TrajectoryExecutor] Initialized with SPF config: {spf_config_path}")
        print(f"[TrajectoryExecutor] Yaw handling: {self.include_yaw}")

    def execute_trajectory(self, trajectory: Trajectory) -> Dict[str, Any]:
        """
        Execute the selected trajectory using AirSim with yaw control

        For each waypoint:
        1. Rotate to desired yaw (if yaw handling enabled)
        2. Move to waypoint using see-point-fly's command conversion
        3. Verify position reached

        Args:
            trajectory: Trajectory to execute

        Returns:
            Dictionary with execution results
        """
        print(f"\n[Executor] Executing trajectory {trajectory.id}")
        print(f"[Executor] Waypoints: {trajectory.get_num_waypoints()}")

        execution_log = []
        start_time = time.time()

        # Ensure yaw_angles exist
        if not trajectory.yaw_angles or len(trajectory.yaw_angles) != len(trajectory.waypoints):
            print("[Executor] ⚠ Yaw angles not properly initialized, using defaults")
            trajectory.__post_init__()  # Regenerate yaw angles

        # Execute each waypoint with its yaw angle
        for i, (waypoint, yaw_angle) in enumerate(zip(trajectory.waypoints, trajectory.yaw_angles)):
            print(f"\n[Executor] Waypoint {i + 1}/{len(trajectory.waypoints)}: {waypoint}, yaw={yaw_angle:.1f}°")

            waypoint_result = self._execute_waypoint(waypoint, yaw_angle, i)
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
        yaw_angle: float,
        index: int
    ) -> Dict[str, Any]:
        """
        Execute a single waypoint with yaw control

        Args:
            waypoint: Waypoint to execute
            yaw_angle: Desired yaw angle in degrees
            index: Waypoint index in trajectory

        Returns:
            Dictionary with waypoint execution results
        """
        waypoint_start = time.time()

        # Phase 1: Yaw rotation (if enabled and significant)
        yaw_executed = False
        if self.include_yaw:
            current_yaw = self._get_current_yaw()
            delta_yaw = self._normalize_angle(yaw_angle - current_yaw)

            if abs(delta_yaw) > self.yaw_threshold:
                print(f"[Executor] Rotating yaw: {current_yaw:.1f}° → {yaw_angle:.1f}° (Δ={delta_yaw:.1f}°)")
                self._execute_yaw_rotation(delta_yaw)
                yaw_executed = True

        # Phase 2: Translation using see-point-fly's logic
        action = ActionPoint(
            dx=waypoint.x,
            dy=waypoint.y,
            dz=waypoint.z,
            action_type="move"
        )

        commands = self.drone_space.action_to_commands(action)
        print(f"[Executor] Generated {len(commands)} AirSim commands")

        for cmd_idx, (cmd_type, params) in enumerate(commands):
            print(f"[Executor] Command {cmd_idx + 1}/{len(commands)}: {cmd_type}")
            self._execute_command(cmd_type, params)

        # Phase 3: Position verification
        if self.config.get('verify_position', True):
            actual_pos = self._get_current_position()
            error = self._compute_position_error(actual_pos, waypoint)

            if error > 0.5:  # 0.5m threshold
                print(f"[Executor] ⚠ Position error {error:.2f}m at waypoint {index}")

        waypoint_duration = time.time() - waypoint_start

        return {
            'waypoint_index': index,
            'waypoint': {'x': waypoint.x, 'y': waypoint.y, 'z': waypoint.z},
            'yaw': yaw_angle,
            'yaw_executed': yaw_executed,
            'num_commands': len(commands),
            'duration_seconds': waypoint_duration,
            'success': True
        }

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

    def _get_current_yaw(self) -> float:
        """
        Get current drone yaw from AirSim

        Returns:
            Current yaw angle in degrees
        """
        try:
            state = self.client.getMultirotorState()
            orientation = state.kinematics_estimated.orientation
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            yaw_radians = airsim.to_eularian_angles(orientation)[2]
            yaw_degrees = math.degrees(yaw_radians)
            return yaw_degrees
        except Exception as e:
            print(f"[Executor] Error getting yaw: {e}")
            return 0.0

    def _get_current_position(self) -> Tuple[float, float, float]:
        """
        Get current drone position from AirSim

        Returns:
            (x, y, z) position tuple in meters
        """
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            return (pos.x_val, pos.y_val, pos.z_val)
        except Exception as e:
            print(f"[Executor] Error getting position: {e}")
            return (0.0, 0.0, 0.0)

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-180, 180] range

        Args:
            angle: Angle in degrees

        Returns:
            Normalized angle in [-180, 180]
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _execute_yaw_rotation(self, delta_yaw: float):
        """
        Execute yaw rotation command

        Args:
            delta_yaw: Yaw change in degrees
        """
        try:
            duration = abs(delta_yaw) / self.base_yaw_rate
            rate = self.base_yaw_rate if delta_yaw > 0 else -self.base_yaw_rate

            self.client.rotateByYawRateAsync(rate, duration).join()
            self.client.hoverAsync().join()

        except Exception as e:
            print(f"[Executor] Error executing yaw rotation: {e}")

    def _compute_position_error(
        self,
        actual: Tuple[float, float, float],
        target: Waypoint
    ) -> float:
        """
        Compute Euclidean distance between actual and target positions

        Args:
            actual: Actual (x, y, z) position
            target: Target waypoint

        Returns:
            Distance in meters
        """
        dx = actual[0] - target.x
        dy = actual[1] - target.y
        dz = actual[2] - target.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
