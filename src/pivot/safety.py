"""
Safety and Feasibility Validation for PIVOT

Provides safety checking and trajectory validation to prevent
unsafe operations and workspace violations.
"""

import math
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from .trajectory import Trajectory, Waypoint


class SafetyChecker:
    """
    Safety and feasibility validation for trajectories

    Performs multi-level safety checks including:
    - Workspace bounds validation
    - Velocity limits
    - Altitude constraints
    - Trajectory feasibility
    - Optional depth-based obstacle gating
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety checker

        Args:
            config: Configuration dictionary with safety parameters
        """
        # Workspace bounds
        bounds = config.get('workspace_bounds', {})
        self.workspace_bounds = {
            'x_min': bounds.get('x_min', -5.0),
            'x_max': bounds.get('x_max', 5.0),
            'y_min': bounds.get('y_min', 0.5),
            'y_max': bounds.get('y_max', 10.0),
            'z_min': bounds.get('z_min', 0.5),
            'z_max': bounds.get('z_max', 5.0),
        }

        # Velocity and altitude limits
        self.max_velocity = config.get('max_velocity', 3.0)  # m/s
        self.min_altitude = config.get('min_altitude', 0.5)  # m
        self.max_altitude = config.get('max_altitude', 10.0)  # m

        # Safety margins
        self.safety_margin = config.get('safety_margin', 0.2)  # 20cm margin
        self.depth_gate_margin = config.get('depth_gate_margin', 0.3)  # meters

        print(f"[SafetyChecker] Initialized with bounds: {self.workspace_bounds}")
        print(f"[SafetyChecker] Max velocity: {self.max_velocity} m/s")
        print(f"[SafetyChecker] Altitude range: [{self.min_altitude}, {self.max_altitude}] m")

    def validate_trajectory(
        self,
        trajectory: Trajectory,
        current_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Multi-level safety validation for a trajectory

        Args:
            trajectory: Trajectory to validate
            current_state: Current drone state (position, velocity, etc.)

        Returns:
            Dictionary with:
                - safe: bool - Whether trajectory is safe
                - violations: List[str] - List of safety violations
                - risk_score: float - Risk score (0.0 = safe, 1.0 = very unsafe)
        """
        violations = []

        # 1. Workspace bounds check
        bounds_violations = self._check_workspace_bounds(trajectory)
        violations.extend(bounds_violations)

        # 2. Velocity limits
        velocity_violations = self._check_velocity_limits(trajectory)
        violations.extend(velocity_violations)

        # 3. Altitude check
        altitude_violations = self._check_altitude(trajectory)
        violations.extend(altitude_violations)

        # 4. Trajectory feasibility (e.g., sudden direction changes)
        feasibility_violations = self._check_feasibility(trajectory)
        violations.extend(feasibility_violations)

        # Calculate risk score
        risk_score = min(len(violations) / 3.0, 1.0)

        return {
            'safe': len(violations) == 0,
            'violations': violations,
            'risk_score': risk_score,
            'trajectory_id': trajectory.id
        }

    def _check_workspace_bounds(self, trajectory: Trajectory) -> List[str]:
        """Check if waypoints are within workspace bounds"""
        violations = []

        for i, wp in enumerate(trajectory.waypoints):
            if not self._in_bounds(wp):
                violations.append(
                    f"Waypoint {i} ({wp}) out of bounds "
                    f"(bounds: x[{self.workspace_bounds['x_min']}, {self.workspace_bounds['x_max']}], "
                    f"y[{self.workspace_bounds['y_min']}, {self.workspace_bounds['y_max']}], "
                    f"z[{self.workspace_bounds['z_min']}, {self.workspace_bounds['z_max']}])"
                )

        return violations

    def _in_bounds(self, waypoint: Waypoint) -> bool:
        """Check if a single waypoint is within bounds (with safety margin)"""
        margin = self.safety_margin

        return (
            self.workspace_bounds['x_min'] + margin <= waypoint.x <= self.workspace_bounds['x_max'] - margin and
            self.workspace_bounds['y_min'] + margin <= waypoint.y <= self.workspace_bounds['y_max'] - margin and
            self.workspace_bounds['z_min'] + margin <= waypoint.z <= self.workspace_bounds['z_max'] - margin
        )

    def _check_velocity_limits(self, trajectory: Trajectory) -> List[str]:
        """Check if trajectory velocity requirements are within limits"""
        violations = []

        if len(trajectory.waypoints) < 2:
            return violations

        for i in range(len(trajectory.waypoints) - 1):
            distance = trajectory.waypoints[i].distance_to(trajectory.waypoints[i + 1])

            # Assume minimum time between waypoints is 1 second
            min_time = 1.0
            required_velocity = distance / min_time

            if required_velocity > self.max_velocity:
                violations.append(
                    f"Segment {i}->{i+1} requires velocity {required_velocity:.2f} m/s > max {self.max_velocity} m/s"
                )

        return violations

    def _check_altitude(self, trajectory: Trajectory) -> List[str]:
        """Check altitude constraints"""
        violations = []

        for i, wp in enumerate(trajectory.waypoints):
            if wp.z < self.min_altitude:
                violations.append(
                    f"Waypoint {i} altitude {wp.z:.2f}m < minimum {self.min_altitude}m"
                )
            if wp.z > self.max_altitude:
                violations.append(
                    f"Waypoint {i} altitude {wp.z:.2f}m > maximum {self.max_altitude}m"
                )

        return violations

    def _check_feasibility(self, trajectory: Trajectory) -> List[str]:
        """Check trajectory feasibility (e.g., sharp turns, sudden changes)"""
        violations = []

        if len(trajectory.waypoints) < 3:
            return violations

        # Check for sharp direction changes
        for i in range(len(trajectory.waypoints) - 2):
            wp1 = trajectory.waypoints[i]
            wp2 = trajectory.waypoints[i + 1]
            wp3 = trajectory.waypoints[i + 2]

            # Calculate angle between segments
            v1 = (wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z)
            v2 = (wp3.x - wp2.x, wp3.y - wp2.y, wp3.z - wp2.z)

            # Dot product and magnitudes
            dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

            if mag1 > 0.01 and mag2 > 0.01:
                cos_angle = dot / (mag1 * mag2)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp
                angle_deg = math.degrees(math.acos(cos_angle))

                # Flag sharp turns (> 120 degrees)
                if angle_deg > 120:
                    violations.append(
                        f"Sharp turn at waypoint {i+1}: {angle_deg:.1f}° direction change"
                    )

        return violations

    def filter_unsafe_candidates(
        self,
        candidates: List[Trajectory],
        current_state: Dict[str, Any] = None,
        depth_map: Optional[np.ndarray] = None,
        overlay: Optional[Any] = None
    ) -> List[Trajectory]:
        """
        Remove unsafe candidates before VLM selection

        Args:
            candidates: List of candidate trajectories
            current_state: Current drone state
            depth_map: Optional depth image (meters) aligned with camera
            overlay: VisualOverlay for projecting waypoints to pixels

        Returns:
            List of safe trajectories (at least one, even if unsafe)
        """
        safe_candidates = []
        filtered_count = 0

        for traj in candidates:
            result = self.validate_trajectory(traj, current_state)

            depth_violation = False
            if depth_map is not None and overlay is not None:
                depth_violation = self._check_depth_gate(traj, depth_map, overlay)

            if result['safe'] and not depth_violation:
                safe_candidates.append(traj)
            else:
                filtered_count += 1
                reasons = result['violations'][:2]
                if depth_violation:
                    reasons.append("depth obstacle along ray")
                print(f"  [Safety] Filtered candidate {traj.id}: {', '.join(reasons)}")

        if not safe_candidates:
            # If all candidates are unsafe, keep the least unsafe one
            print(f"  [Safety] WARNING: All candidates unsafe! Keeping least risky.")
            best_candidate = min(
                candidates,
                key=lambda t: self.validate_trajectory(t, current_state)['risk_score']
            )
            safe_candidates = [best_candidate]

        if filtered_count > 0:
            print(f"  [Safety] Filtered {filtered_count}/{len(candidates)} unsafe candidates")

        return safe_candidates

    def get_safe_bounds(self) -> Dict[str, float]:
        """Get the workspace bounds for use in trajectory generation"""
        return self.workspace_bounds.copy()

    def _check_depth_gate(self, trajectory: Trajectory, depth_map: np.ndarray, overlay: Any) -> bool:
        """
        Simple obstacle gating using a depth map.
        If the measured depth along the candidate pixel ray is closer than the waypoint
        minus a margin, treat as obstructed.
        """
        waypoint = trajectory.get_first_waypoint()
        px, py = overlay.project_3d_to_2d(waypoint)

        try:
            measured_depth = float(depth_map[py, px])
        except Exception:
            return False

        # Invalid depth values are often 0 or very small; ignore gating in that case
        if measured_depth <= 0.01 or np.isinf(measured_depth) or np.isnan(measured_depth):
            return False

        return measured_depth + self.depth_gate_margin < waypoint.y

    def is_position_safe(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is safe

        Args:
            position: (x, y, z) tuple

        Returns:
            True if position is within safe bounds
        """
        wp = Waypoint(x=position[0], y=position[1], z=position[2])
        return self._in_bounds(wp)
