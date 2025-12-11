"""
Trajectory data structures for PIVOT navigation

Defines Waypoint and Trajectory classes for representing
candidate navigation paths in 3D space.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Waypoint:
    """
    Single 3D waypoint in camera-relative coordinates

    Coordinate system (camera-relative):
    - x: Lateral movement (positive = right, negative = left)
    - y: Forward depth (positive = forward, negative = backward)
    - z: Vertical movement (positive = up, negative = down)
    """
    x: float  # Lateral (left/right)
    y: float  # Depth (forward)
    z: float  # Vertical (up/down)

    def distance_to(self, other: 'Waypoint') -> float:
        """Calculate Euclidean distance to another waypoint"""
        return math.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )

    def __str__(self) -> str:
        return f"Waypoint(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


@dataclass
class Trajectory:
    """
    Sequence of waypoints forming a trajectory

    Attributes:
        waypoints: List of Waypoint objects
        id: Unique identifier for VLM selection
        color: RGB tuple for visual overlay
        score: VLM confidence/selection score
        reasoning: VLM's reasoning for selection (optional)
        yaw_angles: List of yaw angles (degrees) at each waypoint
    """
    waypoints: List[Waypoint]
    id: int
    color: Tuple[int, int, int]
    score: float = 0.0
    reasoning: str = ""
    yaw_angles: List[float] = None  # Yaw at each waypoint (degrees)

    def __post_init__(self):
        """Initialize yaw_angles if not provided"""
        if self.yaw_angles is None:
            # Default: yaw points toward each waypoint from origin
            self.yaw_angles = []
            for wp in self.waypoints:
                # Calculate yaw from (0,0) to waypoint
                yaw = math.degrees(math.atan2(wp.x, wp.y))
                self.yaw_angles.append(yaw)

    def get_first_waypoint(self) -> Waypoint:
        """Get the first waypoint in the trajectory"""
        if not self.waypoints:
            raise ValueError("Trajectory has no waypoints")
        return self.waypoints[0]

    def get_last_waypoint(self) -> Waypoint:
        """Get the last waypoint in the trajectory"""
        if not self.waypoints:
            raise ValueError("Trajectory has no waypoints")
        return self.waypoints[-1]

    def get_total_distance(self) -> float:
        """Calculate total distance along the trajectory"""
        if len(self.waypoints) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.waypoints) - 1):
            total += self.waypoints[i].distance_to(self.waypoints[i + 1])
        return total

    def get_num_waypoints(self) -> int:
        """Get number of waypoints in trajectory"""
        return len(self.waypoints)

    def get_duration(self) -> float:
        """
        Estimate execution time in seconds

        Assumes average velocity of 2 m/s and yaw rate of 30 deg/s
        """
        distance = self.get_total_distance()
        travel_time = distance / 2.0  # 2 m/s average velocity

        # Add yaw rotation time
        if self.yaw_angles and len(self.yaw_angles) > 1:
            total_yaw_change = sum(
                abs(self.yaw_angles[i+1] - self.yaw_angles[i])
                for i in range(len(self.yaw_angles) - 1)
            )
            yaw_time = total_yaw_change / 30.0  # 30 deg/s yaw rate
        else:
            yaw_time = 0.0

        return travel_time + yaw_time

    def is_feasible(self, bounds: dict) -> bool:
        """
        Check if trajectory is within workspace bounds

        Args:
            bounds: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max

        Returns:
            True if all waypoints are within bounds
        """
        for wp in self.waypoints:
            if not (bounds['x_min'] <= wp.x <= bounds['x_max']):
                return False
            if not (bounds['y_min'] <= wp.y <= bounds['y_max']):
                return False
            if not (bounds['z_min'] <= wp.z <= bounds['z_max']):
                return False
        return True

    def __str__(self) -> str:
        num_wp = len(self.waypoints)
        dist = self.get_total_distance()
        duration = self.get_duration()
        return f"Trajectory(id={self.id}, waypoints={num_wp}, distance={dist:.2f}, duration={duration:.1f}s, score={self.score:.3f})"
