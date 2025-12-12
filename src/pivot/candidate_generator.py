"""
Candidate Generation for PIVOT

Generates diverse trajectory candidates for VLM selection
and refines them iteratively based on VLM feedback.
"""

import numpy as np
from typing import List, Tuple
from .trajectory import Waypoint, Trajectory


class CandidateGenerator:
    """
    Generates and refines trajectory candidates for PIVOT

    Uses radial sampling in 3D space to create diverse candidates
    that cover the navigation workspace.
    """

    def __init__(self,
                 num_candidates: int = 8,
                 depth_range: Tuple[float, float] = (0.5, 2.0),
                 lateral_range: Tuple[float, float] = (-1.5, 1.5),
                 vertical_range: Tuple[float, float] = (-0.8, 0.8)):
        """
        Initialize candidate generator

        Args:
            num_candidates: Number of trajectory candidates per iteration
            depth_range: Forward distance range in meters (y-axis)
            lateral_range: Left/right range in meters (x-axis)
            vertical_range: Up/down range in meters (z-axis)
        """
        self.num_candidates = num_candidates
        self.depth_range = depth_range
        self.lateral_range = lateral_range
        self.vertical_range = vertical_range

        # Predefined colors for up to 12 candidates
        self.colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
            (255, 192, 203), # Pink
            (128, 128, 0),   # Olive
            (0, 128, 128),   # Teal
            (128, 0, 0),     # Maroon
        ]

    def generate_initial_candidates(self) -> List[Trajectory]:
        """
        Generate initial set of candidates by random sampling from the action space.

        Returns:
            List of Trajectory objects with single waypoints
        """
        candidates: List[Trajectory] = []

        xs = np.random.uniform(self.lateral_range[0], self.lateral_range[1], size=(self.num_candidates,))
        ys = np.random.uniform(self.depth_range[0], self.depth_range[1], size=(self.num_candidates,))
        zs = np.random.uniform(self.vertical_range[0], self.vertical_range[1], size=(self.num_candidates,))

        for i in range(self.num_candidates):
            waypoint = Waypoint(x=float(xs[i]), y=float(ys[i]), z=float(zs[i]))
            color = self.colors[i % len(self.colors)]
            candidates.append(Trajectory(waypoints=[waypoint], id=i + 1, color=color))

        return candidates

    def refine_candidates(self,
                          selected_trajectories: List[Trajectory],
                          iteration: int,
                          scale: float) -> List[Trajectory]:
        """
        Generate refined candidates by fitting a distribution to top-K selections

        Paper-style refinement:
        - Fit an isotropic Gaussian to the selected waypoints
        - Sample new candidates from that distribution

        Args:
            selected_trajectories: Selected trajectories (top-K) from the VLM
            iteration: Current iteration number
            scale: Additional scaling factor for exploration

        Returns:
            List of refined Trajectory objects
        """
        if not selected_trajectories:
            raise ValueError("selected_trajectories must not be empty")

        selected_waypoints = np.array(
            [[t.get_first_waypoint().x, t.get_first_waypoint().y, t.get_first_waypoint().z]
             for t in selected_trajectories],
            dtype=float
        )

        mean = selected_waypoints.mean(axis=0)
        centered = selected_waypoints - mean
        dists = np.linalg.norm(centered, axis=1)

        # Isotropic Gaussian: sigma is scalar; use RMS distance.
        empirical_sigma = float(np.sqrt(np.mean(dists**2))) if len(dists) else 0.0

        # Keep a minimum sigma to avoid collapsing too early.
        min_sigma = 0.05
        sigma = max(min_sigma, empirical_sigma) * max(0.1, float(scale))

        candidates: List[Trajectory] = []

        # Ensure the best (first) selected trajectory is preserved as candidate 1
        best_wp = selected_trajectories[0].get_first_waypoint()
        candidates.append(
            Trajectory(
                waypoints=[Waypoint(x=best_wp.x, y=best_wp.y, z=best_wp.z)],
                id=1,
                color=self.colors[0 % len(self.colors)]
            )
        )

        for i in range(1, self.num_candidates):
            sample = np.random.normal(loc=mean, scale=sigma, size=(3,))
            x = float(np.clip(sample[0], self.lateral_range[0], self.lateral_range[1]))
            y = float(np.clip(sample[1], self.depth_range[0], self.depth_range[1]))
            z = float(np.clip(sample[2], self.vertical_range[0], self.vertical_range[1]))

            waypoint = Waypoint(x=x, y=y, z=z)
            color = self.colors[i % len(self.colors)]
            candidates.append(Trajectory(waypoints=[waypoint], id=i + 1, color=color))

        return candidates

    def get_search_space_volume(self) -> float:
        """Calculate the volume of the search space"""
        x_range = self.lateral_range[1] - self.lateral_range[0]
        y_range = self.depth_range[1] - self.depth_range[0]
        z_range = self.vertical_range[1] - self.vertical_range[0]
        return x_range * y_range * z_range
