"""
Candidate Generation for PIVOT

Generates diverse trajectory candidates for VLM selection
and refines them iteratively based on VLM feedback.
"""

import math
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
                 vertical_range: Tuple[float, float] = (-0.8, 0.8),
                 vlm_client = None,
                 visual_overlay = None):
        """
        Initialize candidate generator

        Args:
            num_candidates: Number of trajectory candidates per iteration
            depth_range: Forward distance range in meters (y-axis)
            lateral_range: Left/right range in meters (x-axis)
            vertical_range: Up/down range in meters (z-axis)
            vlm_client: VLMClient for VLM-based generation (optional)
            visual_overlay: VisualOverlay for inverse projection (optional)
        """
        self.num_candidates = num_candidates
        self.depth_range = depth_range
        self.lateral_range = lateral_range
        self.vertical_range = vertical_range
        self.vlm = vlm_client
        self.overlay = visual_overlay

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

    def generate_initial_candidates(self,
                                    instruction: str = "",
                                    image: np.ndarray = None,
                                    use_vlm: bool = False) -> List[Trajectory]:
        """
        Generate initial set of diverse candidates covering the 3D space

        Can use either VLM-based generation (instruction-conditioned) or
        fixed heuristic (deterministic grid pattern).

        Args:
            instruction: Navigation instruction
            image: Current camera frame (required if use_vlm=True)
            use_vlm: Whether to use VLM-based generation

        Returns:
            List of Trajectory objects with single waypoints
        """
        # VLM-based generation path
        if use_vlm and self.vlm is not None and self.overlay is not None and image is not None:
            print("[CandidateGen] Using VLM-based candidate generation")
            try:
                candidates = self._generate_vlm_based_candidates(image, instruction)
                if candidates and len(candidates) >= 3:
                    return candidates
                else:
                    print("[CandidateGen] VLM generation insufficient, using fallback")
            except Exception as e:
                print(f"[CandidateGen] VLM generation failed: {e}, using fallback")

        # Fallback: fixed heuristic generation
        print("[CandidateGen] Using fixed heuristic generation")
        candidates = []

        if self.num_candidates == 8:
            # Pattern: 3x3 grid minus corners = 8 points
            # Distributes candidates strategically across space
            positions = [
                (0.0, 0.0),      # Center
                (-1.0, 0.0),     # Left
                (1.0, 0.0),      # Right
                (0.0, 1.0),      # Up
                (0.0, -1.0),     # Down
                (-0.7, 0.7),     # Upper left
                (0.7, 0.7),      # Upper right
                (0.0, 0.5),      # Mid-up
            ]
        else:
            # General case: distribute evenly in a circle pattern
            positions = []
            for i in range(self.num_candidates):
                angle = 2 * math.pi * i / self.num_candidates
                # Circle in lateral-vertical plane
                x_norm = math.cos(angle)
                z_norm = math.sin(angle)
                positions.append((x_norm, z_norm))

        # Create candidates from normalized positions
        for i, (x_norm, z_norm) in enumerate(positions):
            # Scale to actual ranges
            x = x_norm * (self.lateral_range[1] - self.lateral_range[0]) / 2
            z = z_norm * (self.vertical_range[1] - self.vertical_range[0]) / 2

            # Use middle of depth range
            y = (self.depth_range[0] + self.depth_range[1]) / 2

            # Clamp to bounds
            x = np.clip(x, self.lateral_range[0], self.lateral_range[1])
            y = np.clip(y, self.depth_range[0], self.depth_range[1])
            z = np.clip(z, self.vertical_range[0], self.vertical_range[1])

            waypoint = Waypoint(x=x, y=y, z=z)
            color = self.colors[i % len(self.colors)]

            trajectory = Trajectory(
                waypoints=[waypoint],
                id=i + 1,  # Start IDs from 1 for clarity
                color=color
            )
            candidates.append(trajectory)

        return candidates

    def refine_candidates(self,
                         best_trajectory: Trajectory,
                         iteration: int,
                         search_radius: float) -> List[Trajectory]:
        """
        Generate refined candidates around the best selection

        Creates new candidates in a sphere around the previously
        selected best trajectory, progressively narrowing the search.

        Args:
            best_trajectory: Previously selected best trajectory
            iteration: Current iteration number
            search_radius: Radius for sampling around best trajectory

        Returns:
            List of refined Trajectory objects
        """
        candidates = []
        best_waypoint = best_trajectory.get_first_waypoint()

        # Generate candidates around the best waypoint
        for i in range(self.num_candidates):
            if i == 0:
                # First candidate is the previous best
                waypoint = Waypoint(
                    x=best_waypoint.x,
                    y=best_waypoint.y,
                    z=best_waypoint.z
                )
            else:
                # Sample in a sphere around the best waypoint
                # Use Fibonacci sphere for uniform distribution
                phi = math.acos(1 - 2 * (i - 1) / (self.num_candidates - 2))
                theta = math.pi * (1 + 5**0.5) * (i - 1)

                # Convert spherical to Cartesian
                dx = search_radius * math.sin(phi) * math.cos(theta)
                dy = search_radius * math.sin(phi) * math.sin(theta)
                dz = search_radius * math.cos(phi)

                # Create new waypoint
                x = best_waypoint.x + dx
                y = best_waypoint.y + dy
                z = best_waypoint.z + dz

                # Clamp to bounds
                x = np.clip(x, self.lateral_range[0], self.lateral_range[1])
                y = np.clip(y, self.depth_range[0], self.depth_range[1])
                z = np.clip(z, self.vertical_range[0], self.vertical_range[1])

                waypoint = Waypoint(x=x, y=y, z=z)

            color = self.colors[i % len(self.colors)]
            trajectory = Trajectory(
                waypoints=[waypoint],
                id=i + 1,
                color=color
            )
            candidates.append(trajectory)

        return candidates

    def get_search_space_volume(self) -> float:
        """Calculate the volume of the search space"""
        x_range = self.lateral_range[1] - self.lateral_range[0]
        y_range = self.depth_range[1] - self.depth_range[0]
        z_range = self.vertical_range[1] - self.vertical_range[0]
        return x_range * y_range * z_range

    def _generate_vlm_based_candidates(self,
                                       image: np.ndarray,
                                       instruction: str) -> List[Trajectory]:
        """
        Generate candidates using VLM to identify relevant regions

        Two-phase approach:
        1. Query VLM for 2D regions relevant to instruction
        2. Generate candidates with depth variation around those regions

        Args:
            image: Camera frame
            instruction: Navigation instruction

        Returns:
            List of trajectory candidates

        Raises:
            ValueError: If VLM returns insufficient regions
            json.JSONDecodeError: If VLM response is invalid JSON
        """
        import json
        import sys
        sys.path.append('/home/as06047/github/see-point-fly/src')
        from spf.clients.vlm_client import VLMClient

        # Phase 1: Query VLM for relevant 2D regions
        prompt = f"""You are analyzing a drone camera view for navigation.

INSTRUCTION: "{instruction}"

Identify 3 relevant regions in the image for this task. For each region, provide:
1. The pixel coordinates (x, y) of the center point
2. A brief label describing what the region represents
3. Priority (1 = most relevant, 2 = alternative, 3 = backup)

Return JSON in this EXACT format:
{{
  "regions": [
    {{"point": [x, y], "label": "description", "priority": 1}},
    {{"point": [x, y], "label": "description", "priority": 2}},
    {{"point": [x, y], "label": "description", "priority": 3}}
  ],
  "reasoning": "Brief explanation of region selection"
}}

Image dimensions: {self.overlay.image_width}x{self.overlay.image_height} pixels
Coordinate system: [0,0] is top-left, [{self.overlay.image_width},{self.overlay.image_height}] is bottom-right

Focus on identifying positions that would help accomplish: "{instruction}"
"""

        # Query VLM
        response_text = self.vlm.generate_response(prompt, image)
        response_text = VLMClient.clean_response_text(response_text)

        # Parse response
        response_json = json.loads(response_text)
        regions = response_json.get('regions', [])
        reasoning = response_json.get('reasoning', '')

        print(f"[VLM] Identified {len(regions)} regions: {reasoning}")

        if len(regions) < 2:
            raise ValueError(f"Insufficient regions identified: {len(regions)}")

        # Phase 2: Generate candidates from regions with depth variation
        candidates = self._generate_candidates_from_regions(regions)

        return candidates

    def _generate_candidates_from_regions(self,
                                         regions: List[dict]) -> List[Trajectory]:
        """
        Generate trajectory candidates from VLM-identified 2D regions

        Strategy:
        - For each 2D region, generate multiple candidates at different depths
        - Distribute 8 candidates across regions: [3, 3, 2] for priorities [1, 2, 3]

        Args:
            regions: List of region dicts with 'point', 'label', 'priority'

        Returns:
            List of trajectory candidates
        """
        candidates = []
        candidate_id = 1

        # Depth samples for variation
        depth_samples = {
            1: [0.8, 1.5, 2.5],   # Priority 1: 3 depths
            2: [0.8, 1.5, 2.5],   # Priority 2: 3 depths
            3: [1.0, 2.0]          # Priority 3: 2 depths
        }

        # Sort regions by priority
        sorted_regions = sorted(regions, key=lambda r: r.get('priority', 999))[:3]

        for region in sorted_regions:
            point = region['point']
            label = region.get('label', 'target')
            priority = region.get('priority', 1)

            # Validate coordinates
            screen_x = int(np.clip(point[0], 0, self.overlay.image_width - 1))
            screen_y = int(np.clip(point[1], 0, self.overlay.image_height - 1))

            print(f"[VLM] Region priority {priority} at ({screen_x}, {screen_y}): {label}")

            # Generate candidates at different depths
            depths = depth_samples.get(priority, [1.5])

            for depth in depths:
                # Inverse project to 3D waypoint
                waypoint = self.overlay.inverse_project_2d_to_3d(screen_x, screen_y, depth)

                # Clamp to workspace bounds
                waypoint.x = np.clip(waypoint.x, self.lateral_range[0], self.lateral_range[1])
                waypoint.y = np.clip(waypoint.y, self.depth_range[0], self.depth_range[1])
                waypoint.z = np.clip(waypoint.z, self.vertical_range[0], self.vertical_range[1])

                # Create trajectory
                color = self.colors[(candidate_id - 1) % len(self.colors)]
                trajectory = Trajectory(
                    waypoints=[waypoint],
                    id=candidate_id,
                    color=color
                )

                candidates.append(trajectory)
                candidate_id += 1

                # Stop if we have enough candidates
                if len(candidates) >= self.num_candidates:
                    break

            if len(candidates) >= self.num_candidates:
                break

        return candidates[:self.num_candidates]
