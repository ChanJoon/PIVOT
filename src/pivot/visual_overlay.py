"""
Visual Overlay for PIVOT

Overlays trajectory candidates on camera images for VLM selection.
Projects 3D waypoints to 2D image coordinates and draws visual markers.
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Optional, Set
from .trajectory import Trajectory, Waypoint


class VisualOverlay:
    """
    Overlays candidate trajectories on drone camera images

    Projects 3D waypoints to 2D image space and draws colored markers
    with IDs and directional arrows for VLM selection.
    """

    def __init__(self,
                 image_width: int = 1920,
                 image_height: int = 1080,
                 fov_horizontal: float = 90.0,
                 fov_vertical: float = 90.0):
        """
        Initialize visual overlay system

        Args:
            image_width: Width of camera image in pixels
            image_height: Height of camera image in pixels
            fov_horizontal: Horizontal field of view in degrees
            fov_vertical: Vertical field of view in degrees
        """
        self.image_width = image_width
        self.image_height = image_height
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical

        # Calculate FOV factors for projection
        self.fov_factor_h = np.tan(np.radians(fov_horizontal / 2))
        self.fov_factor_v = np.tan(np.radians(fov_vertical / 2))

        # Drawing parameters
        self.circle_radius = 20
        # Encode depth with circle size (paper-style cue: closer appears larger).
        self.enable_depth_size_cue = True
        self.circle_radius_min = 14
        self.circle_radius_max = 34
        self.circle_thickness = 3
        self.arrow_thickness = 2
        # Paper-style label: gray translucent fill + number inside + green border for top-K.
        self.gray_fill_bgr = (160, 160, 160)
        self.gray_fill_alpha = 0.55
        self.text_color_bgr = (255, 255, 255)
        self.topk_border_bgr = (0, 255, 0)  # green
        # Some OpenCV builds (e.g., headless) lack FONT_HERSHEY_BOLD; fall back to SIMPLEX.
        self.text_font = getattr(cv2, "FONT_HERSHEY_BOLD", cv2.FONT_HERSHEY_SIMPLEX)
        self.text_scale = 1.0
        self.text_thickness = 2

    @staticmethod
    def _center_text_position(text: str, center: Tuple[int, int], font, scale: float, thickness: int) -> Tuple[int, int]:
        (tw, th), _baseline = cv2.getTextSize(text, font, scale, thickness)
        x = int(center[0] - tw / 2)
        y = int(center[1] + th / 2)
        return x, y

    def _draw_labeled_circle(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        border_bgr: Tuple[int, int, int],
        label: str,
        border_thickness: int,
    ) -> None:
        overlay = image.copy()
        cv2.circle(overlay, center, radius, self.gray_fill_bgr, thickness=-1)
        cv2.addWeighted(overlay, self.gray_fill_alpha, image, 1.0 - self.gray_fill_alpha, 0, image)

        cv2.circle(image, center, radius, border_bgr, thickness=border_thickness)

        pos = self._center_text_position(label, center, self.text_font, self.text_scale, self.text_thickness)
        cv2.putText(image, label, pos, self.text_font, self.text_scale, self.text_color_bgr, self.text_thickness)

    def project_3d_to_2d(self, waypoint: Waypoint) -> Tuple[int, int]:
        """
        Project 3D waypoint to 2D image coordinates using perspective projection

        Uses the same projection as see-point-fly's ActionProjector.
        Camera-relative coordinates:
        - x: lateral (left/right)
        - y: depth (forward)
        - z: vertical (up/down)

        Args:
            waypoint: 3D waypoint in camera-relative coordinates

        Returns:
            Tuple of (pixel_x, pixel_y) in image coordinates
        """
        # Center points
        center_x = self.image_width / 2
        center_y = self.image_height / 2

        # Avoid division by zero
        y = max(waypoint.y, 0.1)

        # Perspective projection with FOV
        x_projected = (waypoint.x / (y * self.fov_factor_h)) * (self.image_width / 2)
        z_projected = (waypoint.z / (y * self.fov_factor_v)) * (self.image_height / 2)

        # Convert to screen coordinates
        # Note: screen Y increases downward, so we negate z_projected
        screen_x = int(center_x + x_projected)
        screen_y = int(center_y - z_projected)

        # Clamp to image bounds
        screen_x = np.clip(screen_x, 0, self.image_width - 1)
        screen_y = np.clip(screen_y, 0, self.image_height - 1)

        return (screen_x, screen_y)

    def inverse_project_2d_to_3d(self,
                                 screen_x: int,
                                 screen_y: int,
                                 depth: float) -> Waypoint:
        """
        Inverse project 2D screen coordinates to 3D waypoint at given depth

        This is the mathematical inverse of project_3d_to_2d().
        Given a 2D pixel location and an assumed depth, computes the
        corresponding 3D waypoint in camera-relative coordinates.

        Args:
            screen_x: Pixel x-coordinate in image
            screen_y: Pixel y-coordinate in image
            depth: Assumed depth (forward distance) in meters

        Returns:
            Waypoint in 3D camera-relative coordinates
        """
        # Validate inputs
        screen_x = np.clip(screen_x, 0, self.image_width - 1)
        screen_y = np.clip(screen_y, 0, self.image_height - 1)
        depth = max(depth, 0.1)  # Avoid zero/negative depth

        # Center points
        center_x = self.image_width / 2
        center_y = self.image_height / 2

        # Normalize to [-1, 1] range
        normalized_x = (screen_x - center_x) / (self.image_width / 2)
        normalized_z = (center_y - screen_y) / (self.image_height / 2)  # Note: Y flipped

        # Apply inverse perspective projection
        # From forward: x_projected = (waypoint.x / (y * fov_factor_h)) * (image_width / 2)
        # Inverse: waypoint.x = normalized_x * y * fov_factor_h
        x = normalized_x * depth * self.fov_factor_h
        z = normalized_z * depth * self.fov_factor_v
        y = depth

        return Waypoint(x=x, y=y, z=z)

    def overlay_candidates(self,
                          image: np.ndarray,
                          trajectories: List[Trajectory],
                          topk_ids: Optional[Set[int]] = None) -> np.ndarray:
        """
        Draw all candidate trajectories on the image

        Visual elements per candidate:
        1. Large colored circle at waypoint location
        2. Numbered label next to circle (trajectory ID)
        3. Arrow from center showing direction
        4. Color-coded for easy VLM reference

        Args:
            image: Input image (BGR format)
            trajectories: List of candidate trajectories to overlay

        Returns:
            Annotated image with visual overlays
        """
        annotated = image.copy()
        center = (self.image_width // 2, self.image_height // 2)
        topk_ids = topk_ids or set()

        depths = [float(t.get_first_waypoint().y) for t in trajectories] if trajectories else []
        min_depth = min(depths) if depths else 0.5
        max_depth = max(depths) if depths else 3.0
        depth_span = max(1e-6, max_depth - min_depth)

        for traj in trajectories:
            # Get first waypoint (for single-waypoint trajectories)
            waypoint = traj.get_first_waypoint()
            waypoint_2d = self.project_3d_to_2d(waypoint)

            # Convert color from RGB to BGR for OpenCV
            color_bgr = (traj.color[2], traj.color[1], traj.color[0])

            radius = self.circle_radius
            if self.enable_depth_size_cue:
                # Smaller circles for farther waypoints; larger for closer.
                t = (float(waypoint.y) - min_depth) / depth_span
                inv = 1.0 - float(np.clip(t, 0.0, 1.0))
                radius = int(round(self.circle_radius_min + inv * (self.circle_radius_max - self.circle_radius_min)))

            border = self.topk_border_bgr if traj.id in topk_ids else color_bgr
            border_thickness = self.circle_thickness + (1 if traj.id in topk_ids else 0)
            self._draw_labeled_circle(
                annotated,
                waypoint_2d,
                radius,
                border_bgr=border,
                label=str(traj.id),
                border_thickness=border_thickness,
            )

            # Draw arrow from center to waypoint
            cv2.arrowedLine(annotated, center, waypoint_2d,
                          color_bgr, self.arrow_thickness,
                          tipLength=0.15)

            # Optional: Draw filled circle at center for reference
            cv2.circle(annotated, center, 5, (128, 128, 128), -1)

        return annotated

    def overlay_trajectories_multi_waypoint(self,
                                           image: np.ndarray,
                                           trajectories: List[Trajectory]) -> np.ndarray:
        """
        Draw multi-waypoint trajectories (for future extension)

        Draws lines connecting waypoints to show full trajectory paths.

        Args:
            image: Input image (BGR format)
            trajectories: List of trajectories with multiple waypoints

        Returns:
            Annotated image with trajectory paths
        """
        annotated = image.copy()

        for traj in trajectories:
            color_bgr = (traj.color[2], traj.color[1], traj.color[0])

            # Project all waypoints
            points_2d = [self.project_3d_to_2d(wp) for wp in traj.waypoints]

            # Draw lines connecting waypoints
            for i in range(len(points_2d) - 1):
                cv2.line(annotated, points_2d[i], points_2d[i+1],
                        color_bgr, 2)

            # Draw circles at each waypoint
            for i, pt in enumerate(points_2d):
                radius = self.circle_radius if i == 0 else 10
                cv2.circle(annotated, pt, radius, color_bgr,
                          self.circle_thickness if i == 0 else 2)

            # Draw ID at first waypoint
            if points_2d:
                label_pos = (points_2d[0][0] + 25, points_2d[0][1])
                cv2.putText(annotated, str(traj.id), label_pos,
                           self.text_font, self.text_scale, color_bgr,
                           self.text_thickness)

        return annotated

    def add_legend(self,
                  image: np.ndarray,
                  trajectories: List[Trajectory]) -> np.ndarray:
        """
        Add a legend showing trajectory IDs and colors

        Args:
            image: Input image
            trajectories: List of trajectories

        Returns:
            Image with legend overlay
        """
        annotated = image.copy()

        # Legend position (top-left corner)
        legend_x = 10
        legend_y = 30
        line_height = 25

        for i, traj in enumerate(trajectories):
            color_bgr = (traj.color[2], traj.color[1], traj.color[0])
            y_pos = legend_y + i * line_height

            # Draw colored square
            cv2.rectangle(annotated,
                         (legend_x, y_pos - 15),
                         (legend_x + 15, y_pos),
                         color_bgr, -1)

            # Draw ID text
            text = f"Option {traj.id}"
            cv2.putText(annotated, text, (legend_x + 25, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def overlay_candidates_with_selection(self,
                                         image: np.ndarray,
                                         trajectories: List[Trajectory],
                                         selected_id: int,
                                         selected_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Draw all candidates and highlight the top-K selections with green borders.

        Notes:
        - `selected_id` is the BEST (rank-1) selection.
        - `selected_ids` (optional) is the full top-K list; when provided, those IDs get green borders.

        Args:
            image: Input image (BGR format)
            trajectories: List of candidate trajectories
            selected_id: ID of the selected trajectory

        Returns:
            Annotated image with selected trajectory highlighted
        """
        topk = set(selected_ids) if selected_ids is not None else {selected_id}
        annotated = self.overlay_candidates(image, trajectories, topk_ids=topk)

        # Best gets an extra outer green ring for clarity.
        for traj in trajectories:
            if traj.id != selected_id:
                continue
            waypoint_2d = self.project_3d_to_2d(traj.get_first_waypoint())
            cv2.circle(annotated, waypoint_2d, self.circle_radius_max + 8, self.topk_border_bgr, thickness=2)
            break

        return annotated
