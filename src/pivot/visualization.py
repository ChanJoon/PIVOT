"""
Enhanced Visualization for PIVOT

Creates paper-style visualizations matching PIVOT paper figures:
- Progressive refinement composites (side-by-side iterations)
- Before/after comparisons
- Iteration progression with confidence scores
- Search space shrinking visualization

Based on PIVOT paper Figures 1-2
"""

import cv2
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


class PivotVisualizer:
    """Create publication-quality PIVOT visualizations"""

    def __init__(self, output_dir: str):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # Main font
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_large = 1.2
        self.font_scale_medium = 1.0
        self.font_scale_small = 0.7
        self.font_thickness_bold = 3  # Thicker = bolder
        self.font_thickness_normal = 2

        # Colors
        self.color_white = (255, 255, 255)
        self.color_green = (0, 255, 0)
        self.color_yellow = (0, 255, 255)
        self.color_red = (0, 0, 255)
        self.color_blue = (255, 0, 0)

    def create_progressive_refinement(self,
                                     iteration_images: List[np.ndarray],
                                     iteration_data: List[Dict],
                                     output_path: str):
        """
        Create side-by-side comparison of all iterations

        Layout: [Iteration 0] [Iteration 1] [Iteration 2] ...
        Each panel shows:
        - Iteration number
        - Selected option ID
        - VLM confidence score
        - Search radius

        Args:
            iteration_images: List of annotated images per iteration
            iteration_data: List of dicts with iteration metadata
            output_path: Output file path
        """
        if not iteration_images:
            print("[Viz] No iteration images to create composite")
            return

        num_iterations = len(iteration_images)
        h, w = iteration_images[0].shape[:2]

        # Target size for each iteration panel (resize for composite)
        target_w = min(640, w)  # Max 640px width per panel
        target_h = int(h * target_w / w)

        # Resize all images
        resized = [cv2.resize(img, (target_w, target_h)) for img in iteration_images]

        # Add metadata overlays to each panel
        for i, (img, data) in enumerate(zip(resized, iteration_data)):
            # Add semi-transparent black background for text
            overlay = img.copy()
            cv2.rectangle(overlay, (5, 5), (target_w - 5, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

            # Iteration label
            label = f"Iteration {i}"
            cv2.putText(img, label, (15, 35),
                       self.font, self.font_scale_medium, self.color_white,
                       self.font_thickness_bold)

            # Selected option
            selected_id = data.get('selected_id', '?')
            selected_text = f"Selected: #{selected_id}"
            cv2.putText(img, selected_text, (15, 70),
                       self.font_small, self.font_scale_small, self.color_yellow,
                       self.font_thickness_normal)

            # Confidence score
            confidence = data.get('confidence', 0.0)
            conf_text = f"Confidence: {confidence:.3f}"
            conf_color = self.color_green if confidence >= 0.9 else self.color_yellow
            cv2.putText(img, conf_text, (15, 100),
                       self.font_small, self.font_scale_small, conf_color,
                       self.font_thickness_normal)

            # Search radius
            radius = data.get('search_radius', 1.0)
            radius_text = f"Radius: {radius:.2f}m"
            cv2.putText(img, radius_text, (15, 130),
                       self.font_small, self.font_scale_small, self.color_blue,
                       self.font_thickness_normal)

        # Stack panels horizontally
        composite = np.hstack(resized)

        # Add title bar at top
        title_height = 60
        title_bar = np.zeros((title_height, composite.shape[1], 3), dtype=np.uint8)
        title_text = f"PIVOT Progressive Refinement ({num_iterations} iterations)"
        text_size = cv2.getTextSize(title_text, self.font, self.font_scale_large,
                                   self.font_thickness_bold)[0]
        text_x = (title_bar.shape[1] - text_size[0]) // 2
        cv2.putText(title_bar, title_text, (text_x, 40),
                   self.font, self.font_scale_large, self.color_white,
                   self.font_thickness_bold)

        # Combine title and composite
        final_composite = np.vstack([title_bar, composite])

        # Save
        cv2.imwrite(str(output_path), final_composite)
        print(f"[Viz] Saved progressive refinement: {output_path}")

    def create_before_after(self,
                           before_image: np.ndarray,
                           after_image: np.ndarray,
                           trajectory_overlay: np.ndarray,
                           output_path: str):
        """
        Create before/after comparison with trajectory visualization

        Layout: [Before] [Trajectory Path] [After]

        Args:
            before_image: Initial frame (no annotations)
            after_image: Frame after execution
            trajectory_overlay: Frame with selected trajectory highlighted
            output_path: Output file path
        """
        h, w = before_image.shape[:2]
        target_w = 640
        target_h = int(h * target_w / w)

        # Resize all three images
        before = cv2.resize(before_image, (target_w, target_h))
        traj = cv2.resize(trajectory_overlay, (target_w, target_h))
        after = cv2.resize(after_image, (target_w, target_h))

        # Add labels to each panel
        labels = [
            ("Before Optimization", self.color_white),
            ("Selected Trajectory", self.color_green),
            ("After Execution", self.color_yellow)
        ]

        for img, (label, color) in zip([before, traj, after], labels):
            # Semi-transparent background
            overlay = img.copy()
            text_size = cv2.getTextSize(label, self.font, self.font_scale_large,
                                       self.font_thickness_bold)[0]
            bg_width = text_size[0] + 30
            cv2.rectangle(overlay, (10, 10), (bg_width, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

            # Label text
            cv2.putText(img, label, (20, 45),
                       self.font, self.font_scale_large, color,
                       self.font_thickness_bold)

        # Stack horizontally
        composite = np.hstack([before, traj, after])

        # Add title
        title_height = 60
        title_bar = np.zeros((title_height, composite.shape[1], 3), dtype=np.uint8)
        title_text = "PIVOT Before/After Comparison"
        text_size = cv2.getTextSize(title_text, self.font, self.font_scale_large,
                                   self.font_thickness_bold)[0]
        text_x = (title_bar.shape[1] - text_size[0]) // 2
        cv2.putText(title_bar, title_text, (text_x, 40),
                   self.font, self.font_scale_large, self.color_white,
                   self.font_thickness_bold)

        final_composite = np.vstack([title_bar, composite])

        # Save
        cv2.imwrite(str(output_path), final_composite)
        print(f"[Viz] Saved before/after comparison: {output_path}")

    def create_search_space_visualization(self,
                                         iterations: List[Dict],
                                         output_path: str):
        """
        Visualize search space shrinking and confidence progression

        Creates a 2-panel plot:
        - Left: Search radius over iterations (shows convergence)
        - Right: VLM confidence over iterations (shows quality)

        Args:
            iterations: List of iteration metadata dicts
            output_path: Output file path (PNG)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Viz] WARNING: matplotlib not installed, skipping search space plot")
            return

        iterations_num = [d['iteration'] for d in iterations]
        radii = [d['search_radius'] for d in iterations]
        confidences = [d['confidence'] for d in iterations]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Search radius (shows refinement)
        ax1.plot(iterations_num, radii, 'o-', linewidth=3, markersize=10, color='#2E86AB')
        ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Search Radius (m)', fontsize=14, fontweight='bold')
        ax1.set_title('Search Space Refinement', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(iterations_num)

        # Add value labels on points
        for i, (x, y) in enumerate(zip(iterations_num, radii)):
            ax1.annotate(f'{y:.2f}m', (x, y), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=10)

        # Plot 2: Confidence (shows quality improvement)
        ax2.plot(iterations_num, confidences, 'o-', linewidth=3, markersize=10, color='#06A77D')
        ax2.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax2.set_ylabel('VLM Confidence', fontsize=14, fontweight='bold')
        ax2.set_title('Confidence Progression', fontsize=16, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.axhline(y=0.9, color='red', linestyle='--', label='Convergence Threshold', alpha=0.7)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticks(iterations_num)
        ax2.legend(fontsize=10)

        # Add value labels on points
        for i, (x, y) in enumerate(zip(iterations_num, confidences)):
            ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Viz] Saved search space plot: {output_path}")

    def highlight_selected_trajectory(self,
                                     base_image: np.ndarray,
                                     selected_id: int,
                                     all_candidates: List,
                                     output_path: str):
        """
        Create visualization with selected trajectory highlighted

        Args:
            base_image: Base camera frame
            selected_id: ID of selected trajectory
            all_candidates: List of all candidate trajectories
            output_path: Output file path
        """
        img = base_image.copy()

        # Find selected candidate
        selected_traj = None
        for cand in all_candidates:
            if cand.id == selected_id:
                selected_traj = cand
                break

        if selected_traj is None:
            print(f"[Viz] WARNING: Selected trajectory {selected_id} not found")
            return

        # Draw all candidates in gray (dim)
        for cand in all_candidates:
            if cand.id != selected_id:
                self._draw_trajectory_dim(img, cand)

        # Draw selected candidate in bright green
        self._draw_trajectory_highlight(img, selected_traj)

        # Add selection label
        label = f"SELECTED: Option {selected_id}"
        text_size = cv2.getTextSize(label, self.font, self.font_scale_large,
                                   self.font_thickness_bold)[0]
        # Draw background
        cv2.rectangle(img, (10, 10), (text_size[0] + 30, 60), (0, 0, 0), -1)
        # Draw text
        cv2.putText(img, label, (20, 45),
                   self.font, self.font_scale_large, self.color_green,
                   self.font_thickness_bold)

        # Save
        cv2.imwrite(str(output_path), img)
        print(f"[Viz] Saved highlighted trajectory: {output_path}")

    def _draw_trajectory_dim(self, img: np.ndarray, trajectory):
        """Draw trajectory in dim gray"""
        # This would require access to the overlay module to project waypoints
        # Simplified version: just mark the general area
        pass

    def _draw_trajectory_highlight(self, img: np.ndarray, trajectory):
        """Draw trajectory in bright highlight color"""
        # This would require access to the overlay module
        # Simplified version
        pass

    def create_iteration_grid(self,
                             iteration_images: List[np.ndarray],
                             grid_cols: int = 3,
                             output_path: str = None):
        """
        Create a grid layout of iteration images

        Useful when you have many iterations (>4) and want a compact view

        Args:
            iteration_images: List of annotated images
            grid_cols: Number of columns in grid
            output_path: Output file path
        """
        if not iteration_images:
            return

        num_images = len(iteration_images)
        grid_rows = (num_images + grid_cols - 1) // grid_cols

        # Target size for each cell
        cell_w, cell_h = 480, 360

        # Resize images
        resized = [cv2.resize(img, (cell_w, cell_h)) for img in iteration_images]

        # Pad with black images if needed
        while len(resized) < grid_rows * grid_cols:
            resized.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

        # Create grid
        rows = []
        for r in range(grid_rows):
            row_images = resized[r * grid_cols:(r + 1) * grid_cols]
            row = np.hstack(row_images)
            rows.append(row)

        grid = np.vstack(rows)

        if output_path:
            cv2.imwrite(str(output_path), grid)
            print(f"[Viz] Saved iteration grid: {output_path}")

        return grid
