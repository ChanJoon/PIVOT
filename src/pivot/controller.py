"""
PIVOT Controller

Implements the main PIVOT algorithm with iterative visual prompting.
Manages the VLM interaction loop, candidate refinement, and trajectory selection.
"""

import os
import json
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Import PIVOT components
from .candidate_generator import CandidateGenerator
from .visual_overlay import VisualOverlay
from .trajectory import Trajectory
from .executor import TrajectoryExecutor
from .vlm_client import VLMClient
from .visualization import PivotVisualizer


class PivotController:
    """
    Main PIVOT controller implementing iterative visual prompting

    Manages the complete PIVOT workflow:
    1. Generate candidate trajectories
    2. Overlay candidates on image
    3. Query VLM for selection
    4. Refine candidates based on selection
    5. Repeat until convergence
    6. Execute selected trajectory
    """

    def __init__(self,
                 airsim_client,
                 vlm_client: VLMClient,
                 config: Dict[str, Any]):
        """
        Initialize PIVOT controller

        Args:
            airsim_client: AirSim MultirotorClient instance
            vlm_client: VLMClient for API calls
            config: Configuration dictionary
        """
        self.client = airsim_client
        self.vlm = vlm_client
        self.config = config

        # Initialize components
        self.candidate_gen = CandidateGenerator(
            num_candidates=config.get('num_candidates', 8),
            depth_range=tuple(config.get('depth_range', [0.5, 2.0])),
            lateral_range=tuple(config.get('lateral_range', [-1.5, 1.5])),
            vertical_range=tuple(config.get('vertical_range', [-0.8, 0.8]))
        )

        self.overlay = VisualOverlay(
            image_width=config.get('image_width', 1920),
            image_height=config.get('image_height', 1080),
            fov_horizontal=config.get('fov_horizontal', 90.0),
            fov_vertical=config.get('fov_vertical', 90.0)
        )

        # In parallel runs we often want "selection only" (no per-instance execution).
        self.execute_selected_trajectory = config.get('execute_selected_trajectory', True)
        self.executor = TrajectoryExecutor(airsim_client, config) if self.execute_selected_trajectory else None

        # Algorithm parameters (paper-style: run for N iterations)
        self.max_iterations = config.get('max_iterations', 3)
        self.refinement_factor = config.get('refinement_factor', 0.5)
        self.vlm_select_k = int(config.get('vlm_select_k', 3))

        # Visualization settings
        self.save_iterations = config.get('save_iterations', True)
        self.output_dir = config.get('output_dir', 'pivot_visualizations')
        self.collect_iteration_artifacts = config.get('collect_iteration_artifacts', False)

        # Create root output directory with timestamp; per-navigation runs go inside this folder
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, self.timestamp)
        if self.save_iterations:
            os.makedirs(self.session_dir, exist_ok=True)

        # Advanced visualizations (paper-style)
        self.enable_advanced_viz = config.get('enable_advanced_visualization', True)

        # Track iteration images for composite visualization (reset per navigation run)
        self.iteration_images = []
        self.initial_frame = None  # Store for before/after comparison
        self.run_counter = 0

        print(f"[PivotController] Initialized with {config.get('num_candidates', 8)} candidates")
        print(f"[PivotController] Max iterations: {self.max_iterations}")
        print(f"[PivotController] Output directory: {self.session_dir}")

    def navigate_with_pivot(self,
                           current_frame: np.ndarray,
                           instruction: str) -> Dict[str, Any]:
        """
        Main PIVOT algorithm - iterative visual prompting navigation

        Args:
            current_frame: Current camera frame (BGR format)
            instruction: Natural language navigation instruction

        Returns:
            Dictionary with:
                - selected_trajectory: Final selected trajectory
                - iterations: Number of iterations performed
                - execution: Execution result
                - iteration_history: List of selections per iteration
        """
        self.run_counter += 1
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.session_dir, f"run_{self.run_counter:03d}_{run_timestamp}")
        if self.save_iterations:
            os.makedirs(run_dir, exist_ok=True)

        # Reset per-run visualization buffers
        self.iteration_images = []
        self.initial_frame = None

        # Create a run-scoped visualizer to avoid overwriting artifacts across runs
        visualizer = PivotVisualizer(run_dir) if self.enable_advanced_viz else None

        print(f"\n{'='*60}")
        print(f"PIVOT NAVIGATION: {instruction}")
        print(f"{'='*60}")
        print(f"[Output] Artifacts for this run: {run_dir}")

        # Phase 1: Generate initial candidates (paper: random proposals from action space)
        print("\n[Phase 1] Generating initial candidates...")
        candidates = self.candidate_gen.generate_initial_candidates()
        print(f"Generated {len(candidates)} initial candidates")

        best_trajectory = None
        search_radius = 1.0  # Initial search radius
        iteration_history = []
        execution_results = []  # Track all executions
        iteration_artifacts = []  # Optional in-memory images for parallel runs

        # Phase 2: Iterative refinement BEFORE execution (paper-aligned)
        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"PIVOT Iteration {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")

            # Overlay candidates on current frame
            annotated_image = self.overlay.overlay_candidates(current_frame, candidates)

            # Store initial frame for before/after comparison
            if iteration == 0 and self.initial_frame is None:
                self.initial_frame = current_frame.copy()

            # Query VLM for selection (paper-style top-K points)
            print(f"[Iteration {iteration + 1}] Querying VLM with {len(candidates)} candidates...")
            selection_result = self._query_vlm_for_selection(
                annotated_image,
                instruction,
                candidates,
                iteration
            )

            best_trajectory = selection_result['selected_trajectory']
            selected_trajectories = selection_result['selected_trajectories']
            reasoning = selection_result['reasoning']
            selected_ids = selection_result['selected_ids']

            # Create selection-highlight image (used for saving and optional composites).
            highlighted_image = self.overlay.overlay_candidates_with_selection(
                current_frame,
                candidates,
                best_trajectory.id,
                selected_ids=selected_ids
            )

            # Store iteration image for composite visualization (paper-style panels)
            if self.enable_advanced_viz:
                self.iteration_images.append(highlighted_image.copy())

            if self.save_iterations:
                # Save both: candidates overlay and selected highlight (matches "후보 -> 선정" flow).
                self._save_iteration_viz(annotated_image, iteration, run_dir, tag="candidates")
                self._save_iteration_viz(highlighted_image, iteration, run_dir, tag="selected")

            if self.collect_iteration_artifacts:
                # Keep compressed images in memory (for parallel: save only chosen instance later).
                ok1, buf1 = cv2.imencode(".jpg", annotated_image)
                ok2, buf2 = cv2.imencode(".jpg", highlighted_image)
                if ok1 and ok2:
                    iteration_artifacts.append({
                        "iteration": iteration + 1,
                        "selected_id": best_trajectory.id,
                        "selected_ids": selected_ids,
                        "candidates_jpg": buf1.tobytes(),
                        "selected_jpg": buf2.tobytes(),
                    })

            # Store iteration result
            iteration_history.append({
                'iteration': iteration + 1,
                'selected_id': best_trajectory.id,
                'selected_ids': selected_ids,
                'reasoning': reasoning,
                'search_radius': search_radius,
                'num_candidates': len(candidates)
            })

            print(f"[Iteration {iteration + 1}] Selected: Option {best_trajectory.id}")
            print(f"[Iteration {iteration + 1}] Reasoning: {reasoning}")

            if iteration == self.max_iterations - 1:
                print(f"\n→ Maximum iterations reached ({self.max_iterations})")
                break

            # Refine candidates by fitting an action distribution to the top-K selected actions (paper-aligned)
            search_radius *= self.refinement_factor
            print(f"\n[Refinement] Updating proposal distribution (scale factor: {search_radius:.3f})")
            candidates = self.candidate_gen.refine_candidates(
                selected_trajectories=selected_trajectories,
                iteration=iteration + 1,
                scale=search_radius
            )
            print(f"[Refinement] Generated {len(candidates)} refined candidates")

            # Save progressive refinement snapshot per iteration (pre-execution)
            if self.save_iterations and self.enable_advanced_viz and visualizer:
                progressive_path = os.path.join(
                    run_dir,
                    f"progressive_refinement_iter_{iteration + 1}.jpg"
                )
                print(f"[Viz] Saving progressive refinement snapshot to {progressive_path}")
                visualizer.create_progressive_refinement(
                    self.iteration_images,
                    iteration_history,
                    progressive_path
                )
            else:
                print("[Viz] Skipping progressive refinement snapshot (advanced viz disabled or unavailable)")

        # Execute the selected trajectory once after refinement (optional; disabled in parallel instances)
        execution_result = {
            'all_executions': [],
            'total_executions': 0,
            'final_trajectory_id': best_trajectory.id if best_trajectory else None,
            'success': True,
            'skipped': not self.execute_selected_trajectory
        }

        if self.execute_selected_trajectory:
            print(f"\n[Execute] Running selected trajectory {best_trajectory.id} after refinement loop")
            if self.executor is None:
                self.executor = TrajectoryExecutor(self.client, self.config)
            execution_result_single = self.executor.execute_trajectory(best_trajectory)
            execution_results.append(execution_result_single)
            execution_result.update({
                'all_executions': execution_results,
                'total_executions': len(execution_results),
                'success': execution_result_single.get('success', True),
                'skipped': False
            })
        else:
            print("\n[Execute] Skipped (execute_selected_trajectory=false)")

        # Save final result
        if self.save_iterations:
            self._save_final_result(
                best_trajectory,
                iteration_history,
                execution_result,
                run_dir,
                visualizer
            )

        print(f"\n✓ PIVOT navigation completed in {len(iteration_history)} iterations")

        return {
            'selected_trajectory': best_trajectory,
            'iterations': len(iteration_history),
            'execution': execution_result,
            'iteration_history': iteration_history,
            'instruction': instruction,
            'iteration_artifacts': iteration_artifacts
        }

    def _query_vlm_for_selection(self,
                                 annotated_image: np.ndarray,
                                 instruction: str,
                                 candidates: List[Trajectory],
                                 iteration: int) -> Dict[str, Any]:
        """
        Query VLM to select best candidate from visual overlay

        Args:
            annotated_image: Image with overlaid candidates
            instruction: Navigation instruction
            candidates: List of candidate trajectories
            iteration: Current iteration number

        Returns:
            Dictionary with selected_trajectory, confidence, reasoning
        """
        candidate_ids = [t.id for t in candidates]
        k = max(1, min(self.vlm_select_k, len(candidate_ids)))

        # Paper-style navigation prompt adapted for a drone:
        # - numbered circles are candidate waypoint destinations
        # - arrows indicate direction from the drone (image center) to the waypoint
        # - choose top-K numbers; skip analysis; return JSON {"points": []}
        prompt = f"""You are a drone that cannot fly through obstacles. This is the image you are seeing right now.
I have annotated it with numbered circles and arrows. Each number represents a candidate waypoint destination you can fly toward.
The arrow shows the direction from the drone (image center) to that waypoint.

Your task is to choose which circles you should pick for the task of: {instruction}

Choose the {k} best candidate numbers.
Do NOT choose routes that would go through obstacles, ground or unsafe gaps.
Skip analysis and provide your answer at the end in a json file of this form:
{{"points": []}}

IMPORTANT:
- "points" must contain ONLY numbers from: {candidate_ids}
- Return "points" ordered from BEST to WORST
"""

        try:
            # Query VLM
            response_text = self.vlm.generate_response(prompt, annotated_image)
            response_text = VLMClient.clean_response_text(response_text)

            # Parse JSON response
            response_json = json.loads(response_text)
            points = response_json.get('points', [])
            if not isinstance(points, list):
                raise ValueError("VLM response 'points' must be a list")

            selected_ids: List[int] = []
            for raw in points:
                try:
                    val = int(raw)
                except (TypeError, ValueError):
                    continue
                if val in candidate_ids and val not in selected_ids:
                    selected_ids.append(val)
                if len(selected_ids) >= k:
                    break

            if not selected_ids:
                print("Warning: VLM returned no valid points; defaulting to middle candidate")
                selected_ids = [candidate_ids[len(candidate_ids) // 2]]

            selected_trajectories = [t for t in candidates if t.id in selected_ids]
            if not selected_trajectories:
                selected_trajectories = [candidates[0]]
                selected_ids = [selected_trajectories[0].id]

            # Best trajectory is the first returned (prompt asks BEST->WORST)
            best = selected_trajectories[0]

            # Provide a compatibility confidence: higher when fewer options are picked.
            confidence = float(response_json.get('confidence', max(0.2, 1.0 / len(selected_trajectories))))
            reasoning = response_json.get('reasoning', '')
            risk = response_json.get('risk_assessment', '')

            best.score = confidence
            best.reasoning = reasoning

            return {
                'selected_trajectory': best,
                'selected_trajectories': selected_trajectories,
                'selected_ids': selected_ids,
                'confidence': confidence,
                'reasoning': reasoning or "Selected top-K candidates",
                'risk_assessment': risk or "Not provided"
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing VLM response: {e}")
            print(f"Raw response: {response_text[:200]}...")

            # Fallback: select middle candidate
            fallback_traj = candidates[len(candidates) // 2]
            fallback_traj.score = 0.3
            return {
                'selected_trajectory': fallback_traj,
                'selected_trajectories': [fallback_traj],
                'selected_ids': [fallback_traj.id],
                'confidence': 0.3,
                'reasoning': 'Fallback due to parse error',
                'risk_assessment': 'Unknown'
            }

    def _save_iteration_viz(self,
                           annotated_image: np.ndarray,
                           iteration: int,
                           run_dir: str,
                           tag: str):
        """Save a visualization image for this iteration."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"iteration_{iteration + 1}_{tag}_{timestamp}.jpg"
        img_path = os.path.join(run_dir, img_filename)

        cv2.imwrite(img_path, annotated_image)

    def _save_final_result(self,
                          trajectory: Trajectory,
                          iteration_history: List[Dict],
                          execution_result: Dict,
                          run_dir: str,
                          visualizer: Optional[PivotVisualizer]):
        """Save final result summary"""
        result_file = os.path.join(run_dir, "final_result.json")

        result_data = {
            'timestamp': self.timestamp,
            'final_trajectory': {
                'id': trajectory.id,
                'waypoint': {
                    'x': trajectory.get_first_waypoint().x,
                    'y': trajectory.get_first_waypoint().y,
                    'z': trajectory.get_first_waypoint().z
                },
                'score': trajectory.score,
                'reasoning': trajectory.reasoning
            },
            'iterations': len(iteration_history),
            'iteration_history': iteration_history,
            'execution': execution_result
        }

        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\nFinal result saved to: {result_file}")

        # Create advanced visualizations if enabled
        if self.enable_advanced_viz and visualizer and len(self.iteration_images) >= 2:
            print("\n[Viz] Generating paper-style visualizations...")

            # Progressive refinement composite (side-by-side iterations)
            composite_path = os.path.join(run_dir, "progressive_refinement.jpg")
            visualizer.create_progressive_refinement(
                self.iteration_images,
                iteration_history,
                composite_path
            )

            # Search space and confidence plots
            plot_path = os.path.join(run_dir, "search_space.png")
            visualizer.create_search_space_visualization(
                iteration_history,
                plot_path
            )
