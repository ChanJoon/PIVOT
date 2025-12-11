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
from typing import List, Dict, Any
from pathlib import Path

# Import from see-point-fly
import sys
sys.path.append('/home/as06047/github/see-point-fly/src')
from spf.clients.vlm_client import VLMClient

# Import PIVOT components
from .candidate_generator import CandidateGenerator
from .visual_overlay import VisualOverlay
from .trajectory import Trajectory
from .executor import TrajectoryExecutor
from .safety import SafetyChecker


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
            vertical_range=tuple(config.get('vertical_range', [-0.8, 0.8])),
            vlm_client=vlm_client,
            visual_overlay=None  # Will set after overlay created
        )

        self.overlay = VisualOverlay(
            image_width=config.get('image_width', 1920),
            image_height=config.get('image_height', 1080),
            fov_horizontal=config.get('fov_horizontal', 90.0),
            fov_vertical=config.get('fov_vertical', 90.0)
        )

        # Link overlay to candidate generator for inverse projection
        self.candidate_gen.overlay = self.overlay

        self.executor = TrajectoryExecutor(airsim_client, config)

        # Initialize safety checker
        self.enable_safety = config.get('enable_safety_checks', True)
        if self.enable_safety:
            self.safety_checker = SafetyChecker(config)
        else:
            self.safety_checker = None

        # Algorithm parameters
        self.max_iterations = config.get('max_iterations', 3)
        self.refinement_factor = config.get('refinement_factor', 0.5)
        self.convergence_threshold = config.get('convergence_threshold', 0.9)

        # Frame re-capture settings
        self.enable_closed_loop = config.get('enable_closed_loop', True)
        self.frame_recapture_interval = config.get('frame_recapture_interval', 1)

        # Visualization settings
        self.save_iterations = config.get('save_iterations', True)
        self.output_dir = config.get('output_dir', 'pivot_visualizations')

        # Create output directory with timestamp
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, self.timestamp)
        if self.save_iterations:
            os.makedirs(self.session_dir, exist_ok=True)

        print(f"[PivotController] Initialized with {config.get('num_candidates', 8)} candidates")
        print(f"[PivotController] Max iterations: {self.max_iterations}")
        print(f"[PivotController] Safety checks: {self.enable_safety}")
        print(f"[PivotController] Closed-loop: {self.enable_closed_loop}")
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
        print(f"\n{'='*60}")
        print(f"PIVOT NAVIGATION: {instruction}")
        print(f"{'='*60}")

        # Phase 1: Generate initial candidates
        print("\n[Phase 1] Generating initial candidates...")
        use_vlm = self.config.get('use_vlm_based_candidates', False)
        candidates = self.candidate_gen.generate_initial_candidates(
            instruction,
            image=current_frame,
            use_vlm=use_vlm
        )
        print(f"Generated {len(candidates)} initial candidates (VLM-based: {use_vlm})")

        # Apply safety filtering
        if self.enable_safety:
            print("\n[Safety] Filtering unsafe candidates...")
            candidates = self.safety_checker.filter_unsafe_candidates(candidates)
            print(f"[Safety] {len(candidates)} safe candidates remaining")

        best_trajectory = None
        search_radius = 1.0  # Initial search radius
        iteration_history = []

        # Phase 2: Iterative refinement loop with frame re-capture
        for iteration in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"PIVOT Iteration {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")

            # Re-capture frame if closed-loop enabled (after first iteration)
            if self.enable_closed_loop and iteration > 0:
                if iteration % self.frame_recapture_interval == 0:
                    print(f"[Closed-Loop] Re-capturing frame for iteration {iteration + 1}")
                    current_frame = self._capture_fresh_frame()
                    print(f"[Closed-Loop] Frame updated")

            # Overlay candidates on current frame (not frozen!)
            annotated_image = self.overlay.overlay_candidates(current_frame, candidates)

            # Save visualization
            if self.save_iterations:
                self._save_iteration_viz(annotated_image, iteration, candidates)

            # Query VLM for selection
            print(f"[Iteration {iteration + 1}] Querying VLM with {len(candidates)} candidates...")
            selection_result = self._query_vlm_for_selection(
                annotated_image,
                instruction,
                candidates,
                iteration
            )

            best_trajectory = selection_result['selected_trajectory']
            confidence = selection_result['confidence']
            reasoning = selection_result['reasoning']

            # Store iteration result
            iteration_history.append({
                'iteration': iteration + 1,
                'selected_id': best_trajectory.id,
                'confidence': confidence,
                'reasoning': reasoning,
                'search_radius': search_radius,
                'num_candidates': len(candidates)
            })

            print(f"[Iteration {iteration + 1}] Selected: Option {best_trajectory.id}")
            print(f"[Iteration {iteration + 1}] Confidence: {confidence:.3f}")
            print(f"[Iteration {iteration + 1}] Reasoning: {reasoning}")

            # Check convergence
            if confidence >= self.convergence_threshold:
                print(f"\n✓ Converged at iteration {iteration + 1} (confidence: {confidence:.3f} >= {self.convergence_threshold})")
                break

            if iteration == self.max_iterations - 1:
                print(f"\n→ Maximum iterations reached ({self.max_iterations})")
                break

            # Refine candidates around best selection
            search_radius *= self.refinement_factor
            print(f"\n[Refinement] Reducing search radius to {search_radius:.3f}")
            candidates = self.candidate_gen.refine_candidates(
                best_trajectory,
                iteration + 1,
                search_radius
            )
            print(f"[Refinement] Generated {len(candidates)} refined candidates")

            # Apply safety filtering to refined candidates
            if self.enable_safety:
                candidates = self.safety_checker.filter_unsafe_candidates(candidates)
                print(f"[Safety] {len(candidates)} safe refined candidates")

        # Phase 3: Execute selected trajectory
        print(f"\n{'='*60}")
        print(f"EXECUTING SELECTED TRAJECTORY")
        print(f"{'='*60}")
        print(f"Trajectory: {best_trajectory}")

        execution_result = self.executor.execute_trajectory(best_trajectory)

        # Save final result
        if self.save_iterations:
            self._save_final_result(
                best_trajectory,
                iteration_history,
                execution_result
            )

        print(f"\n✓ PIVOT navigation completed in {len(iteration_history)} iterations")

        return {
            'selected_trajectory': best_trajectory,
            'iterations': len(iteration_history),
            'execution': execution_result,
            'iteration_history': iteration_history,
            'instruction': instruction
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

        prompt = f"""You are a drone navigation expert. The image shows a drone camera view with {len(candidates)} numbered trajectory options overlaid as colored circles with arrows.

TASK: {instruction}

ITERATION: {iteration + 1}

The colored circles show possible waypoint destinations for the drone. Each option is numbered (IDs: {candidate_ids}).

Analyze each option considering:
1. Alignment with the task "{instruction}"
2. Safety (avoiding obstacles)
3. Efficiency (direct path)
4. Feasibility (reasonable position)

Select the BEST option and provide your reasoning.

Return JSON in this EXACT format:
{{
    "selected_option": <option_number>,
    "confidence": <0.0-1.0>,
    "reasoning": "Brief explanation of why this option is best",
    "risk_assessment": "Any potential risks or concerns"
}}

IMPORTANT:
- selected_option must be one of: {candidate_ids}
- confidence should reflect how certain you are (1.0 = very certain, 0.0 = uncertain)
- Be concise but specific in reasoning
"""

        try:
            # Query VLM
            response_text = self.vlm.generate_response(prompt, annotated_image)
            response_text = VLMClient.clean_response_text(response_text)

            # Parse JSON response
            response_json = json.loads(response_text)

            selected_id = int(response_json['selected_option'])
            confidence = float(response_json.get('confidence', 0.5))
            reasoning = response_json.get('reasoning', 'No reasoning provided')
            risk = response_json.get('risk_assessment', 'No assessment provided')

            # Find selected trajectory
            selected_traj = None
            for t in candidates:
                if t.id == selected_id:
                    selected_traj = t
                    break

            if selected_traj is None:
                print(f"Warning: VLM selected invalid ID {selected_id}, defaulting to first candidate")
                selected_traj = candidates[0]
                confidence = 0.3

            # Update trajectory with VLM feedback
            selected_traj.score = confidence
            selected_traj.reasoning = reasoning

            return {
                'selected_trajectory': selected_traj,
                'confidence': confidence,
                'reasoning': reasoning,
                'risk_assessment': risk
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing VLM response: {e}")
            print(f"Raw response: {response_text[:200]}...")

            # Fallback: select middle candidate
            fallback_traj = candidates[len(candidates) // 2]
            fallback_traj.score = 0.3
            return {
                'selected_trajectory': fallback_traj,
                'confidence': 0.3,
                'reasoning': 'Fallback due to parse error',
                'risk_assessment': 'Unknown'
            }

    def _save_iteration_viz(self,
                           annotated_image: np.ndarray,
                           iteration: int,
                           candidates: List[Trajectory]):
        """Save visualization for this iteration"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"iteration_{iteration + 1}_{timestamp}.jpg"
        img_path = os.path.join(self.session_dir, img_filename)

        cv2.imwrite(img_path, annotated_image)

        # Save candidate metadata
        json_filename = f"iteration_{iteration + 1}_{timestamp}.json"
        json_path = os.path.join(self.session_dir, json_filename)

        metadata = {
            'iteration': iteration + 1,
            'timestamp': timestamp,
            'num_candidates': len(candidates),
            'candidates': [
                {
                    'id': t.id,
                    'waypoint': {
                        'x': t.get_first_waypoint().x,
                        'y': t.get_first_waypoint().y,
                        'z': t.get_first_waypoint().z
                    },
                    'color': t.color
                }
                for t in candidates
            ]
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _save_final_result(self,
                          trajectory: Trajectory,
                          iteration_history: List[Dict],
                          execution_result: Dict):
        """Save final result summary"""
        result_file = os.path.join(self.session_dir, "final_result.json")

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

    def _capture_fresh_frame(self) -> np.ndarray:
        """
        Re-capture frame from AirSim for closed-loop iteration

        Returns:
            Fresh camera frame as numpy array (BGR format)
        """
        import airsim

        try:
            # Capture image from AirSim
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            if responses and len(responses) > 0:
                # Convert to numpy array
                img_response = responses[0]
                img1d = np.frombuffer(img_response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(img_response.height, img_response.width, 3)

                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                return img_bgr
            else:
                print("[Controller] Warning: No image response, using previous frame")
                return None

        except Exception as e:
            print(f"[Controller] Error capturing frame: {e}")
            return None
