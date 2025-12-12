"""
Parallel PIVOT Controller - Ensemble Execution

Runs E parallel PIVOT instances and aggregates results via voting/confidence weighting.

Based on PIVOT paper Section 3.3: "Robust PIVOT with Parallel Calls"
"""

import concurrent.futures
import copy
import os
import time
import cv2
import numpy as np
from typing import List, Dict, Any
from collections import Counter

from .controller import PivotController
from .trajectory import Trajectory, Waypoint
from .visual_overlay import VisualOverlay
from .vlm_client import VLMClient


class ParallelPivotController:
    """
    Runs multiple PIVOT instances in parallel and aggregates results

    Architecture:
    - E parallel instances using ThreadPoolExecutor
    - Each instance has independent VLM client (for thread safety)
    - Shared AirSim client (read-only operations are thread-safe)
    - Aggregation via voting, confidence weighting, or averaging
    """

    def __init__(self,
                 airsim_client,
                 vlm_client,
                 config: Dict[str, Any]):
        """
        Initialize parallel controller

        Args:
            airsim_client: Shared AirSim client
            vlm_client: Base VLM client (will be cloned per instance)
            config: Configuration with 'num_parallel_instances'
        """
        self.client = airsim_client
        self.config = config
        self.vlm = vlm_client
        self.num_instances = config.get('num_parallel_instances', 1)
        self.aggregation_method = config.get('aggregation_method', 'voting')
        self.save_iterations = config.get('save_iterations', True)
        self.output_dir = config.get('output_dir', 'pivot_visualizations')

        # Create a single output directory for the parallel run (suppress per-instance artifacts).
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, self.timestamp)
        if self.save_iterations:
            os.makedirs(self.session_dir, exist_ok=True)

        print(f"\n[ParallelPIVOT] Initializing with {self.num_instances} parallel instances")
        print(f"[ParallelPIVOT] Aggregation method: {self.aggregation_method}")

        # Create E independent controllers
        # Each gets a copy of the VLM client for thread safety
        self.controllers = []
        for i in range(self.num_instances):
            # Clone VLM client (thread-safe copy)
            instance_vlm = self._clone_vlm_client(vlm_client)

            # Create controller instance
            instance_config = dict(config)
            instance_config['save_iterations'] = False
            instance_config['enable_advanced_visualization'] = False
            instance_config['execute_selected_trajectory'] = False
            instance_config['collect_iteration_artifacts'] = True
            controller = PivotController(airsim_client, instance_vlm, instance_config)
            self.controllers.append(controller)

        print(f"[ParallelPIVOT] Created {len(self.controllers)} controller instances")
        self.overlay = VisualOverlay(
            image_width=config.get('image_width', 1920),
            image_height=config.get('image_height', 1080),
            fov_horizontal=config.get('fov_horizontal', 90.0),
            fov_vertical=config.get('fov_vertical', 90.0)
        )

    def _clone_vlm_client(self, vlm_client):
        """
        Clone VLM client for thread safety

        Creates a new VLM client instance with the same configuration
        but independent state to avoid race conditions.

        Args:
            vlm_client: Original VLM client

        Returns:
            Cloned VLM client instance
        """
        # Simple deep copy for now
        # In production, might need custom cloning logic
        try:
            return copy.deepcopy(vlm_client)
        except Exception as e:
            print(f"[ParallelPIVOT] WARNING: VLM client deep copy failed: {e}")
            print(f"[ParallelPIVOT] Using original client (not thread-safe)")
            return vlm_client

    def navigate_with_pivot_parallel(self,
                                     current_frame: np.ndarray,
                                     instruction: str) -> Dict[str, Any]:
        """
        Run E parallel PIVOT instances and aggregate results

        Flow:
        1. Launch E instances in parallel (ThreadPoolExecutor)
        2. Each instance runs full PIVOT refinement loop
        3. Collect all results
        4. Aggregate via voting/confidence/averaging
        5. Execute aggregated trajectory

        Args:
            current_frame: Current camera frame
            instruction: Navigation instruction

        Returns:
            Aggregated result with selected trajectory and execution
        """
        print(f"\n{'='*70}")
        print(f"PARALLEL PIVOT EXECUTION ({self.num_instances} instances)")
        print(f"{'='*70}")
        print(f"Instruction: {instruction}")
        print(f"Aggregation: {self.aggregation_method}")

        # Run instances in parallel using ThreadPoolExecutor
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_instances) as executor:
            print(f"\n[ParallelPIVOT] Launching {self.num_instances} parallel instances...")

            # Submit all instances
            futures = []
            for i, controller in enumerate(self.controllers):
                future = executor.submit(
                    self._run_single_instance,
                    controller, current_frame, instruction, i
                )
                futures.append(future)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[ParallelPIVOT] Instance {result['instance_id']} completed")
                except Exception as e:
                    print(f"[ParallelPIVOT] Instance failed with error: {e}")

        print(f"\n[ParallelPIVOT] All {len(results)} instances completed successfully")

        if not results:
            raise RuntimeError("All parallel PIVOT instances failed!")

        # Aggregate results
        print(f"\n[ParallelPIVOT] Aggregating {len(results)} results...")
        aggregated = self._aggregate_results(results, current_frame, instruction)

        # Execute aggregated trajectory
        print(f"\n[ParallelPIVOT] Executing aggregated trajectory...")
        from .executor import TrajectoryExecutor
        executor_instance = TrajectoryExecutor(self.client, self.config)
        execution_result = executor_instance.execute_trajectory(aggregated['selected_trajectory'])

        aggregated['execution'] = execution_result

        # Save only the chosen instance's iteration artifacts (+ final selection).
        if self.save_iterations:
            run_timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(self.session_dir, f"run_parallel_{run_timestamp}")
            os.makedirs(run_dir, exist_ok=True)

            artifacts = aggregated.get('iteration_artifacts', [])
            for item in artifacts:
                it = item.get('iteration', 0)
                cand_bytes = item.get('candidates_jpg')
                sel_bytes = item.get('selected_jpg')
                if cand_bytes:
                    out_path = os.path.join(run_dir, f"iteration_{it:02d}_candidates.jpg")
                    with open(out_path, "wb") as f:
                        f.write(cand_bytes)
                if sel_bytes:
                    out_path = os.path.join(run_dir, f"iteration_{it:02d}_selected.jpg")
                    with open(out_path, "wb") as f:
                        f.write(sel_bytes)

            final_traj = aggregated['selected_trajectory']
            single = Trajectory(
                waypoints=[final_traj.get_first_waypoint()],
                id=1,
                color=(0, 255, 0)
            )
            final_image = self.overlay.overlay_candidates_with_selection(current_frame, [single], selected_id=1)
            final_path = os.path.join(run_dir, "final_selected.jpg")
            cv2.imwrite(final_path, final_image)
            print(f"[Viz] Saved final selection: {final_path}")

        print(f"\n[ParallelPIVOT] ✓ Parallel execution complete")
        return aggregated

    def _run_single_instance(self,
                            controller: PivotController,
                            frame: np.ndarray,
                            instruction: str,
                            instance_id: int) -> Dict[str, Any]:
        """
        Run single PIVOT instance (executed in parallel thread)

        Args:
            controller: PIVOT controller instance
            frame: Current camera frame
            instruction: Navigation instruction
            instance_id: Instance ID for logging

        Returns:
            PIVOT result with instance_id added
        """
        print(f"[ParallelPIVOT] Instance {instance_id} starting...")

        try:
            # Run full PIVOT refinement
            result = controller.navigate_with_pivot(frame, instruction)

            # Add instance ID for tracking
            result['instance_id'] = instance_id

            # Extract confidence for aggregation
            traj = result['selected_trajectory']
            result['confidence'] = traj.score if hasattr(traj, 'score') else 0.0

            print(f"[ParallelPIVOT] Instance {instance_id} completed (confidence: {result['confidence']:.3f})")

            return result

        except Exception as e:
            print(f"[ParallelPIVOT] Instance {instance_id} FAILED: {e}")
            raise

    def _aggregate_results(self, results: List[Dict], current_frame: np.ndarray, instruction: str) -> Dict[str, Any]:
        """
        Aggregate results from E parallel instances (frame-aware for VLM rerank).
        """
        if self.aggregation_method == 'vlm':
            return self._aggregate_by_vlm_rerank(results, current_frame, instruction)
        # Delegate to the frame-free methods.
        return self._aggregate_results_frame_free(results)

    def _aggregate_results_frame_free(self, results: List[Dict]) -> Dict[str, Any]:
        if self.aggregation_method == 'voting':
            return self._aggregate_by_voting(results)
        if self.aggregation_method == 'confidence':
            return self._aggregate_by_confidence(results)
        if self.aggregation_method == 'average':
            return self._aggregate_by_averaging(results)
        print(f"[ParallelPIVOT] Unknown aggregation method '{self.aggregation_method}', using confidence")
        return self._aggregate_by_confidence(results)

    def _aggregate_by_vlm_rerank(self, results: List[Dict], current_frame: np.ndarray, instruction: str) -> Dict[str, Any]:
        """
        Paper Section 3.3 option (2): query the VLM again to select the best action among E.
        """
        # Build E candidate trajectories from each instance's final selection.
        rerank_candidates: List[Trajectory] = []
        for i, r in enumerate(results):
            traj = r['selected_trajectory']
            wp = traj.get_first_waypoint()
            rerank_candidates.append(
                Trajectory(
                    waypoints=[Waypoint(x=wp.x, y=wp.y, z=wp.z)],
                    id=i + 1,
                    color=(255, 0, 0)  # color overridden by overlay palette anyway
                )
            )

        annotated = self.overlay.overlay_candidates(current_frame, rerank_candidates)
        candidate_ids = [t.id for t in rerank_candidates]

        prompt = f"""You are a drone that cannot fly through obstacles. This is the image you are seeing right now.
I have annotated it with numbered circles and arrows. Each number represents a candidate waypoint destination you can fly toward.
The arrow shows the direction from the drone (image center) to that waypoint.

Your task is to choose which single circle you should pick for the task of: {instruction}

Choose the 1 best candidate number.
Do NOT choose routes that would go through obstacles or unsafe gaps.
Skip analysis and provide your answer at the end in a json file of this form:
{{"points": []}}

IMPORTANT:
- "points" must contain EXACTLY ONE number from: {candidate_ids}
"""

        try:
            response_text = self.vlm.generate_response(prompt, annotated)
            response_text = VLMClient.clean_response_text(response_text)
            import json
            response_json = json.loads(response_text)
            points = response_json.get('points', [])
            if not isinstance(points, list) or not points:
                raise ValueError("Missing points")
            choice = int(points[0])
            if choice not in candidate_ids:
                raise ValueError(f"Invalid choice: {choice}")
        except Exception as e:
            print(f"[ParallelPIVOT] VLM rerank failed ({e}); falling back to confidence")
            return self._aggregate_by_confidence(results)

        chosen_result = results[choice - 1]
        chosen_result = dict(chosen_result)
        chosen_result['aggregation_method'] = 'vlm'
        chosen_result['rerank_choice'] = choice
        return chosen_result

    def _aggregate_by_voting(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Select trajectory by majority vote on waypoint positions

        Groups similar trajectories (within distance threshold) and
        picks the most common one.

        Args:
            results: List of PIVOT results

        Returns:
            Result with majority-voted trajectory
        """
        # Group similar trajectories (within distance threshold)
        trajectory_groups = []
        threshold = 0.5  # 0.5m grouping threshold

        for result in results:
            traj = result['selected_trajectory']
            wp = traj.get_first_waypoint()

            # Find matching group
            matched = False
            for group in trajectory_groups:
                group_wp = group['waypoint']
                dist = wp.distance_to(group_wp)

                if dist < threshold:
                    group['votes'] += 1
                    group['trajectories'].append(traj)
                    group['results'].append(result)
                    matched = True
                    break

            if not matched:
                # Create new group
                trajectory_groups.append({
                    'waypoint': wp,
                    'votes': 1,
                    'trajectories': [traj],
                    'results': [result]
                })

        # Select group with most votes
        best_group = max(trajectory_groups, key=lambda g: g['votes'])

        # Within best group, pick highest confidence
        best_result = max(best_group['results'],
                         key=lambda r: r.get('confidence', 0.0))

        print(f"[Aggregation] Voting: {best_group['votes']}/{len(results)} votes for selected trajectory")

        return {
            'selected_trajectory': best_result['selected_trajectory'],
            'iterations': best_result['iterations'],
            'aggregation_method': 'voting',
            'num_votes': best_group['votes'],
            'num_instances': len(results),
            'all_results': results
        }

    def _aggregate_by_confidence(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Select trajectory with highest VLM confidence

        Args:
            results: List of PIVOT results

        Returns:
            Result with highest confidence trajectory
        """
        best_result = max(results, key=lambda r: r.get('confidence', 0.0))

        print(f"[Aggregation] Confidence: selected trajectory with confidence {best_result['confidence']:.3f}")

        return {
            'selected_trajectory': best_result['selected_trajectory'],
            'iterations': best_result['iterations'],
            'aggregation_method': 'confidence',
            'best_confidence': best_result['confidence'],
            'num_instances': len(results),
            'all_results': results
        }

    def _aggregate_by_averaging(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Average waypoint positions across all trajectories

        Creates a new trajectory with averaged waypoint positions.

        Args:
            results: List of PIVOT results

        Returns:
            Result with averaged trajectory
        """
        # Extract all trajectories
        trajectories = [r['selected_trajectory'] for r in results]

        # Average first waypoint positions
        avg_x = np.mean([t.get_first_waypoint().x for t in trajectories])
        avg_y = np.mean([t.get_first_waypoint().y for t in trajectories])
        avg_z = np.mean([t.get_first_waypoint().z for t in trajectories])

        # Create averaged trajectory
        avg_waypoint = Waypoint(x=avg_x, y=avg_y, z=avg_z)
        avg_trajectory = Trajectory(
            waypoints=[avg_waypoint],
            id=0,  # Aggregated ID
            color=(128, 128, 128),  # Gray
            score=np.mean([r.get('confidence', 0.0) for r in results])
        )

        print(f"[Aggregation] Averaging: averaged {len(results)} trajectories")
        print(f"[Aggregation] Average waypoint: ({avg_x:.2f}, {avg_y:.2f}, {avg_z:.2f})")

        return {
            'selected_trajectory': avg_trajectory,
            'iterations': results[0]['iterations'],  # Use first instance's iteration count
            'aggregation_method': 'average',
            'num_instances': len(results),
            'all_results': results
        }
