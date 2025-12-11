"""
Parallel PIVOT Controller - Ensemble Execution

Runs E parallel PIVOT instances and aggregates results via voting/confidence weighting.

Based on PIVOT paper Section 3.3: "Robust PIVOT with Parallel Calls"
"""

import concurrent.futures
import copy
import numpy as np
from typing import List, Dict, Any
from collections import Counter

from .controller import PivotController
from .trajectory import Trajectory, Waypoint


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
        self.num_instances = config.get('num_parallel_instances', 1)
        self.aggregation_method = config.get('aggregation_method', 'voting')

        print(f"\n[ParallelPIVOT] Initializing with {self.num_instances} parallel instances")
        print(f"[ParallelPIVOT] Aggregation method: {self.aggregation_method}")

        # Create E independent controllers
        # Each gets a copy of the VLM client for thread safety
        self.controllers = []
        for i in range(self.num_instances):
            # Clone VLM client (thread-safe copy)
            instance_vlm = self._clone_vlm_client(vlm_client)

            # Create controller instance
            controller = PivotController(airsim_client, instance_vlm, config)
            self.controllers.append(controller)

        print(f"[ParallelPIVOT] Created {len(self.controllers)} controller instances")

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
        aggregated = self._aggregate_results(results)

        # Execute aggregated trajectory
        print(f"\n[ParallelPIVOT] Executing aggregated trajectory...")
        from .executor import TrajectoryExecutor
        executor_instance = TrajectoryExecutor(self.client, self.config)
        execution_result = executor_instance.execute_trajectory(aggregated['selected_trajectory'])

        aggregated['execution'] = execution_result

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

    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate results from E parallel instances

        Methods:
        - 'voting': Select trajectory that appears most frequently
        - 'confidence': Weight by VLM confidence scores
        - 'average': Average waypoint positions

        Args:
            results: List of PIVOT results from parallel instances

        Returns:
            Aggregated result with selected trajectory
        """
        if self.aggregation_method == 'voting':
            return self._aggregate_by_voting(results)
        elif self.aggregation_method == 'confidence':
            return self._aggregate_by_confidence(results)
        elif self.aggregation_method == 'average':
            return self._aggregate_by_averaging(results)
        else:
            # Fallback: return highest confidence result
            print(f"[ParallelPIVOT] Unknown aggregation method '{self.aggregation_method}', using confidence")
            return self._aggregate_by_confidence(results)

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
