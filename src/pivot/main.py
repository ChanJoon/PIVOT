"""
PIVOT Main Entry Point

Initializes AirSim connection, VLM client, and PIVOT controller,
then runs the main navigation loop.
"""

import argparse
import sys
import time
import yaml
import airsim
import numpy as np
import cv2
from pathlib import Path

# Import PIVOT components
from .controller import PivotController
from .vlm_client import VLMClient


def get_rgb_from_response(img_response):
    """
    Convert AirSim image response to RGB numpy array

    Args:
        img_response: AirSim ImageResponse object

    Returns:
        RGB image as numpy array
    """
    img1d = np.frombuffer(img_response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(img_response.height, img_response.width, 3)
    return img_rgb


def capture_airsim_image(client, camera_name: str = "0"):
    """
    Capture image from AirSim simulator

    Args:
        client: AirSim MultirotorClient
        camera_name: Camera name/ID

    Returns:
        BGR image as numpy array (OpenCV format)
    """
    # Request uncompressed RGB image
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    ])

    if responses:
        img_rgb = get_rgb_from_response(responses[0])
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    else:
        print("Warning: No image response from AirSim")
        return None


def check_camera_settings(client, camera_name: str = "0"):
    """
    Check camera resolution and warn if too low

    Args:
        client: AirSim MultirotorClient
        camera_name: Camera name/ID
    """
    print("\n[Camera Check] Verifying camera settings...")

    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    ])

    if responses:
        width = responses[0].width
        height = responses[0].height
        print(f"[Camera Check] Resolution: {width}x{height}")

        if width < 640 or height < 480:
            print("⚠ WARNING: Camera resolution is very low!")
            print("  Recommended: 1920x1080 or higher")
            print("  See: src/spf/airsim/settings.json.example")
    else:
        print("[Camera Check] ⚠ Could not verify camera settings")


def main():
    """Main entry point for PIVOT navigation"""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="PIVOT: Prompting with Iterative Visual Optimization for Drone Navigation"
    )
    parser.add_argument('--config', type=str, default='config_pivot.yaml',
                       help='Path to configuration file')
    parser.add_argument('--instruction', type=str, default=None,
                       help='Navigation instruction (overrides config)')
    parser.add_argument('--single-shot', action='store_true',
                       help='Run single navigation then exit')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    args = parser.parse_args()

    # Load configuration
    print(f"\n{'='*60}")
    print("PIVOT - Prompting with Iterative Visual Optimization")
    print(f"{'='*60}")

    config_path = args.config
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please create config_pivot.yaml based on the template")
        return 1

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n[Config] Loaded from: {config_path}")
    print(f"[Config] API Provider: {config.get('api_provider', 'gemini')}")
    print(f"[Config] Model: {config.get('model_name', 'default')}")
    print(f"[Config] Max Iterations: {config.get('max_iterations', 3)}")
    print(f"[Config] Candidates: {config.get('num_candidates', 8)}")

    # Initialize AirSim connection
    print(f"\n{'='*60}")
    print("Initializing AirSim Connection")
    print(f"{'='*60}")

    max_attempts = config.get('connection_retries', 5)
    retry_delay = config.get('connection_retry_delay', 5)
    client = None

    for attempt in range(1, max_attempts + 1):
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            print("✓ Connected to AirSim")

            client.enableApiControl(True)
            print("✓ API control enabled")

            client.armDisarm(True)
            print("✓ Drone armed")
            break
        except Exception as e:
            print(f"✗ Failed to connect to AirSim (attempt {attempt}/{max_attempts}): {e}")
            if attempt == max_attempts:
                print("\nMake sure AirSim is running and accessible")
                return 1
            print(f"Retrying in {retry_delay}s... (start AirSim if it is not running)")
            time.sleep(retry_delay)

    # Check camera settings
    camera_name = config.get('camera_name', '0')
    check_camera_settings(client, camera_name)

    # Initialize VLM client (from see-point-fly)
    print(f"\n{'='*60}")
    print("Initializing VLM Client")
    print(f"{'='*60}")

    try:
        vlm_client = VLMClient(
            api_provider=config.get('api_provider', 'gemini'),
            model_name=config.get('model_name', 'gemini-2.5-flash')
        )
        print("✓ VLM client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize VLM client: {e}")
        print("\nCheck your .env file and API keys")
        return 1

    # Initialize PIVOT controller
    print(f"\n{'='*60}")
    print("Initializing PIVOT Controller")
    print(f"{'='*60}")

    try:
        controller = PivotController(client, vlm_client, config)
        print("✓ PIVOT controller initialized")
    except Exception as e:
        print(f"✗ Failed to initialize PIVOT controller: {e}")
        return 1

    # Takeoff
    print(f"\n{'='*60}")
    print("Taking Off")
    print(f"{'='*60}")

    try:
        print("Initiating takeoff...")
        client.takeoffAsync().join()
        print("✓ Takeoff complete")
        time.sleep(2)  # Stabilize after takeoff
    except Exception as e:
        print(f"✗ Takeoff failed: {e}")
        return 1

    print(f"\n{'='*60}")
    print("🚁 PIVOT Navigation Ready")
    print(f"{'='*60}")

    # Get navigation instruction
    instruction = args.instruction or config.get('default_instruction', 'fly to the red car')
    print(f"\nNavigation Instruction: '{instruction}'")

    # Main control loop
    command_loop_delay = config.get('command_loop_delay', 1)
    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            print(f"\n{'#'*60}")
            print(f"NAVIGATION CYCLE {cycle_count}")
            print(f"{'#'*60}")

            # Capture current frame
            print("\n[Capture] Getting image from AirSim...")
            frame = capture_airsim_image(client, camera_name)

            if frame is None:
                print("Error: Failed to capture image")
                break

            print(f"[Capture] Frame captured: {frame.shape[1]}x{frame.shape[0]}")

            # Run PIVOT navigation
            result = controller.navigate_with_pivot(frame, instruction)

            print(f"\n{'='*60}")
            print(f"CYCLE {cycle_count} SUMMARY")
            print(f"{'='*60}")
            print(f"Iterations: {result['iterations']}")
            print(f"Selected: Trajectory {result['selected_trajectory'].id}")
            print(f"Execution: {result['execution']['success']}")

            # Single-shot mode: exit after one navigation
            if args.single_shot:
                print("\n[Single-shot mode] Exiting after one navigation")
                break

            # Wait before next cycle
            if command_loop_delay > 0:
                print(f"\nWaiting {command_loop_delay}s before next cycle...")
                time.sleep(command_loop_delay)

    except KeyboardInterrupt:
        print("\n\n[Interrupted] User requested stop")

    except Exception as e:
        print(f"\n\n[Error] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print(f"\n{'='*60}")
        print("Shutting Down")
        print(f"{'='*60}")

        try:
            print("Landing drone...")
            client.landAsync().join()
            print("✓ Landed")

            time.sleep(2)

            print("Disarming...")
            client.armDisarm(False)
            print("✓ Disarmed")

            print("Disabling API control...")
            client.enableApiControl(False)
            print("✓ API control disabled")

        except Exception as e:
            print(f"Warning: Cleanup error: {e}")

        print(f"\n{'='*60}")
        print(f"PIVOT Session Complete")
        print(f"Total Cycles: {cycle_count}")
        print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
