#!/usr/bin/env python3
"""
Test script for Simforge Client - Move Robot

Tests basic robot movement functionality via the Command Gateway.

Usage:
    python test_move.py --server <SERVER_IP>
    python test_move.py --server 192.168.1.100 --joints 0 -1.57 1.57 0 0 0
"""

import asyncio
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, '..')

from simforge_client import SimforgeClient
from simforge_client.command_client import Pose, MoveFeedback


def feedback_callback(feedback: MoveFeedback):
    """Print movement feedback."""
    print(f"  Progress: {feedback.progress*100:.1f}% - {feedback.status}")


async def test_move_joints(client: SimforgeClient, joints: list):
    """Test joint movement."""
    print(f"\n=== Testing Joint Move ===")
    print(f"Target joints: {joints}")
    
    result = await client.move_robot(
        robot_name="nakul_ur5e",
        target_joints=joints,
        velocity_scale=0.3,
        collision_check=True,
        feedback_callback=feedback_callback,
    )
    
    print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Message: {result.message}")
    print(f"Execution time: {result.execution_time_sec:.2f}s")
    if result.final_joint_positions:
        print(f"Final joints: {[f'{j:.3f}' for j in result.final_joint_positions]}")
    
    return result.success


async def test_move_cartesian(client: SimforgeClient, pose: Pose):
    """Test Cartesian movement."""
    print(f"\n=== Testing Cartesian Move ===")
    print(f"Target pose: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})")
    
    result = await client.move_robot(
        robot_name="nakul_ur5e",
        target_pose=pose,
        velocity_scale=0.2,
        collision_check=True,
        feedback_callback=feedback_callback,
    )
    
    print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Message: {result.message}")
    print(f"Execution time: {result.execution_time_sec:.2f}s")
    
    return result.success


async def test_get_state(client: SimforgeClient):
    """Test getting robot state."""
    print(f"\n=== Testing Get Robot State ===")
    
    state = await client.get_robot_state("ur20")
    
    if state:
        print(f"Robot state received:")
        print(f"  Mode: {state.get('robot_mode', 'unknown')}")
        print(f"  Joints: {state.get('joint_positions', [])}")
        print(f"  Protective stop: {state.get('protective_stop_active', False)}")
        print(f"  Emergency stop: {state.get('emergency_stop_active', False)}")
        return True
    else:
        print("Failed to get robot state")
        return False


async def test_ping(client: SimforgeClient):
    """Test ping latency."""
    print(f"\n=== Testing Ping ===")
    
    latencies = []
    for i in range(5):
        latency = await client.ping()
        latencies.append(latency)
        print(f"  Ping {i+1}: {latency:.1f}ms")
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average latency: {avg_latency:.1f}ms")
    
    return avg_latency < 100  # Pass if < 100ms


async def main():
    parser = argparse.ArgumentParser(description='Test Simforge Client')
    parser.add_argument('--server', type=str, default='localhost',
                        help='Server IP address')
    parser.add_argument('--port', type=int, default=8765,
                        help='Command Gateway port')
    parser.add_argument('--joints', type=float, nargs=6, default=None,
                        help='Target joint positions (6 values)')
    parser.add_argument('--pose', type=float, nargs=3, default=None,
                        help='Target Cartesian pose (x, y, z)')
    
    args = parser.parse_args()
    
    print(f"Connecting to server at {args.server}:{args.port}...")
    
    async with SimforgeClient(
        server_ip=args.server,
        command_port=args.port,
        client_id="test_client"
    ) as client:
        
        print("Connected!")
        
        # Test ping
        await test_ping(client)
        
        # Test get state
        await test_get_state(client)
        
        # Test joint move
        if args.joints:
            joints = args.joints
        else:
            # Default: home position
            joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        
        await test_move_joints(client, joints)
        
        # Test Cartesian move
        if args.pose:
            pose = Pose(x=args.pose[0], y=args.pose[1], z=args.pose[2])
        else:
            pose = Pose(x=0.4, y=0.0, z=0.3)
        
        await test_move_cartesian(client, pose)
        
        print("\n=== All tests completed ===")


if __name__ == '__main__':
    asyncio.run(main())
