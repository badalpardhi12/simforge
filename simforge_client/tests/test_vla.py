#!/usr/bin/env python3
"""
Test script for Simforge Client - VLA Task Execution

Tests VLA-guided task execution.

Usage:
    python test_vla.py --server <SERVER_IP>
    python test_vla.py --server 192.168.1.100 --instruction "pick up the red cube"
"""

import asyncio
import argparse
import sys

sys.path.insert(0, '..')

from simforge_client import SimforgeClient
from simforge_client.command_client import MoveFeedback


def feedback_callback(feedback: MoveFeedback):
    """Print task feedback."""
    print(f"  [{feedback.progress*100:.0f}%] {feedback.status}")


async def test_vla_task(client: SimforgeClient, instruction: str):
    """Test VLA task execution."""
    print(f"\n=== Testing VLA Task Execution ===")
    print(f"Instruction: \"{instruction}\"")
    print()
    
    result = await client.execute_task(
        instruction=instruction,
        robot_name="ur20",
        max_velocity_scale=0.3,
        timeout=120.0,
        feedback_callback=feedback_callback,
    )
    
    print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Message: {result.message}")
    print(f"Execution time: {result.execution_time_sec:.2f}s")
    
    return result.success


async def main():
    parser = argparse.ArgumentParser(description='Test VLA Task Execution')
    parser.add_argument('--server', type=str, default='localhost',
                        help='Server IP address')
    parser.add_argument('--port', type=int, default=8765,
                        help='Command Gateway port')
    parser.add_argument('--instruction', type=str, 
                        default='pick up the red cube',
                        help='Natural language instruction')
    
    args = parser.parse_args()
    
    print(f"Connecting to server at {args.server}:{args.port}...")
    
    async with SimforgeClient(
        server_ip=args.server,
        command_port=args.port,
        client_id="vla_test_client"
    ) as client:
        
        print("Connected!")
        
        # Test VLA task
        await test_vla_task(client, args.instruction)
        
        # Test a few more instructions
        test_instructions = [
            "move to the left",
            "place the object on the table",
        ]
        
        for instruction in test_instructions:
            print("\n" + "="*60)
            await test_vla_task(client, instruction)
        
        print("\n=== All VLA tests completed ===")


if __name__ == '__main__':
    asyncio.run(main())
