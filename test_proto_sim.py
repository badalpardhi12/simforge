#!/usr/bin/env python3
"""
Standalone Proto-Sim Test Script

Run this via SSH to test the proto-sim visualization pipeline.
This script connects directly to the WebSocket server and runs a simple protocol.

Usage:
    python test_proto_sim.py [--server IP] [--port PORT]
    
Example:
    python test_proto_sim.py --server 192.168.1.12 --port 8766
"""

import asyncio
import json
import time
import argparse
import logging

# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'websockets'])
    import websockets


async def send_heartbeat(ws, client_id: str, stop_event: asyncio.Event):
    """Send heartbeats to keep connection alive."""
    seq = 0
    while not stop_event.is_set():
        try:
            heartbeat = {
                'type': 'heartbeat',
                'client_id': client_id,
                'sequence': seq,
                'timestamp': time.time(),
            }
            await ws.send(json.dumps(heartbeat))
            logger.debug(f"Sent heartbeat #{seq}")
            seq += 1
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            break


async def test_proto_sim(server: str, port: int):
    """Run a simple proto-sim test."""
    uri = f"ws://{server}:{port}"
    client_id = f"test_client_{int(time.time())}"
    
    logger.info(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=60) as ws:
            logger.info("Connected!")
            
            # Start heartbeat task
            stop_heartbeat = asyncio.Event()
            heartbeat_task = asyncio.create_task(
                send_heartbeat(ws, client_id, stop_heartbeat)
            )
            
            try:
                # Step 1: Get environment info
                logger.info("=" * 60)
                logger.info("Step 1: Getting environment info...")
                request = {
                    'type': 'rpc',
                    'request_id': 'req_001',
                    'method': 'get_environment_info',
                    'params': {},
                    'client_id': client_id,
                }
                await ws.send(json.dumps(request))
                
                response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(response)
                logger.info(f"Environment info: {json.dumps(data, indent=2)}")
                
                # Step 2: Run a simple proto-sim with just 3 poses
                logger.info("=" * 60)
                logger.info("Step 2: Running proto-sim with 3 poses...")
                
                proto_request = {
                    'type': 'rpc',
                    'request_id': 'req_002',
                    'method': 'run_proto_sim',
                    'client_id': client_id,
                    'params': {
                        'robot_name': 'nakul_ur5e',
                        'target_object': 'face_link',
                        'mode': 'simulation',  # Only simulation, no real robot
                        'horiz': [0],          # Just center
                        'vert': [0],           # Just center
                        'distance': [350],     # One distance (mm)
                        'roll': [-90],         # One roll
                        'pitch': [-30, 0, 30], # Three pitch angles
                        'yaw': [0],            # Just center
                        'idle_time': 1.0,      # 1 second at each pose
                        'move_speed': 0.5,
                        'randomize': False,
                    }
                }
                
                logger.info(f"Sending proto-sim request...")
                await ws.send(json.dumps(proto_request))
                
                # Listen for feedback and result
                pose_count = 0
                start_time = time.time()
                
                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        data = json.loads(response)
                        msg_type = data.get('type', 'unknown')
                        
                        if msg_type == 'rpc_feedback':
                            pose_idx = data.get('current_pose_index', 0)
                            total = data.get('total_poses', 0)
                            status = data.get('status', '')
                            pose_name = data.get('current_pose_name', '')
                            progress = data.get('progress_percent', 0)
                            
                            logger.info(f"  Pose {pose_idx+1}/{total}: {pose_name} - {status} ({progress:.1f}%)")
                            pose_count = pose_idx + 1
                            
                        elif msg_type == 'rpc_result':
                            elapsed = time.time() - start_time
                            logger.info("=" * 60)
                            logger.info(f"Proto-sim completed!")
                            logger.info(f"  Success: {data.get('success')}")
                            logger.info(f"  Message: {data.get('message')}")
                            logger.info(f"  Completed: {data.get('completed')}/{data.get('total')}")
                            logger.info(f"  Time: {elapsed:.1f}s")
                            break
                            
                        elif msg_type == 'error':
                            logger.error(f"Server error: {data.get('message')}")
                            break
                            
                        else:
                            logger.debug(f"Received: {msg_type} - {data}")
                            
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for response, continuing...")
                        continue
                
            finally:
                # Stop heartbeat
                stop_heartbeat.set()
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection closed: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def test_direct_joint_publish(server: str):
    """
    Test publishing joint states directly inside the Docker container.
    This bypasses the WebSocket and tests the ROS 2 pipeline.
    """
    logger.info("=" * 60)
    logger.info("This test requires running inside the Docker container.")
    logger.info("Run: sudo docker exec -it simforge_server_dev bash")
    logger.info("Then: python3 /ros2_ws/src/simforge_server/test_joint_publish.py")


def main():
    parser = argparse.ArgumentParser(description="Test Proto-Sim Pipeline")
    parser.add_argument("--server", default="192.168.1.12", help="Server IP")
    parser.add_argument("--port", type=int, default=8766, help="WebSocket port")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Proto-Sim Test Script")
    logger.info("=" * 60)
    logger.info(f"Server: {args.server}:{args.port}")
    logger.info("")
    
    asyncio.run(test_proto_sim(args.server, args.port))


if __name__ == "__main__":
    main()
