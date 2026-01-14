"""Headless runner that mirrors the proto_sim GUI simulate button."""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simforge_new.core import Backend
from simforge_new.control.proto_simulation import (
    ProtoSimParameters,
    generate_proto_poses,
    execute_proto_sim,
)
from simforge_new.control.session import SimulationSession


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("proto_sim.default")

    session = await SimulationSession.create(
        "simforge_new/environment/presets/face_robot.yaml",
        backend=Backend.GPU,
        logger=logger,
    )
    try:
        robot = next((robot for robot in session.spec.robots if robot.name == "TX2_90XL_1"), None)
        if robot is None:
            raise RuntimeError("Robot 'TX2_90XL_1' not found in spec")

        frame_map = session.reference_frames()
        face_frame = frame_map.get("obj:face_object_0")
        if face_frame is None:
            raise RuntimeError("face_object_0 frame not available")

        params = ProtoSimParameters(
            horiz=[-100.0, 0.0, 100.0],
            vert=[-100.0, 0.0, 100.0],
            distance=[300.0],
            roll=[0.0],
            pitch=[0.0],
            yaw=[0.0],
        )
        poses = generate_proto_poses(params)
        result = await execute_proto_sim(
            session,
            robot,
            frame_map,
            {"face_object_0": "obj:face_object_0"},
            poses,
            idle_timeout=20.0,
            logger=logger,
            progress=None,
        )
        successes = result.total_successes()
        failures = result.total_failures()
        print(f"ProtoSim default run complete: {successes} success / {failures} failure")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
