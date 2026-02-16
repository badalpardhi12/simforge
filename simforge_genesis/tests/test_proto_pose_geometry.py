"""Unit tests for proto_sim pose sampling geometry helpers."""
from __future__ import annotations

import math

from simforge_genesis.control import proto_simulation as sim


def _norm(vec) -> float:
    return math.sqrt(sum(component * component for component in vec))


def test_rot_vec_respects_pitch_for_all_yaw() -> None:
    """Pitch should contribute a vertical component regardless of yaw."""

    pitch_deg = 30.0
    yaw_deg = 90.0
    distance_mm = 300.0

    vec = sim._rot_vec(pitch_deg, yaw_deg, distance_mm)

    expected_z = sim._mm_to_m(distance_mm) * math.sin(math.radians(pitch_deg))
    assert math.isclose(vec[2], expected_z, abs_tol=1e-9)

    expected_length = sim._mm_to_m(distance_mm)
    assert math.isclose(_norm(vec), expected_length, rel_tol=1e-9)


def test_generate_proto_pose_points_tool_towards_pivot() -> None:
    """The sampled orientation should align the tool +Z axis with the pivot."""

    params = sim.ProtoSimParameters(
        horiz=[0.0],
        vert=[0.0],
        distance=[300.0],
        roll=[0.0],
        pitch=[-30.0],
        yaw=[0.0],
    )

    pose = sim.generate_proto_poses(params)[0]

    # Extract the tool's +Z axis from the quaternion-produced rotation matrix.
    rot = sim._quaternion_to_matrix_wxyz(pose.orientation_quat_wxyz)
    tool_z = (rot[0][2], rot[1][2], rot[2][2])

    pivot = (0.0, 0.0, 0.0)
    position = pose.position_m
    direction_to_pivot = tuple(p - c for p, c in zip(pivot, position))
    norm = _norm(direction_to_pivot)
    direction_to_pivot = tuple(component / norm for component in direction_to_pivot)

    for axis_component, pivot_component in zip(tool_z, direction_to_pivot):
        assert math.isclose(axis_component, pivot_component, abs_tol=1e-9)
