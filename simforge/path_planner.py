# path_planner.py
"""OMPL-based planners + time parameterization + Cartesian linear planner."""
from __future__ import annotations

from typing import Callable, List, Tuple, Optional
import numpy as np

try:
    from ompl import base as ob
    from ompl import geometric as og
    HAS_OMPL = True
except Exception:
    HAS_OMPL = False


# ---------- time parameterization (trapezoidal) ----------
def _trap_times(waypoints: np.ndarray, max_vel: float, max_acc: float) -> np.ndarray:
    if waypoints.shape[0] <= 1:
        return np.zeros(waypoints.shape[0], dtype=np.float32)

    # crude per-joint path length accumulation
    seg_dist = np.linalg.norm(np.diff(waypoints, axis=0), axis=1, ord=np.inf)
    total_dist = float(np.sum(seg_dist))
    if total_dist <= 1e-9:
        return np.zeros(waypoints.shape[0], dtype=np.float32)

    # single-scalar timing using worst-case axis (fast + safe)
    t_acc = max_vel / max_acc
    d_acc = 0.5 * max_acc * t_acc * t_acc
    if 2 * d_acc >= total_dist:
        total_time = 2 * np.sqrt(total_dist / max_acc)
    else:
        d_const = total_dist - 2 * d_acc
        t_const = d_const / max_vel
        total_time = 2 * t_acc + t_const

    return np.linspace(0.0, total_time, waypoints.shape[0]).astype(np.float32)


# ---------- OMPL Joint-space RRT-Connect ----------
def ompl_rrt_connect_plan(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: Callable[[np.ndarray], bool],
    timeout_s: float = 3.0,
    range_rad: float = 0.2,
    simplify: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Joint-space RRT-Connect (returns (waypoints[N,DoF], times[N]))"""
    if not HAS_OMPL:
        return None
    dof = int(q_start.shape[0])
    space = ob.RealVectorStateSpace(dof)
    bounds = ob.RealVectorBounds(dof)
    for i in range(dof):
        bounds.setLow(i, float(lower[i]))
        bounds.setHigh(i, float(upper[i]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)

    def _valid(s: ob.State) -> bool:
        q = np.array([s[i] for i in range(dof)], dtype=np.float64)
        return is_state_valid(q)

    si.setStateValidityChecker(ob.StateValidityCheckerFn(_valid))
    si.setup()

    start = ob.State(space); goal = ob.State(space)
    for i in range(dof):
        start[i] = float(q_start[i])
        goal[i]  = float(q_goal[i])

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    planner = og.RRTConnect(si)
    planner.setRange(range_rad)
    planner.setProblemDefinition(pdef)
    planner.setup()

    if not planner.solve(timeout_s):
        return None

    path_geometric = pdef.getSolutionPath()
    if simplify:
        og.PathSimplifier(si).simplifyMax(path_geometric)

    states = path_geometric.getStates()
    waypoints = np.array([[s[i] for i in range(dof)] for s in states], dtype=np.float64)
    times = _trap_times(waypoints, max_vel=1.0, max_acc=2.0)  # execution is re-scalable upstream
    return waypoints, times


# ---------- Cartesian straight-line via per-waypoint IK ----------
def cartesian_linear_plan(
    start_q: np.ndarray,
    start_pose_se3: Tuple[np.ndarray, np.ndarray],  # (pos(3), quat_wxyz(4))
    target_pose_se3: Tuple[np.ndarray, np.ndarray],  # (pos(3), quat_wxyz(4))
    solve_ik: Callable[[np.ndarray, Tuple[np.ndarray, np.ndarray]], Optional[np.ndarray]],
    is_state_valid: Callable[[np.ndarray], bool],
    num_waypoints: int = 200,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Straight-line EEF path → per-waypoint IK → collision check."""
    pos_start, quat_start = start_pose_se3
    pos_goal, quat_goal = target_pose_se3
    pos_start = np.asarray(pos_start, dtype=np.float64)
    quat_start = np.asarray(quat_start, dtype=np.float64)
    pos_goal = np.asarray(pos_goal, dtype=np.float64)
    quat_goal = np.asarray(quat_goal, dtype=np.float64)

    # Build linear interpolation in SE3: linear in XYZ + SLERP in quat (wxyz)
    way_q: List[np.ndarray] = [start_q.copy()]
    q_prev = start_q.copy()

    def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        q1 = q1 / np.linalg.norm(q1); q2 = q2 / np.linalg.norm(q2)
        dot = float(np.dot(q1, q2))
        if dot < 0.0: q2 = -q2; dot = -dot
        if dot > 0.9995:
            out = q1 + t * (q2 - q1)
            return out / np.linalg.norm(out)
        theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin0 = np.sin(theta0)
        theta = theta0 * t
        s0 = np.sin(theta0 - theta) / sin0
        s1 = np.sin(theta) / sin0
        return s0 * q1 + s1 * q2

    # Interpolate from start to goal
    for i in range(1, num_waypoints):
        a = i / (num_waypoints - 1)
        pos_i = (1 - a) * pos_start + a * pos_goal
        quat_i = _slerp(quat_start, quat_goal, a)

        q_next = solve_ik(q_prev, (pos_i, quat_i))
        if q_next is None or not is_state_valid(q_next):
            return None
        way_q.append(q_next)
        q_prev = q_next

    waypoints = np.stack(way_q, axis=0)
    times = _trap_times(waypoints, max_vel=1.0, max_acc=2.0)
    return waypoints, times


# ---------- Simple joint-space linear planner ----------
def plan_joint_path(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    resolution: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple linear interpolation in joint space."""
    waypoints = np.linspace(q_start, q_goal, resolution)
    times = np.linspace(0.0, 1.0, resolution)
    return waypoints, times


__all__ = [
    "ompl_rrt_connect_plan",
    "cartesian_linear_plan",
    "plan_joint_path",
]