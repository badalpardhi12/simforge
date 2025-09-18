# path_planner.py
"""OMPL-based planners + time parameterization."""
from __future__ import annotations

from typing import Callable, List, Tuple, Optional, TYPE_CHECKING, Any
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


# ---------- Core OMPL helpers ----------

def _create_space_information(
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: Callable[[np.ndarray], bool],
) -> Tuple[int, ob.SpaceInformation]:
    dof = int(lower.shape[0])
    space = ob.RealVectorStateSpace(dof)
    bounds = ob.RealVectorBounds(dof)
    for i in range(dof):
        bounds.setLow(i, float(lower[i]))
        bounds.setHigh(i, float(upper[i]))
    space.setBounds(bounds)

    # Finer resolution so OMPL samples/validates denser along edges
    space.setLongestValidSegmentFraction(1.0/200.0)   # ~0.5% segments

    si = ob.SpaceInformation(space)

    def _valid(s: ob.State) -> bool:
        q = np.array([s[i] for i in range(dof)], dtype=np.float64)
        return is_state_valid(q)

    si.setStateValidityChecker(ob.StateValidityCheckerFn(_valid))
    si.setStateValidityCheckingResolution(0.005)      # 0.5% of extent
    si.setup()
    return dof, si


def _check_segment_collision_free(
    q_start: np.ndarray,
    q_end: np.ndarray,
    is_state_valid: Callable[[np.ndarray], bool],
    resolution: int = 10
) -> bool:
    """Check if the straight-line path between two configurations is collision-free."""
    for i in range(1, resolution):
        alpha = i / (resolution - 1)
        q_intermediate = (1 - alpha) * q_start + alpha * q_end
        if not is_state_valid(q_intermediate):
            return False
    return True


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


def ompl_plan_with_factory(
    planner_name: str,
    planner_factory: Callable[[ob.SpaceInformation], og.Planner],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: Callable[[np.ndarray], bool],
    timeout_s: float = 3.0,
    simplify: bool = True,
    configure: Optional[Callable[[og.Planner], None]] = None,
) -> Optional[Tuple[str, np.ndarray, np.ndarray, float]]:
    if not HAS_OMPL:
        return None

    dof, si = _create_space_information(lower, upper, is_state_valid)
    space = si.getStateSpace()

    start = ob.State(space)
    goal = ob.State(space)
    for i in range(dof):
        start[i] = float(q_start[i])
        goal[i] = float(q_goal[i])

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    planner = planner_factory(si)
    if configure is not None:
        configure(planner)
    planner.setProblemDefinition(pdef)
    planner.setup()

    if not planner.solve(timeout_s):
        return None

    path_geometric = pdef.getSolutionPath()
    if simplify and hasattr(og, "PathSimplifier"):
        og.PathSimplifier(si).simplifyMax(path_geometric)

    states = path_geometric.getStates()
    waypoints = np.array([[s[i] for i in range(dof)] for s in states], dtype=np.float64)

    for i in range(len(waypoints) - 1):
        if not _check_segment_collision_free(waypoints[i], waypoints[i + 1], is_state_valid, resolution=20):
            return None

    times = _trap_times(waypoints, max_vel=1.0, max_acc=2.0)
    cost = float(path_geometric.length()) if hasattr(path_geometric, "length") else float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))
    return planner_name, waypoints, times, cost


if TYPE_CHECKING and HAS_OMPL:
    PlannerCallable = Callable[[ob.SpaceInformation], ob.Planner]
    PlannerConfig = Optional[Callable[[ob.Planner], None]]
else:
    PlannerCallable = Callable[[Any], Any]
    PlannerConfig = Optional[Callable[[Any], None]]

PlannerSpec = Tuple[str, PlannerCallable, PlannerConfig]


def ompl_parallel_plans(
    planner_specs: List[PlannerSpec],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    is_state_valid: Callable[[np.ndarray], bool],
    timeout_s: float = 3.0,
    simplify: bool = True,
) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
    if not HAS_OMPL:
        return results

    def _run(spec: PlannerSpec):
        name, factory, config = spec
        return ompl_plan_with_factory(
            name,
            factory,
            q_start,
            q_goal,
            lower,
            upper,
            is_state_valid,
            timeout_s=timeout_s,
            simplify=simplify,
            configure=config,
        )

    with ThreadPoolExecutor(max_workers=len(planner_specs)) as executor:
        future_map = {executor.submit(_run, spec): spec[0] for spec in planner_specs}
        for fut in as_completed(future_map):
            res = fut.result()
            if res is not None:
                results.append(res)

    results.sort(key=lambda item: item[3])
    return results


def default_joint_planner_specs(range_rad: float) -> List[PlannerSpec]:
    if not HAS_OMPL:
        return []
    return [
        (
            "RRTConnect",
            lambda si: og.RRTConnect(si),
            lambda planner: planner.setRange(range_rad),
        ),
        ("BITstar", lambda si: og.BITstar(si), None),
        ("InformedRRTstar", lambda si: og.InformedRRTstar(si), None),
        ("PRMstar", lambda si: og.PRMstar(si), None),
    ]


__all__ = [
    "ompl_plan_with_factory",
    "ompl_parallel_plans",
    "default_joint_planner_specs",
    "plan_joint_path",
]
