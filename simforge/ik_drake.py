# simforge/ik_drake.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np

# Drake imports (pip package: drake)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.geometry import SceneGraph
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.tree import Frame
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

@dataclass
class DrakeIKOptions:
    pos_tolerance_m: float = 1e-3
    rot_tolerance_deg: float = 1.0
    max_random_seeds: int = 12
    seed_noise_rad: float = 0.25           # per-try uniform noise amplitude
    center_bias_weight: float = 1e-2       # pull to joint mid-range
    seed_stick_weight: float = 5e-3        # pull to current/seed q
    timeout_s: float = 0.5                 # per attempt soft cap (solver side)
    allow_position_only_fallback: bool = False  # if True, allow position-only mode
    # NOTE: collision is enforced outside via is_state_valid()

class DrakeIKCache:
    """Builds and caches a Drake MultibodyPlant per robot URDF."""
    def __init__(self, urdf_path: str, base_link: str, ee_link: str):
        self.urdf = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link

        self.scene_graph = SceneGraph()
        self.plant = MultibodyPlant(time_step=0.0)
        parser = Parser(self.plant)
        parser.AddModels(self.urdf)
        # Prefer welding base_link (aligns with Genesis root); skip if URDF already fixes base.
        try:
            self.plant.WeldFrames(
                self.plant.world_frame(),
                self.plant.GetFrameByName(self.base_link),
                RigidTransform()
            )
        except Exception:
            # If the URDF already attaches base to world, skip welding.
            pass
        try:
            self.plant.Finalize()
        except RuntimeError as e:
            # Rebuild cleanly without the extra weld if the URDF already implies a fixed base -> loop error.
            if "loops in the system graph" in str(e):
                self.plant = MultibodyPlant(time_step=0.0)
                Parser(self.plant).AddModels(self.urdf)
                self.plant.Finalize()
            else:
                raise

        # Resolve frames (stick with configured base_link / ee_link to avoid hidden offsets)
        self.base_frame: Frame = self.plant.GetFrameByName(self.base_link)
        self.ee_frame: Frame = self.plant.GetFrameByName(self.ee_link)

        # Limits & centers
        self.lower = self.plant.GetPositionLowerLimits()
        self.upper = self.plant.GetPositionUpperLimits()
        # Replace +/-inf with sane revolute limits
        self.lower = np.where(np.isfinite(self.lower), self.lower, -np.pi)
        self.upper = np.where(np.isfinite(self.upper), self.upper,  np.pi)
        self.q_mid = 0.5 * (self.lower + self.upper)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).flatten()
        n = self.plant.num_positions()
        if q.size < n:
            q = np.concatenate([q, np.zeros(n - q.size, dtype=np.float64)], axis=0)
        elif q.size > n:
            q = q[:n]
        return np.clip(q, self.lower, self.upper)

def _quat_wxyz_to_rot(qwxyz: Tuple[float,float,float,float]) -> RotationMatrix:
    w, x, y, z = map(float, qwxyz)
    norm = (w*w + x*x + y*y + z*z) ** 0.5
    if norm <= 0.0:
        raise ValueError("Invalid quaternion")
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)
    return RotationMatrix(R)

def solve_ik_drake(
    cache: DrakeIKCache,
    q_seed: np.ndarray,
    *,
    target_pos_base_m: Tuple[float,float,float],
    target_quat_base_wxyz: Tuple[float,float,float,float],
    is_state_valid=lambda q: True,
    opts: DrakeIKOptions = DrakeIKOptions(),
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """
    Solve IK so that the EE pose relative to BASE equals (pos, quat) within tolerances.
    Returns (q_sol, info). No ROS required.
    """
    plant = cache.plant
    base = cache.base_frame
    ee = cache.ee_frame

    q_seed = cache.clamp(q_seed)
    # Robust quaternion handling: fall back to identity if invalid to prevent SNOPT crashes
    try:
        R_des = _quat_wxyz_to_rot(target_quat_base_wxyz)
    except Exception:
        R_des = RotationMatrix()
    p_des = np.array(target_pos_base_m, dtype=np.float64)

    # Pre-construct deterministic seed bank
    seeds: List[np.ndarray] = [q_seed.copy(), cache.q_mid.copy(), np.zeros_like(q_seed)]
    # Bias first joint to target yaw in base XY
    try:
        yaw = float(np.arctan2(p_des[1], p_des[0]))
        s = q_seed.copy()
        s[0] = np.clip(yaw, cache.lower[0], cache.upper[0])
        seeds.append(cache.clamp(s))
    except Exception:
        pass

    # Wrist-flip exploration seeds (help satisfy tough orientations)
    if q_seed.size >= 6:
        for d4 in (0.0, np.pi, -np.pi):
            for d5 in (0.0, np.pi, -np.pi):
                for d6 in (0.0, np.pi, -np.pi):
                    s = q_seed.copy()
                    s[-3] = np.clip(s[-3] + d4, cache.lower[-3], cache.upper[-3])
                    s[-2] = np.clip(s[-2] + d5, cache.lower[-2], cache.upper[-2])
                    s[-1] = np.clip(s[-1] + d6, cache.lower[-1], cache.upper[-1])
                    seeds.append(cache.clamp(s))

    # Randomized restarts
    rng = np.random.RandomState(1234)
    for _ in range(int(opts.max_random_seeds)):
        noise = rng.uniform(-opts.seed_noise_rad, opts.seed_noise_rad, size=q_seed.size)
        seeds.append(cache.clamp(q_seed + noise))

    # Tight boxes for position (base-frame)
    tol = float(opts.pos_tolerance_m)
    theta_tol = np.deg2rad(float(opts.rot_tolerance_deg))

    # Try strict orientation first, then relaxed. Optionally allow position-only as a last resort.
    modes = [
        ("strict", theta_tol),
        ("relaxed", max(theta_tol, np.deg2rad(10.0))),
    ]
    if bool(getattr(opts, "allow_position_only_fallback", False)):
        modes.append(("position_only", None))
    last_info: Dict[str, float] = {}
    for ori_mode, ang in modes:
        for idx, s in enumerate(seeds):
            ik = InverseKinematics(plant, with_joint_limits=True)
            q = ik.q()

            # Enforce the EE origin to sit within [p_des - tol, p_des + tol] in the BASE frame
            # Correct Drake call order is: (frameA, p_BQ, frameB, p_AQ_lower, p_AQ_upper)
            tol_vec = np.full(3, tol, dtype=np.float64)
            ik.AddPositionConstraint(
                ee,
                np.zeros(3),   # point fixed in EE frame (origin)
                base,
                p_des - tol_vec,
                p_des + tol_vec,
            )

            # Enforce orientation within angle bound unless position-only mode
            if ang is not None:
                # Desire EE orientation R_des in BASE within bound 'ang'
                # Use Abar = BASE with I, Bbar = EE with R_des^T so that R_AB * R_des^T ≈ I -> R_AB ≈ R_des
                ik.AddOrientationConstraint(
                    base, R_des,                 # desired orientation in BASE
                    ee,   RotationMatrix(),      # actual EE frame
                    ang
                )

            # Quadratic regularization: center + stick to seed
            if opts.center_bias_weight > 0:
                ik.prog().AddQuadraticErrorCost(opts.center_bias_weight * np.eye(q.size), cache.q_mid, q)
            if opts.seed_stick_weight > 0:
                ik.prog().AddQuadraticErrorCost(opts.seed_stick_weight * np.eye(q.size), s, q)

            # Initial guess & solve
            ik.prog().SetInitialGuess(q, s)
            try:
                result = Solve(ik.prog())
            except Exception as e:
                # Guard against Drake/SNOPT exceptions (e.g., degenerate quaternion paths)
                last_info = {"reason": "solver_exception", "error": str(e), "seed_idx": idx, "ori_mode": ori_mode}
                continue
            success = bool(result.is_success())

            if success:
                q_sol = result.GetSolution(q).astype(np.float64)
                q_sol = cache.clamp(q_sol)
                if is_state_valid(q_sol):
                    return q_sol, {"iters": idx+1, "backend": "drake_ik", "status": "success", "ori_mode": ori_mode}
                else:
                    last_info = {"reason": "goal_in_collision", "seed_idx": idx, "ori_mode": ori_mode}
                    continue
            else:
                last_info = {"reason": "solver_failed", "seed_idx": idx, "ori_mode": ori_mode}
        # try next orientation mode
    return None, {"status": "fail", **last_info}
