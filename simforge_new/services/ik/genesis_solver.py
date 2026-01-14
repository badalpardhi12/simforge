"""Genesis-backed inverse kinematics solver.

This module implements a robust multi-seed collision-aware IK solver that:
1. Tries multiple random seeds when the initial solution collides
2. Uses TRAC-IK-style random restarts to escape local minima
3. Returns the first kinematically valid AND collision-free solution

Key configuration parameters (in profile.metadata):
- genesis_ik_collision_retries: Number of random-seed retries if collision detected (default: 8)
- genesis_ik_max_samples: Samples per IK solve for Genesis internal restart (default: 64)
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Tuple, List
import threading

import numpy as np
import torch

import genesis as gs

from .base import IKRequest, IKResult, IKMetrics, IKSolver
from ..util.joint_limits import resolve_joint_limits, wrap_vector_to_limits
from ...core.models import RobotProfile
from ...core.transformations import quaternion_multiply, quaternion_to_rotation_matrix


def _as_float(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_quaternion(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(tuple(float(v) for v in values), dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return arr / norm


def _quaternion_conjugate(values: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in values)
    return np.array([w, -x, -y, -z], dtype=np.float64)


class GenesisIKSolver(IKSolver):
    """Resolve IK targets directly through the Genesis rigid solver."""

    def __init__(
        self,
        *,
        profile: RobotProfile,
        entity,
        logger: Optional[logging.Logger] = None,
        lock: Optional[threading.RLock] = None,
    ) -> None:
        self.profile = profile
        self.entity = entity
        self.logger = logger or logging.getLogger(f"simforge.ik.genesis.{profile.name}")
        self._lock = lock or threading.RLock()

        metadata = profile.metadata or {}
        self._default_pos_tol = _as_float(metadata.get("legacy_control_ik_pos_tolerance_m"), 1e-3)
        self._default_rot_tol_deg = _as_float(metadata.get("legacy_control_ik_rot_tolerance_deg"), 1.0)
        self._max_solver_iters = int(_as_float(metadata.get("genesis_ik_solver_iters"), 50))
        self._max_step_size = _as_float(metadata.get("genesis_ik_max_step"), 0.2)
        self._damping = _as_float(metadata.get("genesis_ik_damping"), 0.05)
        self._max_samples = int(_as_float(metadata.get("genesis_ik_max_samples"), 64))
        # Number of collision-retry attempts with different random seeds
        self._collision_retries = int(_as_float(metadata.get("genesis_ik_collision_retries"), 8))

        self._ee_link = self._resolve_end_effector()
        self._last_solution: Optional[np.ndarray] = None

        mount_quat = _normalize_quaternion(profile.mount.orientation)
        self._mount_pos = np.asarray(tuple(float(v) for v in profile.mount.position), dtype=np.float64)
        self._mount_quat = mount_quat
        self._mount_quat_conj = _quaternion_conjugate(mount_quat)
        self._mount_rot = quaternion_to_rotation_matrix(mount_quat)
        self._mount_rot_T = self._mount_rot.T

        joint_lower, joint_upper = resolve_joint_limits(profile, entity)
        self._joint_lower = joint_lower
        self._joint_upper = joint_upper

    # ------------------------------------------------------------------
    # IKSolver API
    # ------------------------------------------------------------------
    def solve(self, request: IKRequest) -> IKResult:
        """Solve IK with multi-seed collision-aware retries.
        
        This method implements a robust IK strategy that preserves arm configuration:
        1. First try with the provided seed (or last solution / current position)
        2. If kinematically valid but collides, retry with SMALL perturbations of the seed
           (this keeps the arm in a similar configuration, making motion planning easier)
        3. Only returns solutions that are both kinematically valid AND collision-free
        
        Note: We avoid fully random seeds because they often produce solutions in 
        different arm configurations (elbow up vs down, etc.) which the motion 
        planner cannot reach without passing through obstacles.
        """
        if request.target.pose is None:
            return IKResult(success=False, solution=None, raw={"reason": "no_pose"})

        position = tuple(float(v) for v in request.target.pose.position)
        orientation = tuple(float(v) for v in request.target.pose.orientation)
        world_pos, world_quat = self._to_world_pose(position, orientation)

        pos_tol = request.position_tolerance_m or self._default_pos_tol
        rot_tol_deg = request.orientation_tolerance_deg or self._default_rot_tol_deg
        rot_tol = math.radians(rot_tol_deg)

        validator = request.is_state_valid or (lambda _: True)
        max_samples = max(self._max_samples, int(request.max_attempts) if request.max_attempts else 1)
        
        # Build list of seeds: first default, then small perturbations (NOT fully random)
        seeds_to_try = self._generate_seeds(request, self._collision_retries)
        
        best_collision_solution = None
        best_collision_error = None
        total_attempts = 0
        
        for seed_idx, seed in enumerate(seeds_to_try):
            result = self._solve_with_seed(
                seed=seed,
                world_pos=world_pos,
                world_quat=world_quat,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                rot_tol_deg=rot_tol_deg,
                max_samples=max_samples,
                validator=validator,
            )
            total_attempts += 1
            
            if result.success:
                # Found collision-free solution
                if seed_idx > 0:
                    self.logger.debug(
                        "IK found collision-free solution on seed attempt %d/%d for %s",
                        seed_idx + 1, len(seeds_to_try), self.profile.name
                    )
                return result
            
            # Track best kinematically-valid but colliding solution for diagnostics
            if result.raw.get("reason") == "goal_collision":
                if best_collision_solution is None:
                    best_collision_solution = result.raw.get("solution_before_collision")
                    best_collision_error = result.raw
        
        # All seeds failed - return appropriate failure
        if best_collision_solution is not None:
            # Had kinematically valid solutions but all collided
            self.logger.debug(
                "IK tried %d seeds, all collided for %s",
                total_attempts, self.profile.name
            )
            return IKResult(
                success=False,
                solution=None,
                raw={
                    "reason": "goal_collision",
                    "seeds_tried": total_attempts,
                    "pos_err": best_collision_error.get("pos_err"),
                    "rot_err_deg": best_collision_error.get("rot_err_deg"),
                },
            )
        
        # No kinematically valid solution found
        return IKResult(
            success=False,
            solution=None,
            raw={"reason": "out_of_tolerance", "seeds_tried": total_attempts},
        )
    
    def _generate_seeds(self, request: IKRequest, num_perturbations: int) -> List[np.ndarray]:
        """Generate seeds: first the default, then small perturbations to stay in same arm config.
        
        We use SMALL perturbations (not fully random seeds) because:
        - Random seeds often produce solutions in different arm configurations
        - Different arm configurations require passing through obstacles to reach
        - Small perturbations keep the arm in a similar configuration space
        - This makes motion planning much more likely to succeed
        
        The perturbation magnitude increases with each retry, starting small.
        """
        seeds = []
        
        # First seed: use request seed, last solution, or current position
        default_seed = self._seed_for_request(request)
        seeds.append(default_seed)
        
        # Generate progressively larger perturbations of the default seed
        # Start with small perturbations (±0.1 rad ≈ ±6°) and increase
        for i in range(num_perturbations):
            # Perturbation magnitude increases: 0.1, 0.2, 0.3, ... radians
            magnitude = 0.1 * (i + 1)
            perturbation = np.random.uniform(-magnitude, magnitude, size=default_seed.shape)
            perturbed_seed = default_seed + perturbation
            
            # Clip to joint limits if available
            if self._joint_lower is not None and self._joint_upper is not None:
                perturbed_seed = np.clip(perturbed_seed, self._joint_lower, self._joint_upper)
            
            seeds.append(perturbed_seed)
        
        return seeds
    
    def _solve_with_seed(
        self,
        seed: np.ndarray,
        world_pos: np.ndarray,
        world_quat: np.ndarray,
        pos_tol: float,
        rot_tol: float,
        rot_tol_deg: float,
        max_samples: int,
        validator,
    ) -> IKResult:
        """Attempt IK solve with a specific seed."""
        pos_tensor = self._tensor(world_pos, pad_to_dof=False)
        quat_tensor = self._tensor(world_quat, pad_to_dof=False)
        init_tensor = self._tensor(seed, pad_to_dof=True)

        n_envs = int(getattr(getattr(self.entity, "_solver", None), "n_envs", 0) or 0)
        if n_envs > 1:
            pos_tensor = pos_tensor.unsqueeze(0)
            quat_tensor = quat_tensor.unsqueeze(0)
            init_tensor = init_tensor.unsqueeze(0)

        try:
            with self._lock:
                qpos, error = self.entity.inverse_kinematics(
                    link=self._ee_link,
                    pos=pos_tensor,
                    quat=quat_tensor,
                    init_qpos=init_tensor,
                    respect_joint_limit=True,
                    max_samples=max_samples,
                    max_solver_iters=self._max_solver_iters,
                    damping=self._damping,
                    pos_tol=pos_tol,
                    rot_tol=rot_tol,
                    max_step_size=self._max_step_size,
                    return_error=True,
                )
        except Exception as exc:
            self.logger.exception("Genesis IK failed for %s: %s", self.profile.name, exc)
            return IKResult(success=False, solution=None, raw={"reason": "exception", "error": str(exc)})

        qpos_tensor = qpos if isinstance(qpos, torch.Tensor) else torch.as_tensor(qpos)
        if qpos_tensor.ndim > 1:
            qpos_tensor = qpos_tensor[0]
        error_tensor = error if isinstance(error, torch.Tensor) else torch.as_tensor(error)
        if error_tensor.ndim > 1:
            error_tensor = error_tensor[0]

        pos_err = float(torch.norm(error_tensor[:3]).item())
        rot_err_deg = float(torch.norm(error_tensor[3:]).item() * 180.0 / math.pi)

        within_limits = pos_err <= pos_tol + 1e-6 and rot_err_deg <= rot_tol_deg + 1e-6
        
        if not within_limits:
            return IKResult(
                success=False,
                solution=None,
                raw={"reason": "out_of_tolerance", "pos_err": pos_err, "rot_err_deg": rot_err_deg},
            )
        
        init_np = init_tensor.detach().cpu().numpy() if isinstance(init_tensor, torch.Tensor) else np.asarray(seed, dtype=np.float64)
        if init_np.ndim > 1:
            init_np = init_np[0]
        solution_np = qpos_tensor.detach().cpu().numpy().astype(np.float64)
        solution_np = self._wrap_solution(solution_np, init_np)
        solution = tuple(float(v) for v in solution_np)

        # Validate collision
        if validator(solution):
            # Success! Update last solution for continuity
            self._last_solution = np.asarray(solution, dtype=np.float64)
            metrics = IKMetrics(
                position_error_m=pos_err,
                orientation_error_deg=rot_err_deg,
                attempts=max_samples,
                slack_used=False,
                diagnostics=("genesis",),
            )
            return IKResult(success=True, solution=solution, metrics=metrics, raw={"backend": "genesis"})

        # Kinematically valid but collides
        return IKResult(
            success=False,
            solution=None,
            raw={
                "reason": "goal_collision",
                "pos_err": pos_err,
                "rot_err_deg": rot_err_deg,
                "solution_before_collision": solution,
            },
        )

    def _wrap_solution(self, solution: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if reference is None or reference.size == 0:
            reference = solution
        wrapped = wrap_vector_to_limits(solution, reference, self._joint_lower, self._joint_upper)
        if wrapped.shape != solution.shape:
            wrapped = wrapped.reshape(solution.shape)
        return wrapped

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def forward_kinematics(self, joints: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        joints_tensor = self._tensor(joints, pad_to_dof=True)
        with self._lock:
            original = self.entity.get_qpos()
            try:
                self.entity.set_qpos(joints_tensor, zero_velocity=False)
                link_local = getattr(self._ee_link, "idx_local", None)
                if link_local is None:
                    link_idx = getattr(self._ee_link, "idx", None)
                    link_start = getattr(self.entity, "_link_start", 0)
                    if link_idx is not None:
                        link_local = int(link_idx) - int(link_start)
                if link_local is None:
                    link_local = 0

                try:
                    pos_tensor = self.entity.get_links_pos(int(link_local))
                    quat_tensor = self.entity.get_links_quat(int(link_local))
                except Exception as exc:
                    self.logger.debug("FK link query fallback for %s: %s", self.profile.name, exc)
                    name = getattr(self._ee_link, "name", None)
                    if name is not None:
                        pos_tensor = self.entity.get_links_pos(name)
                        quat_tensor = self.entity.get_links_quat(name)
                    else:
                        raise
            finally:
                self.entity.set_qpos(original, zero_velocity=False)

        pos = pos_tensor.detach().cpu().numpy()
        if pos.ndim > 1:
            pos = pos[0]
        quat = quat_tensor.detach().cpu().numpy()
        if quat.ndim > 1:
            quat = quat[0]
        pos_base, quat_base = self._to_base_pose(pos, quat)
        return pos_base.astype(np.float64), quat_base.astype(np.float64)

    def _resolve_end_effector(self):
        link_name = (self.profile.end_effector_link or "").strip()
        if link_name:
            try:
                return self.entity.get_link(name=link_name)
            except Exception:
                self.logger.warning(
                    "End-effector link '%s' not found on %s; falling back to final link",
                    link_name,
                    self.profile.name,
                )
        links = getattr(self.entity, "links", None)
        if links:
            return links[-1]
        raise RuntimeError(f"Genesis entity for {self.profile.name} has no links")

    def _seed_for_request(self, request: IKRequest) -> np.ndarray:
        if request.seed is not None:
            seed = np.asarray(request.seed, dtype=np.float64)
        elif self._last_solution is not None:
            seed = self._last_solution.copy()
        else:
            current = self.entity.get_qpos()
            if isinstance(current, torch.Tensor):
                if current.ndim > 1:
                    current = current[0]
                seed = current.detach().cpu().numpy()
            else:
                seed = np.asarray(current, dtype=np.float64)
        return self._align_vector(seed)

    def _align_vector(self, values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(tuple(float(v) for v in values), dtype=np.float64)
        dof = int(getattr(self.entity, "n_qs", arr.size))
        if arr.size < dof:
            arr = np.pad(arr, (0, dof - arr.size), constant_values=0.0)
        elif arr.size > dof:
            arr = arr[:dof]
        return arr

    def _tensor(self, values: Sequence[float], *, pad_to_dof: bool) -> torch.Tensor:
        if pad_to_dof:
            arr = self._align_vector(values)
        else:
            arr = np.asarray(tuple(float(v) for v in values), dtype=np.float64)
        return torch.as_tensor(arr, dtype=gs.tc_float, device=gs.device)

    def _to_world_pose(
        self,
        position_base: Sequence[float],
        orientation_base: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_base = np.asarray(tuple(float(v) for v in position_base), dtype=np.float64)
        quat_base = _normalize_quaternion(orientation_base)
        world_pos = self._mount_rot.dot(pos_base) + self._mount_pos
        world_quat = quaternion_multiply(self._mount_quat, quat_base)
        return world_pos, world_quat

    def _to_base_pose(
        self,
        position_world: Sequence[float],
        orientation_world: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_world = np.asarray(tuple(float(v) for v in position_world), dtype=np.float64)
        quat_world = _normalize_quaternion(orientation_world)
        pos_base = self._mount_rot_T.dot(pos_world - self._mount_pos)
        quat_base = quaternion_multiply(self._mount_quat_conj, quat_world)
        return pos_base, _normalize_quaternion(quat_base)


__all__ = ["GenesisIKSolver"]
