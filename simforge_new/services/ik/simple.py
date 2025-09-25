"""Lightweight IK solver used for tests and fallback scenarios."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .base import IKRequest, IKResult, IKMetrics, IKSolver


@dataclass(frozen=True)
class SimpleIKSettings:
    """Configuration for :class:`SimpleIKSolver`."""

    default_dof: int = 6


class SimpleIKSolver(IKSolver):
    """Returns seeded joint targets without calling external solvers."""

    def __init__(self, settings: SimpleIKSettings | None = None) -> None:
        self.settings = settings or SimpleIKSettings()

    def solve(self, request: IKRequest) -> IKResult:
        solution = self._resolve_solution(request)
        if solution is None:
            return IKResult(success=False, solution=None, metrics=None, raw={"reason": "no_seed"})
        if request.is_state_valid and not request.is_state_valid(solution):
            return IKResult(success=False, solution=None, metrics=None, raw={"reason": "invalid_state"})
        metrics = IKMetrics(
            position_error_m=0.0,
            orientation_error_deg=0.0,
            attempts=1,
            slack_used=False,
            diagnostics=("simple",),
        )
        return IKResult(success=True, solution=solution, metrics=metrics, raw={"solver": "simple"})

    def _resolve_solution(self, request: IKRequest):
        if request.seed is not None:
            return tuple(float(v) for v in request.seed)
        if request.robot.initial_joint_positions_deg:
            return tuple(math.radians(float(v)) for v in request.robot.initial_joint_positions_deg)
        metadata = request.robot.metadata or {}
        dof_hint = metadata.get("dof")
        try:
            dof = int(dof_hint) if dof_hint is not None else self.settings.default_dof
        except (TypeError, ValueError):
            dof = self.settings.default_dof
        return tuple(0.0 for _ in range(max(1, dof)))


__all__ = ["SimpleIKSettings", "SimpleIKSolver"]
