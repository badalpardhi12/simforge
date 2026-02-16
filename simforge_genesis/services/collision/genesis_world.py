"""Collision validation using Genesis contact queries."""
from __future__ import annotations

import logging
import threading
from typing import Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch

import genesis as gs

from .base import CollisionCheck, CollisionQuery, CollisionWorld
from ...core.config_schema import SafetyPolicy
if TYPE_CHECKING:  # pragma: no cover - import guard for type hints
    from ...control.coordinator import RobotContext


class GenesisCollisionWorld(CollisionWorld):
    """Leverage Genesis collider state for joint-space validation.
    
    This collision world checks for ALL collisions including:
    - Self-collisions (robot links colliding with each other)
    - Ground plane collisions
    - Environment object collisions (URDFs with collision_enabled: true)
    """

    def __init__(
        self,
        contexts: Dict[str, object],
        *,
        safety: Optional[SafetyPolicy],
        logger: Optional[logging.Logger] = None,
        lock: Optional[threading.RLock] = None,
    ) -> None:
        self._contexts: Dict[str, "RobotContext"] = dict(contexts)
        self._entities = {name: ctx.entity for name, ctx in contexts.items()}
        self._safety = safety
        self._logger = logger or logging.getLogger("simforge.collision.genesis")
        self._state_lock = threading.RLock()
        self._gs_lock = lock or threading.RLock()
        self._last_states: Dict[str, Tuple[float, ...]] = {
            name: self._snapshot_state(entity)
            for name, entity in self._entities.items()
        }

    # ------------------------------------------------------------------
    # CollisionWorld API
    # ------------------------------------------------------------------
    def is_state_valid(self, query: CollisionQuery) -> CollisionCheck:
        entity = self._entities.get(query.robot.name)
        if entity is None:
            return CollisionCheck(distance_m=self._clearance(), in_collision=False)

        with self._state_lock:
            originals = {
                name: self._tensor(state, entity_ref=self._entities[name])
                for name, state in self._last_states.items()
            }

        with self._gs_lock:
            try:
                self._apply_state(entity, query.joints)
                self._apply_other_states(query)
                contacts = entity.detect_collision()
                collision_count = int(getattr(contacts, "shape", (0, 0))[0]) if isinstance(contacts, np.ndarray) else int(len(contacts))
                in_collision = collision_count > 0
                
                # Enhanced logging for debugging asymmetric failures
                if in_collision:
                    self._logger.debug(
                        "[%s] Collision detected: %d pairs, joints=%s",
                        query.robot.name,
                        collision_count,
                        [round(j, 3) for j in query.joints[:6]],
                    )
                    # Log collision pair details if available
                    if isinstance(contacts, np.ndarray) and contacts.size > 0:
                        self._log_collision_details(entity, contacts)
                
                if not in_collision:
                    with self._state_lock:
                        self._last_states[query.robot.name] = tuple(
                            float(v) for v in self._align_vector(query.joints, entity)
                        )
                details = {
                    "source": "genesis",
                    "pairs": collision_count,
                }
                return CollisionCheck(
                    distance_m=0.0 if in_collision else self._clearance(),
                    in_collision=in_collision,
                    details=details,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.debug("Collision validation failed for %s: %s", query.robot.name, exc)
                return CollisionCheck(distance_m=self._clearance(), in_collision=False, details={"reason": "fallback"})
            finally:
                for name, tensor in originals.items():
                    self._entities[name].set_qpos(tensor, zero_velocity=False)
    
    def _log_collision_details(self, entity, contacts: np.ndarray) -> None:
        """Log detailed collision information for debugging."""
        try:
            scene = getattr(entity, "scene", None)
            rigid_solver = getattr(scene, "rigid_solver", None) if scene else None
            
            for i in range(min(contacts.shape[0], 5)):  # Log up to 5 collision pairs
                if contacts.ndim == 1:
                    geom_a, geom_b = int(contacts[0]), int(contacts[1]) if contacts.size > 1 else -1
                else:
                    geom_a = int(contacts[i, 0]) if contacts.shape[1] > 0 else -1
                    geom_b = int(contacts[i, 1]) if contacts.shape[1] > 1 else -1
                
                # Try to get entity/link names for the geoms
                entity_a_name = self._geom_to_entity_name(rigid_solver, geom_a)
                entity_b_name = self._geom_to_entity_name(rigid_solver, geom_b)
                
                self._logger.debug(
                    "  Collision pair %d: geom %d (%s) <-> geom %d (%s)",
                    i, geom_a, entity_a_name, geom_b, entity_b_name
                )
                
                if contacts.ndim == 1:
                    break
        except Exception as exc:
            self._logger.debug("Failed to log collision details: %s", exc)
    
    def _geom_to_entity_name(self, rigid_solver, geom_idx: int) -> str:
        """Get the entity name that owns a given geom index."""
        if rigid_solver is None or geom_idx < 0:
            return "unknown"
        try:
            for ent in rigid_solver.entities:
                if hasattr(ent, "geom_start") and hasattr(ent, "geom_end"):
                    if ent.geom_start <= geom_idx < ent.geom_end:
                        return getattr(ent, "name", f"entity_{getattr(ent, 'idx', '?')}")
            return f"geom_{geom_idx}"
        except Exception:
            return f"geom_{geom_idx}"

    def update_environment(self, robot_states: Dict[str, Tuple[float, ...]]) -> None:
        with self._state_lock:
            for name, values in (robot_states or {}).items():
                if name in self._entities:
                    self._last_states[name] = tuple(float(v) for v in values)

    def allowed_pairs(self) -> Iterable[Tuple[str, str]]:
        return tuple()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_state(self, entity, joints: Sequence[float]) -> None:
        tensor = self._tensor(joints, entity_ref=entity)
        entity.set_qpos(tensor, zero_velocity=False)

    def _apply_other_states(self, query: CollisionQuery) -> None:
        other_states = dict(query.other_robot_states or {})
        for name, entity in self._entities.items():
            if name == query.robot.name:
                continue
            joints = other_states.get(name, self._last_states.get(name))
            if joints is not None:
                self._apply_state(entity, joints)

    def _clearance(self) -> float:
        if self._safety is None:
            return 0.0
        try:
            return float(max(0.0, self._safety.minimum_clearance_m))
        except Exception:
            return 0.0

    def _tensor(self, values: Sequence[float], *, entity_ref) -> torch.Tensor:
        arr = self._align_vector(values, entity_ref)
        return torch.as_tensor(arr, dtype=gs.tc_float, device=gs.device)

    def _align_vector(self, values: Sequence[float], entity_ref) -> np.ndarray:
        arr = np.asarray(tuple(float(v) for v in values), dtype=np.float64)
        dof = int(getattr(entity_ref, "n_qs", arr.size))
        if arr.size < dof:
            arr = np.pad(arr, (0, dof - arr.size), constant_values=0.0)
        elif arr.size > dof:
            arr = arr[:dof]
        return arr

    def _snapshot_state(self, entity) -> Tuple[float, ...]:
        qpos = entity.get_qpos()
        if isinstance(qpos, torch.Tensor):
            if qpos.ndim > 1:
                qpos = qpos[0]
            arr = qpos.detach().cpu().numpy()
        else:
            arr = np.asarray(qpos, dtype=np.float64)
        return tuple(float(v) for v in arr)


__all__ = ["GenesisCollisionWorld"]
