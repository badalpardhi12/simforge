"""Simulation session orchestration."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..core import Backend, Command
from ..core.commands import (
    CartesianMoveCommand as _CartesianMoveCommand,
    JointTargetsCommand as _JointTargetsCommand,
)
from ..core.models import EnvironmentSpec, RobotProfile, RobotState
from ..core.events import RobotStateSnapshot, EventTopic
from ..environment import load_environment
from ..infrastructure.genesis.builders import build_scene, SceneBuildResult
from ..infrastructure.genesis.io import get_joint_positions
from ..infrastructure import set_joint_positions
from ..services import (
    IKSolver,
    MotionPlanner,
    StateEstimator,
    StateSubscription,
    PollingStateEstimator,
    CollisionWorld,
    DrakeIKSolver,
    OMPLMotionPlanner,
    FCLCollisionWorld,
)
from .event_bus import EventBus
from .command_bus import CommandBus
from .coordinator import RobotCoordinator, RobotContext
from .synchronized_trajectory import SynchronizedTrajectoryManager
from simforge.transformations import rpy_to_quaternion


@dataclass
class SimulationResources:
    spec: EnvironmentSpec
    scene_result: SceneBuildResult
    client: object
    event_bus: EventBus
    command_bus: CommandBus


@dataclass(frozen=True)
class ServiceFactories:
    """Factories for per-robot services used by the session."""

    make_ik_solver: Callable[[RobotProfile], IKSolver]
    make_motion_planner: Callable[[RobotProfile], MotionPlanner]
    make_state_estimator: Callable[[Dict[str, "RobotContext"]], StateEstimator]
    make_collision_world: Callable[[Dict[str, "RobotContext"]], CollisionWorld]

    @classmethod
    def default(cls, *, spec: EnvironmentSpec) -> "ServiceFactories":
        dt = spec.scene.dt
        safety_policy = spec.config.policies.safety
        default_freq = 1.0 / dt if dt > 0 else 60.0
        service_logger = logging.getLogger("simforge.services")
        collision_world_box: list[Optional[CollisionWorld]] = [None]
        ik_cache: Dict[str, IKSolver] = {}
        planner_cache: Dict[str, MotionPlanner] = {}

        def _ik(robot: RobotProfile) -> IKSolver:
            if robot.name in ik_cache:
                return ik_cache[robot.name]

            try:
                solver = DrakeIKSolver(robot, logger=service_logger.getChild(f"ik.{robot.name}"))
            except Exception as exc:  # pragma: no cover - surface configuration issues
                raise RuntimeError(f"Failed to initialise Drake IK for {robot.name}") from exc

            ik_cache[robot.name] = solver
            return solver

        def _planner(robot: RobotProfile) -> MotionPlanner:
            if robot.name in planner_cache:
                return planner_cache[robot.name]

            world_ref = collision_world_box[0]
            if world_ref is None:
                raise RuntimeError("Collision world must be initialised before creating planners")

            try:
                planner = OMPLMotionPlanner(
                    robot,
                    collision_world=world_ref,
                    logger=service_logger.getChild(f"planner.{robot.name}"),
                )
            except Exception as exc:  # pragma: no cover - surface configuration issues
                raise RuntimeError(f"Failed to initialise OMPL planner for {robot.name}") from exc

            planner_cache[robot.name] = planner
            return planner

        def _state_estimator(contexts: Dict[str, "RobotContext"]) -> StateEstimator:
            entities = {name: ctx.entity for name, ctx in contexts.items()}
            profiles = {name: ctx.profile for name, ctx in contexts.items()}
            return PollingStateEstimator(entities, profiles, default_frequency_hz=default_freq)

        def _collision_world(contexts: Dict[str, "RobotContext"]) -> CollisionWorld:
            profiles = {name: ctx.profile for name, ctx in contexts.items()}
            try:
                world = FCLCollisionWorld(
                    profiles,
                    spec=spec,
                    safety=safety_policy,
                    logger=service_logger.getChild("collision"),
                )
            except Exception as exc:  # pragma: no cover - surface configuration issues
                raise RuntimeError("Failed to initialise FCL collision world") from exc
            collision_world_box[0] = world
            return world

        return cls(
            make_ik_solver=_ik,
            make_motion_planner=_planner,
            make_state_estimator=_state_estimator,
            make_collision_world=_collision_world,
        )


class SimulationSession:
    """Coordinates environment construction and command dispatch."""

    def __init__(
        self,
        resources: SimulationResources,
        logger: logging.Logger,
        *,
        service_factories: Optional[ServiceFactories] = None,
    ) -> None:
        self.resources = resources
        self.logger = logger
        self._sync_trajectory_manager = SynchronizedTrajectoryManager(logger)
        self._coordinators: Dict[str, RobotCoordinator] = {}
        self._factories = service_factories or ServiceFactories.default(spec=self.resources.spec)
        self._closed = False
        self._lock = asyncio.Lock()
        self._state_estimator: Optional[StateEstimator] = None
        self._state_tasks: Dict[str, asyncio.Task] = {}
        self._collision_world: Optional[CollisionWorld] = None
        self._latest_states: Dict[str, RobotState] = {}
        self._simulation_task: Optional[asyncio.Task] = None
        self._reference_frames: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._initialise_robot_coordinators()
        self._start_simulation_loop()

    @classmethod
    async def create_from_spec(
        cls,
        spec: EnvironmentSpec,
        *,
        backend: Backend = Backend.GPU,
        logger: Optional[logging.Logger] = None,
        service_factories: Optional[ServiceFactories] = None,
    ) -> "SimulationSession":
        from ..infrastructure.genesis.client import GenesisClient

        logger = logger or logging.getLogger("simforge.session")
        client = GenesisClient(backend, logger)
        scene_result = build_scene(client, spec, logger)
        event_bus = EventBus()
        command_bus = CommandBus(event_bus, spec.robot_names)
        resources = SimulationResources(
            spec=spec,
            scene_result=scene_result,
            client=client,
            event_bus=event_bus,
            command_bus=command_bus,
        )
        return cls(resources, logger, service_factories=service_factories)

    @classmethod
    async def create(
        cls,
        config_path: str,
        *,
        backend: Backend = Backend.GPU,
        logger: Optional[logging.Logger] = None,
        service_factories: Optional[ServiceFactories] = None,
    ) -> "SimulationSession":
        spec = load_environment(config_path)
        return await cls.create_from_spec(
            spec,
            backend=backend,
            logger=logger,
            service_factories=service_factories,
        )

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
        await self.resources.command_bus.stop_all("session_close")
        # Stop simulation loop
        if self._simulation_task:
            self._simulation_task.cancel()
            try:
                await self._simulation_task
            except asyncio.CancelledError:
                pass
        await asyncio.gather(
            *(coordinator.stop() for coordinator in self._coordinators.values()),
            return_exceptions=True,
        )
        for task in self._state_tasks.values():
            task.cancel()
        if self._state_tasks:
            await asyncio.gather(*self._state_tasks.values(), return_exceptions=True)
        try:
            destroy_scene = getattr(self.resources.scene_result.scene, "destroy", None)
            if callable(destroy_scene):
                destroy_scene()
        except Exception as exc:  # pragma: no cover - Genesis internals
            self.logger.debug("Scene destroy reported: %s", exc)
        try:
            self.resources.client.destroy()
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Genesis client destroy raised: %s", exc)
        self.logger.info("Simulation session closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
        return False

    @property
    def spec(self) -> EnvironmentSpec:
        return self.resources.spec

    @property
    def command_bus(self) -> CommandBus:
        return self.resources.command_bus

    @property
    def event_bus(self) -> EventBus:
        return self.resources.event_bus

    @property
    def scene(self):  # pragma: no cover - passthrough property
        return self.resources.scene_result.scene

    @property
    @property
    def coordinators(self) -> Dict[str, RobotCoordinator]:
        return dict(self._coordinators)

    @property
    def state_estimator(self) -> Optional[StateEstimator]:
        return self._state_estimator

    @property
    def collision_world(self) -> Optional[CollisionWorld]:
        return self._collision_world

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    async def send_command(self, command: Command) -> str:
        """Submit a command to the bus and return its identifier."""
        robot_name = getattr(command, "robot", "") or getattr(command, "robot_name", "") or "<unknown>"
        if isinstance(command, _CartesianMoveCommand):
            pos = ", ".join(f"{float(v):.3f}" for v in command.position_m)
            orient = ", ".join(f"{float(v):.2f}" for v in command.orientation_deg)
            self.logger.info(
                "[%s] Queue cartesian move frame=%s pos=[%s]m rpy=[%s]° duration=%.2fs",
                robot_name,
                command.reference_frame or "base",
                pos,
                orient,
                float(command.duration),
            )
        elif isinstance(command, _JointTargetsCommand):
            preview = ", ".join(f"{float(v):.1f}" for v in command.values_deg[:6])
            if len(command.values_deg) > 6:
                preview += ", …"
            metadata = command.metadata or {}
            source = str(metadata.get("source", "command"))
            tag = "direct" if metadata.get("direct_joint_set") else "planned"
            message_args = (
                robot_name,
                source,
                tag,
                preview,
                float(command.duration),
            )
            if metadata.get("direct_joint_set") and source == "gui_slider":
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "[%s] Queue joint targets (%s/%s): [%s]° duration=%.2fs",
                        *message_args,
                    )
            else:
                self.logger.info(
                    "[%s] Queue joint targets (%s/%s): [%s]° duration=%.2fs",
                    *message_args,
                )
        else:
            self.logger.info(
                "[%s] Queue command %s priority=%s",
                robot_name,
                type(command).__name__,
                getattr(command.priority, "name", command.priority),
            )
        await self.command_bus.submit(command)
        return command.command_id

    async def pop_recent_trajectory(self, robot: str) -> Optional[Dict[str, Any]]:
        """Return the most recent trajectory planned for ``robot``."""
        return self._sync_trajectory_manager.pop_recent_trajectory(robot)

    async def clear_recent_trajectories(self, robot: str) -> None:
        """Clear buffered trajectory data for ``robot``."""
        self._sync_trajectory_manager.clear_recent_trajectories(robot)

    def get_latest_state(self, robot: str) -> Optional[RobotState]:
        """Return the most recent state sample for a robot, if available."""
        return self._latest_states.get(robot)

    async def wait_for_robot_state(self, robot: str, timeout: float = 5.0) -> RobotState:
        """Block until a robot state sample is available or timeout expires."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            state = self._latest_states.get(robot)
            if state is not None:
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for state from robot '{robot}'")
            await asyncio.sleep(0.05)

    async def wait_until_idle(
        self,
        robot: str,
        *,
        timeout: float = 10.0,
        settle_time: float = 0.25,
    ) -> None:
        """Wait until the synchronized trajectory manager reports the robot idle."""
        deadline = time.monotonic() + max(0.0, timeout)
        settled_since: Optional[float] = None
        while True:
            active = self._sync_trajectory_manager.has_active_trajectory(robot)
            coordinator = self._coordinators.get(robot)
            busy = coordinator.has_active_motion() if coordinator else False
            try:
                pending = self.command_bus.pending_count(robot)
            except KeyError:
                pending = 0

            if not active and not busy and pending == 0:
                if settled_since is None:
                    settled_since = time.monotonic()
                elif time.monotonic() - settled_since >= settle_time:
                    return
            else:
                settled_since = None

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Robot '{robot}' did not become idle within {timeout:.1f}s")
            await asyncio.sleep(0.05)

    def reference_frames(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return known static reference frames (world/object)."""
        return {name: (pos.copy(), quat.copy()) for name, (pos, quat) in self._reference_frames.items()}

    def _build_reference_frames(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        frames: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "world": (
                np.zeros(3, dtype=np.float64),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
        }

        for obj in self.resources.spec.world.objects:
            name = (obj.name or "").strip()
            if not name:
                continue
            position = np.asarray(obj.pose_position, dtype=np.float64)
            roll, pitch, yaw = (math.radians(float(v)) for v in obj.pose_orientation_rpy)
            quat = np.asarray(rpy_to_quaternion(roll, pitch, yaw), dtype=np.float64)
            frames[f"obj:{name.lower()}"] = (position, quat)

        return frames

    def _initialise_robot_coordinators(self) -> None:
        contexts: Dict[str, RobotContext] = {}
        safety = self.resources.spec.config.policies.safety
        frames = self._build_reference_frames()
        self._reference_frames = frames
        for robot in self.resources.spec.robots:
            entity = self.resources.scene_result.robot_entities.get(robot.name)
            if entity is None:
                raise KeyError(f"Genesis entity for robot '{robot.name}' not found")
            joint_count = self._infer_joint_count(robot, entity)
            ctx = RobotContext(
                profile=robot,
                entity=entity,
                joint_count=joint_count,
                logger=self.logger.getChild(f"robot.{robot.name}"),
                safety_policy=safety,
            )
            contexts[robot.name] = ctx
            if robot.initial_joint_positions_deg:
                initial_rad = np.deg2rad(np.asarray(robot.initial_joint_positions_deg, dtype=np.float64))
                if initial_rad.size:
                    initial_slice = initial_rad[:joint_count]
                    self._sync_trajectory_manager.hold_position(robot.name, initial_slice)
                    set_joint_positions(entity, initial_slice.tolist(), joint_count, ctx.logger)
        if contexts:
            self._collision_world = self._factories.make_collision_world(contexts)
            for name, ctx in contexts.items():
                robot = ctx.profile
                coordinator = RobotCoordinator(
                    ctx,
                    ik_solver=self._factories.make_ik_solver(robot),
                    planner=self._factories.make_motion_planner(robot),
                    sync_trajectory_manager=self._sync_trajectory_manager,
                    command_bus=self.resources.command_bus,
                    event_bus=self.resources.event_bus,
                    collision_world=self._collision_world,
                    state_provider=self._other_robot_states,
                    frame_transforms=frames,
                    logger=self.logger.getChild(f"coordinator.{robot.name}"),
                )
                coordinator.start()
                self._coordinators[name] = coordinator
            self._state_estimator = self._factories.make_state_estimator(contexts)
            self._start_state_tasks(contexts)

    def _infer_joint_count(self, robot: RobotProfile, entity: object) -> int:
        joints = get_joint_positions(entity, self.logger, prefer_struct=True)
        if joints.size:
            return int(joints.size)
        if robot.initial_joint_positions_deg:
            return len(robot.initial_joint_positions_deg)
        metadata = robot.metadata or {}
        if "dof" in metadata:
            try:
                return int(metadata["dof"])
            except (TypeError, ValueError):
                pass
        return 6

    def _start_state_tasks(self, contexts: Dict[str, RobotContext]) -> None:
        if self._state_estimator is None:
            return
        loop = asyncio.get_event_loop()
        for name, ctx in contexts.items():
            dt = float(self.resources.spec.scene.dt)
            frequency = 1.0 / dt if dt > 0.0 else 60.0
            subscription = StateSubscription(robot=ctx.profile, frequency_hz=frequency)
            task = loop.create_task(self._stream_robot_state(subscription))
            self._state_tasks[name] = task

    async def _stream_robot_state(self, subscription: StateSubscription) -> None:
        assert self._state_estimator is not None
        async for state in self._state_estimator.states(subscription):
            self._latest_states[state.name] = state
            if self._collision_world is not None:
                cache = {name: tuple(s.joint_positions) for name, s in self._latest_states.items()}
                self._collision_world.update_environment(cache)
            event = RobotStateSnapshot(
                topic=EventTopic.ROBOT_STATE,
                robot=state.name,
                state=state,
                timestamp_s=state.timestamp_s,
            )
            await self.event_bus.publish(event)

    def _other_robot_states(self, robot: str) -> Dict[str, Tuple[float, ...]]:
        return {
            name: tuple(state.joint_positions)
            for name, state in self._latest_states.items()
            if name != robot
        }

    def _start_simulation_loop(self) -> None:
        """Start the continuous simulation loop that steps Genesis."""
        if self._simulation_task is not None:
            return
        self._simulation_task = asyncio.create_task(self._simulation_loop())

    def _update_robot_trajectories(self, current_time: float) -> None:
        """Update robot joint positions based on active trajectories (like legacy MovementController)."""
        from ..infrastructure import set_joint_positions
        
        # Debug: Log all active trajectories
        active_robots = self._sync_trajectory_manager.get_active_robots()
        if active_robots:
            self.logger.debug(f"Active trajectories at t={current_time:.3f}: {active_robots}")
        
        # Process all robots that have either active trajectories or are holding final positions
        all_controlled_robots = set(active_robots)
        
        # Also check robots with final positions (holding position after trajectory completion)
        for robot_name in self._coordinators.keys():
            if self._sync_trajectory_manager.get_current_positions(robot_name, current_time) is not None:
                all_controlled_robots.add(robot_name)
        
        for robot_name in all_controlled_robots:
            try:
                # Get the robot entity and context
                robot_entity = None
                joint_count = 6  # Default
                coordinator_ref = None
                
                for coordinator in self._coordinators.values():
                    if coordinator.ctx.profile.name == robot_name:
                        robot_entity = coordinator.ctx.entity
                        joint_count = coordinator.ctx.joint_count
                        coordinator_ref = coordinator
                        break
                
                if robot_entity is None:
                    self.logger.warning(f"No entity found for robot {robot_name}")
                    continue
                
                # Get current joint positions from trajectory or final position
                q = self._sync_trajectory_manager.get_current_positions(robot_name, current_time)
                if q is not None:
                    # Apply joint positions to Genesis entity (like legacy set_robot_joints)
                    set_joint_positions(robot_entity, q.tolist(), joint_count, self.logger)
                    
                    # Only log active trajectories, not position holding
                    if robot_name in active_robots:
                        self.logger.debug(f"Applied trajectory joints to {robot_name}: {np.round(q, 3)}")
                        
                        # Check if trajectory is done and log completion only once
                        if self._sync_trajectory_manager.is_trajectory_done(robot_name, current_time):
                            self.logger.info(f"Trajectory execution completed for {robot_name}")
                            if coordinator_ref is not None:
                                coordinator_ref.on_trajectory_completed(np.asarray(q, dtype=np.float64))
                        
            except Exception as exc:
                self.logger.error(f"Failed to update trajectory for {robot_name}: {exc}")
                # Stop the problematic trajectory to prevent repeated errors
                self._sync_trajectory_manager.stop_trajectory(robot_name)

    async def _simulation_loop(self) -> None:
        """Continuously step the Genesis simulation at the specified rate."""
        dt = float(self.resources.spec.scene.dt)
        self.logger.info(f"Starting simulation loop with dt={dt}s")
        
        try:
            while True:
                start_time = asyncio.get_event_loop().time()
                current_time = time.time()  # Use time.time() like legacy system
                
                # Update trajectories for all robots (like legacy system)
                self._update_robot_trajectories(current_time)
                
                # Step the Genesis scene
                try:
                    self.resources.scene_result.scene.step()
                except Exception as exc:
                    self.logger.error(f"Genesis scene step failed: {exc}")
                    break
                
                # Maintain timing
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0.0, dt - elapsed)
                if sleep_time > 0.0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("Simulation loop cancelled")
            raise
        except Exception as exc:
            self.logger.error(f"Simulation loop error: {exc}")
            raise


__all__ = ["SimulationSession", "SimulationResources", "ServiceFactories"]
