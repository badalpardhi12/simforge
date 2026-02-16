from typing import Dict, Any, Optional
import numpy as np
import asyncio
from ..core.commands import (
    Command, MoveJointCommand, MoveCartesianCommand, 
    SetJointPositionsCommand, CommandType
)
from ..core.events import Event
from .event_bus import EventBus


class CommandProcessor:
    """Processes commands and generates trajectories."""
    
    def __init__(self, 
                 event_bus: EventBus,
                 ik_solver = None,
                 planner = None):
        self.event_bus = event_bus
        self.ik_solver = ik_solver
        self.planner = planner
        self._processing = False
        
    async def process(self, command: Command) -> Dict[str, Any]:
        """Process a command and emit appropriate events."""
        self._processing = True
        result = {'success': False, 'error': None}
        
        try:
            if isinstance(command, MoveJointCommand):
                result = await self._process_joint_move(command)
            elif isinstance(command, MoveCartesianCommand):
                result = await self._process_cartesian_move(command)
            elif isinstance(command, SetJointPositionsCommand):
                result = await self._process_joint_positions(command)
            else:
                result['error'] = f"Unknown command type: {command.command_type}"
                
        except Exception as e:
            result['error'] = str(e)
            self.event_bus.emit(Event("command_error", {
                "command": command,
                "error": str(e)
            }))
        finally:
            self._processing = False
            
        return result
    
    async def _process_joint_move(self, cmd: MoveJointCommand) -> Dict[str, Any]:
        """Process single joint movement command."""
        # For now, directly emit the movement event
        self.event_bus.emit(Event("joint_move_requested", {
            "robot": cmd.robot_name,
            "joint_index": cmd.joint_index,
            "target": cmd.target_position,
            "duration": cmd.duration
        }))
        
        return {'success': True}
    
    async def _process_cartesian_move(self, cmd: MoveCartesianCommand) -> Dict[str, Any]:
        """Process Cartesian movement command."""
        if self.ik_solver is None:
            return {'success': False, 'error': 'IK solver not available'}
            
        # Solve IK for target pose
        target_joints = self.ik_solver.solve(cmd.target_pose)
        
        if target_joints is None:
            self.event_bus.emit(Event("ik_failed", {"command": cmd}))
            return {'success': False, 'error': 'IK solution not found'}
        
        # If planning is available and collision checking is requested
        if self.planner and cmd.use_collision_checking:
            # Get current joint positions (this would come from robot state)
            current_joints = np.zeros_like(target_joints)  # Placeholder
            
            path = self.planner.plan(current_joints, target_joints)
            
            if path is None:
                self.event_bus.emit(Event("planning_failed", {"command": cmd}))
                return {'success': False, 'error': 'Motion planning failed'}
            
            # Emit trajectory
            self.event_bus.emit(Event("trajectory_ready", {
                "robot": cmd.robot_name,
                "path": path.tolist(),
                "duration": cmd.duration
            }))
        else:
            # Direct movement without planning
            self.event_bus.emit(Event("cartesian_move_requested", {
                "robot": cmd.robot_name,
                "target_joints": target_joints.tolist(),
                "duration": cmd.duration
            }))
        
        return {'success': True}
    
    async def _process_joint_positions(self, cmd: SetJointPositionsCommand) -> Dict[str, Any]:
        """Process joint positions command."""
        self.event_bus.emit(Event("joint_positions_requested", {
            "robot": cmd.robot_name,
            "positions": cmd.positions,
            "duration": cmd.duration,
            "synchronous": cmd.synchronous
        }))
        
        return {'success': True}