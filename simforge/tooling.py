"""End effector tooling management for robots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import logging

from .transformations import rpy_to_rotation_matrix, quaternion_to_rotation_matrix

try:
    import trimesh
    import fcl
    HAS_GEOMETRY = True
except ImportError:
    HAS_GEOMETRY = False


@dataclass
class ToolDefinition:
    """Definition of an end effector tool."""
    name: str
    urdf_path: str
    tcp_offset: Optional[Tuple[float, float, float, float, float, float, float]] = None  # x,y,z,qw,qx,qy,qz
    attach_offset: Optional[Tuple[float, float, float, float, float, float]] = None  # x,y,z,roll,pitch,yaw
    
    def __post_init__(self):
        """Validate and normalize paths."""
        self.urdf_path = str(Path(self.urdf_path).resolve())
        
    def get_tcp_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get TCP offset as position and quaternion."""
        if self.tcp_offset:
            pos = np.array(self.tcp_offset[:3], dtype=np.float64)
            quat = np.array(self.tcp_offset[3:], dtype=np.float64)
            # Normalize quaternion
            quat = quat / np.linalg.norm(quat)
            return pos, quat
        return np.zeros(3), np.array([1, 0, 0, 0])  # Identity
    
    def get_attach_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get attachment offset as position and rotation matrix."""
        if self.attach_offset:
            pos = np.array(self.attach_offset[:3], dtype=np.float64)
            R = rpy_to_rotation_matrix(
                self.attach_offset[3], 
                self.attach_offset[4], 
                self.attach_offset[5]
            )
            return pos, R
        return np.zeros(3), np.eye(3)


class ToolManager:
    """Manages end effector tools for robots."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.tools: Dict[str, ToolDefinition] = {}
        self._tool_geometries: Dict[str, List[Any]] = {}  # Cache parsed geometries
        
    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
        
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self.tools.get(name)
    
    def parse_tool_geometry(self, tool_name: str, shrink_factor: float = 1.0) -> List[Dict[str, Any]]:
        """Parse tool URDF and extract collision geometry.
        
        Returns a list of geometry dictionaries with:
        - 'local_R': rotation matrix relative to tool base
        - 'local_t': translation relative to tool base
        - 'geometry': FCL collision geometry object
        """
        if not HAS_GEOMETRY:
            self.logger.warning("trimesh/FCL not available - cannot parse tool geometry")
            return []
            
        tool = self.tools.get(tool_name)
        if not tool:
            self.logger.warning(f"Tool '{tool_name}' not found")
            return []
            
        # Check cache
        cache_key = f"{tool_name}_{shrink_factor}"
        if cache_key in self._tool_geometries:
            return self._tool_geometries[cache_key]
            
        geometries = []
        
        try:
            # Parse URDF
            tree = ET.parse(tool.urdf_path)
            root = tree.getroot()
            base_dir = Path(tool.urdf_path).parent
            
            # Find tool_base_link collision geometry
            for link in root.findall("link"):
                link_name = link.get("name", "")
                if "base" not in link_name.lower():
                    continue
                    
                for collision in link.findall("collision"):
                    geom_elem = collision.find("geometry")
                    if geom_elem is None:
                        continue
                        
                    # Get local transform
                    origin = collision.find("origin")
                    if origin is not None:
                        xyz = origin.get("xyz", "0 0 0").split()
                        rpy = origin.get("rpy", "0 0 0").split()
                        local_t = np.array([float(x) for x in xyz], dtype=np.float64)
                        local_R = rpy_to_rotation_matrix(
                            float(rpy[0]), float(rpy[1]), float(rpy[2])
                        )
                    else:
                        local_t = np.zeros(3)
                        local_R = np.eye(3)
                    
                    # Parse geometry
                    fcl_geom = None
                    
                    # Handle mesh
                    mesh = geom_elem.find("mesh")
                    if mesh is not None:
                        filename = mesh.get("filename", "")
                        if filename:
                            mesh_path = base_dir / filename
                            if mesh_path.exists():
                                try:
                                    tm = trimesh.load(mesh_path, force="mesh", process=False)
                                    vertices = np.asarray(tm.vertices, dtype=np.float64) * shrink_factor
                                    faces = np.asarray(tm.faces, dtype=np.int32)
                                    
                                    bvh = fcl.BVHModel()
                                    bvh.beginModel(vertices.shape[0], faces.shape[0])
                                    bvh.addSubModel(vertices, faces)
                                    bvh.endModel()
                                    fcl_geom = bvh
                                except Exception as e:
                                    self.logger.debug(f"Failed to load mesh {mesh_path}: {e}")
                    
                    # Handle cylinder
                    cylinder = geom_elem.find("cylinder")
                    if cylinder is not None and fcl_geom is None:
                        radius = float(cylinder.get("radius", "0.01")) * shrink_factor
                        length = float(cylinder.get("length", "0.1")) * shrink_factor
                        fcl_geom = fcl.Cylinder(radius, length)
                    
                    # Handle box
                    box = geom_elem.find("box")
                    if box is not None and fcl_geom is None:
                        size = box.get("size", "0.1 0.1 0.1").split()
                        sx = float(size[0]) * shrink_factor
                        sy = float(size[1]) * shrink_factor
                        sz = float(size[2]) * shrink_factor
                        fcl_geom = fcl.Box(sx, sy, sz)
                    
                    # Handle sphere
                    sphere = geom_elem.find("sphere")
                    if sphere is not None and fcl_geom is None:
                        radius = float(sphere.get("radius", "0.01")) * shrink_factor
                        fcl_geom = fcl.Sphere(radius)
                    
                    if fcl_geom is not None:
                        geometries.append({
                            'local_R': local_R,
                            'local_t': local_t,
                            'geometry': fcl_geom
                        })
                        
        except Exception as e:
            self.logger.error(f"Failed to parse tool geometry for '{tool_name}': {e}")
            
        # Cache result
        self._tool_geometries[cache_key] = geometries
        return geometries
    
    def get_tool_tcp_offset(self, tool_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get TCP offset for a tool (position and quaternion)."""
        tool = self.tools.get(tool_name)
        if tool:
            return tool.get_tcp_transform()
        return None
        
    def attach_tool_to_robot_urdf(
        self, 
        robot_urdf_path: str, 
        tool_name: str,
        end_effector_link: str,
        output_path: Optional[str] = None
    ) -> str:
        """Create a combined URDF with tool attached to robot.
        
        This creates a new URDF file that includes both robot and tool,
        with the tool attached via a fixed joint.
        """
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        # Parse robot URDF
        robot_tree = ET.parse(robot_urdf_path)
        robot_root = robot_tree.getroot()
        
        # Parse tool URDF
        tool_tree = ET.parse(tool.urdf_path)
        tool_root = tool_tree.getroot()
        
        # Get attach transform
        attach_pos, attach_R = tool.get_attach_transform()
        
        # Convert rotation matrix to RPY for URDF
        # Simple conversion (may need more robust implementation)
        rpy = [0, 0, 0]  # Simplified for now
        if not np.allclose(attach_R, np.eye(3)):
            # Extract Euler angles from rotation matrix
            rpy[0] = np.arctan2(attach_R[2, 1], attach_R[2, 2])  # roll
            rpy[1] = np.arctan2(-attach_R[2, 0], np.sqrt(attach_R[2, 1]**2 + attach_R[2, 2]**2))  # pitch
            rpy[2] = np.arctan2(attach_R[1, 0], attach_R[0, 0])  # yaw
        
        # Rename tool links to avoid conflicts
        tool_prefix = f"{tool_name}_"
        for link in tool_root.findall("link"):
            old_name = link.get("name", "")
            new_name = tool_prefix + old_name
            link.set("name", new_name)
            
        # Update joint references
        for joint in tool_root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is not None:
                old_link = parent.get("link", "")
                parent.set("link", tool_prefix + old_link)
            if child is not None:
                old_link = child.get("link", "")
                child.set("link", tool_prefix + old_link)
        
        # Add tool links to robot
        for link in tool_root.findall("link"):
            robot_root.append(link)
            
        # Add tool joints to robot
        for joint in tool_root.findall("joint"):
            robot_root.append(joint)
            
        # Create attachment joint
        attach_joint = ET.SubElement(robot_root, "joint")
        attach_joint.set("name", f"{tool_name}_attach_joint")
        attach_joint.set("type", "fixed")
        
        parent_elem = ET.SubElement(attach_joint, "parent")
        parent_elem.set("link", end_effector_link)
        
        child_elem = ET.SubElement(attach_joint, "child")
        child_elem.set("link", tool_prefix + "tool_base_link")
        
        origin_elem = ET.SubElement(attach_joint, "origin")
        origin_elem.set("xyz", f"{attach_pos[0]} {attach_pos[1]} {attach_pos[2]}")
        origin_elem.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
        
        # Save combined URDF
        if output_path is None:
            robot_path = Path(robot_urdf_path)
            output_path = robot_path.parent / f"{robot_path.stem}_with_{tool_name}.urdf"
        
        robot_tree.write(output_path, encoding="utf-8", xml_declaration=True)
        self.logger.info(f"Created combined URDF with tool at: {output_path}")
        
        return str(output_path)


def create_tool_from_config(config: Dict[str, Any]) -> ToolDefinition:
    """Create a ToolDefinition from configuration dictionary."""
    return ToolDefinition(
        name=config.get("name", "unnamed_tool"),
        urdf_path=config.get("urdf", ""),
        tcp_offset=config.get("tcp_offset"),
        attach_offset=config.get("attach_offset")
    )
