"""
Collision Checking Service for Simforge Server

Uses python-fcl for collision detection between the robot and environment objects.
This provides collision checking capabilities for:
- Path validation before execution
- Continuous collision monitoring
- IK solution validation

This is a lightweight alternative to cuMotion for development on x86.
On Jetson Thor, this can be replaced with cuMotion for GPU-accelerated planning.
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    import fcl
    HAS_FCL = True
except ImportError:
    HAS_FCL = False
    fcl = None

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    trimesh = None


logger = logging.getLogger(__name__)


@dataclass
class CollisionObject:
    """Represents a collision object in the scene."""
    name: str
    geometry: Any  # fcl.CollisionGeometry
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    fcl_object: Any = None  # fcl.CollisionObject
    
    def update_transform(self, transform: np.ndarray) -> None:
        """Update the object's transform."""
        self.transform = transform
        if self.fcl_object and HAS_FCL:
            # Extract rotation and translation
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            self.fcl_object.setTransform(fcl.Transform(rotation, translation))


@dataclass
class CollisionResult:
    """Result of a collision check."""
    in_collision: bool
    colliding_pairs: List[Tuple[str, str]] = field(default_factory=list)
    min_distance: float = float('inf')
    closest_points: Optional[Tuple[np.ndarray, np.ndarray]] = None


class CollisionChecker:
    """
    Collision checking service using python-fcl.
    
    Features:
    - Robot self-collision checking
    - Robot-environment collision checking
    - Distance queries for path optimization
    - Batch collision checking for trajectory validation
    """
    
    def __init__(
        self,
        urdf_path: Optional[str] = None,
        mesh_base_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the collision checker.
        
        Args:
            urdf_path: Path to the robot URDF file
            mesh_base_path: Base path for resolving mesh file paths
        """
        if not HAS_FCL:
            raise ImportError("python-fcl is required. Install with: pip install python-fcl")
        
        self.urdf_path = urdf_path
        self.mesh_base_path = mesh_base_path or (Path(urdf_path).parent if urdf_path else None)
        
        # Collision objects
        self.robot_links: Dict[str, CollisionObject] = {}
        self.environment_objects: Dict[str, CollisionObject] = {}
        
        # Self-collision pairs to check (adjacent links are typically excluded)
        self.self_collision_pairs: List[Tuple[str, str]] = []
        
        # Links to ignore for self-collision (adjacent links)
        self.ignored_self_collision_pairs: set = set()
        
        # Collision manager for broad-phase
        self.manager = fcl.DynamicAABBTreeCollisionManager()
        
        if urdf_path:
            self._load_robot_from_urdf(urdf_path)
    
    def _load_robot_from_urdf(self, urdf_path: str) -> None:
        """Load robot collision geometry from URDF."""
        logger.info(f"Loading robot collision geometry from {urdf_path}")
        
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # Parse links and their collision geometry
        for link_elem in root.findall("link"):
            link_name = link_elem.get("name")
            if not link_name:
                continue
            
            collision_elem = link_elem.find("collision")
            if collision_elem is None:
                continue
            
            geometry_elem = collision_elem.find("geometry")
            if geometry_elem is None:
                continue
            
            # Parse origin
            origin_elem = collision_elem.find("origin")
            origin_transform = np.eye(4)
            if origin_elem is not None:
                xyz = origin_elem.get("xyz", "0 0 0").split()
                rpy = origin_elem.get("rpy", "0 0 0").split()
                origin_transform = self._pose_to_matrix(
                    [float(x) for x in xyz],
                    [float(r) for r in rpy],
                )
            
            # Parse geometry
            geom = self._parse_geometry(geometry_elem, link_name)
            if geom is not None:
                obj = CollisionObject(
                    name=link_name,
                    geometry=geom,
                    transform=origin_transform,
                )
                obj.fcl_object = fcl.CollisionObject(geom, fcl.Transform())
                self.robot_links[link_name] = obj
                logger.debug(f"Loaded collision geometry for link: {link_name}")
        
        # Parse joints to determine adjacent links (for self-collision filtering)
        for joint_elem in root.findall("joint"):
            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            if parent_elem is not None and child_elem is not None:
                parent_link = parent_elem.get("link")
                child_link = child_elem.get("link")
                if parent_link and child_link:
                    # Ignore collision between adjacent links
                    pair = tuple(sorted([parent_link, child_link]))
                    self.ignored_self_collision_pairs.add(pair)
        
        # Generate self-collision pairs
        link_names = list(self.robot_links.keys())
        for i, link_a in enumerate(link_names):
            for link_b in link_names[i+1:]:
                pair = tuple(sorted([link_a, link_b]))
                if pair not in self.ignored_self_collision_pairs:
                    self.self_collision_pairs.append((link_a, link_b))
        
        logger.info(f"Loaded {len(self.robot_links)} robot links, {len(self.self_collision_pairs)} self-collision pairs")
    
    def _parse_geometry(self, geometry_elem: ET.Element, link_name: str) -> Optional[Any]:
        """Parse geometry element and return FCL geometry."""
        # Check for mesh
        mesh_elem = geometry_elem.find("mesh")
        if mesh_elem is not None:
            filename = mesh_elem.get("filename", "")
            scale_str = mesh_elem.get("scale", "1 1 1")
            scale = [float(s) for s in scale_str.split()]
            return self._load_mesh_geometry(filename, scale, link_name)
        
        # Check for box
        box_elem = geometry_elem.find("box")
        if box_elem is not None:
            size_str = box_elem.get("size", "1 1 1")
            size = [float(s) for s in size_str.split()]
            return fcl.Box(*size)
        
        # Check for sphere
        sphere_elem = geometry_elem.find("sphere")
        if sphere_elem is not None:
            radius = float(sphere_elem.get("radius", "0.1"))
            return fcl.Sphere(radius)
        
        # Check for cylinder
        cylinder_elem = geometry_elem.find("cylinder")
        if cylinder_elem is not None:
            radius = float(cylinder_elem.get("radius", "0.1"))
            length = float(cylinder_elem.get("length", "1.0"))
            return fcl.Cylinder(radius, length)
        
        return None
    
    def _load_mesh_geometry(
        self,
        filename: str,
        scale: List[float],
        link_name: str,
    ) -> Optional[Any]:
        """Load mesh geometry from file."""
        if not HAS_TRIMESH:
            logger.warning(f"trimesh not available, using box approximation for {link_name}")
            return fcl.Box(0.1, 0.1, 0.1)
        
        # Resolve file path
        if filename.startswith("package://"):
            # Handle package:// URIs
            package_path = filename.replace("package://", "")
            parts = package_path.split("/", 1)
            if len(parts) == 2 and self.mesh_base_path:
                # Try to find the mesh relative to base path
                mesh_path = self.mesh_base_path / parts[1]
            else:
                mesh_path = Path(filename)
        elif self.mesh_base_path:
            mesh_path = self.mesh_base_path / filename
        else:
            mesh_path = Path(filename)
        
        if not mesh_path.exists():
            logger.warning(f"Mesh file not found: {mesh_path}, using box approximation for {link_name}")
            return fcl.Box(0.1, 0.1, 0.1)
        
        try:
            mesh = trimesh.load(str(mesh_path))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # Apply scale
            mesh.apply_scale(scale)
            
            # Convert to FCL BVH
            vertices = np.array(mesh.vertices, dtype=np.float64)
            faces = np.array(mesh.faces, dtype=np.int32)
            
            bvh = fcl.BVHModel()
            bvh.beginModel(len(vertices), len(faces))
            bvh.addSubModel(vertices, faces)
            bvh.endModel()
            
            return bvh
            
        except Exception as e:
            logger.warning(f"Failed to load mesh {mesh_path}: {e}, using box approximation")
            return fcl.Box(0.1, 0.1, 0.1)
    
    def _pose_to_matrix(self, position: List[float], rpy: List[float]) -> np.ndarray:
        """Convert position and RPY to 4x4 transformation matrix."""
        roll, pitch, yaw = rpy
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])
        
        R = Rz @ Ry @ Rx
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        return T
    
    def add_environment_object(
        self,
        name: str,
        geometry_type: str,
        params: Dict[str, Any],
        transform: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add an environment object for collision checking.
        
        Args:
            name: Unique name for the object
            geometry_type: One of 'box', 'sphere', 'cylinder', 'mesh'
            params: Geometry parameters (e.g., {'size': [1, 1, 1]} for box)
            transform: 4x4 transformation matrix
        """
        if geometry_type == "box":
            geom = fcl.Box(*params.get("size", [1, 1, 1]))
        elif geometry_type == "sphere":
            geom = fcl.Sphere(params.get("radius", 0.1))
        elif geometry_type == "cylinder":
            geom = fcl.Cylinder(
                params.get("radius", 0.1),
                params.get("length", 1.0),
            )
        elif geometry_type == "mesh" and HAS_TRIMESH:
            mesh_path = params.get("path")
            if mesh_path:
                geom = self._load_mesh_geometry(
                    mesh_path,
                    params.get("scale", [1, 1, 1]),
                    name,
                )
            else:
                logger.warning(f"No mesh path provided for {name}")
                return
        else:
            logger.warning(f"Unknown geometry type: {geometry_type}")
            return
        
        if geom is None:
            return
        
        obj = CollisionObject(
            name=name,
            geometry=geom,
            transform=transform if transform is not None else np.eye(4),
        )
        
        if transform is not None:
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            obj.fcl_object = fcl.CollisionObject(geom, fcl.Transform(rotation, translation))
        else:
            obj.fcl_object = fcl.CollisionObject(geom, fcl.Transform())
        
        self.environment_objects[name] = obj
        logger.info(f"Added environment object: {name} ({geometry_type})")
    
    def update_robot_transforms(self, link_transforms: Dict[str, np.ndarray]) -> None:
        """
        Update robot link transforms (typically from forward kinematics).
        
        Args:
            link_transforms: Dict mapping link name to 4x4 transform matrix
        """
        for link_name, transform in link_transforms.items():
            if link_name in self.robot_links:
                self.robot_links[link_name].update_transform(transform)
    
    def check_collision(
        self,
        check_self_collision: bool = True,
        check_environment: bool = True,
    ) -> CollisionResult:
        """
        Check for collisions.
        
        Args:
            check_self_collision: Whether to check robot self-collision
            check_environment: Whether to check robot-environment collision
        
        Returns:
            CollisionResult with collision status and details
        """
        colliding_pairs: List[Tuple[str, str]] = []
        min_distance = float('inf')
        
        # Self-collision check
        if check_self_collision:
            for link_a, link_b in self.self_collision_pairs:
                obj_a = self.robot_links.get(link_a)
                obj_b = self.robot_links.get(link_b)
                if obj_a is None or obj_b is None:
                    continue
                
                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()
                
                ret = fcl.collide(obj_a.fcl_object, obj_b.fcl_object, request, result)
                
                if result.is_collision:
                    colliding_pairs.append((link_a, link_b))
        
        # Environment collision check
        if check_environment:
            for link_name, link_obj in self.robot_links.items():
                for env_name, env_obj in self.environment_objects.items():
                    request = fcl.CollisionRequest()
                    result = fcl.CollisionResult()
                    
                    ret = fcl.collide(link_obj.fcl_object, env_obj.fcl_object, request, result)
                    
                    if result.is_collision:
                        colliding_pairs.append((link_name, env_name))
                    
                    # Distance query
                    dist_request = fcl.DistanceRequest()
                    dist_result = fcl.DistanceResult()
                    
                    fcl.distance(link_obj.fcl_object, env_obj.fcl_object, dist_request, dist_result)
                    
                    if dist_result.min_distance < min_distance:
                        min_distance = dist_result.min_distance
        
        return CollisionResult(
            in_collision=len(colliding_pairs) > 0,
            colliding_pairs=colliding_pairs,
            min_distance=min_distance,
        )
    
    def check_trajectory(
        self,
        trajectory: List[Dict[str, np.ndarray]],
        check_self_collision: bool = True,
        check_environment: bool = True,
    ) -> Tuple[bool, int, CollisionResult]:
        """
        Check an entire trajectory for collisions.
        
        Args:
            trajectory: List of link_transforms dicts for each waypoint
            check_self_collision: Whether to check robot self-collision
            check_environment: Whether to check robot-environment collision
        
        Returns:
            Tuple of (all_clear, first_collision_index, collision_result)
        """
        for i, link_transforms in enumerate(trajectory):
            self.update_robot_transforms(link_transforms)
            result = self.check_collision(check_self_collision, check_environment)
            
            if result.in_collision:
                return False, i, result
        
        return True, -1, CollisionResult(in_collision=False)


# Factory function for creating collision checker from URDF
def create_collision_checker_from_urdf(
    urdf_path: str,
    mesh_base_path: Optional[str] = None,
) -> CollisionChecker:
    """Create a collision checker from a URDF file."""
    return CollisionChecker(urdf_path=urdf_path, mesh_base_path=mesh_base_path)
