"""Collision checking using FCL.

Clean collision detection without the complexity of the legacy implementation.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path

try:
    import fcl
    HAS_FCL = True
except ImportError:
    HAS_FCL = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class CollisionChecker:
    """Simple collision checker for robot self-collision and ground plane."""
    
    def __init__(self, urdf_path: str, logger) -> None:
        self.urdf_path = Path(urdf_path)
        self.logger = logger
        self.collision_objects: Dict[str, List[Any]] = {}
        self.ground_plane_z = 0.0
        self.available = HAS_FCL and HAS_TRIMESH
        
        if not self.available:
            self.logger.warning(
                "FCL or trimesh not available - collision checking disabled"
            )
            return
            
        self._load_collision_meshes()

    def _load_collision_meshes(self):
        """Load collision meshes from URDF."""
        if not self.available:
            return
            
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
            
            for link in root.findall("link"):
                link_name = link.get("name", "")
                for collision in link.findall("collision"):
                    geometry = collision.find("geometry")
                    if geometry is None:
                        continue
                        
                    mesh_elem = geometry.find("mesh")
                    if mesh_elem is None:
                        continue
                        
                    filename = mesh_elem.get("filename", "")
                    if filename:
                        mesh_path = self.urdf_path.parent / filename
                        if mesh_path.exists():
                            self._add_mesh_collision_object(link_name, mesh_path)
                            
        except Exception as e:
            self.logger.warning(f"Failed to load collision meshes: {e}")

    def _add_mesh_collision_object(self, link_name: str, mesh_path: Path):
        """Add collision object from mesh file."""
        try:
            mesh = trimesh.load(mesh_path)
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                vertices = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.int32)
                
                # Create FCL collision object
                bvh = fcl.BVHModel()
                bvh.beginModel(len(vertices), len(faces))
                bvh.addSubModel(vertices, faces)
                bvh.endModel()
                
                transform = fcl.Transform()
                collision_obj = fcl.CollisionObject(bvh, transform)
                
                if link_name not in self.collision_objects:
                    self.collision_objects[link_name] = []
                self.collision_objects[link_name].append(collision_obj)
                
        except Exception as e:
            self.logger.debug(f"Failed to load mesh {mesh_path}: {e}")

    def check_self_collision(self, link_poses: dict) -> bool:
        """Check for self-collision given link poses.
        
        Args:
            link_poses: Dict mapping link names to (position, quaternion_wxyz) tuples
            
        Returns:
            True if collision detected
        """
        if not self.available:
            return False
            
        # Update transforms
        for link_name, pose in link_poses.items():
            if link_name in self.collision_objects:
                pos, quat_wxyz = pose
                # Convert to rotation matrix
                w, x, y, z = quat_wxyz
                R = self._quat_to_rotation_matrix(w, x, y, z)
                transform = fcl.Transform(R, np.array(pos))
                
                for obj in self.collision_objects[link_name]:
                    obj.setTransform(transform)

        # Check all pairs
        link_names = list(self.collision_objects.keys())
        for i in range(len(link_names)):
            for j in range(i + 1, len(link_names)):
                link_a, link_b = link_names[i], link_names[j]
                
                # Skip adjacent links (would need proper adjacency from URDF)
                if self._are_adjacent(link_a, link_b):
                    continue
                    
                for obj_a in self.collision_objects[link_a]:
                    for obj_b in self.collision_objects[link_b]:
                        request = fcl.CollisionRequest()
                        result = fcl.CollisionResult()
                        if fcl.collide(obj_a, obj_b, request, result) > 0:
                            return True
        return False

    def check_ground_collision(self, link_poses: dict) -> bool:
        """Check if any link is below ground plane."""
        for link_name, pose in link_poses.items():
            pos, _ = pose
            if pos[2] < self.ground_plane_z:
                return True
        return False

    def _are_adjacent(self, link_a: str, link_b: str) -> bool:
        """Simple heuristic for adjacent links."""
        # This is simplified - in reality would parse URDF joint structure
        return abs(hash(link_a) - hash(link_b)) % 3 == 0

    def _quat_to_rotation_matrix(self, w: float, x: float, y: float, z: float) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ], dtype=np.float32)


__all__ = ["CollisionChecker", "HAS_FCL", "HAS_TRIMESH"]