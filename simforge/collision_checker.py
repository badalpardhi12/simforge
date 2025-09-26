# simforge/collision_checker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
from pathlib import Path
import numpy as np
from threading import Lock
import xml.etree.ElementTree as ET

from .transformations import rpy_to_rotation_matrix

try:
    import fcl
    HAS_FCL = True
except Exception:
    HAS_FCL = False

try:
    import trimesh
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False

try:
    import pinocchio as pin
    HAS_PIN = True
except Exception:
    HAS_PIN = False


@dataclass
class _LinkGeom:
    """Collision geometry anchored to a link."""

    local_R: np.ndarray
    local_t: np.ndarray
    geom: "fcl.CollisionGeometry"
    obj: Optional[fcl.CollisionObject] = None
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def collision_object(self) -> fcl.CollisionObject:
        with self._lock:
            if self.obj is None:
                try:
                    self.geom.thisown = False  # avoid double free when object is destroyed
                except AttributeError:
                    pass
                self.obj = fcl.CollisionObject(self.geom, fcl.Transform(np.eye(3), np.zeros(3)))
            return self.obj


class CollisionChecker:
    """FCL collision checker with proper URDF collision origins/scales and optional shrink."""

    def __init__(
        self,
        urdf_path: str,
        logger,
        *,
        base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        allowed_link_pairs: Optional[Iterable[Tuple[str, str]]] = None,
        world_allowed_pairs: Optional[Iterable[Tuple[str, str]]] = None,
        world_boxes: Optional[Iterable[Tuple[str, Tuple[float,float,float], Tuple[float,float,float], Tuple[float,float,float]]]] = None,
        ground_plane_z: float = 0.0,
        collision_mesh_shrink: float = 1.0,
        world_meshes: Optional[Iterable[Tuple[str, str, Tuple[float, float, float], Tuple[float, float, float]]]] = None,
    ) -> None:
        self.logger = logger
        self.urdf_path = Path(urdf_path)
        self.ground_plane_z = float(ground_plane_z)
        self.available = HAS_FCL and HAS_TRIMESH
        if not self.available:
            self.logger.warning("FCL/trimesh not available - collision checking disabled")
            return

        # Robot base pose in WORLD; all world objects will be re-expressed in this BASE frame
        self._base_t = np.array(base_position, dtype=np.float64)
        self._base_R = rpy_to_rotation_matrix(*base_orientation_rpy)

        self._shrink = float(collision_mesh_shrink)
        self.robot_geoms: Dict[str, List[_LinkGeom]] = {}
        self.env_objs: Dict[str, fcl.CollisionObject] = {}
        # External robots registered as dynamic obstacles: name -> (link -> geoms)
        self.env_robot_geoms: Dict[str, Dict[str, List[_LinkGeom]]] = {}
        self.allowed_link_pairs: Set[Tuple[str, str]] = set()
        self.allowed_world_pairs: Set[Tuple[str, str]] = set()

        if allowed_link_pairs:
            for a, b in allowed_link_pairs:
                self.allowed_link_pairs.add(tuple(sorted((a, b))))
        if world_allowed_pairs:
            for a, b in world_allowed_pairs:
                # keep order (robotLink, obj:name)
                self.allowed_world_pairs.add((a, b))

        self.world_mesh_geoms: Dict[str, List[_LinkGeom]] = {}
        self._world_geom_refs: List[fcl.CollisionGeometry] = []

        self._load_urdf()
        self._load_world(world_boxes or [], world_meshes or [])

    # ---------- URDF loader with <origin> and mesh <scale> ----------
    def _load_urdf(self) -> None:
        try:
            root = ET.parse(self.urdf_path).getroot()
        except Exception as e:
            self.logger.warning(f"URDF parse failed: {e}")
            return

        base_dir = self.urdf_path.parent

        # Build a quick map of adjacent (parent, child) links to ignore self-collisions on connected links
        try:
            for joint in root.findall("joint"):
                parent = joint.find("parent")
                child = joint.find("child")
                if parent is not None and child is not None:
                    pa = parent.get("link", ""); ch = child.get("link", "")
                    if pa and ch:
                        self.allowed_link_pairs.add(tuple(sorted((pa, ch))))
        except Exception:
            pass

        for link in root.findall("link"):
            lname = link.get("name", "")
            L: List[_LinkGeom] = []

            for coll in link.findall("collision"):
                geom = coll.find("geometry")
                if geom is None:
                    continue

                # local origin (relative to link frame)
                origin = coll.find("origin")
                if origin is not None:
                    xyz = origin.get("xyz", "0 0 0").split()
                    rpy = origin.get("rpy", "0 0 0").split()
                    t_local = np.array([float(x) for x in xyz], dtype=np.float64)
                    R_local = rpy_to_rotation_matrix(*(float(a) for a in rpy))
                else:
                    t_local = np.zeros(3, dtype=np.float64)
                    R_local = np.eye(3)

                # mesh
                mesh = geom.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename", "")
                    if not filename:
                        continue
                    mesh_path = (base_dir / filename).resolve()
                    if not mesh_path.exists():
                        self.logger.debug(f"mesh not found: {mesh_path}")
                        continue
                    scale_attr = mesh.get("scale", None)
                    scale_vec = np.ones(3, dtype=np.float64)
                    if scale_attr:
                        try:
                            scale_vec = np.array([float(v) for v in scale_attr.split()], dtype=np.float64)
                        except Exception:
                            pass
                    try:
                        tm = trimesh.load(mesh_path, force="mesh", process=False)  # keep raw coords
                        V = np.asarray(tm.vertices, dtype=np.float64)
                        V = (V * scale_vec) * self._shrink
                        F = np.asarray(tm.faces, dtype=np.int32)
                        bvh = fcl.BVHModel()
                        bvh.beginModel(V.shape[0], F.shape[0])
                        bvh.addSubModel(V, F)
                        bvh.endModel()
                        L.append(_LinkGeom(R_local, t_local, bvh))
                    except Exception as e:
                        self.logger.debug(f"failed to load mesh {mesh_path}: {e}")
                    continue

                # box primitive (rare in these URDFs but support it)
                box = geom.find("box")
                if box is not None:
                    size_attr = box.get("size", None)
                    if not size_attr:
                        continue
                    sx, sy, sz = [float(v) for v in size_attr.split()]
                    bx = fcl.Box(sx*self._shrink, sy*self._shrink, sz*self._shrink)
                    L.append(_LinkGeom(R_local, t_local, bx))
                    continue

                cylinder = geom.find("cylinder")
                if cylinder is not None:
                    radius = float(cylinder.get("radius", "0.01")) * self._shrink
                    length = float(cylinder.get("length", "0.1")) * self._shrink
                    cy = fcl.Cylinder(radius, length)
                    L.append(_LinkGeom(R_local, t_local, cy))
                    continue

                sphere = geom.find("sphere")
                if sphere is not None:
                    radius = float(sphere.get("radius", "0.01")) * self._shrink
                    sp = fcl.Sphere(radius)
                    L.append(_LinkGeom(R_local, t_local, sp))
                    continue

            if L:
                self.robot_geoms[lname] = L

    def _load_world(self, boxes, meshes):
        # Express world boxes in the robot BASE frame used by Pinocchio/IK/OMPL
        Rb_T = self._base_R.T
        tb = self._base_t
        for name, shape_info, pos, rpy in boxes:
            if isinstance(shape_info, dict) and shape_info.get("type") == "sphere":
                radius = float(shape_info.get("radius", 0.0))
                geom = fcl.Sphere(radius)
            else:
                sx, sy, sz = [float(x) for x in shape_info]
                geom = fcl.Box(sx, sy, sz)

            Rw = rpy_to_rotation_matrix(*rpy)
            tw = np.array(pos, dtype=np.float64)
            R_rel = Rb_T @ Rw
            t_rel = Rb_T @ (tw - tb)
            try:
                geom.thisown = False
            except AttributeError:
                pass
            co = fcl.CollisionObject(geom, fcl.Transform(R_rel, t_rel))
            self._world_geom_refs.append(geom)
            self.env_objs[f"obj:{name}"] = co

        for name, urdf_path, pos, rpy in meshes:
            try:
                env_geoms = self._build_geoms_from_urdf(urdf_path)
            except Exception as exc:
                self.logger.warning(f"Failed to load world URDF '{urdf_path}': {exc}")
                continue

            Rw = rpy_to_rotation_matrix(*rpy)
            tw = np.array(pos, dtype=np.float64)
            R_rel_base = Rb_T @ Rw
            t_rel_base = Rb_T @ (tw - tb)

            stored_geoms: List[_LinkGeom] = []
            for link, geoms in env_geoms.items():
                for geom in geoms:
                    obj_name = f"obj:{name}:{link}"
                    R_obj = R_rel_base @ geom.local_R
                    t_obj = t_rel_base + R_rel_base @ geom.local_t
                    co = geom.collision_object()
                    co.setTransform(fcl.Transform(R_obj, t_obj))
                    self.env_objs[obj_name] = co
                    stored_geoms.append(geom)

            if stored_geoms:
                self.world_mesh_geoms[name] = stored_geoms

    # ---------- external robots registration and updates ----------
    def register_env_robot(self, name: str, urdf_path: str) -> None:
        """Register another robot's collision geometry as dynamic obstacles."""
        if not self.available:
            return
        try:
            self.env_robot_geoms[name] = self._build_geoms_from_urdf(urdf_path)
            # Count links to verify tool is included
            link_count = len(self.env_robot_geoms[name])
            self.logger.info(f"Registered environment robot: {name} with {link_count} links from {urdf_path}")
        except Exception as e:
            self.logger.warning(f"Failed to register env robot '{name}': {e}")

    def _build_geoms_from_urdf(self, urdf_path: str) -> Dict[str, List[_LinkGeom]]:
        """Load collision geometry from an arbitrary URDF into _LinkGeom map without altering self.robot_geoms."""
        mapping: Dict[str, List[_LinkGeom]] = {}
        try:
            root = ET.parse(str(urdf_path)).getroot()
        except Exception as e:
            self.logger.warning(f"URDF parse failed (env robot): {e}")
            return mapping

        base_dir = Path(urdf_path).parent
        for link in root.findall("link"):
            lname = link.get("name", "")
            L: List[_LinkGeom] = []
            for coll in link.findall("collision"):
                geom = coll.find("geometry")
                if geom is None:
                    continue
                origin = coll.find("origin")
                if origin is not None:
                    xyz = origin.get("xyz", "0 0 0").split()
                    rpy = origin.get("rpy", "0 0 0").split()
                    t_local = np.array([float(x) for x in xyz], dtype=np.float64)
                    R_local = rpy_to_rotation_matrix(*(float(a) for a in rpy))
                else:
                    t_local = np.zeros(3, dtype=np.float64)
                    R_local = np.eye(3)

                mesh = geom.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename", "")
                    if not filename:
                        continue
                    mesh_path = (base_dir / filename).resolve()
                    if not mesh_path.exists():
                        self.logger.debug(f"[env] mesh not found: {mesh_path}")
                        continue
                    scale_attr = mesh.get("scale", None)
                    scale_vec = np.ones(3, dtype=np.float64)
                    if scale_attr:
                        try:
                            scale_vec = np.array([float(v) for v in scale_attr.split()], dtype=np.float64)
                        except Exception:
                            pass
                    try:
                        tm = trimesh.load(mesh_path, force="mesh", process=False)
                        V = np.asarray(tm.vertices, dtype=np.float64)
                        V = (V * scale_vec) * self._shrink
                        F = np.asarray(tm.faces, dtype=np.int32)
                        bvh = fcl.BVHModel()
                        bvh.beginModel(V.shape[0], F.shape[0])
                        bvh.addSubModel(V, F)
                        bvh.endModel()
                        L.append(_LinkGeom(R_local, t_local, bvh))
                    except Exception as e:
                        self.logger.debug(f"[env] failed to load mesh {mesh_path}: {e}")
                    continue

                box = geom.find("box")
                if box is not None:
                    size_attr = box.get("size", None)
                    if not size_attr:
                        continue
                    sx, sy, sz = [float(v) for v in size_attr.split()]
                    bx = fcl.Box(sx*self._shrink, sy*self._shrink, sz*self._shrink)
                    L.append(_LinkGeom(R_local, t_local, bx))
                    continue

                cylinder = geom.find("cylinder")
                if cylinder is not None:
                    radius = float(cylinder.get("radius", "0.01")) * self._shrink
                    length = float(cylinder.get("length", "0.1")) * self._shrink
                    cy = fcl.Cylinder(radius, length)
                    L.append(_LinkGeom(R_local, t_local, cy))
                    continue

                sphere = geom.find("sphere")
                if sphere is not None:
                    radius = float(sphere.get("radius", "0.01")) * self._shrink
                    sp = fcl.Sphere(radius)
                    L.append(_LinkGeom(R_local, t_local, sp))
                    continue

            if L:
                mapping[lname] = L
        return mapping

    def update_env_robot_from_pin(
        self,
        name: str,
        model: "pin.Model",
        data: "pin.Data",
        q: np.ndarray,
        base_position: Tuple[float, float, float],
        base_orientation_rpy: Tuple[float, float, float],
    ) -> None:
        """Update transforms of an already-registered env robot from its Pinocchio state."""
        if not (self.available and HAS_PIN):
            return
        if name not in self.env_robot_geoms:
            # Not registered; nothing to do.
            return

        local_data = model.createData()
        pin.forwardKinematics(model, local_data, q)
        pin.updateFramePlacements(model, local_data)

        # This checker’s BASE pose in WORLD
        Rb_T = self._base_R.T
        tb = self._base_t
        # External robot base (WORLD)
        R_ext = rpy_to_rotation_matrix(*base_orientation_rpy)
        t_ext = np.array(base_position, dtype=np.float64)

        linkmap = self.env_robot_geoms[name]
        for lname, geoms in linkmap.items():
            fid = model.getFrameId(lname)
            if fid != model.nframes:
                M_link = local_data.oMf[fid]
            else:
                try:
                    jid = model.getJointId(lname)
                    M_link = local_data.oMi[jid]
                except Exception:
                    continue
            R_link_ext = M_link.rotation
            t_link_ext = M_link.translation

            # Link frame in WORLD coordinates
            R_world = R_ext @ R_link_ext
            t_world = t_ext + R_ext @ t_link_ext

            for g in geoms:
                R_geom_world = R_world @ g.local_R
                t_geom_world = t_world + R_world @ g.local_t

                R_rel = Rb_T @ R_geom_world
                t_rel = Rb_T @ (t_geom_world - tb)

                obj = g.collision_object()
                obj.setTransform(fcl.Transform(R_rel, t_rel))

    # ---------- check collision given a Pinocchio state ----------
    def in_collision_from_pin(self, model: "pin.Model", data: "pin.Data", q: np.ndarray) -> bool:
        if not (self.available and HAS_PIN):
            return False

        # Use a fresh Data to avoid shared-state hazards when called from multiple threads
        # (Pinocchio Data is not thread-safe across concurrent FK/Jacobian ops)
        local_data = model.createData()
        pin.forwardKinematics(model, local_data, q)
        pin.updateFramePlacements(model, local_data)

        # This checker's BASE pose in WORLD
        Rb_T = self._base_R.T
        tb = self._base_t


        # update robot link geoms with (link * local) transforms in BASE frame
        for lname, geoms in self.robot_geoms.items():
            # try frame first (most URDF importers create frames for links)
            fid = model.getFrameId(lname)
            if fid != model.nframes:  # valid frame id check
                M_link = local_data.oMf[fid]
            else:
                # fallback: try joint id (the body's parent joint)
                try:
                    jid = model.getJointId(lname)
                    M_link = local_data.oMi[jid]
                except Exception:
                    # if we cannot resolve this link, skip its geoms
                    continue

            R_link_w = M_link.rotation
            t_link_w = M_link.translation

            # For fixed-base Pinocchio models, frame placements are already in BASE
            # (no additional base transform needed for this robot's own links)
            R_link_b = R_link_w
            t_link_b = t_link_w


            for g in geoms:
                R = R_link_b @ g.local_R
                t = t_link_b + R_link_b @ g.local_t
                obj = g.collision_object()
                obj.setTransform(fcl.Transform(R, t))

        # self collisions
        lnames = list(self.robot_geoms.keys())
        for i in range(len(lnames)):
            for j in range(i+1, len(lnames)):
                a, b = lnames[i], lnames[j]
                if tuple(sorted((a, b))) in self.allowed_link_pairs:
                    continue
                for oa in self.robot_geoms[a]:
                    oa_obj = oa.collision_object()
                    for ob in self.robot_geoms[b]:
                        ob_obj = ob.collision_object()
                        req = fcl.CollisionRequest()
                        res = fcl.CollisionResult()
                        if fcl.collide(oa_obj, ob_obj, req, res) > 0:
                            return True

        # robot vs world
        for lname, geoms in self.robot_geoms.items():
            for oname, env in self.env_objs.items():
                if (lname, oname) in self.allowed_world_pairs:
                    continue
                # Skip base link collisions with ground plane (base should sit on ground)
                if ("base" in lname.lower() and "plane" in oname.lower()):
                    continue
                for oa in geoms:
                    oa_obj = oa.collision_object()
                    req = fcl.CollisionRequest()
                    res = fcl.CollisionResult()
                    if fcl.collide(oa_obj, env, req, res) > 0:
                        return True

        # robot vs external robots (treated as environment obstacles)
        if self.env_robot_geoms:
            for lname, geoms in self.robot_geoms.items():
                for env_robot, linkmap in self.env_robot_geoms.items():
                    for elname, env_geoms in linkmap.items():
                        for oa in geoms:
                            oa_obj = oa.collision_object()
                            for ob in env_geoms:
                                ob_obj = ob.collision_object()
                                req = fcl.CollisionRequest()
                                res = fcl.CollisionResult()
                                if fcl.collide(oa_obj, ob_obj, req, res) > 0:
                                    # Log tool collisions specifically
                                    if "tool" in lname.lower() or "tool" in elname.lower():
                                        self.logger.warning(f"TOOL COLLISION DETECTED: {lname} <-> {env_robot}/{elname}")
                                        return True
                                    else:
                                        self.logger.debug(f"Non-tool collision: {lname} <-> {env_robot}/{elname}")
                                        return True

        # ground plane quick check (optional) - t_link is already in BASE for fixed-base models
        if self.ground_plane_z != 0.0:
            for lname in self.robot_geoms.keys():
                fid = model.getFrameId(lname)
                if fid == len(model.frames):
                    continue
                # For fixed-base models, link placements are already in BASE frame
                t_link_b = local_data.oMf[fid].translation
                if float(t_link_b[2]) < self.ground_plane_z - 1e-3:  # small tolerance
                    return True

        return False

    def check_ground_collision(self, poses: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]) -> bool:
        """Check if any link is below ground plane."""
        for link_name, (pos, quat) in poses.items():
            if pos[2] < self.ground_plane_z:
                return True
        return False
