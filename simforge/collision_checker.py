# simforge/collision_checker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

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


def _rpy_to_R(roll, pitch, yaw) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [ -sp ,         cp*sr   ,         cp*cr   ]], dtype=np.float64)


@dataclass
class _LinkGeom:
    # per-collision geometry for a link: local transform relative to link frame
    local_R: np.ndarray
    local_t: np.ndarray
    obj: fcl.CollisionObject


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
    ) -> None:
        self.logger = logger
        self.urdf_path = Path(urdf_path)
        self.available = HAS_FCL and HAS_TRIMESH
        if not self.available:
            self.logger.warning("FCL/trimesh not available - collision checking disabled")
            return

        # Robot base pose in WORLD; all world objects will be re-expressed in this BASE frame
        self._base_t = np.array(base_position, dtype=np.float64)
        self._base_R = _rpy_to_R(*base_orientation_rpy)

        self._shrink = float(collision_mesh_shrink)
        self.robot_geoms: Dict[str, List[_LinkGeom]] = {}
        self.env_objs: Dict[str, fcl.CollisionObject] = {}
        self.allowed_link_pairs: Set[Tuple[str, str]] = set()
        self.allowed_world_pairs: Set[Tuple[str, str]] = set()
        self.ground_plane_z = float(ground_plane_z)

        if allowed_link_pairs:
            for a, b in allowed_link_pairs:
                self.allowed_link_pairs.add(tuple(sorted((a, b))))
        if world_allowed_pairs:
            for a, b in world_allowed_pairs:
                # keep order (robotLink, obj:name)
                self.allowed_world_pairs.add((a, b))

        self._load_urdf()
        self._load_world(world_boxes or [])

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
                    R_local = _rpy_to_R(*(float(a) for a in rpy))
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
                        # apply scale (per-axis) and shrink (uniform)
                        V = (V * scale_vec) * self._shrink
                        F = np.asarray(tm.faces, dtype=np.int32)
                        bvh = fcl.BVHModel()
                        bvh.beginModel(V.shape[0], F.shape[0])
                        bvh.addSubModel(V, F)
                        bvh.endModel()
                        co = fcl.CollisionObject(bvh, fcl.Transform(np.eye(3), np.zeros(3)))
                        L.append(_LinkGeom(R_local, t_local, co))
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
                    co = fcl.CollisionObject(bx, fcl.Transform(np.eye(3), np.zeros(3)))
                    L.append(_LinkGeom(R_local, t_local, co))
                    continue

            if L:
                self.robot_geoms[lname] = L

    def _load_world(self, boxes):
        # Express world boxes in the robot BASE frame used by Pinocchio/IK/OMPL
        Rb_T = self._base_R.T
        tb = self._base_t
        for name, size, pos, rpy in boxes:
            sx, sy, sz = [float(x) for x in size]
            bx = fcl.Box(sx, sy, sz)
            Rw = _rpy_to_R(*rpy)
            tw = np.array(pos, dtype=np.float64)
            R_rel = Rb_T @ Rw
            t_rel = Rb_T @ (tw - tb)
            self.env_objs[f"obj:{name}"] = fcl.CollisionObject(bx, fcl.Transform(R_rel, t_rel))

    # ---------- check collision given a Pinocchio state ----------
    def in_collision_from_pin(self, model: "pin.Model", data: "pin.Data", q: np.ndarray) -> bool:
        if not (self.available and HAS_PIN):
            return False

        # Use a fresh Data to avoid shared-state hazards when called from multiple threads
        # (Pinocchio Data is not thread-safe across concurrent FK/Jacobian ops)
        local_data = model.createData()
        pin.forwardKinematics(model, local_data, q)
        pin.updateFramePlacements(model, local_data)

        # update robot link geoms with (link * local) transforms
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

            R_link = M_link.rotation
            t_link = M_link.translation
            for g in geoms:
                # world transform = link * local
                R = R_link @ g.local_R
                t = t_link + R_link @ g.local_t
                g.obj.setTransform(fcl.Transform(R, t))

        # self collisions
        lnames = list(self.robot_geoms.keys())
        for i in range(len(lnames)):
            for j in range(i+1, len(lnames)):
                a, b = lnames[i], lnames[j]
                if tuple(sorted((a, b))) in self.allowed_link_pairs:
                    continue
                for oa in self.robot_geoms[a]:
                    for ob in self.robot_geoms[b]:
                        req = fcl.CollisionRequest()
                        res = fcl.CollisionResult()
                        if fcl.collide(oa.obj, ob.obj, req, res) > 0:
                            return True

        # robot vs world
        for lname, geoms in self.robot_geoms.items():
            for oname, env in self.env_objs.items():
                if (lname, oname) in self.allowed_world_pairs:
                    continue
                for oa in geoms:
                    req = fcl.CollisionRequest()
                    res = fcl.CollisionResult()
                    if fcl.collide(oa.obj, env, req, res) > 0:
                        return True

        # ground plane quick check (optional)
        if self.ground_plane_z != 0.0:
            for lname in self.robot_geoms.keys():
                fid = model.getFrameId(lname)
                if fid == len(model.frames):
                    continue
                if float(local_data.oMf[fid].translation[2]) < self.ground_plane_z:
                    return True

        return False

    def check_ground_collision(self, poses: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]) -> bool:
        """Check if any link is below ground plane."""
        for link_name, (pos, quat) in poses.items():
            if pos[2] < self.ground_plane_z:
                return True
        return False
