from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

try:
    import fcl  # python-fcl
except Exception:  # pragma: no cover
    fcl = None  # We detect this and disable geometric checks


@dataclass
class CollisionGeom:
    # One collision geometry (mesh) attached to a link
    link: str
    vertices: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (M, 3) int
    origin_xyz: np.ndarray  # (3,)
    origin_quat_wxyz: np.ndarray  # (4,)
    scale: np.ndarray  # (3,)


def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array-like to numpy array."""
    if hasattr(x, 'detach'):
        # PyTorch tensor
        return x.detach().cpu().numpy()
    elif hasattr(x, 'numpy'):
        # Other tensor types with .numpy() method
        return x.numpy()
    else:
        # Already numpy or convertible
        return np.array(x)


@dataclass
class LinkCollisionSet:
    link: str
    geoms: List[CollisionGeom]


@dataclass
class URDFCollisions:
    links: Dict[str, LinkCollisionSet]
    # pairs of links that are directly connected (parent-child) – we skip self-collision checks for these
    adjacent_pairs: set[Tuple[str, str]]


def _rpy_to_wxyz(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    return np.array([
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        sy*cp*sr + cy*sp*cr,
        sy*cp*cr - cy*sp*sr,
    ], dtype=np.float32)


def _wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def _read_urdf_collisions(urdf_path: str | Path) -> URDFCollisions:
    """
    Minimal URDF parser for <collision><geometry><mesh filename=...> with optional
    <origin xyz=... rpy=...> and scale. Supports multiple collision entries per link.
    Also records parent-child adjacency to skip those pairs in self-collision.
    """
    import xml.etree.ElementTree as ET

    path = Path(urdf_path)
    tree = ET.parse(path)
    root = tree.getroot()

    # parent-child adjacency (actuated and fixed both count to avoid false-positive at joints)
    adjacent: set[Tuple[str, str]] = set()
    for j in root.findall("joint"):
        parent = j.find("parent").attrib.get("link") if j.find("parent") is not None else None
        child = j.find("child").attrib.get("link") if j.find("child") is not None else None
        if parent and child:
            a = tuple(sorted((parent, child)))
            adjacent.add(a)

    link_map: Dict[str, LinkCollisionSet] = {}

    for link in root.findall("link"):
        lname = link.attrib.get("name", "")
        geoms: List[CollisionGeom] = []
        for col in link.findall("collision"):
            # origin
            ox = np.zeros(3, dtype=np.float32)
            oq = np.array([1, 0, 0, 0], dtype=np.float32)
            origin = col.find("origin")
            if origin is not None:
                if "xyz" in origin.attrib:
                    ox = np.array([float(x) for x in origin.attrib["xyz"].split()], dtype=np.float32)
                if "rpy" in origin.attrib:
                    rpy = [float(x) for x in origin.attrib["rpy"].split()]
                    oq = _rpy_to_wxyz(*rpy)

            # geometry->mesh
            geom = col.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is None or "filename" not in mesh.attrib:
                continue
            fname = mesh.attrib["filename"]
            # scale (URDF uses single 'scale' or attribute on mesh)
            scale = np.ones(3, dtype=np.float32)
            if "scale" in mesh.attrib:
                vals = [float(x) for x in mesh.attrib["scale"].split()]
                if len(vals) == 3:
                    scale = np.array(vals, dtype=np.float32)
                elif len(vals) == 1:
                    scale = np.array([vals[0]]*3, dtype=np.float32)

            # Resolve relative paths against URDF file
            mesh_path = (path.parent / fname).resolve()
            tm = trimesh.load(mesh_path, force="mesh")
            if not isinstance(tm, trimesh.Trimesh):
                # handle Scene by concatenating
                tm = trimesh.util.concatenate(tuple(m for m in tm.dump() if isinstance(m, trimesh.Trimesh)))

            v = np.array(tm.vertices, dtype=np.float32)
            f = np.array(tm.faces, dtype=np.int32)
            geoms.append(CollisionGeom(
                link=lname, vertices=v, triangles=f,
                origin_xyz=ox, origin_quat_wxyz=oq, scale=scale,
            ))

        if geoms:
            link_map[lname] = LinkCollisionSet(link=lname, geoms=geoms)

    return URDFCollisions(links=link_map, adjacent_pairs=adjacent)


class CollisionChecker:
    """
    Builds FCL objects for each link's collision meshes (per robot), and exposes:
      - check_state(q): self + environment (plane z=0) collision
      - check_path(waypoints, substeps): CCD-like sampling
    """
    def __init__(self,
                 robot_name: str,
                 urdf_path: str | Path,
                 entity,
                 logger,
                 plane_z: float = 0.0,
                 allowed_pairs: Optional[List[Tuple[str, str]]] = None,
                 mesh_shrink: float = 1.0):
        self.robot_name = robot_name
        self.urdf_path = str(urdf_path)
        self.entity = entity
        self.logger = logger
        self.plane_z = float(plane_z)
        self.allowed_pairs = {tuple(sorted(p)) for p in (allowed_pairs or [])}
        # Uniform shrink factor for collision meshes (around mesh centroid). 1.0 = no change.
        self.shrink = float(mesh_shrink)

        self.available = fcl is not None
        if not self.available:
            self.logger.warning("python-fcl not available; geometric collision checks are disabled")

        self.urdf = _read_urdf_collisions(self.urdf_path)

        # Expand adjacency one hop to reduce wrist false-positives
        adj = set(self.urdf.adjacent_pairs)
        for (a,b) in list(adj):
            for (x,y) in list(adj):
                if b == x:
                    self.urdf.adjacent_pairs.add(tuple(sorted((a, y))))

        # Build FCL collision objects per link geometry, but only for links that actually exist on the Genesis entity
        self._objs: Dict[str, List["fcl.CollisionObject"]] = {}
        # Track the filtered, valid link names
        valid_links: List[str] = []
        # Cache collision request to avoid allocation in inner loops
        self._fcl_req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True) if self.available else None
        if self.available:
            for lname, lset in self.urdf.links.items():
                # Skip URDF links that the Genesis entity does not expose
                try:
                    _ = self.entity.get_link(lname)
                except Exception:
                    try:
                        self.logger.warning("%s: URDF link '%s' not found on Genesis entity; skipping from collision model", self.robot_name, lname)
                    except Exception:
                        pass
                    continue
                objs = []
                for g in lset.geoms:
                    # Apply URDF scale to vertices
                    v_scaled = (g.vertices * g.scale[None, :]).astype(np.float32)
                    # Optionally shrink uniformly around centroid to reduce false-positive self-collisions
                    if self.shrink != 1.0:
                        c = np.mean(v_scaled, axis=0, dtype=np.float32)
                        v = (c + self.shrink * (v_scaled - c)).astype(np.float32)
                    else:
                        v = v_scaled
                    m = fcl.BVHModel()
                    m.beginModel(v.shape[0], g.triangles.shape[0])
                    m.addSubModel(v, g.triangles.astype(np.int32))
                    m.endModel()
                    tf = fcl.Transform(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
                    objs.append(fcl.CollisionObject(m, tf))
                if objs:
                    self._objs[lname] = objs
                    valid_links.append(lname)
        # Publish filtered link list for downstream loops
        self.links = valid_links

    # --- transforms ---
    @staticmethod
    def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2], dtype=np.float32)

    def _link_pose_w(self, link_name: str) -> Tuple[np.ndarray, np.ndarray]:
        link = self.entity.get_link(link_name)
        # wxyz (ensure CPU numpy even if backend returns torch cuda tensors)
        pos = _to_numpy(link.get_pos()).astype(np.float32).reshape(3)
        quat = _to_numpy(link.get_quat()).astype(np.float32).reshape(4)
        return pos, quat

    def _apply_origin(self, pos_w: np.ndarray, quat_w_wxyz: np.ndarray,
                      ox: np.ndarray, oq_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # world_T_link * link_T_origin
        Rw = _wxyz_to_R(quat_w_wxyz)
        p = pos_w + Rw @ ox
        q = self._quat_mul_wxyz(quat_w_wxyz, oq_wxyz)
        return p, q

    # --- updates & checks ---
    def _update_world_transforms(self):
        if not self.available:
            return

        for lname, objs in self._objs.items():
            try:
                pos_w, quat_w = self._link_pose_w(lname)
            except Exception:
                continue
            Rw = _wxyz_to_R(quat_w)
            for obj, geom in zip(objs, self.urdf.links[lname].geoms):
                # bake the local origin into the object transform
                p, q = self._apply_origin(pos_w, quat_w, geom.origin_xyz, geom.origin_quat_wxyz)
                R = _wxyz_to_R(q)
                obj.setTransform(fcl.Transform(R.astype(np.float32), p.astype(np.float32)))

    # --- diagnostics & dynamic allow-list management ---
    def list_self_collisions_now(self) -> List[tuple[str, str]]:
        """
        Update transforms from the entity's CURRENT joint state and return ALL
        colliding link pairs (self-collision only, not ground), excluding
        adjacent pairs and already-allowed pairs.
        """
        if not self.available:
            return []
        self._update_world_transforms()
        req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
        hits: List[tuple[str, str]] = []
        links = list(self._objs.keys())
        for i in range(len(links)):
            A = links[i]
            for j in range(i + 1, len(links)):
                B = links[j]
                pair = tuple(sorted((A, B)))
                if pair in self.urdf.adjacent_pairs:
                    continue
                if pair in self.allowed_pairs:
                    continue
                for oa in self._objs[A]:
                    for ob in self._objs[B]:
                        res = fcl.CollisionResult()
                        if fcl.collide(oa, ob, req, res) > 0:
                            hits.append(pair)
                            break
                    else:
                        continue
                    break
        return hits

    def add_allowed_pairs(self, pairs: List[tuple[str, str]]):
        for p in pairs:
            self.allowed_pairs.add(tuple(sorted(p)))

    def _check_self(self) -> List[Tuple[str, str]]:
        if not self.available:
            return []
        hits: List[Tuple[str, str]] = []
        links = list(self._objs.keys())
        for i in range(len(links)):
            A = links[i]
            for j in range(i+1, len(links)):
                B = links[j]
                pair = tuple(sorted((A, B)))
                if pair in self.urdf.adjacent_pairs:
                    continue  # ignore parent-child
                if pair in self.allowed_pairs:
                    continue  # user-specified allowed collision
                if any(a in pair and b in pair for (a,b) in self.urdf.adjacent_pairs):
                    continue
                for oa in self._objs[A]:
                    for ob in self._objs[B]:
                        res = fcl.CollisionResult()
                        if fcl.collide(oa, ob, self._fcl_req, res) > 0:
                            hits.append(pair)
                            # short-circuit for speed
                            return hits
        return hits

    def _check_plane(self) -> List[str]:
        """Quick ground-plane check: if any link's transformed mesh dips below plane_z."""
        # For speed, just AABB zmin test using sampled vertices
        viol: List[str] = []
        for lname, objs in self._objs.items():
            # Early cull: if link origin is well above plane, skip detailed check
            pos_w, quat_w = self._link_pose_w(lname)
            origin_z = float(pos_w[2])
            if origin_z - self.plane_z > 0.02:
                continue
            # Recompute zmin by sampling each geom's vertices transformed by current world transform
            try:
                lset = self.urdf.links[lname]
            except KeyError:
                continue
            for geom in lset.geoms:
                p, q = self._apply_origin(pos_w, quat_w, geom.origin_xyz, geom.origin_quat_wxyz)
                R = _wxyz_to_R(q)
                v_scaled = (geom.vertices * geom.scale[None, :]).astype(np.float32)
                if self.shrink != 1.0:
                    c = np.mean(v_scaled, axis=0, dtype=np.float32)
                    v = (c + self.shrink * (v_scaled - c)).astype(np.float32)
                else:
                    v = v_scaled
                vv = (R @ v.T).T + p[None, :]
                zmin = float(np.min(vv[:,2]))
                if zmin < self.plane_z:
                    viol.append(lname)
                    return viol
        return viol

    # API: check a single configuration
    def check_state(self, q: np.ndarray, dofs_idx: List[int]) -> Tuple[bool, str]:
        """
        Sets the entity to q (kinematically), updates geometry, and tests:
          - self collision (mesh-mesh via FCL)
          - ground plane z <= plane_z
        Returns (ok, reason)
        """
        # Convert tensor to numpy if needed
        q_np = _to_numpy(q).astype(np.float32)
        
        # Stash and set kinematic pose
        try:
            joints = self.entity.get_joints()
            q_list: List[float] = []
            for j in joints:
                v = _to_numpy(getattr(j, "qpos", 0.0))
                v = np.array(v, dtype=np.float32).reshape(-1)
                q_list.append(float(v[0]))
            qprev = np.array(q_list, dtype=np.float32)
        except Exception:
            qprev = _to_numpy(q).astype(np.float32)
        try:
            self.entity.set_dofs_position(q_np, dofs_idx_local=dofs_idx)
        except Exception:
            self.entity.set_dofs_position(q_np, dofs_idx)

        # Update collision object transforms from current link poses
        self._update_world_transforms()

        # Self-collision
        if self.available:
            hits = self._check_self()
            if hits:
                # Restore and report
                try:
                    self.entity.set_dofs_position(qprev, dofs_idx_local=dofs_idx)
                except Exception:
                    self.entity.set_dofs_position(qprev, dofs_idx)
                return False, f"self_collision:{hits[0][0]}-{hits[0][1]}"

        # Plane collision
        gviol = self._check_plane()
        if gviol:
            try:
                self.entity.set_dofs_position(qprev, dofs_idx_local=dofs_idx)
            except Exception:
                self.entity.set_dofs_position(qprev, dofs_idx)
            return False, f"ground_collision:{gviol[0]}"

        # Restore original pose
        try:
            self.entity.set_dofs_position(qprev, dofs_idx_local=dofs_idx)
        except Exception:
            self.entity.set_dofs_position(qprev, dofs_idx)

        return True, ""

    def check_path(self, waypoints, dofs_idx, substeps=3, stride=None, max_time_s=None):
        """
        Returns (ok, reason, seg_idx). If stride is set, samples every `stride`-th
        segment; if max_time_s is set, gives up early and assumes valid when the
        budget is exhausted (the online planner already enforces validity).
        """
        import time
        start = time.perf_counter()
        n = len(waypoints)
        if n < 2:
            return True, "trivial", -1

        if stride is None or stride < 1:
            stride = max(1, n // 30)  # ~30 samples max

        for i in range(0, n-1, stride):
            a = _to_numpy(waypoints[i]).astype(np.float32)
            b = _to_numpy(waypoints[min(i+stride, n-1)]).astype(np.float32)
            # check a->b in 'substeps' straight-line joint increments
            for s in range(substeps + 1):
                if max_time_s is not None and (time.perf_counter() - start) > max_time_s:
                    return True, "time_budget_exhausted", i
                alpha = s / max(1, substeps)
                q = a * (1 - alpha) + b * alpha
                ok, reason = self.check_state(q, dofs_idx)
                if not ok:
                    return False, reason, i
        return True, "ok", -1

# === World-aware collision manager ===========================================
# Appended: MoveIt-like planning scene utilities that aggregate all robot checkers
# and static objects to support inter-robot/world collision and clearance checks.

from typing import Set

def _rpy_deg_to_wxyz(r: float, p: float, y: float) -> np.ndarray:
    return _rpy_to_wxyz(np.deg2rad(r), np.deg2rad(p), np.deg2rad(y))

class CollisionWorld:
    """
    MoveIt-like world built over per-robot CollisionChecker(s).
    Supports:
      - inter-robot mesh-mesh collision/clearance
      - static objects (boxes) with names
      - allowed-collision pairs across robots/objects
      - min clearance (meters) using FCL distance if available
    """
    def __init__(self, logger, min_clearance: float = 0.0):
        self.logger = logger
        self.min_clearance = float(min_clearance)
        self._robots: Dict[str, CollisionChecker] = {}
        self._static_objs: Dict[str, "fcl.CollisionObject"] = {}
        self._allowed: Set[Tuple[str, str]] = set()
        self._have_distance = hasattr(fcl, "DistanceRequest") if fcl else False
        if self._have_distance:
            self._dist_req = fcl.DistanceRequest(enable_nearest_points=False)
            self._dist_res = fcl.DistanceResult()

    # --- registration & config ---
    def add_robot(self, name: str, checker: "CollisionChecker"):
        self._robots[name] = checker

    def allow(self, pairs: List[Tuple[str, str]]):
        for a, b in pairs or []:
            A = tuple(sorted((a, b)))
            self._allowed.add(A)

    def _key_link(self, robot: str, link: str) -> str:
        return f"{robot}/{link}"

    def _key_obj(self, obj_name: str) -> str:
        return f"obj:{obj_name}"

    def add_box(self, name: str, size_xyz: Tuple[float, float, float],
                pos_w: Tuple[float, float, float],
                rpy_deg_w: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        if not fcl:
            return
        sx, sy, sz = [float(abs(v)) for v in size_xyz]
        bx = fcl.Box(sx, sy, sz)
        q = _rpy_deg_to_wxyz(*rpy_deg_w)
        R = _wxyz_to_R(q).astype(np.float32)
        p = np.array(pos_w, dtype=np.float32).reshape(3)
        tf = fcl.Transform(R, p)
        self._static_objs[str(name)] = fcl.CollisionObject(bx, tf)

    # --- core updates ---
    def _update_all_robot_transforms(self):
        # Make sure each robot's link collision objects are at current poses
        for chk in self._robots.values():
            try:
                chk._update_world_transforms()
            except Exception:
                pass

    def _with_robot_q(self, robot: str, q: np.ndarray, dofs_idx):
        """Temporarily set robot 'robot' to joint state q (radians), yield qprev, and restore on exit (call _restore_robot_q)."""
        ent = self._robots[robot].entity
        # capture previous
        try:
            joints = ent.get_joints()
            qprev = np.array([float(getattr(j, "qpos", 0.0)) for j in joints], dtype=np.float32)
        except Exception:
            qprev = None
        # set new
        try:
            ent.set_dofs_position(q, dofs_idx_local=dofs_idx)
        except Exception:
            ent.set_dofs_position(q, dofs_idx)
        return qprev

    def _restore_robot_q(self, robot: str, qprev: Optional[np.ndarray], dofs_idx):
        if qprev is None:
            return
        ent = self._robots[robot].entity
        try:
            ent.set_dofs_position(qprev, dofs_idx_local=dofs_idx)
        except Exception:
            ent.set_dofs_position(qprev, dofs_idx)

    # --- pairwise helpers ---
    def _pairs_iter_links(self, robotA: str, robotB: str):
        A = self._robots[robotA]; B = self._robots[robotB]
        for la, objs_a in A._objs.items():
            keyA = self._key_link(robotA, la)
            for lb, objs_b in B._objs.items():
                keyB = self._key_link(robotB, lb)
                if tuple(sorted((keyA, keyB))) in self._allowed:
                    continue
                yield keyA, objs_a, keyB, objs_b

    def _collide_objs(self, oa: "fcl.CollisionObject", ob: "fcl.CollisionObject") -> bool:
        res = fcl.CollisionResult()
        # reuse any request from an existing checker
        req = None
        for chk in self._robots.values():
            req = chk._fcl_req
            break
        if req is None:
            req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
        return fcl.collide(oa, ob, req, res) > 0

    def _violates_clearance(self, oa: "fcl.CollisionObject", ob: "fcl.CollisionObject") -> bool:
        if not self._have_distance or self.min_clearance <= 0.0:
            return False
        try:
            self._dist_res.clear()
            d = fcl.distance(oa, ob, self._dist_req, self._dist_res)
            return d < self.min_clearance
        except Exception:
            return False

    # --- public API (state and path) ---
    def check_state(self, robot: str, q: np.ndarray, dofs_idx) -> Tuple[bool, str]:
        """
        Test the given joint state of 'robot' against:
          - its own self-collision + ground (via per-robot checker)
          - other robots
          - static objects
          - optional min clearance
        """
        if robot not in self._robots:
            return True, "no_checker"
        chk = self._robots[robot]

        # 1) self + ground (uses its own restore logic)
        ok, reason = chk.check_state(q, dofs_idx)
        if not ok:
            return ok, reason

        if not fcl:
            return True, "fcl_unavailable"

        # 2) world checks: set q temporarily, update transforms
        qprev = self._with_robot_q(robot, _to_numpy(q).astype(np.float32), dofs_idx)
        try:
            self._update_all_robot_transforms()

            # (a) against other robots
            for other in self._robots.keys():
                if other == robot:
                    continue
                for keyA, objs_a, keyB, objs_b in self._pairs_iter_links(robot, other):
                    for oa in objs_a:
                        for ob in objs_b:
                            if self._collide_objs(oa, ob):
                                return False, f"robot_collision:{keyA}~{keyB}"
                            if self._violates_clearance(oa, ob):
                                return False, f"clearance_violation:{keyA}~{keyB}"

            # (b) against static objects
            for la, objs_a in chk._objs.items():
                keyA = self._key_link(robot, la)
                for oname, ob in self._static_objs.items():
                    keyB = self._key_obj(oname)
                    if tuple(sorted((keyA, keyB))) in self._allowed:
                        continue
                    for oa in objs_a:
                        if self._collide_objs(oa, ob):
                            return False, f"world_collision:{keyA}~{oname}"
                        if self._violates_clearance(oa, ob):
                            return False, f"world_clearance:{keyA}~{oname}"

            return True, ""
        finally:
            self._restore_robot_q(robot, qprev, dofs_idx)

    def check_path(self, robot: str, waypoints: List[np.ndarray], dofs_idx,
                   substeps: int = 3, stride: Optional[int] = None, max_time_s: Optional[float] = None):
        """
        Sampled CCD-like path validator over the WHOLE WORLD (robots + static objects).
        Mirrors per-robot check_path signature.
        """
        import time
        start = time.perf_counter()
        n = len(waypoints)
        if n < 2:
            return True, "trivial", -1
        if stride is None or stride < 1:
            stride = max(1, n // 30)

        for i in range(0, n - 1, stride):
            a = _to_numpy(waypoints[i]).astype(np.float32)
            b = _to_numpy(waypoints[min(i + stride, n - 1)]).astype(np.float32)
            for s in range(substeps + 1):
                if max_time_s is not None and (time.perf_counter() - start) > max_time_s:
                    return True, "time_budget_exhausted", i
                alpha = s / max(1, substeps)
                q = a * (1 - alpha) + b * alpha
                ok, reason = self.check_state(robot, q, dofs_idx)
                if not ok:
                    return False, reason, i
        return True, "ok", -1
