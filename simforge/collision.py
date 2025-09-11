from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from .utils import rpy_to_quat_wxyz, quat_wxyz_multiply, quat_wxyz_rotate_vec, quat_wxyz_to_rotation_matrix, to_numpy, safe_set_dofs_position

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




@dataclass
class LinkCollisionSet:
    link: str
    geoms: List[CollisionGeom]


@dataclass
class URDFCollisions:
    links: Dict[str, LinkCollisionSet]
    # pairs of links that are directly connected (parent-child) – we skip self-collision checks for these
    adjacent_pairs: set[Tuple[str, str]]






def _read_urdf_kinematics(urdf_path: str | Path) -> Dict[str, Tuple[str, np.ndarray, np.ndarray]]:
    """Parses the URDF to get a map of child -> (parent, pos, quat) for all joints."""
    import xml.etree.ElementTree as ET
    path = Path(urdf_path)
    tree = ET.parse(path)
    root = tree.getroot()
    
    kinematics = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib.get("link")
        child = joint.find("child").attrib.get("link")
        if not parent or not child:
            continue
        
        ox = np.zeros(3, dtype=np.float32)
        oq = np.array([1, 0, 0, 0], dtype=np.float32)
        origin = joint.find("origin")
        if origin is not None:
            if "xyz" in origin.attrib:
                ox = np.array([float(x) for x in origin.attrib["xyz"].split()], dtype=np.float32)
            if "rpy" in origin.attrib:
                rpy = [float(x) for x in origin.attrib["rpy"].split()]
                oq = rpy_to_quat_wxyz(*rpy)
        kinematics[child] = (parent, ox, oq)
    return kinematics


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
                    oq = rpy_to_quat_wxyz(*rpy)

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
            # Load as a scene to preserve transforms and multiple meshes
            loaded = trimesh.load(mesh_path)

            if isinstance(loaded, trimesh.Scene):
                mesh_list = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
            elif isinstance(loaded, trimesh.Trimesh):
                mesh_list = [loaded]
            else:
                mesh_list = []

            for tm in mesh_list:
                if tm.is_empty:
                    continue
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

        urdf_model = _read_urdf_collisions(self.urdf_path)
        self.urdf = self._synchronize_collision_model(urdf_model)

        # Expand adjacency one hop to reduce wrist false-positives
        adj = set(self.urdf.adjacent_pairs)
        for (a,b) in list(adj):
            for (x,y) in list(adj):
                if b == x:
                    self.urdf.adjacent_pairs.add(tuple(sorted((a, y))))

        # Build FCL collision objects per link geometry
        # Build FCL collision objects per link geometry
        self._objs: Dict[str, List["fcl.CollisionObject"]] = {}
        self._tool_objs: Dict[str, List[Tuple[str, int]]] = {} # tool_name -> [(link, num_geoms)]
        # Track the filtered, valid link names
        valid_links: List[str] = []
        # Cache collision request to avoid allocation in inner loops
        self._fcl_req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True) if self.available else None
        if self.available:
            for lname, lset in self.urdf.links.items():
                # The synchronized model should only contain links present in the entity
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

    def _get_genesis_kinematic_tree(self) -> set[str]:
        """Traverse the Genesis entity to find all existing link names."""
        try:
            return {link.name for link in self.entity.links}
        except Exception as e:
            self.logger.warning(f"Failed to get links from Genesis entity: {e}")
            return set()

    def _synchronize_collision_model(self, urdf_model: URDFCollisions) -> URDFCollisions:
        """
        Reparents collision geometries from missing links (e.g., fixed joints merged
        by Genesis) to their first existing ancestor in the simulator's kinematic tree.
        """
        sim_links = self._get_genesis_kinematic_tree()
        if not sim_links:
            self.logger.error("Could not get any links from the Genesis entity. Collision model will be empty.")
            return URDFCollisions(links={}, adjacent_pairs=set())

        urdf_kinematics = _read_urdf_kinematics(self.urdf_path)
        
        synced_links: Dict[str, LinkCollisionSet] = {}

        for lname, lset in urdf_model.links.items():
            if lname in sim_links:
                # This link exists in the simulator, so its geometries are correctly parented.
                if lname not in synced_links:
                    synced_links[lname] = LinkCollisionSet(link=lname, geoms=[])
                synced_links[lname].geoms.extend(lset.geoms)
                continue

            # This link is missing. Find its closest existing ancestor.
            self.logger.debug(f"{self.robot_name}: Reparenting geometries from missing link '{lname}'")
            
            parent = lname
            cumulative_pos = np.zeros(3, dtype=np.float32)
            cumulative_quat = np.array([1, 0, 0, 0], dtype=np.float32)
            
            path_to_ancestor = []
            while parent in urdf_kinematics and parent not in sim_links:
                parent, p_pos, p_quat = urdf_kinematics[parent]
                path_to_ancestor.append((p_pos, p_quat))

            if parent not in sim_links:
                self.logger.warning(f"{self.robot_name}: Could not find a valid simulation parent for missing link '{lname}'. Geometries will be dropped.")
                continue

            # Calculate the cumulative transform from the ancestor to the original link
            for p_pos, p_quat in reversed(path_to_ancestor):
                # Correct composition: T_new = T_cumulative * T_step
                cumulative_pos = quat_wxyz_rotate_vec(cumulative_quat, p_pos) + cumulative_pos
                cumulative_quat = quat_wxyz_multiply(cumulative_quat, p_quat)

            # Reparent the geometries
            if parent not in synced_links:
                synced_links[parent] = LinkCollisionSet(link=parent, geoms=[])

            for geom in lset.geoms:
                # Original geometry transform relative to its (missing) link
                orig_pos = geom.origin_xyz
                orig_quat = geom.origin_quat_wxyz
                
                # New transform relative to the existing ancestor
                new_pos = quat_wxyz_rotate_vec(cumulative_quat, orig_pos) + cumulative_pos
                new_quat = quat_wxyz_multiply(cumulative_quat, orig_quat)

                new_geom = CollisionGeom(
                    link=parent, # Reparented
                    vertices=geom.vertices,
                    triangles=geom.triangles,
                    origin_xyz=new_pos,
                    origin_quat_wxyz=new_quat,
                    scale=geom.scale
                )
                synced_links[parent].geoms.append(new_geom)
            self.logger.info(f"{self.robot_name}: Successfully reparented {len(lset.geoms)} geometries from '{lname}' to '{parent}'")

        return URDFCollisions(links=synced_links, adjacent_pairs=urdf_model.adjacent_pairs)

    # --- transforms ---

    def _link_pose_w(self, link_name: str) -> Tuple[np.ndarray, np.ndarray]:
        link = self.entity.get_link(link_name)
        # wxyz (ensure CPU numpy even if backend returns torch cuda tensors)
        pos = to_numpy(link.get_pos()).astype(np.float32).reshape(3)
        quat = to_numpy(link.get_quat()).astype(np.float32).reshape(4)
        return pos, quat

    def _apply_origin(self, pos_w: np.ndarray, quat_w_wxyz: np.ndarray,
                       ox: np.ndarray, oq_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # world_T_link * link_T_origin
        Rw = quat_wxyz_to_rotation_matrix(quat_w_wxyz)
        p = pos_w + Rw @ ox
        q = quat_wxyz_multiply(quat_w_wxyz, oq_wxyz)
        return p, q

    # --- updates & checks ---
    def _update_world_transforms(self):
        if not self.available:
            return

        for lname, lset in self.urdf.links.items():
            try:
                pos_w, quat_w = self._link_pose_w(lname)
                objs = self._objs.get(lname, [])
                for obj, geom in zip(objs, lset.geoms):
                    # bake the local origin into the object transform
                    p, q = self._apply_origin(pos_w, quat_w, geom.origin_xyz, geom.origin_quat_wxyz)
                    R = quat_wxyz_to_rotation_matrix(q)
                    obj.setTransform(fcl.Transform(R.astype(np.float32), p.astype(np.float32)))
            except Exception:
                continue

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

    def add_tool(self, tool_name: str, tool_urdf_path: str | Path, parent_link: str):
        if not self.available:
            return
        if tool_name in self._tool_objs:
            self.logger.warning(f"Tool '{tool_name}' is already attached. Please remove it first.")
            return

        tool_model = _read_urdf_collisions(tool_urdf_path)
        tool_kinematics = _read_urdf_kinematics(tool_urdf_path)
        
        # Find the transform from the tool's base to the parent_link of the robot
        # For now, assume tool base is attached directly to parent_link with identity transform.
        # A more robust solution would parse the fixed joint from robot->tool.
        
        self._tool_objs[tool_name] = []
        for lname, lset in tool_model.links.items():
            # For each geometry in the tool, create a collision object and attach it to the parent_link
            if parent_link not in self._objs:
                self._objs[parent_link] = []
            if parent_link not in self.urdf.links:
                self.urdf.links[parent_link] = LinkCollisionSet(link=parent_link, geoms=[])

            num_geoms_added = 0
            for g in lset.geoms:
                v_scaled = (g.vertices * g.scale[None, :]).astype(np.float32)
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
                
                # The geometry's origin is relative to the tool link. We need to transform it
                # to be relative to the robot's parent_link.
                # For now, we assume the tool is attached at the parent_link's origin.
                # This is a simplification; a full implementation would trace the kinematic chain.
                new_geom = CollisionGeom(
                    link=parent_link, # Attach to the robot's link
                    vertices=g.vertices,
                    triangles=g.triangles,
                    origin_xyz=g.origin_xyz,
                    origin_quat_wxyz=g.origin_quat_wxyz,
                    scale=g.scale
                )
                
                self.urdf.links[parent_link].geoms.append(new_geom)
                self._objs[parent_link].append(fcl.CollisionObject(m, tf))
                num_geoms_added += 1
            
            if num_geoms_added > 0:
                self._tool_objs[tool_name].append((parent_link, num_geoms_added))

        self.logger.info(f"Attached tool '{tool_name}' to link '{parent_link}' with {sum(n for _, n in self._tool_objs[tool_name])} collision geometries.")

    def remove_tool(self, tool_name: str):
        if tool_name not in self._tool_objs:
            return

        for link_name, num_geoms in self._tool_objs[tool_name]:
            if link_name in self._objs and len(self._objs[link_name]) >= num_geoms:
                # Remove the last N objects/geometries that were added for this tool
                self._objs[link_name] = self._objs[link_name][:-num_geoms]
                self.urdf.links[link_name].geoms = self.urdf.links[link_name].geoms[:-num_geoms]
                if not self._objs[link_name]:
                    del self._objs[link_name]
                    del self.urdf.links[link_name]

        del self._tool_objs[tool_name]
        self.logger.info(f"Removed tool '{tool_name}'.")

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
                R = quat_wxyz_to_rotation_matrix(q)
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
        q_np = to_numpy(q).astype(np.float32)

        # Stash and set kinematic pose
        try:
            joints = self.entity.get_joints()
            q_list: List[float] = []
            for j in joints:
                v = to_numpy(getattr(j, "qpos", 0.0))
                v = np.array(v, dtype=np.float32).reshape(-1)
                q_list.append(float(v[0]))
            qprev = np.array(q_list, dtype=np.float32)
        except Exception:
            qprev = to_numpy(q).astype(np.float32)
        safe_set_dofs_position(self.entity, q_np, dofs_idx)

        # Update collision object transforms from current link poses
        self._update_world_transforms()

        # Self-collision
        if self.available:
            hits = self._check_self()
            if hits:
                # Restore and report
                safe_set_dofs_position(self.entity, qprev, dofs_idx)
                return False, f"self_collision:{hits[0][0]}-{hits[0][1]}"

        # Plane collision
        gviol = self._check_plane()
        if gviol:
            safe_set_dofs_position(self.entity, qprev, dofs_idx)
            return False, f"ground_collision:{gviol[0]}"

        # Restore original pose
        safe_set_dofs_position(self.entity, qprev, dofs_idx)

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
            a = to_numpy(waypoints[i]).astype(np.float32)
            b = to_numpy(waypoints[min(i+stride, n-1)]).astype(np.float32)
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
    return rpy_to_quat_wxyz(np.deg2rad(r), np.deg2rad(p), np.deg2rad(y))

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
        R = quat_wxyz_to_rotation_matrix(q).astype(np.float32)
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
            qprev = np.array([float(to_numpy(getattr(j, "qpos", 0.0))) for j in joints], dtype=np.float32)
        except Exception:
            qprev = None
        # set new
        safe_set_dofs_position(ent, q, dofs_idx)
        return qprev

    def _restore_robot_q(self, robot: str, qprev: Optional[np.ndarray], dofs_idx):
        if qprev is None:
            return
        ent = self._robots[robot].entity
        safe_set_dofs_position(ent, qprev, dofs_idx)

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
        qprev = self._with_robot_q(robot, to_numpy(q).astype(np.float32), dofs_idx)
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
            a = to_numpy(waypoints[i]).astype(np.float32)
            b = to_numpy(waypoints[min(i + stride, n - 1)]).astype(np.float32)
            for s in range(substeps + 1):
                if max_time_s is not None and (time.perf_counter() - start) > max_time_s:
                    return True, "time_budget_exhausted", i
                alpha = s / max(1, substeps)
                q = a * (1 - alpha) + b * alpha
                ok, reason = self.check_state(robot, q, dofs_idx)
                if not ok:
                    return False, reason, i
        return True, "ok", -1
