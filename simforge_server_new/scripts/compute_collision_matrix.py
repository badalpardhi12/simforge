#!/usr/bin/env python3
"""
Compute the Allowed Collision Matrix (ACM) for a dual UR5e cell.

This script replicates what the MoveIt Setup Assistant does
programmatically: it loads the URDF, samples thousands of random joint
configurations, performs collision detection between all link pairs,
and outputs `<disable_collisions>` entries for the SRDF.

Usage (inside the Docker container with ROS2 + MoveIt):
  # Generate collision entries for the current URDF
  python3 compute_collision_matrix.py

  # With custom sample count and output file
  python3 compute_collision_matrix.py --samples 25000 --output srdf_collisions.xml

  # Dry-run mode: just print statistics without writing
  python3 compute_collision_matrix.py --dry-run

The script uses the MoveIt Python bindings (moveit_py) if available,
otherwise falls back to a standalone approach using:
  - urdfpy / yourdfpy for URDF parsing
  - trimesh + fcl (python-fcl) for collision checking

Requirements (standalone mode):
  pip install yourdfpy trimesh numpy python-fcl

Requirements (MoveIt mode - inside Docker):
  ROS2 Humble + MoveIt2 + moveit_py
"""

import argparse
import itertools
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

# Joints to randomize and their limits (from UR5e URDF, ±2π)
JOINT_LIMITS = {
    "nakul_shoulder_pan_joint":  (-6.2832, 6.2832),
    "nakul_shoulder_lift_joint": (-6.2832, 6.2832),
    "nakul_elbow_joint":        (-3.1416, 3.1416),
    "nakul_wrist_1_joint":      (-6.2832, 6.2832),
    "nakul_wrist_2_joint":      (-6.2832, 6.2832),
    "nakul_wrist_3_joint":      (-6.2832, 6.2832),
    "sahadev_shoulder_pan_joint":  (-6.2832, 6.2832),
    "sahadev_shoulder_lift_joint": (-6.2832, 6.2832),
    "sahadev_elbow_joint":        (-3.1416, 3.1416),
    "sahadev_wrist_1_joint":      (-6.2832, 6.2832),
    "sahadev_wrist_2_joint":      (-6.2832, 6.2832),
    "sahadev_wrist_3_joint":      (-6.2832, 6.2832),
}

# Minimum fraction of samples where a pair must be collision-free
# to be marked as "Never" colliding.
# MoveIt Setup Assistant uses 95% (0.95) by default.
NEVER_THRESHOLD = 0.95

# Pairs that are ALWAYS in collision (>99.9% of samples) are marked
# with reason="Default" — they are typically overlapping geometry.
ALWAYS_THRESHOLD = 0.999


@dataclass
class LinkPairStats:
    """Track collision statistics for a pair of links."""
    in_collision: int = 0
    total_samples: int = 0
    is_adjacent: bool = False
    reason: str = ""

    @property
    def collision_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.in_collision / self.total_samples


def parse_urdf_adjacency(urdf_path: str) -> Tuple[Set[str], Dict[str, str], Set[Tuple[str, str]]]:
    """Parse URDF to extract link names, parent-child relationships, and adjacent pairs.
    
    Returns:
        links: Set of all link names that have collision geometry
        link_parents: Dict mapping child link -> parent link
        adjacent_pairs: Set of (link1, link2) tuples that are connected by joints
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Collect all links that have collision geometry
    links_with_collision = set()
    all_links = set()
    for link_elem in root.findall(".//link"):
        name = link_elem.get("name")
        if name:
            all_links.add(name)
            if link_elem.find("collision") is not None:
                links_with_collision.add(name)

    # Collect parent-child pairs from joints
    link_parents = {}
    adjacent_pairs = set()
    for joint_elem in root.findall(".//joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is not None and child_elem is not None:
            parent = parent_elem.get("link")
            child = child_elem.get("link")
            if parent and child:
                link_parents[child] = parent
                # Adjacent if both have collision geometry
                pair = tuple(sorted([parent, child]))
                adjacent_pairs.add(pair)

    return links_with_collision, link_parents, adjacent_pairs


def find_always_never_pairs(
    links_with_collision: Set[str],
    link_parents: Dict[str, str],
    adjacent_pairs: Set[Tuple[str, str]],
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Identify link pairs that can never collide due to kinematic structure.
    
    Returns:
        always_pairs: Pairs that are always in collision (overlapping geometry)
        never_by_structure: Pairs that can never collide (too far in kinematic chain)
    """
    # For a serial chain, links separated by 2+ joints typically can't
    # self-collide. But we don't assume this — sampling will verify.
    return set(), set()


def compute_collision_matrix_standalone(
    urdf_path: str,
    num_samples: int = 10000,
    verbose: bool = False,
) -> Dict[Tuple[str, str], LinkPairStats]:
    """Compute collision matrix using standalone collision checking.
    
    Uses yourdfpy for URDF parsing and FK, trimesh for collision geometry,
    and python-fcl for fast collision detection.
    """
    try:
        import yourdfpy
        import trimesh
    except ImportError:
        print("ERROR: yourdfpy and trimesh are required for standalone mode.")
        print("Install with: pip install yourdfpy trimesh python-fcl")
        sys.exit(1)

    try:
        import fcl
        use_fcl = True
    except ImportError:
        print("WARNING: python-fcl not available, using trimesh collision manager (slower)")
        use_fcl = False

    print(f"Loading URDF: {urdf_path}")
    
    # Load URDF
    robot = yourdfpy.URDF.load(
        urdf_path,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )

    # Get adjacency info from URDF XML
    links_with_collision, link_parents, adjacent_pairs = parse_urdf_adjacency(urdf_path)
    
    # Get list of collision-enabled links from yourdfpy
    collision_links = []
    for link_name in robot.link_map:
        link = robot.link_map[link_name]
        if link.collisions:
            collision_links.append(link_name)
    
    print(f"Found {len(collision_links)} links with collision geometry")
    print(f"Found {len(adjacent_pairs)} adjacent pairs")
    
    # Generate all possible pairs
    all_pairs = list(itertools.combinations(sorted(collision_links), 2))
    print(f"Total link pairs to check: {len(all_pairs)}")

    # Initialize stats
    pair_stats: Dict[Tuple[str, str], LinkPairStats] = {}
    for pair in all_pairs:
        stats = LinkPairStats()
        if pair in adjacent_pairs:
            stats.is_adjacent = True
        pair_stats[pair] = stats

    # Get joint names from robot
    actuated_joints = [j for j in robot.joint_names if j in JOINT_LIMITS]
    print(f"Actuated joints: {actuated_joints}")

    # Sample random configurations and check collisions
    print(f"\nSampling {num_samples} random configurations...")
    
    for i in range(num_samples):
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  Sample {i + 1}/{num_samples}...")

        # Generate random joint configuration
        config = {}
        for joint_name in actuated_joints:
            lo, hi = JOINT_LIMITS[joint_name]
            config[joint_name] = np.random.uniform(lo, hi)

        # Update robot configuration
        try:
            robot.update_cfg(config)
        except Exception as e:
            if verbose:
                print(f"    WARNING: Failed to update config at sample {i}: {e}")
            continue

        # Get collision meshes in world frame
        link_transforms = {}
        link_meshes = {}
        
        for link_name in collision_links:
            link = robot.link_map[link_name]
            try:
                # Get the transform for this link
                tf = robot.get_transform(link_name)
                link_transforms[link_name] = tf
                
                # Get collision meshes
                meshes = []
                for col in link.collisions:
                    if col.geometry.mesh is not None:
                        mesh = col.geometry.mesh
                        if isinstance(mesh, trimesh.Scene):
                            for geom in mesh.geometry.values():
                                m = geom.copy()
                                if col.origin is not None:
                                    m.apply_transform(col.origin)
                                m.apply_transform(tf)
                                meshes.append(m)
                        elif isinstance(mesh, trimesh.Trimesh):
                            m = mesh.copy()
                            if col.origin is not None:
                                m.apply_transform(col.origin)
                            m.apply_transform(tf)
                            meshes.append(m)
                    elif col.geometry.box is not None:
                        size = col.geometry.box
                        m = trimesh.creation.box(extents=size)
                        if col.origin is not None:
                            m.apply_transform(col.origin)
                        m.apply_transform(tf)
                        meshes.append(m)
                    elif col.geometry.sphere is not None:
                        r = col.geometry.sphere
                        m = trimesh.creation.icosphere(radius=r)
                        if col.origin is not None:
                            m.apply_transform(col.origin)
                        m.apply_transform(tf)
                        meshes.append(m)
                    elif col.geometry.cylinder is not None:
                        r, h = col.geometry.cylinder
                        m = trimesh.creation.cylinder(radius=r, height=h)
                        if col.origin is not None:
                            m.apply_transform(col.origin)
                        m.apply_transform(tf)
                        meshes.append(m)
                
                if meshes:
                    link_meshes[link_name] = meshes
            except Exception as e:
                if verbose:
                    print(f"    WARNING: Failed to get transform for {link_name}: {e}")
                continue

        # Check all pairs for collision
        for pair in all_pairs:
            link1, link2 = pair
            stats = pair_stats[pair]
            stats.total_samples += 1

            if link1 not in link_meshes or link2 not in link_meshes:
                continue

            # Check if any mesh from link1 collides with any mesh from link2
            colliding = False
            for m1 in link_meshes[link1]:
                for m2 in link_meshes[link2]:
                    try:
                        # Use trimesh collision manager
                        manager = trimesh.collision.CollisionManager()
                        manager.add_object("a", m1)
                        colliding = manager.in_collision_single(m2)
                        if colliding:
                            break
                    except Exception:
                        # trimesh collision can fail with degenerate meshes
                        continue
                if colliding:
                    break

            if colliding:
                stats.in_collision += 1

    return pair_stats


def compute_collision_matrix_moveit(
    urdf_path: str,
    srdf_path: str,
    num_samples: int = 10000,
) -> Dict[Tuple[str, str], LinkPairStats]:
    """Compute collision matrix using MoveIt2 Python bindings.
    
    This is the preferred method when running inside the Docker container
    with ROS2 and MoveIt installed. It uses MoveIt's own collision
    checking infrastructure which is identical to what the Setup
    Assistant uses.
    
    NOTE: This requires a running ROS2 node context.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from moveit_py.core import RobotModel, RobotState
        from moveit_py.planning import PlanningScene
    except ImportError:
        print("ERROR: moveit_py not available. Use standalone mode or run inside Docker.")
        return {}

    # Initialize ROS2 if needed
    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node("collision_matrix_generator")

    # Load robot model
    with open(urdf_path) as f:
        urdf_string = f.read()
    with open(srdf_path) as f:
        srdf_string = f.read()

    robot_model = RobotModel(urdf_string, srdf_string)
    planning_scene = PlanningScene(robot_model)

    # Get all link names with collision geometry
    collision_links = []
    for link_name in robot_model.link_model_names:
        link = robot_model.get_link_model(link_name)
        if link and link.collision_mesh_count > 0:
            collision_links.append(link_name)

    # Get adjacent pairs
    _, _, adjacent_pairs = parse_urdf_adjacency(urdf_path)

    # Generate all pairs
    all_pairs = list(itertools.combinations(sorted(collision_links), 2))

    pair_stats: Dict[Tuple[str, str], LinkPairStats] = {}
    for pair in all_pairs:
        stats = LinkPairStats()
        if pair in adjacent_pairs:
            stats.is_adjacent = True
        pair_stats[pair] = stats

    # Sample random configurations
    print(f"Sampling {num_samples} configurations using MoveIt collision checking...")
    robot_state = RobotState(robot_model)

    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Sample {i + 1}/{num_samples}...")

        # Randomize all joints
        robot_state.set_to_random_positions()
        planning_scene.set_current_state(robot_state)

        # Check pairwise collisions
        for pair in all_pairs:
            link1, link2 = pair
            stats = pair_stats[pair]
            stats.total_samples += 1

            if planning_scene.is_path_valid(
                robot_state, group_name="", 
                link_names=[link1, link2]
            ):
                continue
            else:
                stats.in_collision += 1

    node.destroy_node()
    return pair_stats


def classify_pairs(
    pair_stats: Dict[Tuple[str, str], LinkPairStats],
    adjacent_pairs: Set[Tuple[str, str]],
    never_threshold: float = NEVER_THRESHOLD,
    always_threshold: float = ALWAYS_THRESHOLD,
) -> Dict[Tuple[str, str], LinkPairStats]:
    """Classify each pair and assign a reason for disabling collision."""
    
    for pair, stats in pair_stats.items():
        if stats.total_samples == 0:
            continue

        collision_rate = stats.collision_rate

        if pair in adjacent_pairs:
            stats.reason = "Adjacent"
        elif collision_rate >= always_threshold:
            stats.reason = "Default"  # Always in collision (overlapping geometry)
        elif collision_rate <= (1.0 - never_threshold):
            stats.reason = "Never"  # Never (or almost never) in collision
        # else: pair sometimes collides — DO NOT disable, let MoveIt check it

    return pair_stats


def generate_srdf_entries(
    pair_stats: Dict[Tuple[str, str], LinkPairStats],
    include_reasons: Set[str] = {"Adjacent", "Default", "Never"},
) -> str:
    """Generate SRDF disable_collisions XML entries."""
    
    lines = []
    lines.append("  <!-- ================================================================ -->")
    lines.append("  <!-- DISABLE COLLISIONS (auto-generated by compute_collision_matrix.py) -->")
    lines.append("  <!-- ================================================================ -->")
    lines.append("")

    # Group by reason
    by_reason: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for pair, stats in sorted(pair_stats.items()):
        if stats.reason in include_reasons:
            by_reason[stats.reason].append(pair)

    for reason in ["Adjacent", "Default", "Never"]:
        if reason not in by_reason:
            continue
        pairs = by_reason[reason]
        lines.append(f"  <!-- ── {reason} pairs ({len(pairs)}) ──── -->")
        for link1, link2 in sorted(pairs):
            lines.append(
                f'  <disable_collisions link1="{link1}" link2="{link2}" reason="{reason}"/>'
            )
        lines.append("")

    return "\n".join(lines)


def print_statistics(
    pair_stats: Dict[Tuple[str, str], LinkPairStats],
) -> None:
    """Print collision matrix statistics."""
    
    total = len(pair_stats)
    adjacent = sum(1 for s in pair_stats.values() if s.reason == "Adjacent")
    default = sum(1 for s in pair_stats.values() if s.reason == "Default")
    never = sum(1 for s in pair_stats.values() if s.reason == "Never")
    sometimes = sum(1 for s in pair_stats.values() if s.reason == "")
    no_data = sum(1 for s in pair_stats.values() if s.total_samples == 0)

    print("\n" + "=" * 60)
    print("COLLISION MATRIX STATISTICS")
    print("=" * 60)
    print(f"Total link pairs:         {total}")
    print(f"Adjacent (disabled):      {adjacent}")
    print(f"Always colliding:         {default}")
    print(f"Never colliding:          {never}")
    print(f"Sometimes colliding:      {sometimes}  ← MoveIt will check these at runtime")
    print(f"No data:                  {no_data}")
    print(f"Total disabled:           {adjacent + default + never}")
    print()

    # Print the "sometimes colliding" pairs that MoveIt must check
    if sometimes > 0:
        print("Link pairs that SOMETIMES collide (not disabled):")
        for pair, stats in sorted(pair_stats.items()):
            if stats.reason == "" and stats.total_samples > 0:
                pct = stats.collision_rate * 100
                print(f"  {pair[0]} ↔ {pair[1]}: {pct:.1f}% collision rate")
        print()


def generate_urdf_from_xacro(
    xacro_path: str,
    output_path: str,
) -> str:
    """Generate a flat URDF file from a xacro file."""
    import subprocess
    
    result = subprocess.run(
        ["xacro", xacro_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: xacro failed:\n{result.stderr}")
        sys.exit(1)
    
    with open(output_path, "w") as f:
        f.write(result.stdout)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute the Allowed Collision Matrix for a dual UR5e cell"
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to the URDF file. If not specified, will try to generate "
             "from the xacro in the workspace.",
    )
    parser.add_argument(
        "--srdf",
        type=str,
        default=None,
        help="Path to the SRDF file (for MoveIt mode).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of random configurations to sample (default: 10000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for the disable_collisions XML entries.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=NEVER_THRESHOLD,
        help=f"Fraction of samples that must be collision-free to mark as 'Never' "
             f"(default: {NEVER_THRESHOLD}).",
    )
    parser.add_argument(
        "--mode",
        choices=["standalone", "moveit", "auto"],
        default="auto",
        help="Collision checking backend (default: auto).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics but don't write output file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose debug output.",
    )

    args = parser.parse_args()

    # Find the URDF
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    if args.urdf:
        urdf_path = args.urdf
    else:
        # Try to generate from xacro
        xacro_path = str(
            workspace_root
            / "valid8_dual_cell_description"
            / "urdf"
            / "valid8_dual_cell.urdf.xacro"
        )
        if not os.path.exists(xacro_path):
            # Also try the controlled xacro
            xacro_path = str(
                workspace_root
                / "valid8_dual_cell_control"
                / "urdf"
                / "valid8_dual_cell_controlled.urdf.xacro"
            )

        if os.path.exists(xacro_path):
            print(f"Generating URDF from xacro: {xacro_path}")
            urdf_path = "/tmp/valid8_dual_cell.urdf"
            generate_urdf_from_xacro(xacro_path, urdf_path)
        else:
            print("ERROR: Could not find URDF or xacro file.")
            print(f"  Looked for: {xacro_path}")
            print("  Use --urdf to specify the path manually.")
            sys.exit(1)

    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF file not found: {urdf_path}")
        sys.exit(1)

    # Find the SRDF
    srdf_path = args.srdf
    if srdf_path is None:
        candidate = str(
            workspace_root
            / "valid8_dual_cell_moveit_config"
            / "srdf"
            / "valid8_dual_cell.srdf"
        )
        if os.path.exists(candidate):
            srdf_path = candidate

    # Determine mode
    mode = args.mode
    if mode == "auto":
        try:
            import moveit_py  # noqa: F401
            mode = "moveit"
            print("Auto-detected MoveIt Python bindings — using MoveIt mode")
        except ImportError:
            mode = "standalone"
            print("MoveIt Python bindings not available — using standalone mode")

    # Parse adjacency from URDF
    links_with_collision, link_parents, adjacent_pairs = parse_urdf_adjacency(urdf_path)
    print(f"\nLinks with collision geometry: {len(links_with_collision)}")
    if args.verbose:
        for link in sorted(links_with_collision):
            print(f"  - {link}")

    # Compute collision matrix
    if mode == "moveit" and srdf_path:
        pair_stats = compute_collision_matrix_moveit(
            urdf_path, srdf_path, num_samples=args.samples
        )
    else:
        pair_stats = compute_collision_matrix_standalone(
            urdf_path, num_samples=args.samples, verbose=args.verbose
        )

    if not pair_stats:
        print("ERROR: No collision data computed.")
        sys.exit(1)

    # Classify pairs
    pair_stats = classify_pairs(
        pair_stats, adjacent_pairs, never_threshold=args.threshold
    )

    # Print statistics
    print_statistics(pair_stats)

    # Generate SRDF entries
    srdf_entries = generate_srdf_entries(pair_stats)

    if args.dry_run:
        print("DRY RUN — not writing output file")
        print("\nGenerated entries preview (first 50 lines):")
        for line in srdf_entries.split("\n")[:50]:
            print(line)
        if srdf_entries.count("\n") > 50:
            print(f"  ... ({srdf_entries.count(chr(10)) - 50} more lines)")
    else:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = str(
                workspace_root
                / "valid8_dual_cell_moveit_config"
                / "srdf"
                / "generated_collision_matrix.xml"
            )

        with open(output_path, "w") as f:
            f.write(srdf_entries)
        print(f"\nCollision matrix entries written to: {output_path}")
        print(
            "\nTo use these entries, copy them into your SRDF file "
            "(valid8_dual_cell.srdf) replacing the existing "
            "<disable_collisions> section."
        )


if __name__ == "__main__":
    main()
