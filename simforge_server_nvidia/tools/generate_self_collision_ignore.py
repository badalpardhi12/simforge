#!/usr/bin/env python3
"""
Generate cuRobo self_collision_ignore from a MoveIt SRDF file.

MoveIt's Setup Assistant generates the Allowed Collision Matrix (ACM) by
sampling thousands of random joint configurations and checking which
link-pairs actually collide, are always in collision, are adjacent, or
never collide.  The result is stored in the SRDF as
``<disable_collisions>`` entries.

cuRobo uses a ``self_collision_ignore`` dictionary to skip self-collision
checks between specified link pairs.  This script reads the SRDF, filters
to the links that cuRobo actually has collision spheres for, and outputs
the cuRobo-format YAML snippet.

Usage
-----
    python generate_self_collision_ignore.py \
        --srdf ../../simforge_server_new/valid8_dual_cell_moveit_config/srdf/valid8_dual_cell.srdf \
        --prefix nakul_ \
        --collision-links nakul_shoulder_link nakul_upper_arm_link \
            nakul_forearm_link nakul_wrist_1_link nakul_wrist_2_link \
            nakul_wrist_3_link nakul_tool_base_link

The output is a YAML-format ``self_collision_ignore`` section ready
to paste into the cuRobo robot config.
"""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Set


def parse_srdf_disable_collisions(
    srdf_path: str, prefix: str
) -> Dict[str, Set[str]]:
    """Parse all <disable_collisions> entries for a given prefix.

    Returns a bidirectional mapping:
        { link_a: {link_b, link_c}, link_b: {link_a}, ... }
    """
    tree = ET.parse(srdf_path)
    root = tree.getroot()

    disabled: Dict[str, Set[str]] = defaultdict(set)
    for dc in root.findall("disable_collisions"):
        l1 = dc.get("link1", "")
        l2 = dc.get("link2", "")
        reason = dc.get("reason", "")

        # Only consider links with our prefix
        if not l1.startswith(prefix) or not l2.startswith(prefix):
            continue

        disabled[l1].add(l2)
        disabled[l2].add(l1)

    return disabled


def build_curobo_ignore(
    disabled: Dict[str, Set[str]],
    collision_links: List[str],
) -> Dict[str, List[str]]:
    """Build the cuRobo self_collision_ignore dictionary.

    cuRobo's self_collision_ignore lists which links to SKIP collision
    checking for.  The mapping is bidirectional — listing A→B is
    sufficient (no need to also list B→A).

    We only include pairs where BOTH links are in the collision_links
    list (cuRobo only has spheres for those links).
    """
    coll_set = set(collision_links)
    result: Dict[str, List[str]] = {}

    # Process in order of collision_links for deterministic output
    seen_pairs: Set[tuple] = set()

    for link in collision_links:
        ignore_list = []
        for other in collision_links:
            if other == link:
                continue
            pair = tuple(sorted([link, other]))
            if pair in seen_pairs:
                continue
            if other in disabled.get(link, set()):
                ignore_list.append(other)
                seen_pairs.add(pair)
        if ignore_list:
            result[link] = ignore_list

    return result


def pairs_not_disabled(
    disabled: Dict[str, Set[str]],
    collision_links: List[str],
) -> List[tuple]:
    """Find link pairs that ARE checked for collision (not disabled).

    These are the pairs where MoveIt determined that real collisions
    can happen — cuRobo should also check them.
    """
    enabled = []
    n = len(collision_links)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = collision_links[i], collision_links[j]
            if b not in disabled.get(a, set()):
                enabled.append((a, b))
    return enabled


def format_yaml(ignore: Dict[str, List[str]], indent: int = 6) -> str:
    """Format the self_collision_ignore dict as YAML."""
    lines = []
    pad = " " * indent
    for link, others in ignore.items():
        others_str = ", ".join(f'"{o}"' for o in others)
        lines.append(f'{pad}{link}: [{others_str}]')
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cuRobo self_collision_ignore from SRDF"
    )
    parser.add_argument(
        "--srdf", required=True,
        help="Path to the MoveIt SRDF file"
    )
    parser.add_argument(
        "--prefix", required=True,
        help="Link prefix to filter (e.g. 'nakul_')"
    )
    parser.add_argument(
        "--collision-links", nargs="+", required=True,
        help="List of cuRobo collision_link_names"
    )
    args = parser.parse_args()

    print(f"Parsing SRDF: {args.srdf}")
    print(f"Prefix: {args.prefix}")
    print(f"Collision links: {args.collision_links}")
    print()

    disabled = parse_srdf_disable_collisions(args.srdf, args.prefix)

    # Show what SRDF says
    print("=" * 60)
    print("SRDF disabled pairs (for our collision links):")
    print("=" * 60)
    coll_set = set(args.collision_links)
    for link in sorted(disabled.keys()):
        if link in coll_set:
            others_in_scope = sorted(disabled[link] & coll_set)
            if others_in_scope:
                print(f"  {link} → {others_in_scope}")
    print()

    # Build cuRobo ignore dict
    ignore = build_curobo_ignore(disabled, args.collision_links)

    # Show enabled pairs (where collisions WILL be checked)
    enabled = pairs_not_disabled(disabled, args.collision_links)
    print("=" * 60)
    print("ENABLED pairs (MoveIt says these CAN collide):")
    print("cuRobo WILL check self-collision for these pairs.")
    print("=" * 60)
    for a, b in enabled:
        print(f"  ✓ {a} ↔ {b}")
    print()

    # Output YAML
    print("=" * 60)
    print("cuRobo self_collision_ignore (paste into robot config):")
    print("=" * 60)
    print()
    print("    self_collision_ignore:")
    print(format_yaml(ignore))
    print()


if __name__ == "__main__":
    main()
