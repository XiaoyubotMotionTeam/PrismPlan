# -*- coding: utf-8 -*-
"""Convert an upstream MotionBenchMaker problem tarball into the VAMP-format
JSON that ``gpu_planning.benchmark.loaders.load_vamp_problems`` already reads.

Upstream ships each robot's MBM problems as ``problems.tar.bz2`` holding raw
MoveIt YAML pairs (``scene####.yaml`` + ``request####.yaml``) per scene family.
VAMP's own ``problem_tar_to_pkl_json.py`` converts these, but it imports the
compiled ``vamp`` module (for its transform helpers and a validity check),
which we do not have. This script is a dependency-free equivalent: it emits the
same ``{"problems": {scene: [ {start, goals, box/sphere/cylinder, ...} ]}}``
layout our loader consumes, using only numpy + PyYAML.

Two conventions are pinned here, both cross-checked against the robometrics
Panda set (see ``--cross-check``):

* MoveIt pose orientations are ROS order ``[x, y, z, w]``.
* Obstacle orientation is emitted as ``orientation_euler_xyz`` in the fixed-axis
  (extrinsic) XYZ convention, i.e. ``R = Rz @ Ry @ Rx`` -- the convention
  ``benchmark.problem.euler_xyz_to_quat_wxyz`` documents and inverts.

Unlike the upstream script we do not stamp a ``valid`` flag from a collision
check: we have no independent validator, so every problem is emitted with
``valid: true`` and start/goal feasibility is left to the benchmark harness,
which reports StartInCollision itself. Problem counts can therefore differ
slightly from upstream's pre-filtered ones; the harness output is the
authority.

Usage:
    python3 tools/mbm_tar_to_json.py --robot baxter \
        --tar  /path/to/baxter/problems.tar.bz2 \
        --out  data/mbm_vamp/baxter_problems.json
"""

import argparse
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

try:                                   # PyYAML's C loader is ~10x faster here
    from yaml import CSafeLoader as _Loader
except ImportError:                    # pragma: no cover
    from yaml import SafeLoader as _Loader


# Actuated joints, in the order the planners expect q to arrive in. Taken from
# VAMP's robot headers (src/impl/vamp/robots/<robot>.hh, `joint_names`), which
# is the same order the pRRTC constant tables were generated against.
JOINT_NAMES = {
    "panda": [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ],
    "fetch": [
        "torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint",
        "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint",
        "wrist_flex_joint", "wrist_roll_joint",
    ],
    "baxter": [
        "left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2",
        "right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2",
    ],
}


def quat_xyzw_to_matrix(q):
    """ROS-order [x,y,z,w] unit quaternion -> 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def pose_to_matrix(pose):
    """MoveIt pose (dict or absent) -> 4x4 homogeneous transform."""
    T = np.eye(4)
    if not pose:
        return T
    pos = pose.get("position", [0.0, 0.0, 0.0])
    ori = pose.get("orientation", [0.0, 0.0, 0.0, 1.0])
    if isinstance(pos, dict):
        pos = [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]
    if isinstance(ori, dict):
        ori = [ori.get("x", 0.0), ori.get("y", 0.0),
               ori.get("z", 0.0), ori.get("w", 1.0)]
    T[:3, :3] = quat_xyzw_to_matrix(ori)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def matrix_to_euler_xyz(T):
    """4x4 (or 3x3) -> fixed-axis XYZ Euler triple, inverse of R = Rz@Ry@Rx."""
    R = np.asarray(T)[:3, :3]
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, float(sy)))
    ry = np.arcsin(sy)
    if abs(sy) < 1.0 - 1e-9:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:                                  # gimbal lock: fold rz into rx
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return [float(rx), float(ry), float(rz)]


def parse_scene(doc):
    """MoveIt planning-scene YAML -> {'box': [...], 'sphere': [...], 'cylinder': [...]}."""
    out = {"box": [], "sphere": [], "cylinder": []}
    n_dropped_meshes = 0
    world = (doc or {}).get("world") or {}
    for co in world.get("collision_objects") or []:
        base = pose_to_matrix(co.get("pose"))
        prims = co.get("primitives") or []
        poses = co.get("primitive_poses") or []
        n_dropped_meshes += len(co.get("meshes") or [])
        for k, prim in enumerate(prims):
            prim_pose = pose_to_matrix(poses[k] if k < len(poses) else None)
            T = base @ prim_pose
            kind = prim["type"]
            dims = [float(d) for d in prim["dimensions"]]
            obj = {
                "name": co.get("id", f"obj_{k}"),
                "position": [float(v) for v in T[:3, 3]],
                "orientation_euler_xyz": matrix_to_euler_xyz(T),
            }
            if kind == "sphere":
                obj["radius"] = dims[0]
            elif kind == "cylinder":
                # MoveIt CYLINDER dimensions are [height, radius].
                obj["length"] = dims[0]
                obj["radius"] = dims[1]
            elif kind == "box":
                obj["half_extents"] = [d / 2.0 for d in dims]
            else:
                raise ValueError(f"unsupported primitive type {kind!r}")
            out[kind].append(obj)
    # Mesh obstacles are silently unrepresentable here. A scene that is entirely
    # meshes (MBM's kitchen / table_bars) would otherwise convert to empty free
    # space and score a meaningless 100%, so the count is surfaced to the caller.
    out["_dropped_meshes"] = n_dropped_meshes
    return out


def parse_request(doc, joints):
    """MoveIt motion-plan request YAML -> {'start': [...], 'goals': [[...]]}.

    The recorded start state carries more joints than the planner drives (head,
    grippers), so both start and goal are re-indexed onto `joints`.
    """
    js = doc["start_state"]["joint_state"]
    name_to_pos = dict(zip(js["name"], js["position"]))
    missing = [j for j in joints if j not in name_to_pos]
    if missing:
        raise ValueError(f"start state missing joints: {missing}")
    start = [float(name_to_pos[j]) for j in joints]

    jc = doc["goal_constraints"][0]["joint_constraints"]
    goal_map = {e["joint_name"]: float(e["position"]) for e in jc}
    missing = [j for j in joints if j not in goal_map]
    if missing:
        raise ValueError(f"goal constraints missing joints: {missing}")
    goal = [goal_map[j] for j in joints]

    return {"start": start, "goals": [goal]}


_MEMBER_RE = re.compile(r"^(scene|request)(\d+)\.yaml$")


def convert(tar_path, robot):
    joints = JOINT_NAMES[robot]
    scenes, requests = defaultdict(dict), defaultdict(dict)

    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            family, filename = parts[-2], parts[-1]
            # Anchored so `scene_sensed0048.yaml` (the octomap-derived variant)
            # cannot be mistaken for `scene0048.yaml`, and so the digit-free
            # `config.yaml` is skipped rather than raising on the index parse.
            m = _MEMBER_RE.match(filename)
            if not m:
                continue
            family = family.replace(f"_{robot}", "")
            idx = int(m.group(2))
            doc = yaml.load(tar.extractfile(member).read(), Loader=_Loader)
            if m.group(1) == "scene":
                scenes[family][idx] = parse_scene(doc)
            else:
                requests[family][idx] = parse_request(doc, joints)

    problems = {}
    dropped = {}
    for family in sorted(scenes):
        paired = sorted(set(scenes[family]) & set(requests[family]))
        rows = []
        n_meshes = 0
        for idx in paired:
            row = {"index": idx, "problem": family, "valid": True}
            scene = dict(scenes[family][idx])
            n_meshes += scene.pop("_dropped_meshes", 0)
            row.update(scene)
            row.update(requests[family][idx])
            rows.append(row)
        problems[family] = rows
        dropped[family] = n_meshes
    return {"robot": robot, "joints": joints, "problems": problems,
            "_dropped_meshes": dropped}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=sorted(JOINT_NAMES))
    ap.add_argument("--tar", required=True, help="upstream problems.tar.bz2")
    ap.add_argument("--out", required=True, help="output .json")
    args = ap.parse_args()

    data = convert(args.tar, args.robot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)

    total = sum(len(v) for v in data["problems"].values())
    print(f"wrote {args.out}")
    print(f"  robot={args.robot} dof={len(data['joints'])} "
          f"families={len(data['problems'])} problems={total}")
    for k in sorted(data["problems"]):
        rows = data["problems"][k]
        n_obs = np.mean([len(r["box"]) + len(r["sphere"]) + len(r["cylinder"])
                         for r in rows]) if rows else 0
        n_mesh = data["_dropped_meshes"].get(k, 0)
        warn = f"  !! {n_mesh} mesh obstacles DROPPED" if n_mesh else ""
        if n_mesh and n_obs == 0:
            warn += " -- scene is empty, DO NOT BENCHMARK"
        print(f"    {k:42s} {len(rows):4d} problems, "
              f"{n_obs:5.1f} obstacles avg{warn}")


if __name__ == "__main__":
    main()
