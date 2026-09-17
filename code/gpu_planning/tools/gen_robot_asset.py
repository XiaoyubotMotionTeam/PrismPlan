# -*- coding: utf-8 -*-
"""Generate self-contained robot asset YAMLs from the C++ compile-time robot data.

The C++ package hardcodes each robot's kinematic + collision model as
``__constant__`` arrays in ``cpp/src/robots/<robot>.cuh`` (fixed transforms,
collision spheres, sphere->joint map, joint types, self-collision ranges) and
its joint limits as scale factors in ``cpp/src/planning/Robots.hh``. Those
arrays are the ground truth the runtime FK / collision kernels were validated
against, so we lift them verbatim into plain YAML assets that the pure-Python
``robot_model_loader`` can read with zero external kinematics dependency.

Three robots are supported, matching the MotionBenchMaker platforms:

  panda   7-DoF   8 joints   59 spheres   serial chain
  fetch   8-DoF   9 joints  111 spheres   serial chain
  baxter 14-DoF  15 joints   75 spheres   BRANCHING tree (two arms off the base)

Baxter is the whole-body composite case: its kinematic tree branches at the
torso, so ``<robot>_joint_parents`` / ``_joint_id_to_dof`` / ``_T_memory_idx`` /
``_dfs_order`` are emitted to drive the tree-walking runtime FK. Serial robots
carry the same four fields with the degenerate values (parent[i] = i-1,
dof[i] = i-1, dfs = identity), which makes the tree FK reduce exactly to the
old serial loop -- so Panda results stay bit-identical.

Usage:
    python3 tools/gen_robot_asset.py              # regenerate all robots
    python3 tools/gen_robot_asset.py baxter       # just one
"""

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ROBOTS_DIR = _ROOT / "cpp" / "src" / "robots"
_OUT_DIR = _ROOT / "config" / "robots"

# Robots.hh holds the joint limits as scale factors. The repo's own copy may
# only carry the structs it compiles; the retained limits source provides the
# provenance data for robots driven only through the runtime path.
_LIMIT_SOURCES = [
    _ROOT / "cpp" / "src" / "planning" / "Robots.hh",
    _ROOT / "cpp" / "src" / "planning" / "Robots_limits_source.hh",
]

ROBOTS = {
    "panda": {"struct": "Panda", "n_dof": 7},
    "fetch": {"struct": "Fetch", "n_dof": 8},
    "baxter": {"struct": "Baxter", "n_dof": 14},
}


def _floats(text):
    """Extract all C float literals (strips trailing f) from a block of text."""
    out = []
    for tok in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?f?", text):
        t = tok.rstrip("f")
        if t in ("", "-", "+"):
            continue
        out.append(float(t))
    return out


def _ints(text):
    return [int(x) for x in re.findall(r"-?\d+", text)]


def _brace_block(src, anchor):
    """Return the text between the first '{' after `anchor` and its matching '}'."""
    i = src.index(anchor)
    start = src.index("{", i)
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1:j]
    raise ValueError(f"unbalanced braces after {anchor!r}")


def _opt_block(src, anchor):
    """`_brace_block` but returns None when the anchor is absent."""
    try:
        return _brace_block(src, anchor)
    except ValueError:
        return None
    except Exception:
        return None


def parse_robot_cuh(path, robot):
    """Parse one <robot>.cuh into the flat host arrays the YAML asset stores."""
    src = path.read_text()
    up = robot.upper()

    def define(name):
        m = re.search(rf"#define\s+{name}\s+(\d+)", src)
        if m is None:
            raise ValueError(f"{path.name}: missing #define {name}")
        return int(m.group(1))

    n_spheres = define(f"{up}_SPHERE_COUNT")
    n_joints = define(f"{up}_JOINT_COUNT")

    # spheres: float4 {x,y,z,r} * n_spheres
    sph = _floats(_brace_block(src, f"{robot}_spheres_array"))
    assert len(sph) == n_spheres * 4, (robot, len(sph), n_spheres * 4)

    # fixed transforms: n_joints * 16 row-major (strip comments -- they hold digits)
    ft = _floats(re.sub(r"//[^\n]*", "", _brace_block(src, f"{robot}_fixed_transforms")))
    assert len(ft) == n_joints * 16, (robot, len(ft), n_joints * 16)

    s2j = _ints(_brace_block(src, f"{robot}_sphere_to_joint"))
    assert len(s2j) == n_spheres, (robot, len(s2j), n_spheres)

    jt = _ints(_brace_block(src, f"{robot}_joint_types"))
    assert len(jt) == n_joints, (robot, len(jt), n_joints)

    cc = _ints(_brace_block(src, f"{robot}_self_cc_ranges"))
    assert len(cc) % 3 == 0, (robot, len(cc))
    self_cc = [cc[i:i + 3] for i in range(0, len(cc), 3)]

    # --- tree topology (present only for branching robots; synthesise otherwise) ---
    parents_blk = _opt_block(src, f"{robot}_joint_parents")
    if parents_blk is not None:
        joint_parents = _ints(parents_blk)
        joint_id_to_dof = _ints(_brace_block(src, f"{robot}_joint_id_to_dof"))
        t_mem = _ints(_brace_block(src, f"{robot}_T_memory_idx"))
        dfs = _ints(_brace_block(src, f"{robot}_dfs_order"))
    else:
        # Serial chain: joint i hangs off joint i-1 and consumes DoF i-1.
        joint_parents = [max(i - 1, 0) for i in range(n_joints)]
        joint_id_to_dof = [i - 1 for i in range(n_joints)]
        t_mem = [0] + [1] * (n_joints - 1)
        dfs = list(range(n_joints))

    for nm, arr in (("joint_parents", joint_parents),
                    ("joint_id_to_dof", joint_id_to_dof),
                    ("T_memory_idx", t_mem),
                    ("dfs_order", dfs)):
        assert len(arr) == n_joints, (robot, nm, len(arr), n_joints)

    # Sanity: parents must precede children in DFS order, and slot reuse must be safe.
    seen = set()
    for j in dfs:
        assert joint_parents[j] in seen or j == dfs[0], (
            f"{robot}: joint {j} visited before its parent {joint_parents[j]}")
        seen.add(j)

    n_t_slots = max(t_mem) + 1

    return {
        "n_dof": max(joint_id_to_dof) + 1,
        "n_joints": n_joints,
        "fixed_transforms": ft,
        "joint_types": jt,
        "spheres": [sph[i:i + 4] for i in range(0, len(sph), 4)],
        "sphere_to_joint": s2j,
        "self_cc_ranges": self_cc,
        "joint_parents": joint_parents,
        "joint_id_to_dof": joint_id_to_dof,
        "t_memory_idx": t_mem,
        "dfs_order": dfs,
        "n_t_slots": n_t_slots,
        "is_tree": parents_blk is not None,
    }


def parse_limits(struct_name, n_dof):
    """Joint limits: lower = s_a, upper = s_a + s_m (from the Robots.hh scale_cfg)."""
    for path in _LIMIT_SOURCES:
        if not path.exists():
            continue
        src = path.read_text()
        m = re.search(rf"struct\s+{struct_name}\b", src)
        if m is None:
            continue
        start = m.start()
        nxt = re.search(r"\n\s*struct\s+\w+", src[start + 1:])
        end = start + 1 + nxt.start() if nxt else len(src)
        body = src[start:end]
        s_m = _floats(_brace_block(body, "get_s_m"))
        s_a = _floats(_brace_block(body, "get_s_a"))
        assert len(s_m) == len(s_a) == n_dof, (struct_name, len(s_m), len(s_a), n_dof)
        return s_a, [a + m_ for a, m_ in zip(s_a, s_m)], path.name
    raise ValueError(f"no struct {struct_name} found in {[p.name for p in _LIMIT_SOURCES]}")


def _fmt_num(x):
    # compact but exact enough for float32 round-trip
    return repr(float(x))


def emit_yaml(name, data, lower, upper, limit_src, out_path):
    L = []
    L.append(f"# Auto-generated by tools/gen_robot_asset.py from")
    L.append(f"# cpp/src/robots/{name}.cuh (+ {limit_src} joint limits).")
    L.append("# Self-contained robot model asset for gpu_planning.robot_model_loader.")
    L.append("# Do not edit by hand; re-run the generator instead.")
    L.append(f"name: {name}")
    L.append(f"n_dof: {data['n_dof']}")
    L.append(f"n_joints: {data['n_joints']}")
    L.append("")
    L.append("# joint_types: index 0 is the fixed base (ignored by FK); the rest")
    L.append("# are actuated. 5 = Z_ROT, 4 = Y_ROT, 3 = X_ROT, 0..2 = PRISM, -1 = FIXED.")
    L.append("joint_types: [" + ", ".join(str(v) for v in data["joint_types"]) + "]")
    L.append("")
    L.append("joint_lower: [" + ", ".join(_fmt_num(v) for v in lower) + "]")
    L.append("joint_upper: [" + ", ".join(_fmt_num(v) for v in upper) + "]")
    L.append("")
    L.append("# --- kinematic tree topology ---")
    L.append("# joint_parents[i]  : parent joint of i (root points at itself)")
    L.append("# joint_id_to_dof[i]: which entry of q drives joint i (-1 = fixed)")
    L.append("# t_memory_idx[i]   : scratch slot holding joint i's world transform")
    L.append("# dfs_order         : visit order; every parent precedes its children")
    L.append(f"# is_tree: {data['is_tree']}  (false => plain serial chain)")
    L.append(f"n_t_slots: {data['n_t_slots']}")
    L.append("joint_parents: [" + ", ".join(str(v) for v in data["joint_parents"]) + "]")
    L.append("joint_id_to_dof: [" + ", ".join(str(v) for v in data["joint_id_to_dof"]) + "]")
    L.append("t_memory_idx: [" + ", ".join(str(v) for v in data["t_memory_idx"]) + "]")
    L.append("dfs_order: [" + ", ".join(str(v) for v in data["dfs_order"]) + "]")
    L.append("")
    L.append("# fixed_transforms: n_joints blocks of a 4x4 row-major matrix.")
    L.append("fixed_transforms:")
    ft = data["fixed_transforms"]
    for j in range(data["n_joints"]):
        L.append(f"  # joint {j}")
        for r in range(4):
            row = ft[j * 16 + r * 4: j * 16 + r * 4 + 4]
            L.append("  - [" + ", ".join(_fmt_num(v) for v in row) + "]")
    L.append("")
    L.append("# collision spheres in link-local frame: [x, y, z, radius].")
    L.append("spheres:")
    for s in data["spheres"]:
        L.append("  - [" + ", ".join(_fmt_num(v) for v in s) + "]")
    L.append("")
    L.append("sphere_to_joint: [" + ", ".join(str(v) for v in data["sphere_to_joint"]) + "]")
    L.append("")
    L.append("# self_cc_ranges: [sphere_i, range_start, range_end] (end inclusive).")
    L.append("self_cc_ranges:")
    for c in data["self_cc_ranges"]:
        L.append(f"  - [{c[0]}, {c[1]}, {c[2]}]")
    L.append("")
    out_path.write_text("\n".join(L))


def build(name):
    spec = ROBOTS[name]
    cuh = _ROBOTS_DIR / f"{name}.cuh"
    data = parse_robot_cuh(cuh, name)
    assert data["n_dof"] == spec["n_dof"], (name, data["n_dof"], spec["n_dof"])
    lower, upper, limit_src = parse_limits(spec["struct"], spec["n_dof"])
    out = _OUT_DIR / f"{name}.yaml"
    emit_yaml(name, data, lower, upper, limit_src, out)
    kind = "tree" if data["is_tree"] else "serial"
    print(f"wrote {out}")
    print(f"  n_dof={data['n_dof']} n_joints={data['n_joints']} "
          f"spheres={len(data['spheres'])} self_cc={len(data['self_cc_ranges'])} "
          f"{kind} n_t_slots={data['n_t_slots']}")


def main():
    names = sys.argv[1:] or list(ROBOTS)
    for n in names:
        if n not in ROBOTS:
            raise SystemExit(f"unknown robot {n!r}; known: {list(ROBOTS)}")
        build(n)


if __name__ == "__main__":
    main()
