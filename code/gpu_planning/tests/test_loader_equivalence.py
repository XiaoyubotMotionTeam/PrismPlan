# -*- coding: utf-8 -*-
"""Equivalence test: robot_model_loader output vs. C++ ground truth.

The authoritative reference is the
compile-time model in ``cpp/src/robots/panda.cuh`` (+ joint limits in
``Robots.hh``) that the CUDA kernels were validated against. This test parses
those C++ arrays directly (reusing the generator's parser) and asserts the
loader's flat fields are bit-identical.

Run:
    python3 tests/test_loader_equivalence.py
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "tools"))

from gpu_planning.substrate.robot_model_loader import load_robot_asset  # noqa: E402
import gen_robot_asset as gen  # noqa: E402


def main():
    asset = load_robot_asset()

    cuh = gen.parse_robot_cuh(gen._ROBOTS_DIR / "panda.cuh", "panda")
    lower, upper, _ = gen.parse_limits("Panda", 7)

    checks = []

    def chk(name, a, b, exact=True):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        ok = np.array_equal(a, b) if exact else np.allclose(a, b, atol=1e-6)
        checks.append((name, ok, a.shape, b.shape))

    chk("n_dof", asset["n_dof"], cuh["n_dof"])
    chk("n_joints", asset["n_joints"], cuh["n_joints"])
    chk("joint_types", asset["joint_types"], cuh["joint_types"])
    chk("fixed_transforms", asset["fixed_transforms"], cuh["fixed_transforms"])
    chk("sphere_to_joint", asset["sphere_to_joint"], cuh["sphere_to_joint"])

    # spheres: loader is flat [n*4]; cuh["spheres"] is [n][4]
    cuh_spheres_flat = [v for s in cuh["spheres"] for v in s]
    chk("spheres_flat", asset["spheres_flat"], cuh_spheres_flat)

    # self_cc_ranges: loader flat [n*3]; cuh is [n][3]
    cuh_cc_flat = [v for c in cuh["self_cc_ranges"] for v in c]
    chk("self_cc_ranges_flat", asset["self_cc_ranges_flat"], cuh_cc_flat)

    # joint limits round-trip through float32 in the YAML
    chk("joint_lower", asset["joint_lower"], lower, exact=False)
    chk("joint_upper", asset["joint_upper"], upper, exact=False)

    all_ok = True
    for name, ok, sa, sb in checks:
        status = "OK  " if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name:24s} loader{tuple(sa)} vs ref{tuple(sb)}")

    # Sanity on the derived approx model (no C++ ground truth to compare, but
    # the gate must be non-empty or self-collision is silently disabled).
    n_approx = len(asset["approx_sphere_to_joint"])
    print(f"  [info] approx spheres = {n_approx} (one per joint w/ spheres)")
    print(f"  [info] approx self_cc entries = "
          f"{len(asset['approx_self_cc_ranges_flat']) // 3}")
    assert n_approx > 0, "approx model empty"
    assert asset["approx_self_cc_ranges_flat"], "approx self_cc gate empty"

    if all_ok:
        print("\nALL FIELDS BIT-EQUIVALENT to panda.cuh / Robots.hh")
        return 0
    print("\nMISMATCH DETECTED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
