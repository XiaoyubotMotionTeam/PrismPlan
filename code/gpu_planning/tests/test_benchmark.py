# -*- coding: utf-8 -*-
"""Tests for the benchmark harness that need no GPU / no ``prrtc``.

Covers the pure-Python pieces: the Euler->quaternion helper, the conservative
OBB over-approximation (every primitive must be fully enclosed by its OBB), the
VAMP sample loader, and the metric aggregation. Runs under pytest, or directly:

    python3 tests/test_benchmark.py
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gpu_planning.benchmark import (  # noqa: E402
    BenchObstacle,
    aggregate,
    euler_xyz_to_quat_wxyz,
    load_vamp_problems,
    obstacle_to_obb,
)
from gpu_planning.benchmark.runner import TrialRecord  # noqa: E402

_SAMPLE = _ROOT / "examples" / "data" / "sample_mbm_problems.json"


# --------------------------------------------------------------------------- #
# Euler -> quaternion
# --------------------------------------------------------------------------- #
def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_euler_identity():
    q = euler_xyz_to_quat_wxyz([0, 0, 0])
    assert np.allclose(q, [1, 0, 0, 0])


def test_euler_is_unit_and_matches_Rz_Ry_Rx():
    # R = Rz @ Ry @ Rx applied to a vector; verify against explicit matrices.
    ex, ey, ez = 0.3, -0.7, 1.1
    q = euler_xyz_to_quat_wxyz([ex, ey, ez])
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9

    def Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    R_expected = Rz(ez) @ Ry(ey) @ Rx(ex)
    assert np.allclose(_quat_to_R(q), R_expected, atol=1e-9)


# --------------------------------------------------------------------------- #
# Conservative OBB over-approximation
# --------------------------------------------------------------------------- #
def _obb_contains_points(obb, pts):
    """True if every world point lies inside the OBB (with tiny tolerance)."""
    pos = np.asarray(obb["position"])
    half = np.asarray(obb["dims"]) / 2.0
    R = _quat_to_R(np.asarray(obb["quaternion"]))
    local = (pts - pos) @ R  # world->local (R orthonormal)
    return np.all(np.abs(local) <= half + 1e-9)


def test_box_obb_is_exact():
    obs = BenchObstacle(kind="box", name="b",
                        position=np.array([0.1, 0.2, 0.3]),
                        half_extents=np.array([0.2, 0.3, 0.4]))
    obb = obstacle_to_obb(obs)
    assert np.allclose(obb["dims"], [0.4, 0.6, 0.8])


def test_sphere_obb_encloses():
    r = 0.15
    obs = BenchObstacle(kind="sphere", name="s",
                        position=np.array([0.5, -0.2, 0.3]), radius=r)
    obb = obstacle_to_obb(obs)
    assert np.allclose(obb["dims"], [2 * r, 2 * r, 2 * r])
    # sample the sphere surface; all must be inside the OBB
    u = np.linspace(0, np.pi, 8)
    v = np.linspace(0, 2 * np.pi, 8)
    pts = []
    c = np.asarray(obs.position)
    for a in u:
        for b in v:
            pts.append(c + r * np.array(
                [np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]))
    assert _obb_contains_points(obb, np.array(pts))


def test_cylinder_obb_encloses_rotated():
    r, length = 0.05, 0.5
    q = euler_xyz_to_quat_wxyz([0.4, 0.2, -0.6])
    center = np.array([0.3, 0.1, 0.4])
    obs = BenchObstacle(kind="cylinder", name="c", position=center,
                        quat_wxyz=q, radius=r, length=length)
    obb = obstacle_to_obb(obs)
    assert np.allclose(obb["dims"], [2 * r, 2 * r, length])
    # sample the cylinder surface (axis = local +z), transform to world, check
    R = _quat_to_R(q)
    pts = []
    for zc in np.linspace(-length / 2, length / 2, 5):
        for th in np.linspace(0, 2 * np.pi, 12):
            local = np.array([r * np.cos(th), r * np.sin(th), zc])
            pts.append(center + R @ local)
    assert _obb_contains_points(obb, np.array(pts))


# --------------------------------------------------------------------------- #
# VAMP sample loader
# --------------------------------------------------------------------------- #
def test_sample_loader_parses():
    probs = load_vamp_problems(str(_SAMPLE))
    assert len(probs) == 3
    scenes = {p.scene for p in probs}
    assert scenes == {"box", "cage"}
    for p in probs:
        assert p.start.shape == (7,)
        assert p.goal.shape == (7,)
    cage = [p for p in probs if p.scene == "cage"][0]
    kinds = sorted(o.kind for o in cage.obstacles)
    assert kinds == ["box", "cylinder", "sphere"]


def test_sample_loader_scene_filter_and_cap():
    probs = load_vamp_problems(str(_SAMPLE), scenes=["box"], max_per_scene=1)
    assert len(probs) == 1
    assert probs[0].scene == "box"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _rec(planner, scene, ok, solve_ms, cost):
    return TrialRecord(
        planner=planner, scene=scene, problem_index=0, trial=0,
        success=ok, status="ok" if ok else "fail", solve_ms=solve_ms,
        gpu_ms=float("nan"), waypoints=10, total_time=1.0, cost=cost,
        expansions=1, edges_evaluated=1)


def test_aggregate_success_rate_and_stats():
    recs = [
        _rec("A", "box", True, 10.0, 1.0),
        _rec("A", "box", False, 20.0, float("nan")),
        _rec("A", "cage", True, 30.0, 3.0),
        _rec("B", "box", True, 5.0, 2.0),
    ]
    summ = aggregate(recs)
    assert summ["overall"]["attempts"] == 4
    assert summ["overall"]["successes"] == 3
    a = summ["by_planner"]["A"]["all"]
    assert a["attempts"] == 3 and a["successes"] == 2
    assert abs(a["success_rate"] - 2 / 3) < 1e-9
    # solve_ms mean over successful A trials = (10 + 30) / 2 = 20
    assert abs(a["solve_ms"]["mean"] - 20.0) < 1e-9
    # cost mean over successful A trials with finite cost = (1 + 3) / 2 = 2
    assert abs(a["cost"]["mean"] - 2.0) < 1e-9
    assert "box" in summ["by_planner"]["A"]["by_scene"]


def _main():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
