"""CPU-only checks for the equal-budget contract; no GPU execution is mocked
as a performance result. The fake native settings expose only the budget field.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gpu_planning import (  # noqa: E402
    CHOMPPlanner, MHAStarPlanner, MITStarPlanner, PRRTCPlanner,
    STOMPPlanner, WPASEPlanner, PlanningResult,
)
from gpu_planning.benchmark import load_vamp_problems  # noqa: E402
from gpu_planning.benchmark import runner  # noqa: E402

_PLANNERS = [PRRTCPlanner, MITStarPlanner, MHAStarPlanner,
             WPASEPlanner, STOMPPlanner, CHOMPPlanner]


@pytest.fixture
def native_settings(monkeypatch):
    def settings():
        return SimpleNamespace(time_limit_ms=-1.0)

    monkeypatch.setitem(sys.modules, "prrtc", SimpleNamespace(**{
        name: settings for name in (
            "Settings", "MITStarSettings", "MHAStarSettings",
            "WPASESettings", "STOMPSettings", "CHOMPSettings")
    }))


@pytest.mark.parametrize("planner_class", _PLANNERS)
@pytest.mark.parametrize("empty_config", [False, True])
def test_budget_reaches_solver_settings(planner_class, empty_config,
                                       native_settings):
    planner = planner_class(config={} if empty_config else None)
    planner.set_time_budget_ms(2000)
    assert planner._build_settings().time_limit_ms == 2000.0
    # A subsequent override must reach the next solve's settings as well.
    planner.set_time_budget_ms(1000)
    assert planner._build_settings().time_limit_ms == 1000.0


@pytest.fixture
def fake_scene(monkeypatch):
    scene = SimpleNamespace(update_world=lambda obstacles: None)
    monkeypatch.setattr(runner.PlanningScene, "from_robot_asset",
                        lambda *args, **kwargs: scene)
    return scene


@pytest.mark.parametrize("budget_ms", [None, 2000.0])
def test_runner_sets_budget_before_initialization_and_records_it(
        budget_ms, native_settings, fake_scene, monkeypatch, tmp_path):
    initialized = []
    solved = []
    closed = []

    def expected(planner):
        if budget_ms is not None:
            return budget_ms
        return 4000.0 if isinstance(planner, PRRTCPlanner) else 1000.0

    def initialize(planner, scene, **kwargs):
        assert scene is fake_scene
        assert planner._build_settings().time_limit_ms == expected(planner)
        initialized.append(type(planner))
        return True

    def plan(planner, request):
        assert planner._build_settings().time_limit_ms == expected(planner)
        solved.append(type(planner))
        return PlanningResult(success=True, solve_time=0.001)

    for cls in _PLANNERS:
        monkeypatch.setattr(cls, "initialize", initialize)
        monkeypatch.setattr(cls, "plan", plan)
        monkeypatch.setattr(cls, "shutdown", lambda p: closed.append(type(p)))

    problems = load_vamp_problems(str(
        _ROOT / "examples/data/sample_mbm_problems.json"))[:1]
    records, summary = runner.run_benchmark(
        [(cls.__name__, cls) for cls in _PLANNERS], problems,
        device="cpu", budget_ms=budget_ms, out_dir=str(tmp_path))
    assert initialized == solved == closed == _PLANNERS
    assert len(records) == 6
    assert summary["config"]["budget_ms"] == budget_ms
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["config"]["budget_ms"] == budget_ms


def test_equal_budget_rejects_unsupported_planner(fake_scene):
    class UnsupportedPlanner:
        def initialize(self, *args, **kwargs):
            pytest.fail("unsupported planner must fail before initialization")

    with pytest.raises(TypeError, match="no set_time_budget_ms"):
        runner.run_benchmark([("unsupported", UnsupportedPlanner)], [],
                             device="cpu", budget_ms=2000)
