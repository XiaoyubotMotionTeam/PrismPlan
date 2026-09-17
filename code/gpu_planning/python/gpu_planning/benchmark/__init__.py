# -*- coding: utf-8 -*-
"""Benchmark harness for the substrate-consistent planners.

Loads MotionBenchMaker problems (VAMP ``.pkl``/``.json`` or the Robometrics
datasets), normalises their box/sphere/cylinder geometry into conservative
OBBs, and runs every planner over them on the shared GPU collision substrate,
emitting comparable per-trial and aggregate metrics.
"""

from __future__ import annotations

from .problem import (
    BenchObstacle,
    BenchProblem,
    euler_xyz_to_quat_wxyz,
    obstacle_to_obb,
    obstacles_to_obbs,
)
from .loaders import load_robometrics_problems, load_vamp_problems
from .runner import TrialRecord, aggregate, run_benchmark

__all__ = [
    "BenchObstacle",
    "BenchProblem",
    "euler_xyz_to_quat_wxyz",
    "obstacle_to_obb",
    "obstacles_to_obbs",
    "load_vamp_problems",
    "load_robometrics_problems",
    "TrialRecord",
    "aggregate",
    "run_benchmark",
]
