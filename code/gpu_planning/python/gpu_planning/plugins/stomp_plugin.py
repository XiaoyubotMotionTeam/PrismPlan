# -*- coding: utf-8 -*-
"""STOMP — Stochastic Trajectory Optimization for Motion Planning.

Gradient-free trajectory optimisation: samples noisy rollouts, scores them on
the shared GPU sum-hinge collision cost, and updates by a probability-weighted
average. Joint-space start→goal only, fully GPU-resident.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..base.planner_base import PlannerType, PlanningRequest, PlanningResult
from ._substrate_planner_base import SubstratePlannerBase

logger = logging.getLogger(__name__)


class STOMPPlanner(SubstratePlannerBase):
    """Gradient-free stochastic trajectory optimiser (STOMP)."""

    _DEFAULT_YAML = "config/planners/stomp.yaml"

    def __init__(self, config=None, config_path=None):
        super().__init__(PlannerType.STOMP, config=config,
                         config_path=config_path)

    def _init_buffers(self) -> None:
        import prrtc
        rm = self._robot_model
        self._solver_bufs = prrtc.STOMPBuffers.create(
            n_dof=rm.n_dof,
            num_timesteps=self._config.get("stomp_num_timesteps", 51),
            num_rollouts_new=self._config.get("stomp_num_rollouts_new", 30),
            num_rollouts_old=self._config.get("stomp_num_rollouts_old", 20),
            num_batch=self._config.get("stomp_num_batch", 1),
        )

    def _build_settings(self):
        import prrtc
        settings = prrtc.STOMPSettings()
        for key, val in self._config.items():
            if key.startswith("stomp_"):
                attr = key[len("stomp_"):]
                if hasattr(settings, attr):
                    setattr(settings, attr, val)
        if "collision_margin" in self._config:
            settings.collision_margin = self._config["collision_margin"]
        return settings

    def plan(self, request: PlanningRequest) -> PlanningResult:
        err = self._guard_request(request)
        if err is not None:
            return err

        import prrtc
        start_time = time.time()
        start = np.asarray(request.start_joint_state, dtype=np.float32)
        goal = np.asarray(request.target_joint_state, dtype=np.float32)
        settings = self._build_settings()

        try:
            with self._solve_context() as scene_coll:
                if scene_coll is None:
                    return PlanningResult(
                        success=False, status="PlanningError",
                        error_message="STOMP requires scene collision data "
                                      "(the world has no obstacles)",
                        solve_time=time.time() - start_time)

                flags = self._preflight(scene_coll, np.vstack([start, goal]))
                # Colliding goal is fatal; colliding start is only a warning —
                # STOMP can optimise its way out of a start penetration.
                if bool(flags[1]):
                    return PlanningResult(
                        success=False, status="GoalInCollision",
                        error_message="STOMP goal configuration in collision",
                        solve_time=time.time() - start_time)
                if bool(flags[0]):
                    logger.warning("[STOMP] start in collision; optimising out")

                result = prrtc.stomp_solve_scene(
                    start, goal, scene_coll, settings,
                    self._robot_model, bufs=self._solver_bufs)
        except Exception as e:  # noqa: BLE001
            logger.exception("[STOMP] solver raised")
            return PlanningResult(
                success=False, status="PlanningError",
                error_message=f"STOMP planning failed: {e}",
                solve_time=time.time() - start_time)

        solve_time = time.time() - start_time
        info = {"solver": "stomp",
                "iterations_run": result.iterations_run,
                "final_state_cost": result.final_state_cost,
                "final_control_cost": result.final_control_cost,
                "final_total_cost": result.final_total_cost,
                "wall_time_ms": result.wall_time_ms}
        if not result.solved:
            logger.warning("[STOMP] no solution: iters=%s state_cost=%.4g",
                           result.iterations_run, result.final_state_cost)
            return PlanningResult(
                success=False, status="PlanningFailed",
                error_message="STOMP did not reach a collision-free trajectory "
                              f"(final_state_cost={result.final_state_cost:.4g})",
                solve_time=solve_time, extra_info=info)

        path = np.asarray(result.path, dtype=np.float32)
        logger.info("[STOMP] solved: iters=%s total_cost=%.4g waypoints=%d",
                    result.iterations_run, result.final_total_cost, len(path))
        info["geometric_waypoints"] = len(path)
        return self._finalize(path, request, solve_time, extra_info=info)
