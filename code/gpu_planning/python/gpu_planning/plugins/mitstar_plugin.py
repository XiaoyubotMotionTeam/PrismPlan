# -*- coding: utf-8 -*-
"""MIT* — anytime asymptotically-optimal sampling planner.

Batch-informed almost-sure optimal sampling on the shared GPU substrate.
Joint-space start→goal only. Slim sibling of the production plugin: no
external discrepancy diagnostics, no path-revalidation pass.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..base.planner_base import PlannerType, PlanningRequest, PlanningResult
from ._substrate_planner_base import SubstratePlannerBase

logger = logging.getLogger(__name__)


class MITStarPlanner(SubstratePlannerBase):
    """Anytime asymptotically-optimal GPU sampling planner (MIT*)."""

    _DEFAULT_YAML = "config/planners/mitstar.yaml"

    def __init__(self, config=None, config_path=None):
        super().__init__(PlannerType.MITSTAR, config=config,
                         config_path=config_path)

    def _init_buffers(self) -> None:
        import prrtc
        rm = self._robot_model
        self._solver_bufs = prrtc.MITStarBuffers.create(
            n_dof=rm.n_dof,
            batch_size=self._config.get("mitstar_batch_size", 64),
            max_nodes=self._config.get("mitstar_max_nodes", 50000),
            max_edges_per_node=self._config.get("mitstar_max_edges_per_node", 32),
            max_neighbors=self._config.get("mitstar_max_neighbors", 32),
            m_reverse_eval=self._config.get("mitstar_m_reverse_eval", 128),
            m_forward_eval=self._config.get("mitstar_m_forward_eval", 512),
            enable_clearance=self._config.get("mitstar_clearance_weight", 0.0) > 0,
        )

    def _build_settings(self):
        import prrtc
        settings = prrtc.MITStarSettings()
        settings.batch_size = self._config.get("mitstar_batch_size", 64)
        settings.time_limit_ms = self._config.get("mitstar_time_limit_ms", 1000.0)
        settings.max_neighbors = self._config.get("mitstar_max_neighbors", 32)
        settings.m_forward_eval = self._config.get("mitstar_m_forward_eval", 512)
        settings.m_reverse_eval = self._config.get("mitstar_m_reverse_eval", 128)
        settings.valid_segment_length = self._config.get(
            "mitstar_valid_segment_length", 0.025)
        settings.max_nodes = self._config.get("mitstar_max_nodes", 50000)
        settings.max_edges_per_node = self._config.get(
            "mitstar_max_edges_per_node", 32)
        for key in ("mitstar_eta_knn", "mitstar_gamma_rgg", "mitstar_use_eis",
                    "mitstar_adaptive_batch", "mitstar_min_batch_size",
                    "mitstar_initial_sparse_factor",
                    "mitstar_initial_suboptimality",
                    "mitstar_collision_margin",
                    "mitstar_shortcut_collision_margin",
                    "mitstar_early_exit_ms", "mitstar_clearance_weight",
                    "mitstar_clearance_epsilon",
                    "mitstar_max_clearance_penalty", "mitstar_shortcut_path"):
            if key in self._config:
                setattr(settings, key.removeprefix("mitstar_"), self._config[key])
        return settings

    def plan(self, request: PlanningRequest) -> PlanningResult:
        err = self._guard_request(request)
        if err is not None:
            return err

        import prrtc
        start_time = time.time()
        start = np.asarray(request.start_joint_state, dtype=np.float32)
        goal = np.asarray(request.target_joint_state, dtype=np.float32)
        goals = goal.reshape(1, -1)
        settings = self._build_settings()

        try:
            with self._solve_context() as scene_coll:
                if scene_coll is None:
                    return PlanningResult(
                        success=False, status="PlanningError",
                        error_message="MIT* requires scene collision data "
                                      "(the world has no obstacles)",
                        solve_time=time.time() - start_time)

                flags = self._preflight(scene_coll, np.vstack([start, goal]))
                if bool(flags[0]):
                    return PlanningResult(
                        success=False, status="StartInCollision",
                        error_message="MIT* start configuration in collision",
                        solve_time=time.time() - start_time)
                if bool(flags[1]):
                    return PlanningResult(
                        success=False, status="GoalInCollision",
                        error_message="MIT* goal configuration in collision",
                        solve_time=time.time() - start_time)

                result = prrtc.mitstar_solve_scene(
                    start, goals, scene_coll, settings,
                    self._robot_model, bufs=self._solver_bufs)
        except Exception as e:  # noqa: BLE001
            logger.exception("[MIT*] solver raised")
            return PlanningResult(
                success=False, status="PlanningError",
                error_message=f"MIT* planning failed: {e}",
                solve_time=time.time() - start_time)

        solve_time = time.time() - start_time
        if not result.solved:
            logger.warning("[MIT*] no solution: nodes=%s batches=%s wall=%.1fms",
                           result.total_nodes, result.total_batches,
                           result.wall_time_ms)
            return PlanningResult(
                success=False, status="PlanningFailed",
                error_message="MIT* could not find a solution",
                solve_time=solve_time,
                extra_info={"solver": "mitstar",
                            "total_nodes": result.total_nodes,
                            "total_batches": result.total_batches,
                            "wall_time_ms": result.wall_time_ms})

        path = np.asarray(result.path, dtype=np.float32)
        logger.info("[MIT*] solved: cost=%.4f nodes=%s batches=%s waypoints=%d",
                    result.cost, result.total_nodes, result.total_batches,
                    len(path))
        return self._finalize(
            path, request, solve_time,
            extra_info={"solver": "mitstar", "cost": result.cost,
                        "wall_time_ms": result.wall_time_ms,
                        "total_nodes": result.total_nodes,
                        "total_batches": result.total_batches,
                        "cost_history": list(result.cost_history),
                        "time_history_ms": list(result.time_history_ms),
                        "geometric_waypoints": len(path)})
