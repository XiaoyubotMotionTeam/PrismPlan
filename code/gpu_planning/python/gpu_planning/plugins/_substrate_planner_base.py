# -*- coding: utf-8 -*-
"""Shared base for the six substrate-consistent planner plugins.

Every planner in this package (pRRTC, MIT*, MHA*, wPA*SE, STOMP, CHOMP) sits on
the *same* GPU substrate: one robot model (:mod:`gpu_planning.substrate.robot_model_loader`)
and one obstacle world (:class:`gpu_planning.substrate.planning_scene.PlanningScene`)
feeding the identical per-sphere batched collision kernel. This class captures
everything that is genuinely shared so each concrete plugin only has to build
its solver ``Settings`` and call its ``*_solve_scene`` entry point:

  * YAML config discovery + flattening,
  * building the ``prrtc.RobotModelBuilder`` from the robot asset,
  * handing out the scene's ``SceneCollisionDataBuilder`` under the right locks,
  * a start/goal pre-flight collision gate,
  * arc-length trapezoidal time parameterisation of a geometric path,
  * result assembly.

This is the slim, self-contained descendant of the production ``PRRTCPlugin``: no
constraints, no mesh path, no MIT* pRRTC-seeding, no external cross-check
diagnostics, and no per-call ``extra_params["solver"]`` dispatch — the six
solvers are six sibling classes instead.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base.planner_base import (
    PlannerBase,
    PlannerCapability,
    PlannerType,
    PlanningRequest,
    PlanningResult,
)
from ..substrate.robot_model_loader import build_robot_model

logger = logging.getLogger(__name__)

# Package root: .../gpu_planning/python/gpu_planning/plugins/_substrate_planner_base.py
#   parents[0]=plugins  [1]=gpu_planning  [2]=python  [3]=<repo>/code/gpu_planning
_PKG_ROOT = Path(__file__).resolve().parents[3]


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested per-solver YAML sections into prefixed flat keys.

    ``{"output_dt": 0.02, "prrtc": {"num_new_configs": 600},
       "mitstar": {"batch_size": 512}}`` becomes
    ``{"output_dt": 0.02, "num_new_configs": 600, "mitstar_batch_size": 512}``.

    The ``prrtc`` section is promoted without a prefix (its keys are the plain
    pRRTC ``Settings`` names); ``mitstar`` / ``stomp`` / ``chomp`` / ``mhastar`` /
    ``wpase`` sections are prefixed so ``plan()`` can pull them with ``<name>_``
    keys. Unknown nested sections are dropped; top-level scalars pass through.
    """
    flat: Dict[str, Any] = {}
    for key, val in config.items():
        if key in ("mitstar", "stomp", "chomp", "mhastar", "wpase") and isinstance(val, dict):
            for k, v in val.items():
                flat[f"{key}_{k}"] = v
        elif key == "prrtc" and isinstance(val, dict):
            flat.update(val)
        elif isinstance(val, dict):
            pass  # unknown section — skip
        else:
            flat[key] = val
    return flat


def _densify_path(path: np.ndarray, step: float) -> np.ndarray:
    """Linearly subdivide an ``(N, dof)`` path so adjacent waypoints are
    ``<= step`` apart in L2. Endpoints are preserved exactly. A collision-free
    path stays collision-free under linear subdivision.
    """
    p = np.asarray(path, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] < 2:
        return p.astype(np.float32)
    step = max(float(step), 1e-3)
    out = [p[0]]
    for i in range(1, p.shape[0]):
        a, b = p[i - 1], p[i]
        seg = float(np.linalg.norm(b - a))
        n_sub = int(np.ceil(seg / step))
        if n_sub <= 1:
            out.append(b)
            continue
        for k in range(1, n_sub + 1):
            t = k / n_sub
            out.append((1.0 - t) * a + t * b)
    return np.asarray(out, dtype=np.float32)


class SubstratePlannerBase(PlannerBase):
    """Common machinery for the five substrate planner plugins.

    Subclasses set a class ``_DEFAULT_YAML`` and implement :meth:`plan`
    (and optionally :meth:`_init_buffers` for pre-allocated GPU buffers).
    """

    #: pRRTC / MIT* / STOMP / CHOMP / MHA* CUDA kernels use per-process
    #: ``__device__`` global state, so ALL solves across ALL five planners
    #: are serialised on this one process-wide lock.
    _solver_lock = threading.Lock()

    #: Subclasses override with e.g. ``"config/planners/prrtc.yaml"``.
    _DEFAULT_YAML: Optional[str] = None

    def __init__(self, planner_type: PlannerType,
                 config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[str] = None):
        if config is not None:
            cfg = _flatten_config(config)
        else:
            yaml_path = config_path or self._find_default_yaml()
            if yaml_path is not None:
                import yaml
                with open(yaml_path, encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                cfg = _flatten_config(raw)
            else:
                logger.warning(
                    "%s: no YAML config found (%s) — using hardcoded defaults",
                    type(self).__name__, self._DEFAULT_YAML)
                cfg = {}
        cfg.setdefault("output_dt", 0.02)
        cfg.setdefault("default_max_velocity", 1.0)
        cfg.setdefault("default_max_acceleration", 2.0)

        super().__init__(planner_type, cfg)
        self._scene = None
        self._robot_model = None      # prrtc.RobotModelBuilder
        self._solver_bufs = None      # planner-specific, allocated in _init_buffers

    # ------------------------------------------------------------------ #
    # Config discovery
    # ------------------------------------------------------------------ #
    @classmethod
    def _find_default_yaml(cls) -> Optional[str]:
        if cls._DEFAULT_YAML is None:
            return None
        p = _PKG_ROOT / cls._DEFAULT_YAML
        return str(p) if p.is_file() else None

    # ------------------------------------------------------------------ #
    # Wall-clock budget override
    # ------------------------------------------------------------------ #
    def set_time_budget_ms(self, budget_ms: float) -> int:
        """Overwrite this planner's wall-clock budget, returning #keys written.

        The per-solver YAMLs each carry their own ``time_limit_ms``, and those
        defaults are *not* equal across planners, so a benchmark that wants an
        equal-budget comparison must say so explicitly. Call this after
        construction and before :meth:`initialize`: every ``_build_settings``
        reads ``self._config`` inside :meth:`plan`, so the new value is picked
        up on the next solve.

        Every config key ending in ``time_limit_ms`` is rewritten. If the
        loaded config carries none (e.g. the YAML was missing), the canonical
        key for this planner is inserted rather than silently doing nothing --
        a no-op here would look like an honoured budget in the results.
        """
        keys = [k for k in self._config if k.endswith("time_limit_ms")]
        if not keys:
            keys = [self._canonical_budget_key()]
        for key in keys:
            self._config[key] = float(budget_ms)
        return len(keys)

    @classmethod
    def _canonical_budget_key(cls) -> str:
        """The flattened config key this planner reads its budget from.

        Mirrors :func:`_flatten_config`: the ``prrtc`` section is promoted
        unprefixed, every other section is prefixed with its own name.
        """
        section = (Path(cls._DEFAULT_YAML).stem
                   if cls._DEFAULT_YAML else "")
        if section in ("mitstar", "stomp", "chomp", "mhastar", "wpase"):
            return f"{section}_time_limit_ms"
        return "time_limit_ms"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def initialize(self, scene, **kwargs) -> bool:
        """Build the robot model from the asset and bind the obstacle scene.

        ``scene`` must be a :class:`PlanningScene`. The robot asset path is
        taken from ``kwargs['asset_path']`` or config key ``robot_asset``
        (default: the loader's bundled panda).
        """
        from ..substrate.planning_scene import PlanningScene

        if not isinstance(scene, PlanningScene):
            logger.error("%s.initialize requires a PlanningScene, got %s",
                         type(self).__name__, type(scene).__name__)
            return False

        try:
            import prrtc  # noqa: F401
        except ImportError:
            logger.error("%s: prrtc module not found — build the pybind "
                         "module first", type(self).__name__)
            return False

        self._scene = scene
        self._robot_id = scene.robot_id

        asset_path = kwargs.get("asset_path") or self._config.get("robot_asset")
        try:
            self._robot_model = build_robot_model(asset_path)
        except Exception as e:  # noqa: BLE001
            logger.error("%s: failed to build robot model: %s",
                         type(self).__name__, e)
            return False

        try:
            self._init_buffers()
        except Exception as e:  # noqa: BLE001
            logger.warning("%s: buffer pre-allocation failed, will use "
                           "per-solve alloc: %s", type(self).__name__, e)

        self._is_initialized = True
        return True

    def _init_buffers(self) -> None:
        """Allocate reusable GPU buffers (override in subclasses that need them)."""
        return None

    def shutdown(self) -> None:
        if self._solver_bufs is not None:
            try:
                self._solver_bufs.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._solver_bufs = None
        super().shutdown()

    def get_capabilities(self) -> List[PlannerCapability]:
        return [
            PlannerCapability.JOINT_GOAL,
            PlannerCapability.COLLISION_CHECK,
            PlannerCapability.GPU_ACCELERATED,
        ]

    # ------------------------------------------------------------------ #
    # Shared solve helpers
    # ------------------------------------------------------------------ #
    @contextmanager
    def _solve_context(self):
        """Serialise the solve process-wide and hold the scene lock while the
        ``SceneCollisionDataBuilder`` and its device tensors are live.

        Yields the builder (or ``None`` when the world is empty).
        """
        with self._solver_lock:
            with self._scene._voxel_plan_lock:
                yield self._scene.build_scene_collision_data()

    def _guard_request(self, request: PlanningRequest) -> Optional[PlanningResult]:
        """Return an error result if the request can't be planned, else None."""
        if not self._is_initialized:
            return PlanningResult(
                success=False, status="NotInitialized",
                error_message=f"{type(self).__name__} not initialized")
        if request.target_joint_state is None:
            return PlanningResult(
                success=False, status="InvalidRequest",
                error_message=f"{self.planner_type.value} requires "
                              "target_joint_state")
        return None

    def _preflight(self, scene_coll, configs: np.ndarray) -> np.ndarray:
        """Batch start/goal collision check via the shared kernel.

        Returns a bool array (True = in collision), one per input config.
        """
        import prrtc
        return prrtc.check_collision_scene(
            configs=np.asarray(configs, dtype=np.float32),
            scene=scene_coll,
            robot=self._robot_model,
            collision_margin=0.0,
        )

    def _finalize(self, path: np.ndarray, request: PlanningRequest,
                  solve_time: float, extra_info: Dict[str, Any]) -> PlanningResult:
        """Time-parameterise a geometric path into a success result."""
        num_joints = int(np.asarray(request.start_joint_state).ravel().shape[0])
        trajectory, velocities, total_time = self._time_parameterize(
            np.asarray(path, dtype=np.float32), request, num_joints)
        return PlanningResult(
            success=True,
            trajectory=trajectory,
            velocities=velocities,
            dt=self._config["output_dt"],
            total_time=total_time,
            solve_time=solve_time,
            status="Success",
            extra_info=extra_info,
        )

    # ------------------------------------------------------------------ #
    # Time parameterisation (shared arc-length trapezoidal profile)
    # ------------------------------------------------------------------ #
    def _time_parameterize(
        self, path: np.ndarray, request: PlanningRequest, num_joints: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Apply a trapezoidal velocity profile and resample at uniform ``dt``.

        Returns ``(trajectory, velocities, total_time)``.
        """
        dt = self._config["output_dt"]
        vs = max(request.velocity_scaling, 0.01)
        as_ = max(request.acceleration_scaling, 0.01)
        v_max = self._config["default_max_velocity"] * vs
        a_max = self._config["default_max_acceleration"] * as_

        if request.max_velocity is not None:
            v_limits = np.asarray(request.max_velocity) * request.velocity_scaling
        else:
            v_limits = np.full(num_joints, v_max)
        if request.max_acceleration is not None:
            a_limits = np.asarray(request.max_acceleration) * request.acceleration_scaling
        else:
            a_limits = np.full(num_joints, a_max)

        n_wp = len(path)
        if n_wp < 2:
            traj = path.copy().reshape(1, -1)
            return traj, np.zeros_like(traj), 0.0

        seg_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumul = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_arc = cumul[-1]
        if total_arc < 1e-10:
            traj = path[0:1].copy()
            return traj, np.zeros_like(traj), 0.0

        # Bottleneck scalar speed/accel: tightest joint limit over all segments.
        max_scalar_v = float("inf")
        max_scalar_a = float("inf")
        for j in range(num_joints):
            for k in range(n_wp - 1):
                seg_len = seg_lengths[k]
                if seg_len < 1e-10:
                    continue
                ratio = abs(path[k + 1, j] - path[k, j]) / seg_len
                if ratio > 1e-10:
                    max_scalar_v = min(max_scalar_v, v_limits[j] / ratio)
                    max_scalar_a = min(max_scalar_a, a_limits[j] / ratio)
        if not np.isfinite(max_scalar_v) or max_scalar_v < 1e-10:
            max_scalar_v = v_max
        if not np.isfinite(max_scalar_a) or max_scalar_a < 1e-10:
            max_scalar_a = a_max

        v_s = max_scalar_v
        a_s = max_scalar_a
        t_accel = v_s / a_s
        d_accel = 0.5 * a_s * t_accel ** 2
        if 2 * d_accel <= total_arc:
            t_cruise = (total_arc - 2 * d_accel) / v_s
            total_time = 2 * t_accel + t_cruise
        else:
            t_accel = np.sqrt(total_arc / a_s)
            v_s = a_s * t_accel
            total_time = 2 * t_accel
        if total_time < dt:
            total_time = dt

        num_points = int(np.ceil(total_time / dt)) + 1
        t_arr = np.linspace(0, total_time, num_points)

        s_arr = np.empty(num_points)
        ds_arr = np.empty(num_points)
        for i, t in enumerate(t_arr):
            if t <= t_accel:
                s_arr[i] = 0.5 * a_s * t ** 2
                ds_arr[i] = a_s * t
            elif t <= total_time - t_accel:
                s_arr[i] = d_accel + v_s * (t - t_accel)
                ds_arr[i] = v_s
            else:
                t_d = t - (total_time - t_accel)
                s_arr[i] = total_arc - 0.5 * a_s * (t_accel - t_d) ** 2
                ds_arr[i] = a_s * (t_accel - t_d)
        s_arr = np.clip(s_arr, 0.0, total_arc)

        trajectory = np.zeros((num_points, num_joints))
        velocities = np.zeros((num_points, num_joints))
        for i in range(num_points):
            s = s_arr[i]
            idx = int(np.clip(np.searchsorted(cumul, s, side="right") - 1,
                              0, n_wp - 2))
            seg_len = seg_lengths[idx]
            alpha = np.clip((s - cumul[idx]) / seg_len, 0.0, 1.0) \
                if seg_len > 1e-10 else 0.0
            trajectory[i] = path[idx] + alpha * (path[idx + 1] - path[idx])
            dq_ds = (path[idx + 1] - path[idx]) / seg_len \
                if seg_len > 1e-10 else np.zeros(num_joints)
            velocities[i] = dq_ds * ds_arr[i]

        trajectory[0] = path[0]
        trajectory[-1] = path[-1]
        velocities[0] = 0.0
        velocities[-1] = 0.0
        return trajectory, velocities, total_time

    # ------------------------------------------------------------------ #
    @abstractmethod
    def plan(self, request: PlanningRequest) -> PlanningResult:
        """Solve a joint-space start→goal request. Implemented per planner."""
        raise NotImplementedError
