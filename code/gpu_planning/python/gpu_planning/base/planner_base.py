# -*- coding: utf-8 -*-
"""Planner plugin base class and shared request/result types.

Slim, self-contained version of the production ``planner_base``: the planner
zoo is the six substrate-consistent planners from the paper (all GPU,
joint-space start→goal), with no hybrid production types and no
Cartesian/IK path (the open-source package ships no IK solver).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..substrate.planning_scene import PlanningScene


class PlannerType(Enum):
    """The substrate-consistent planners sharing the GPU collision base."""
    PRRTC = "prrtc"        # sampling: GPU-parallel RRT-Connect
    MITSTAR = "mitstar"    # anytime sampling: MIT* (asymptotically optimal)
    MHASTAR = "mhastar"    # search: Multi-Heuristic A*
    WPASE = "wpase"        # search: weighted Parallel A* for Slow Expansions
    STOMP = "stomp"        # optimization, gradient-free
    CHOMP = "chomp"        # optimization, gradient-based


class PlannerCapability(Enum):
    """Capability flags that planners can advertise."""
    JOINT_GOAL = "joint_goal"               # Supports joint space target
    WARM_START = "warm_start"               # Supports warm-start replanning
    COLLISION_CHECK = "collision_check"     # Built-in collision checking
    TIME_OPTIMAL = "time_optimal"           # Time-optimal trajectories
    VELOCITY_PROFILE = "velocity_profile"   # Velocity curve control
    GPU_ACCELERATED = "gpu_accelerated"     # GPU acceleration


@dataclass
class PlanningRequest:
    """Unified joint-space planning request."""
    robot_id: str
    start_joint_state: np.ndarray
    target_joint_state: Optional[np.ndarray] = None      # Joint space goal

    # Velocity/acceleration constraints
    velocity_scaling: float = 1.0
    acceleration_scaling: float = 1.0
    max_velocity: Optional[np.ndarray] = None            # Per-joint velocity limits
    max_acceleration: Optional[np.ndarray] = None        # Per-joint acceleration limits

    # Planning options
    timeout: float = 10.0
    collision_check: bool = True

    # Extended parameters (planner-specific)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Coerce list inputs to arrays."""
        if isinstance(self.start_joint_state, list):
            self.start_joint_state = np.array(self.start_joint_state)
        if isinstance(self.target_joint_state, list):
            self.target_joint_state = np.array(self.target_joint_state)
        if isinstance(self.max_velocity, list):
            self.max_velocity = np.array(self.max_velocity)
        if isinstance(self.max_acceleration, list):
            self.max_acceleration = np.array(self.max_acceleration)


@dataclass
class PlanningResult:
    """Unified planning result: trajectory data plus status."""
    success: bool
    trajectory: Optional[np.ndarray] = None              # Shape: (N, num_joints)
    velocities: Optional[np.ndarray] = None              # Shape: (N, num_joints)
    accelerations: Optional[np.ndarray] = None           # Shape: (N, num_joints)
    dt: float = 0.02                                     # Time step between waypoints
    total_time: float = 0.0                              # Total trajectory duration
    solve_time: float = 0.0                              # Planning computation time

    # Status information
    status: str = ""
    error_message: str = ""

    # Extended information (planner-specific)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_waypoints(self) -> int:
        if self.trajectory is not None:
            return len(self.trajectory)
        return 0

    @property
    def num_joints(self) -> int:
        if self.trajectory is not None and len(self.trajectory) > 0:
            return self.trajectory.shape[1]
        return 0


class PlannerBase(ABC):
    """Abstract base class for all substrate planner plugins.

    Implementations must provide ``initialize()``, ``plan()`` and
    ``get_capabilities()``. ``replan()``, ``update_world()`` and ``shutdown()``
    are optional overrides.
    """

    def __init__(self, planner_type: PlannerType, config: Dict[str, Any]):
        self._planner_type = planner_type
        self._config = config
        self._is_initialized = False
        self._robot_id: Optional[str] = None

    @property
    def planner_type(self) -> PlannerType:
        return self._planner_type

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def robot_id(self) -> Optional[str]:
        return self._robot_id

    @abstractmethod
    def initialize(self, scene: "PlanningScene", **kwargs) -> bool:
        """Initialize the planner with a :class:`PlanningScene`."""
        pass

    @abstractmethod
    def plan(self, request: PlanningRequest) -> PlanningResult:
        """Execute planning and return a trajectory result."""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[PlannerCapability]:
        """Return the list of supported capabilities."""
        pass

    def replan(self, request: PlanningRequest) -> PlanningResult:
        """Warm-start replanning (default: calls plan)."""
        return self.plan(request)

    def update_world(self, obstacles: List[Dict[str, Any]]) -> bool:
        """Update collision world."""
        return True

    def shutdown(self) -> None:
        """Clean up resources."""
        self._is_initialized = False

    def supports(self, capability: PlannerCapability) -> bool:
        """Check if planner supports a capability."""
        return capability in self.get_capabilities()
