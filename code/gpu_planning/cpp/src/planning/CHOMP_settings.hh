#pragma once

#include <vector>

#include "pRRTC_settings.hh"

// CHOMP (Ratliff 2009) settings — GPU gradient-based trajectory optimizer.
//
// This is the OPTIMIZATION-paradigm representative that is GRADIENT-BASED,
// contrasting with STOMP (derivative-free / sampled rollouts) on the SAME
// shared per-sphere collision-kinematics substrate. Both evaluate the identical
// world/self hinge cost (scene_sum_{esdf,obb}_hinge_runtime + self_min_hinge_
// runtime over fk_runtime); CHOMP additionally consumes the NUMERICAL GRADIENT
// of that exact same cost (central finite differences on the substrate), so its
// collision signal can never diverge from the shared boolean anchor used by the
// sampling/search planners.
//
// `num_dimensions` is intentionally omitted: the CUDA solver takes DOF from
// RobotModel.n_dof so it can never disagree with the active robot.
// `time_limit_ms` / `collision_margin` are inherited from pRRTC_settings.
struct CHOMP_settings : pRRTC_settings {
    // --- Trajectory discretisation ---
    int num_timesteps = 51;            // optimised geometric waypoint count (T)

    // --- Iteration control ---
    int num_iterations = 100;          // gradient descent needs more steps than STOMP
    int num_iterations_after_valid = 5;  // extra polishing iterations once collision-free

    // --- Update / metric ---
    // Covariant CHOMP step: theta <- theta - step_size * A^{-1} grad, where the
    // smoothness metric A is the SAME 5-point acceleration matrix R that STOMP
    // uses (see chomp_precompute). step_size folds in CHOMP's 1/eta learning rate.
    float step_size = 0.3f;
    float smoothness_weight = 1.0f;    // lambda on the smoothness gradient (R theta)
    float collision_cost_weight = 20.0f;  // matches STOMP so cost scales are comparable

    // --- Finite-difference obstacle gradient ---
    // Joint-space perturbation for the central difference of the substrate cost.
    float fd_epsilon = 1.0e-3f;

    // --- Smooth collision activation bands (meters) ---
    float world_collision_margin = 0.02f;  // world (ESDF/OBB) hinge band
    float self_collision_margin = 0.01f;    // self-collision hinge band

    // --- Cost-matrix time step (for R; != output dt) ---
    float delta_t = 0.1f;

    // --- Convergence / validity ---
    float convergence_eps = 1.0e-4f;   // flat-cost streak threshold for early stop
    float collision_free_tol = 1.0e-6f;

    // --- Validity-gate densification ---
    // Joint-space L2 step for sub-waypoint collision interpolation in the binary
    // validity gate (mirrors STOMP / MoveIt2 COL_CHECK_DISTANCE). <=0 disables.
    float collision_check_step = 0.05f;
};
