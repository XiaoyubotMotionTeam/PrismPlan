#pragma once

#include <vector>

#include "pRRTC_settings.hh"

// STOMP (Stochastic Trajectory Optimization for Motion Planning) settings.
//
// Ported from the Python `stomp_core._StompSolver` config (prrtc.yaml `stomp:`
// block).  `num_dimensions` is intentionally omitted here — the CUDA solver
// takes the DOF from `RobotModel.n_dof` so it can never disagree with the
// active robot.  `time_limit_ms` / `collision_margin` are inherited from the
// base `pRRTC_settings`.
//
// Passed to kernels as scalar arguments (NOT __constant__), matching the
// MITStar_settings convention.
struct STOMP_settings : pRRTC_settings {
    // --- Problem / batch shape ---
    int num_batch = 1;                 // PTP = 1; kept as a dim for future multi-start

    // --- Trajectory discretisation ---
    int num_timesteps = 51;            // optimised geometric waypoint count (T)

    // --- Iteration control ---
    int num_iterations = 20;
    int num_iterations_after_valid = 3;  // extra polishing iterations once collision-free

    // --- Rollout population ---
    int num_rollouts_new = 30;         // freshly sampled rollouts per iteration
    int num_rollouts_old = 20;         // top-k lowest-cost rollouts reused across iterations

    // --- Cost model ---
    float delta_t = 0.1f;              // dt for the internal acceleration R matrix (!= output_dt)
    float control_cost_weight = 0.1f;
    float collision_cost_weight = 20.0f;
    float exponentiated_cost_sensitivity = 2.0f;  // h in exp(-h*(cost-min)/(max-min))

    // --- Smooth collision activation bands (meters) ---
    float world_collision_margin = 0.02f;  // world (ESDF) hinge band
    float self_collision_margin = 0.01f;   // self-collision hinge band

    // --- Convergence / validity ---
    float convergence_eps = 1.0e-4f;   // flat-cost streak threshold for early stop
    float collision_free_tol = 1.0e-6f;  // collision cost <= this => "valid"

    // --- Noise sampling ---
    // Global multiplier on the exploration noise amplitude. The sampling
    // covariance L Lᵀ = R⁻¹ is rescaled so its max entry == 1/T (see
    // compute_cholesky_L); this scale multiplies the resulting Cholesky factor
    // L, so noise stddev scales linearly with noise_scale. 1.0 = unchanged
    // baseline; >1 widens the Cartesian noise cloud (more exploration, helps
    // escape shallow collisions); <1 narrows it. Config key: stomp_noise_scale.
    float noise_scale = 1.0f;

    // --- Validity-gate densification ---
    // Joint-space L2 step for sub-waypoint collision interpolation in the
    // binary validity gate (mirrors MoveIt2 COL_CHECK_DISTANCE). <=0 disables.
    float collision_check_step = 0.05f;
};
