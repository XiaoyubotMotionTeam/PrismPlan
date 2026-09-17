#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Pre-allocated GPU buffers for CHOMP solve calls.
// Follows the STOMPBuffers pattern: static create() allocates every device
// array once and sets owns_memory=true; destroy() guards on owns_memory, frees,
// and clears the flag. All sizes are fixed by CHOMP_settings + RobotModel at
// create() time and stay constant across solve calls.
//
// CHOMP is far leaner than STOMP: no rollout population, no RNG states, no
// top-k reuse, no sampling covariance L, no projection M. It keeps only the
// single working trajectory, its gradient, the smoothness metric R and its
// inverse A^{-1}, and scalar cost/validity scratch.
//
// Shape symbols:
//   D = n_dof         (from RobotModel, NOT settings)
//   T = num_timesteps
struct CHOMPBuffers {
    bool owns_memory = false;

    // --- Smoothness quadratic form R (row-major, T*T) ---
    // 5-point acceleration control-cost matrix; smoothness gradient = R theta.
    // Identical metric to STOMP's d_R (see chomp_precompute / stomp_precompute).
    float* d_R = nullptr;

    // --- Covariant preconditioner A^{-1} = (R + eps I)^{-1} (row-major, T*T) ---
    // CHOMP step: theta <- theta - step_size * A^{-1} grad.
    float* d_Ainv = nullptr;

    // --- Working + best trajectories (D*T) ---
    float* d_theta = nullptr;          // current trajectory (updated every iteration)
    float* d_best  = nullptr;          // lowest-cost VALID trajectory seen (returned)

    // --- Obstacle gradient (D*T) ---
    // grad[d,t] = collision_cost_weight * d(hinge cost)/d(q_{t,d}), central FD.
    float* d_obs_grad = nullptr;

    // --- Cost breakdown (mirrors STOMP naming for a comparable cost signal) ---
    float* d_state_costs   = nullptr;  // (D*T) per-(d,t) obstacle cost, D-broadcast
    float* d_control_costs = nullptr;  // (D*T) per-(d,t) smoothness cost
    float* d_state_cost    = nullptr;  // (1) scalar obstacle cost
    float* d_control_cost  = nullptr;  // (1) scalar smoothness cost
    float* d_total_cost    = nullptr;  // (1) scalar total cost

    // --- Endpoints (D each) ---
    float* d_start = nullptr;
    float* d_goal  = nullptr;

    // --- Binary validity flag written by the densified collision gate ---
    int* d_collision = nullptr;

    // --- Pinned host mirrors read once per iteration for loop control ---
    int*   h_valid = nullptr;       // 1 => trajectory collision-free
    float* h_total_cost = nullptr;  // current total cost

    // Recorded parameters
    int n_dof = 0;
    int num_timesteps = 0;

    // Allocate all GPU buffers.
    static CHOMPBuffers create(int n_dof, int num_timesteps);

    // Free all GPU buffers. Safe to call multiple times.
    void destroy();
};
