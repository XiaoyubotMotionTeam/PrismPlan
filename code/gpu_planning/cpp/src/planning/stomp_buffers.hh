#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// Pre-allocated GPU buffers for STOMP solve calls.
// Follows the SolverBuffers pattern used by MITStarBuffers:
//   static create() allocates every device array once and sets owns_memory=true;
//   destroy() guards on owns_memory, frees, and clears the flag.
//
// All sizes are fixed by STOMP_settings + RobotModel at create() time and stay
// constant across solve calls, so the plugin can allocate once and reuse.
//
// Shape symbols (see STOMP_settings):
//   Bt = num_batch                 (PTP = 1)
//   D  = n_dof                     (from RobotModel, NOT settings.num_dimensions)
//   T  = num_timesteps
//   Rn = num_rollouts_new
//   Ro = num_rollouts_old
//   R  = Rn + Ro + 1               (new + reused + previous-optimum slot)
//
// Slot layout inside the R axis (mirrors stomp_core._StompSolver):
//   [0, Rn)        freshly sampled rollouts (costs recomputed each iteration)
//   [Rn, R-1)      reused top-k rollouts (costs carried from previous iteration)
//   [R-1]          previous optimum (noise == 0)
struct STOMPBuffers {
    bool owns_memory = false;

    // --- Control-cost Cholesky factor L (lower-triangular, T*T) ---
    // R_inv = L L^T; device sampling draws z~N(0,I) then noise = L z.
    float* d_L = nullptr;

    // --- Update-smoothing projection matrix M (row-major, T*T) ---
    // Kalakrishnan STOMP: candidate = optimum + M @ (weighted noise average).
    float* d_M = nullptr;

    // --- Raw control-cost quadratic form R (row-major, T*T) ---
    // Per-rollout smoothness cost = theta^T R theta, the SAME R that L and M
    // derive from (5-point acceleration metric).
    float* d_R = nullptr;

    // --- Current optimum + candidate trajectories (Bt*D*T) ---
    float* d_parameters_optimized = nullptr;   // working optimum (updated every iteration)
    float* d_candidate = nullptr;              // proposed update after M projection
    float* d_delta = nullptr;                  // raw per-(d,t) weighted noise average, pre-M
    float* d_best_optimum = nullptr;           // lowest-cost VALID optimum seen (the returned path)

    // --- Rollout population, shape (Bt*R*D*T) ---
    float* d_stored_rollouts = nullptr;        // clamped rollouts (opt + noise)
    float* d_noise = nullptr;                  // rollout - opt (endpoints pinned to 0)
    float* d_control_costs = nullptr;          // smoothness cost per element
    float* d_state_costs = nullptr;            // collision cost, D-broadcast
    float* d_total_costs = nullptr;            // state + control per element

    // --- Previous-iteration snapshots for aliasing-free top-k reuse ---
    // Copied from the d_* arrays above before P1/P1b overwrite them, so the
    // gather (arbitrary source r -> reuse slot) never reads a clobbered slot.
    float* d_prev_stored_rollouts = nullptr;   // (Bt*R*D*T)
    float* d_prev_noise = nullptr;             // (Bt*R*D*T)
    float* d_prev_control_costs = nullptr;     // (Bt*R*D*T)
    float* d_prev_state_costs = nullptr;       // (Bt*R*D*T)
    float* d_prev_total_cost = nullptr;        // (Bt*R) scalar per rollout (top-k key)

    // --- Per-rollout scalar total cost (Bt*R), the top-k ranking key ---
    float* d_total_cost = nullptr;

    // --- Local per-(d,t) rollout weights, shape (Bt*D*T*R) ---
    float* d_probabilities = nullptr;

    // --- Cached optimum cost breakdown ---
    float* d_parameters_state_costs = nullptr;    // (Bt*D*T) elementwise collision cost
    float* d_parameters_control_costs = nullptr;  // (Bt*D*T) elementwise smoothness cost
    float* d_parameters_state_cost = nullptr;     // (Bt) scalar collision cost
    float* d_parameters_control_cost = nullptr;   // (Bt) scalar smoothness cost
    float* d_parameters_total_cost = nullptr;     // (Bt) scalar total cost

    // --- Candidate cost breakdown ---
    float* d_candidate_state_costs = nullptr;     // (Bt*D*T)
    float* d_candidate_control_costs = nullptr;   // (Bt*D*T)
    float* d_candidate_total_cost = nullptr;      // (Bt) scalar

    // --- Top-k reuse indices (into the previous R set), size R ---
    int* d_topk_idx = nullptr;

    // --- Endpoints (D each) ---
    float* d_start = nullptr;
    float* d_goal = nullptr;

    // --- Per-(rollout,dim,timestep) RNG states, size Rn*D*T ---
    curandState* d_rng_states = nullptr;

    // --- Binary validity flag written by the densified collision gate (Bt) ---
    int* d_collision = nullptr;

    // --- Pinned host mirrors read once per iteration for loop control ---
    int*   h_valid = nullptr;       // 1 => optimum collision-free (Bt)
    float* h_total_cost = nullptr;  // optimum total cost (Bt)

    // Recorded parameters
    int n_dof = 0;
    int num_timesteps = 0;
    int num_rollouts_new = 0;
    int num_rollouts_old = 0;
    int num_rollouts_all = 0;   // Rn + Ro + 1
    int num_batch = 0;

    // Allocate all GPU buffers.
    static STOMPBuffers create(
        int n_dof, int num_timesteps,
        int num_rollouts_new, int num_rollouts_old,
        int num_batch = 1);

    // Free all GPU buffers. Safe to call multiple times.
    void destroy();
};
