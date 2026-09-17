#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "halton_state.hh"

// Pre-allocated GPU buffers for pRRTC solve calls.
// Follows RobotModelDevice pattern: static create() / explicit destroy().
//
// Thread-safety: caller must hold PRRTCPlugin._solver_lock (only one solve
// at a time), so a single SolverBuffers instance suffices per process.
//
// All buffer sizes are determined by settings + RobotModel at create() time
// and remain constant across solve calls.
struct SolverBuffers {
    bool owns_memory = false;

    // Tree storage (2 trees: forward + backward)
    float* d_nodes_ptrs[2] = {nullptr, nullptr};
    int*   d_parents_ptrs[2] = {nullptr, nullptr};
    float* d_radii_ptrs[2] = {nullptr, nullptr};

    // Device arrays holding pointers to tree storage (kernel reads these)
    float** d_nodes = nullptr;
    int**   d_parents = nullptr;
    float** d_radii = nullptr;

    // RNG states
    curandState* rng_states = nullptr;
    HaltonState_runtime* halton_states = nullptr;

    // Mesh collision joint transforms (conditional, nullptr if disabled)
    float* d_mesh_transforms = nullptr;

    // Host-pinned solved flag
    int* h_solved = nullptr;

    // Recorded parameters (for debug/validation)
    int max_samples = 0;
    int num_new_configs = 0;
    int granularity = 0;
    int n_dof = 0;
    int n_joints = 0;
    bool mesh_enabled = false;

    // Allocate all GPU buffers. Returns a SolverBuffers that owns_memory.
    static SolverBuffers create(
        int max_samples, int n_dof,
        int num_new_configs, int granularity,
        int n_joints, bool enable_mesh);

    // Free all GPU buffers. Safe to call multiple times (idempotent).
    void destroy();
};
