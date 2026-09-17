#pragma once

// ============================================================================
// GPU batched edge evaluation for (Ge)PA*SE-style MHA* search.
//
// One block per edge (from_cfg -> to_cfg). Each block cooperatively validates
// the straight-line interpolation between the two configs and returns
//   valid[e]     : 1 if the whole edge is collision-free, else 0
//   cost[e]      : Euclidean config-space length if valid, else +inf
//   clearance[e] : min ESDF/OBB clearance over the edge (only if requested)
//
// Reuses pRRTC's runtime FK + two-phase (approx sphere -> detailed sphere)
// scene + self collision primitives verbatim, so behaviour matches pRRTC's
// scene EXTEND edge check. ESDF/OBB (nvblox voxel + OBB) path only; mesh (BVH)
// mode has a placeholder insertion point but is not implemented here.
//
// Thread layout mirrors pRRTC exactly:
//   tid in [0,64)   thread_ind = tid % 4   batch_ind = tid / 4  (16 slots)
// Interpolation waypoints per edge may exceed BATCH_SIZE(16); we loop over
// ceil(granularity / BATCH_SIZE) substeps, packing 16 waypoints per substep.
// ============================================================================

#include "src/planning/runtime_kinematics.cuh"
#include "src/planning/robot_model.cuh"
#include "src/collision/scene_collision.cuh"
#include "src/collision/two_phase_cc.cuh"

#include <float.h>
#include <cstdint>

#ifndef BATCH_SIZE
#define BATCH_SIZE 16
#endif

namespace ppln::search {

struct EdgeEvalParams {
    int   granularity;        // interpolation points per edge (endpoint inclusive)
    float collision_margin;   // added to every sphere radius during CC
    bool  check_self;         // also run self-collision check
    bool  compute_clearance;  // also compute min ESDF/OBB clearance (forces detailed FK)
    bool  enable_mesh;        // A-scope: placeholder, must be false for now
    // Configuration-space spacing between checks. When > 0 it REPLACES
    // `granularity`: each edge gets ceil(len / resolution) points, so every edge
    // is sampled at the same resolution regardless of length. A lattice edge and
    // a goal-connection edge an order of magnitude longer then get proportionate
    // check counts -- with a fixed count the short edge is oversampled (wasted
    // GPU time) while the long one is undersampled, which would let a path
    // tunnel through an obstacle and be reported as a success.
    float resolution;
};

// Points to place on an edge of configuration-space length `len`.
__device__ inline int edge_granularity(const EdgeEvalParams& p, float len) {
    if (!(p.resolution > 0.0f)) return p.granularity;
    int g = (int)ceilf(len / p.resolution);
    if (g < 2) g = 2;
    if (g > 1024) g = 1024;   // bound the worst-case substep loop
    return g;
}


// One block per edge, launch with 64 threads:  evaluate_edges_batch<<<num_edges, 64>>>
__global__ void evaluate_edges_batch(
    const float* __restrict__ from_cfgs,   // [num_edges * dim]
    const float* __restrict__ to_cfgs,     // [num_edges * dim]
    int num_edges,
    int dim,
    ppln::collision::SceneCollisionData scene,
    ppln::RobotModel model,
    EdgeEvalParams params,
    uint8_t* __restrict__ valid,           // [num_edges]
    float*   __restrict__ cost,            // [num_edges]
    float*   __restrict__ clearance)       // [num_edges] (may be nullptr)
{
    const int edge = blockIdx.x;
    if (edge >= num_edges) return;

    const int tid        = threadIdx.x;         // [0,64)
    const int thread_ind = tid % 4;
    const int batch_ind  = tid / 4;             // waypoint slot within a substep

    // Shared scratch, sized to match the pRRTC scene kernel.
    __align__(16) __shared__ float sphere_pos[6000];        // n_spheres * 16 * 3
    __align__(16) __shared__ float sphere_pos_approx[2500];  // n_approx_spheres * 16 * 3
    __align__(16) __shared__ volatile int link_CC[640];      // joint_in_collision scratch
    __align__(16) __shared__ float T[16 * 2 * 16];           // per-slot 4x4 FK accum
    __shared__ ppln::collision::TwoPhaseFlags flags;          // per-substep block CC flags
    __shared__ unsigned int edge_bad[1];                     // accumulates across substeps
    __shared__ float wp_clr[BATCH_SIZE];                     // per-waypoint clearance
    __shared__ float block_clr[1];                           // running min clearance

    const float* from = from_cfgs + edge * dim;
    const float* to   = to_cfgs   + edge * dim;

    if (tid == 0) {
        edge_bad[0]  = 0u;
        block_clr[0] = FLT_MAX;
    }
    __syncthreads();

    // Uniform across the block (same inputs), so the substep count below is
    // uniform too and the __syncthreads inside the loop stay well-formed.
    float len2 = 0.0f;
    for (int i = 0; i < dim; ++i) { float d = to[i] - from[i]; len2 += d * d; }
    const float len = sqrtf(len2);

    const int G = edge_granularity(params, len);
    const int num_substeps = (G + BATCH_SIZE - 1) / BATCH_SIZE;
    for (int b = 0; b < num_substeps; ++b) {
        // Global waypoint index j in [1, G]; j == G is exactly to_cfg.
        // Out-of-range (padding) slots collapse to from_cfg, which is a search
        // node assumed collision-free, so warp-collective __any_sync early-exit
        // in the CC primitives is never tripped by padding.
        const int   j       = b * BATCH_SIZE + batch_ind + 1;
        const bool  inrange = (j <= G);
        const float t       = inrange ? (float)j / (float)G : 0.0f;

        float interp_cfg[ppln::MAX_DIM];
        for (int i = 0; i < dim; ++i)
            interp_cfg[i] = from[i] + t * (to[i] - from[i]);
        __syncthreads();

        // The shared anchor (src/collision/two_phase_cc.cuh) — the same routine
        // MIT*'s node and edge kernels call.
        ppln::collision::TwoPhaseResult res = ppln::collision::two_phase_cc(
            model, scene, interp_cfg, sphere_pos, sphere_pos_approx, T,
            link_CC, &flags, tid, params.collision_margin, params.check_self);

        if (tid == 0 && res.collision) atomicOr(&edge_bad[0], 1u);
        __syncthreads();

        // ---- clearance (optional): needs detailed sphere_pos ----
        if (params.compute_clearance) {
            if (!res.did_full_fk) {
                ppln::collision::fk_runtime(model, interp_cfg, sphere_pos, T, tid);
                __syncthreads();
            }
            // scene_min_clearance_runtime reduces within the 4-lane group and
            // replicates the per-waypoint min onto all 4 lanes.
            float wmin = ppln::collision::scene_min_clearance_runtime(
                model, sphere_pos, scene, tid, params.collision_margin);
            if (thread_ind == 0)
                wp_clr[batch_ind] = inrange ? wmin : FLT_MAX;
            __syncthreads();
            if (tid == 0) {
                float m = block_clr[0];
                for (int bi = 0; bi < BATCH_SIZE; ++bi)
                    if (b * BATCH_SIZE + bi + 1 <= G)
                        m = fminf(m, wp_clr[bi]);
                block_clr[0] = m;
            }
            __syncthreads();
        }
    }

    // ---- outputs (tid 0 only) ----
    if (tid == 0) {
        bool ok = (edge_bad[0] == 0u);
        valid[edge] = ok ? 1 : 0;
        cost[edge] = ok ? len : FLT_MAX;
        if (clearance != nullptr)
            clearance[edge] = params.compute_clearance ? block_clr[0] : FLT_MAX;
    }
}

} // namespace ppln::search
