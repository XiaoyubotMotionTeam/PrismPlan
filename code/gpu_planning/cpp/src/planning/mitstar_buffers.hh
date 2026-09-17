#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// Pre-allocated GPU buffers for MITStar solve calls.
// Follows SolverBuffers pattern: static create() / explicit destroy().
//
// All buffer sizes are determined by MITStarSettings + RobotModel at create()
// time and remain constant across solve calls.
struct MITStarBuffers {
    bool owns_memory = false;

    // --- Per-node arrays ---
    float*    d_nodes = nullptr;
    float*    d_fwd_g = nullptr;
    float*    d_rev_g = nullptr;
    int*      d_fwd_parent = nullptr;
    int*      d_rev_parent = nullptr;
    float*    d_lb_ctc = nullptr;
    float*    d_lb_ctg = nullptr;
    uint8_t*  d_pruned = nullptr;
    int*      d_fwd_expand_tag = nullptr;
    float*    d_clearance = nullptr;

    // --- Edge adjacency ---
    int*      d_adj_targets = nullptr;
    float*    d_adj_weights = nullptr;
    uint8_t*  d_adj_cc_status = nullptr;
    int*      d_adj_checks_done = nullptr;
    int*      d_adj_count = nullptr;

    // --- Clearance-conditional ---
    float*    d_adj_l2_dist = nullptr;        // nullptr if clearance disabled
    float*    d_sample_clearance = nullptr;    // nullptr if clearance disabled

    // --- Reverse queue ---
    int*      d_rq_src = nullptr;
    int*      d_rq_dst = nullptr;
    float*    d_rq_key_cost = nullptr;
    float*    d_rq_key_effort = nullptr;
    int*      d_rq_size = nullptr;

    // --- Forward queue ---
    int*      d_fq_src = nullptr;
    int*      d_fq_dst = nullptr;
    float*    d_fq_key_lb_cost = nullptr;
    float*    d_fq_key_est_cost = nullptr;
    float*    d_fq_key_est_effort = nullptr;
    int*      d_fq_size = nullptr;
    int*      d_fq_tag = nullptr;

    // --- Reverse queue sort buffers ---
    int*      d_rq_src_sorted = nullptr;
    int*      d_rq_dst_sorted = nullptr;
    float*    d_rq_key_cost_sorted = nullptr;
    float*    d_rq_key_effort_sorted = nullptr;

    // --- Forward queue sort buffers ---
    int*      d_fq_src_sorted = nullptr;
    int*      d_fq_dst_sorted = nullptr;
    float*    d_fq_key_lb_cost_sorted = nullptr;
    float*    d_fq_key_est_cost_sorted = nullptr;
    float*    d_fq_key_est_effort_sorted = nullptr;
    int*      d_fq_tag_sorted = nullptr;

    // --- Sampling buffers ---
    float*    d_new_samples = nullptr;
    uint8_t*  d_cc_results_sample = nullptr;

    // --- NN buffers ---
    int*      d_nn_counts = nullptr;
    int*      d_nn_indices = nullptr;

    // --- Edge CC results ---
    uint8_t*  d_edge_cc_results = nullptr;

    // --- Newly reached / invalidated / expand ---
    int*      d_newly_reached = nullptr;
    int*      d_n_newly_reached = nullptr;
    int*      d_rev_invalidated = nullptr;
    int*      d_n_rev_invalidated = nullptr;
    int*      d_expand_nodes = nullptr;

    // --- Propagation flag ---
    int*      d_changed = nullptr;

    // --- Start/goal/CL matrix ---
    float*    d_start_cfg = nullptr;
    float*    d_best_goal_cfg = nullptr;
    float*    d_CL_matrix = nullptr;

    // --- Halton + RNG ---
    void*     d_halton_states = nullptr;  // HaltonState_rt* (opaque to header)
    curandState* d_rng_states = nullptr;

    // --- Diagnostics ---
    int*      d_diag = nullptr;

    // --- CUB sort/reduce workspaces ---
    void*     d_sort_temp_rq = nullptr;
    void*     d_sort_temp_fq = nullptr;
    void*     d_reduce_temp = nullptr;
    void*     d_cosort_temp = nullptr;
    float*    d_min_rq_cost = nullptr;
    float*    d_min_fq_lb_cost = nullptr;

    // CUB workspace sizes (needed for sort/reduce calls)
    size_t    sort_temp_bytes_rq = 0;
    size_t    sort_temp_bytes_fq = 0;
    size_t    reduce_temp_bytes = 0;
    size_t    cosort_temp_bytes = 0;

    // Recorded parameters
    int n_dof = 0;
    int batch_size = 0;
    int max_nodes = 0;
    int max_edges_per_node = 0;
    int max_neighbors = 0;
    int m_reverse_eval = 0;
    int m_forward_eval = 0;
    bool clearance_enabled = false;

    // Allocate all GPU buffers.
    static MITStarBuffers create(
        int n_dof, int batch_size, int max_nodes,
        int max_edges_per_node, int max_neighbors,
        int m_reverse_eval, int m_forward_eval,
        bool enable_clearance = false);

    // Free all GPU buffers. Safe to call multiple times.
    void destroy();
};
