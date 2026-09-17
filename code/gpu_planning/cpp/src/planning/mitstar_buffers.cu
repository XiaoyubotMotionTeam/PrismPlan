// IMPORTANT: cub/cub.cuh must be included BEFORE utils.cuh because utils.cuh
// defines #define M 4 which conflicts with CUB's NVTX3 template parameters.
#include <cub/cub.cuh>

#include "mitstar_buffers.hh"
#include "robot_model.cuh"  // ppln::MAX_DIM
#include <cstdio>
#include <algorithm>

// Local HaltonState_rt matching MITStar.cu's definition
struct MITStar_HaltonState_rt {
    float b[ppln::MAX_DIM];
    float n[ppln::MAX_DIM];
    float d[ppln::MAX_DIM];
};

MITStarBuffers MITStarBuffers::create(
    int n_dof, int batch_size, int max_nodes,
    int max_edges_per_node, int max_neighbors,
    int m_reverse_eval, int m_forward_eval,
    bool enable_clearance)
{
    MITStarBuffers b;
    b.n_dof = n_dof;
    b.batch_size = batch_size;
    b.max_nodes = max_nodes;
    b.max_edges_per_node = max_edges_per_node;
    b.max_neighbors = max_neighbors;
    b.m_reverse_eval = m_reverse_eval;
    b.m_forward_eval = m_forward_eval;
    b.clearance_enabled = enable_clearance;

    const size_t adj_total = (size_t)max_nodes * max_edges_per_node;
    const int max_rq = m_reverse_eval * 8;
    const int max_fq = m_forward_eval * 8;
    const int max_eval = std::max({max_rq, max_fq, m_reverse_eval, m_forward_eval});

    // --- Per-node arrays ---
    cudaMalloc(&b.d_nodes,          (size_t)max_nodes * n_dof * sizeof(float));
    cudaMalloc(&b.d_fwd_g,          (size_t)max_nodes * sizeof(float));
    cudaMalloc(&b.d_rev_g,          (size_t)max_nodes * sizeof(float));
    cudaMalloc(&b.d_fwd_parent,     (size_t)max_nodes * sizeof(int));
    cudaMalloc(&b.d_rev_parent,     (size_t)max_nodes * sizeof(int));
    cudaMalloc(&b.d_lb_ctc,         (size_t)max_nodes * sizeof(float));
    cudaMalloc(&b.d_lb_ctg,         (size_t)max_nodes * sizeof(float));
    cudaMalloc(&b.d_pruned,         (size_t)max_nodes * sizeof(uint8_t));
    cudaMalloc(&b.d_fwd_expand_tag, (size_t)max_nodes * sizeof(int));
    cudaMalloc(&b.d_clearance,      (size_t)max_nodes * sizeof(float));

    // --- Edge adjacency ---
    cudaMalloc(&b.d_adj_targets,     adj_total * sizeof(int));
    cudaMalloc(&b.d_adj_weights,     adj_total * sizeof(float));
    cudaMalloc(&b.d_adj_cc_status,   adj_total * sizeof(uint8_t));
    cudaMalloc(&b.d_adj_checks_done, adj_total * sizeof(int));
    cudaMalloc(&b.d_adj_count,       (size_t)max_nodes * sizeof(int));

    // --- Clearance-conditional ---
    if (enable_clearance) {
        cudaMalloc(&b.d_adj_l2_dist,      adj_total * sizeof(float));
        cudaMalloc(&b.d_sample_clearance,  (size_t)batch_size * sizeof(float));
    }

    // --- Reverse queue ---
    cudaMalloc(&b.d_rq_src,        (size_t)max_rq * sizeof(int));
    cudaMalloc(&b.d_rq_dst,        (size_t)max_rq * sizeof(int));
    cudaMalloc(&b.d_rq_key_cost,   (size_t)max_rq * sizeof(float));
    cudaMalloc(&b.d_rq_key_effort, (size_t)max_rq * sizeof(float));
    cudaMalloc(&b.d_rq_size,       sizeof(int));

    // --- Forward queue ---
    cudaMalloc(&b.d_fq_src,            (size_t)max_fq * sizeof(int));
    cudaMalloc(&b.d_fq_dst,            (size_t)max_fq * sizeof(int));
    cudaMalloc(&b.d_fq_key_lb_cost,    (size_t)max_fq * sizeof(float));
    cudaMalloc(&b.d_fq_key_est_cost,   (size_t)max_fq * sizeof(float));
    cudaMalloc(&b.d_fq_key_est_effort, (size_t)max_fq * sizeof(float));
    cudaMalloc(&b.d_fq_size,           sizeof(int));
    cudaMalloc(&b.d_fq_tag,            (size_t)max_fq * sizeof(int));

    // --- RQ sort buffers ---
    cudaMalloc(&b.d_rq_src_sorted,        (size_t)max_rq * sizeof(int));
    cudaMalloc(&b.d_rq_dst_sorted,        (size_t)max_rq * sizeof(int));
    cudaMalloc(&b.d_rq_key_cost_sorted,   (size_t)max_rq * sizeof(float));
    cudaMalloc(&b.d_rq_key_effort_sorted, (size_t)max_rq * sizeof(float));

    // --- FQ sort buffers ---
    cudaMalloc(&b.d_fq_src_sorted,            (size_t)max_fq * sizeof(int));
    cudaMalloc(&b.d_fq_dst_sorted,            (size_t)max_fq * sizeof(int));
    cudaMalloc(&b.d_fq_key_lb_cost_sorted,    (size_t)max_fq * sizeof(float));
    cudaMalloc(&b.d_fq_key_est_cost_sorted,   (size_t)max_fq * sizeof(float));
    cudaMalloc(&b.d_fq_key_est_effort_sorted, (size_t)max_fq * sizeof(float));
    cudaMalloc(&b.d_fq_tag_sorted,            (size_t)max_fq * sizeof(int));

    // --- Sampling buffers ---
    cudaMalloc(&b.d_new_samples,      (size_t)batch_size * n_dof * sizeof(float));
    cudaMalloc(&b.d_cc_results_sample,(size_t)batch_size * sizeof(uint8_t));

    // --- NN buffers ---
    cudaMalloc(&b.d_nn_counts,  (size_t)batch_size * sizeof(int));
    cudaMalloc(&b.d_nn_indices, (size_t)batch_size * max_neighbors * sizeof(int));

    // --- Edge CC results ---
    cudaMalloc(&b.d_edge_cc_results, (size_t)max_eval * sizeof(uint8_t));

    // --- Newly reached / invalidated / expand ---
    cudaMalloc(&b.d_newly_reached,     (size_t)max_nodes * sizeof(int));
    cudaMalloc(&b.d_n_newly_reached,   sizeof(int));
    cudaMalloc(&b.d_rev_invalidated,   (size_t)max_nodes * sizeof(int));
    cudaMalloc(&b.d_n_rev_invalidated, sizeof(int));
    cudaMalloc(&b.d_expand_nodes,      (size_t)max_nodes * sizeof(int));

    // --- Propagation flag ---
    cudaMalloc(&b.d_changed, sizeof(int));

    // --- Start/goal/CL matrix ---
    cudaMalloc(&b.d_start_cfg,     n_dof * sizeof(float));
    cudaMalloc(&b.d_best_goal_cfg, n_dof * sizeof(float));
    cudaMalloc(&b.d_CL_matrix,     n_dof * n_dof * sizeof(float));

    // --- Halton + RNG ---
    MITStar_HaltonState_rt* halton_ptr;
    cudaMalloc(&halton_ptr, batch_size * sizeof(MITStar_HaltonState_rt));
    b.d_halton_states = halton_ptr;
    cudaMalloc(&b.d_rng_states, batch_size * sizeof(curandState));

    // --- Diagnostics ---
    cudaMalloc(&b.d_diag, 6 * sizeof(int));

    // --- CUB sort workspace for RQ ---
    b.sort_temp_bytes_rq = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, b.sort_temp_bytes_rq,
        b.d_rq_key_cost, b.d_rq_key_cost_sorted,
        b.d_rq_src, b.d_rq_src_sorted, max_rq);
    cudaMalloc(&b.d_sort_temp_rq, std::max(b.sort_temp_bytes_rq, (size_t)64));

    // --- CUB sort workspace for FQ ---
    b.sort_temp_bytes_fq = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, b.sort_temp_bytes_fq,
        b.d_fq_key_lb_cost, b.d_fq_key_lb_cost_sorted,
        b.d_fq_src, b.d_fq_src_sorted, max_fq);
    cudaMalloc(&b.d_sort_temp_fq, std::max(b.sort_temp_bytes_fq, (size_t)64));

    // --- CUB reduce workspace ---
    size_t reduce_temp_bytes_rq = 0;
    cub::DeviceReduce::Min(nullptr, reduce_temp_bytes_rq,
        b.d_rq_key_cost, b.d_rq_key_cost_sorted, max_rq);
    size_t reduce_temp_bytes_fq = 0;
    cub::DeviceReduce::Min(nullptr, reduce_temp_bytes_fq,
        b.d_fq_key_lb_cost, b.d_fq_key_lb_cost_sorted, max_fq);
    b.reduce_temp_bytes = std::max(reduce_temp_bytes_rq, reduce_temp_bytes_fq);
    cudaMalloc(&b.d_reduce_temp, std::max(b.reduce_temp_bytes, (size_t)64));

    // --- Device scalars for reduce ---
    cudaMalloc(&b.d_min_rq_cost,    sizeof(float));
    cudaMalloc(&b.d_min_fq_lb_cost, sizeof(float));

    // --- CUB cosort workspace ---
    size_t cosort_temp_bytes_float = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, cosort_temp_bytes_float,
        b.d_fq_key_lb_cost, b.d_fq_key_lb_cost_sorted,
        b.d_fq_key_est_cost, b.d_fq_key_est_cost_sorted, max_fq);
    size_t cosort_temp_bytes_int = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, cosort_temp_bytes_int,
        b.d_fq_key_lb_cost, b.d_fq_key_lb_cost_sorted,
        b.d_fq_dst, b.d_fq_dst_sorted, max_fq);
    b.cosort_temp_bytes = std::max({cosort_temp_bytes_float, cosort_temp_bytes_int, (size_t)64});
    cudaMalloc(&b.d_cosort_temp, b.cosort_temp_bytes);

    b.owns_memory = true;
    return b;
}

void MITStarBuffers::destroy() {
    if (!owns_memory) return;

    cudaFree(d_nodes);
    cudaFree(d_fwd_g);
    cudaFree(d_rev_g);
    cudaFree(d_fwd_parent);
    cudaFree(d_rev_parent);
    cudaFree(d_lb_ctc);
    cudaFree(d_lb_ctg);
    cudaFree(d_pruned);
    cudaFree(d_clearance);
    cudaFree(d_adj_targets);
    cudaFree(d_adj_weights);
    cudaFree(d_adj_cc_status);
    cudaFree(d_adj_checks_done);
    cudaFree(d_adj_count);
    if (d_adj_l2_dist) cudaFree(d_adj_l2_dist);
    if (d_sample_clearance) cudaFree(d_sample_clearance);
    cudaFree(d_rq_src);
    cudaFree(d_rq_dst);
    cudaFree(d_rq_key_cost);
    cudaFree(d_rq_key_effort);
    cudaFree(d_rq_size);
    cudaFree(d_fq_src);
    cudaFree(d_fq_dst);
    cudaFree(d_fq_key_lb_cost);
    cudaFree(d_fq_key_est_cost);
    cudaFree(d_fq_key_est_effort);
    cudaFree(d_fq_size);
    cudaFree(d_fq_tag);
    cudaFree(d_fq_tag_sorted);
    cudaFree(d_fwd_expand_tag);
    cudaFree(d_rq_src_sorted);
    cudaFree(d_rq_dst_sorted);
    cudaFree(d_rq_key_cost_sorted);
    cudaFree(d_rq_key_effort_sorted);
    cudaFree(d_fq_src_sorted);
    cudaFree(d_fq_dst_sorted);
    cudaFree(d_fq_key_lb_cost_sorted);
    cudaFree(d_fq_key_est_cost_sorted);
    cudaFree(d_fq_key_est_effort_sorted);
    cudaFree(d_new_samples);
    cudaFree(d_cc_results_sample);
    cudaFree(d_nn_counts);
    cudaFree(d_nn_indices);
    cudaFree(d_edge_cc_results);
    cudaFree(d_newly_reached);
    cudaFree(d_n_newly_reached);
    cudaFree(d_rev_invalidated);
    cudaFree(d_n_rev_invalidated);
    cudaFree(d_expand_nodes);
    cudaFree(d_changed);
    cudaFree(d_start_cfg);
    cudaFree(d_best_goal_cfg);
    cudaFree(d_CL_matrix);
    cudaFree(d_halton_states);
    cudaFree(d_rng_states);
    cudaFree(d_diag);
    cudaFree(d_sort_temp_rq);
    cudaFree(d_sort_temp_fq);
    cudaFree(d_reduce_temp);
    cudaFree(d_cosort_temp);
    cudaFree(d_min_rq_cost);
    cudaFree(d_min_fq_lb_cost);

    owns_memory = false;
}
