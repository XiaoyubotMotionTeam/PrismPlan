// ============================================================================
// MITStar.cu — GPU-parallel MIT* (Multi-Informed Trees) planner
// Dual-search architecture (Strub & Gammell 2022, Zhang et al. 2025):
//   - Reverse search (sparse CC) computes admissible cost-to-go heuristics
//   - Forward search (full CC) finds validated solutions guided by reverse heuristics
//   - CPU orchestrates interleaving; GPU runs batch kernels
// ============================================================================

// IMPORTANT: cub/cub.cuh must be included BEFORE utils.cuh because utils.cuh
// defines #define M 4 which conflicts with CUB's NVTX3 template parameters.
#include <cub/cub.cuh>

#include "Planners.hh"
#include "utils.cuh"
#include "robot_model.cuh"
#include "MITStar_settings.hh"
#include "mitstar_buffers.hh"
#include "runtime_kinematics.cuh"
#include "src/collision/scene_collision.cuh"
#include "src/collision/two_phase_cc.cuh"
#include "src/planning/shortcut.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <cfloat>
#include <cstdio>

namespace MITStar {
    using namespace ppln;
    using namespace ppln::collision;
    using namespace ppln::device_utils;

    // ========================================================================
    // Constants
    // ========================================================================
    constexpr int MITSTAR_BLOCK_SIZE = 256;
    constexpr float INF_COST = 1e20f;

    // Edge CC status
    constexpr uint8_t EDGE_UNKNOWN   = 0;
    constexpr uint8_t EDGE_WHITELIST = 1;   // confirmed free
    constexpr uint8_t EDGE_BLACKLIST = 2;   // confirmed collision

    // ========================================================================
    // Halton state (runtime, variable dimension)
    // ========================================================================
    struct HaltonState_rt {
        float b[MAX_DIM];
        float n[MAX_DIM];
        float d[MAX_DIM];
    };

    __device__ void halton_next_rt(HaltonState_rt& state, float* result, int dim) {
        for (int i = 0; i < dim; i++) {
            float xf = state.d[i] - state.n[i];
            if (xf == 1.0f) {
                state.d[i] = floorf(state.d[i] * state.b[i]);
                state.n[i] = 1.0f;
            } else {
                float y = floorf(state.d[i] / state.b[i]);
                while (xf <= y) {
                    y = floorf(y / state.b[i]);
                }
                state.n[i] = floorf((state.b[i] + 1.0f) * y) - xf;
            }
            result[i] = state.n[i] / state.d[i];
        }
    }

    __device__ void shuffle_array_rt(float *array, int n, curandState &state) {
        for (int i = n - 1; i > 0; i--) {
            int j = curand(&state) % (i + 1);
            float temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    __device__ __forceinline__ void atomicMinFloat(float* addr, float val) {
        unsigned int* addr_as_uint = (unsigned int*)addr;
        unsigned int old = *addr_as_uint;
        unsigned int assumed;
        do {
            assumed = old;
            float old_f = __uint_as_float(assumed);
            if (val >= old_f) return;
            old = atomicCAS(addr_as_uint, assumed, __float_as_uint(val));
        } while (assumed != old);
    }

    // ========================================================================
    // Kernel 1: init_rng_kernel
    // ========================================================================
    __global__ void init_rng_mitstar(curandState* states, unsigned long seed, int count) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) return;
        curand_init(seed + idx, idx, 0, &states[idx]);
    }

    // ========================================================================
    // Kernel 2: init_halton_kernel
    // ========================================================================
    __global__ void init_halton_mitstar(HaltonState_rt* states, curandState* cr_states, int count, int dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) return;

        float primes[16] = {
            3.f, 5.f, 7.f, 11.f, 13.f, 17.f, 19.f, 23.f,
            29.f, 31.f, 37.f, 41.f, 43.f, 47.f, 53.f, 59.f
        };
        if (idx != 0) shuffle_array_rt(primes, 16, cr_states[idx]);

        for (int i = 0; i < dim; i++) {
            states[idx].b[i] = primes[i];
            states[idx].n[i] = 0.0f;
            states[idx].d[i] = 1.0f;
        }
    }

    // ========================================================================
    // Kernel 3: sample_batch_kernel
    // ========================================================================
    // Halton + informed/EIS rejection sampling.
    // informed_bound = min(solution_cost, eis_cost).
    __global__ void sample_batch_kernel(
        float* d_new_samples,       // [batch_size * dim]
        HaltonState_rt* halton_states,
        curandState* d_rng_states,   // for independent radius sampling
        const RobotModel model,
        int batch_size,
        int dim,
        bool use_informed,
        float informed_bound,       // min(solution_cost, eis_cost)
        const float* d_start,       // [dim]
        const float* d_goal,        // [dim] (best goal)
        const float* d_CL,          // [dim * dim] rotation+scale matrix
        float c_min                  // L2(start, goal)
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size) return;

        float sample[MAX_DIM];
        halton_next_rt(halton_states[idx], sample, dim);

        if (use_informed && d_CL != nullptr && informed_bound < INF_COST) {
            // Informed sampling: uniform point in prolate hyperspheroid
            float dir[MAX_DIM];
            float norm_sq = 0.0f;
            for (int i = 0; i < dim; i++) {
                dir[i] = sample[i] * 2.0f - 1.0f;
                norm_sq += dir[i] * dir[i];
            }
            if (norm_sq > 1e-10f) {
                float inv_norm = rsqrtf(norm_sq);
                for (int i = 0; i < dim; i++) dir[i] *= inv_norm;
            } else {
                dir[0] = 1.0f;
                for (int i = 1; i < dim; i++) dir[i] = 0.0f;
            }

            // r = U^(1/d) for uniform in d-ball — use independent RNG for radius
            float u = fmaxf(curand_uniform(&d_rng_states[idx]), 1e-7f);
            float r = powf(u, 1.0f / (float)dim);

            for (int i = 0; i < dim; i++) {
                sample[i] = r * dir[i];
            }

            // Scale to prolate hyperspheroid
            float r1 = informed_bound * 0.5f;
            float r_rest = sqrtf(fmaxf(informed_bound * informed_bound - c_min * c_min, 0.0f)) * 0.5f;

            float center[MAX_DIM];
            for (int i = 0; i < dim; i++) {
                center[i] = (d_start[i] + d_goal[i]) * 0.5f;
            }

            float scaled[MAX_DIM];
            scaled[0] = sample[0] * r1;
            for (int i = 1; i < dim; i++) {
                scaled[i] = sample[i] * r_rest;
            }

            // Rotate: sample = CL * scaled + center
            for (int i = 0; i < dim; i++) {
                float val = center[i];
                for (int j = 0; j < dim; j++) {
                    val += d_CL[i * dim + j] * scaled[j];
                }
                sample[i] = val;
            }

            clamp_to_limits_runtime(model, sample);
        } else {
            // Uniform sampling: scale [0,1] to joint limits
            scale_cfg_runtime(model, sample);
        }

        for (int i = 0; i < dim; i++) {
            d_new_samples[idx * dim + i] = sample[i];
        }
    }

    // ========================================================================
    // Kernel 4: batch_cc_kernel
    // ========================================================================
    // Point CC for new samples. 4 threads per sample (cooperative FK).
    __global__ void batch_cc_kernel(
        const float* d_new_samples,
        uint8_t* d_cc_results,       // 0=free, 1=collision
        int n_samples,
        const RobotModel model,
        const SceneCollisionData scene,
        int dim,
        float collision_margin,
        float* d_sample_clearance,   // [n_samples] output clearance (NULL when disabled)
        float clearance_weight       // > 0 to compute clearance; 0 to skip
    ) {
        int sample_idx = blockIdx.x;
        if (sample_idx >= n_samples) return;
        int tid = threadIdx.x;
        if (tid >= 4) return;

        extern __shared__ float smem[];
        int n_spheres = model.n_spheres;
        int n_approx = model.n_approx_spheres;
        float* sphere_pos = smem;
        float* approx_sphere_pos = sphere_pos + n_spheres * BATCH_SIZE * 3;
        float* T = approx_sphere_pos + n_approx * BATCH_SIZE * 3;
        volatile int* joint_in_collision = (volatile int*)(T + BATCH_SIZE * ppln::FK_T_SLOTS * 16);
        float* q = (float*)(&joint_in_collision[BATCH_SIZE * 20]);

        if (tid == 0) {
            for (int i = 0; i < dim; i++) {
                q[i] = d_new_samples[sample_idx * dim + i];
            }
        }
        __syncthreads();

        for (int i = tid; i < 20; i += 4) {
            joint_in_collision[i] = 0;
        }
        __syncthreads();

        // The shared anchor — same routine the edge kernels and pRRTC's edge
        // eval call, so a sample this planner admits is admissible for every
        // other paradigm by construction.
        __shared__ ppln::collision::TwoPhaseFlags s_flags;
        ppln::collision::TwoPhaseResult res = ppln::collision::two_phase_cc(
            model, scene, q, sphere_pos, approx_sphere_pos, T,
            joint_in_collision, &s_flags, tid, collision_margin,
            /*check_self=*/true);
        if (res.collision) {
            if (tid == 0) d_cc_results[sample_idx] = 1;
            return;
        }

        // Compute ESDF clearance for passing samples (only when enabled)
        if (d_sample_clearance != nullptr && clearance_weight > 0.0f) {
            if (!res.did_full_fk) {
                fk_runtime(model, q, sphere_pos, T, tid);
                __syncthreads();
            }
            float clearance = scene_min_clearance_runtime(model, sphere_pos, scene, tid, collision_margin);
            if (tid == 0) d_sample_clearance[sample_idx] = clearance;
        } else {
            if (tid == 0 && d_sample_clearance != nullptr) d_sample_clearance[sample_idx] = 1000.0f;
        }

        if (tid == 0) d_cc_results[sample_idx] = 0;
    }

    // ========================================================================
    // Kernel 4b: compute_node_clearance_kernel
    // ========================================================================
    // Compute ESDF clearance for a set of nodes and write to d_clearance.
    // 4 threads per node (cooperative FK, then scene_min_clearance_runtime).
    // Only runs when clearance_weight > 0.
    __global__ void compute_node_clearance_kernel(
        const float* d_nodes,
        float* d_clearance,
        const int* d_node_indices,   // which nodes to compute (NULL = sequential from node_start)
        int node_start,              // used when d_node_indices == NULL
        int n_nodes,
        const RobotModel model,
        const SceneCollisionData scene,
        int dim,
        float collision_margin
    ) {
        int local_idx = blockIdx.x;
        if (local_idx >= n_nodes) return;
        int tid = threadIdx.x;
        if (tid >= 4) return;

        int node_idx = d_node_indices ? d_node_indices[local_idx] : (node_start + local_idx);

        // Shared memory layout: only what fk_runtime + scene_min_clearance need.
        // sphere_pos: [n_spheres * BATCH_SIZE * 3]  — FK output, clearance input
        // T:          [BATCH_SIZE * ppln::FK_T_SLOTS * 16]              — FK transform scratch
        // q:          [dim]                          — joint config
        extern __shared__ float smem[];
        int n_spheres = model.n_spheres;
        float* sphere_pos = smem;
        float* T = sphere_pos + n_spheres * BATCH_SIZE * 3;
        float* q = T + BATCH_SIZE * ppln::FK_T_SLOTS * 16;

        if (tid == 0) {
            for (int i = 0; i < dim; i++) {
                q[i] = d_nodes[node_idx * dim + i];
            }
        }
        __syncthreads();

        // Full FK for sphere positions
        fk_runtime(model, q, sphere_pos, T, tid);
        __syncthreads();

        float clearance = scene_min_clearance_runtime(model, sphere_pos, scene, tid, collision_margin);

        if (tid == 0) {
            d_clearance[node_idx] = clearance;
        }
    }

    // shortcut_cc_kernel moved to src/planning/shortcut.cuh

    // ========================================================================
    // Kernel 5: init_state_kernel
    // ========================================================================
    // For new nodes: compute lb_ctc=L2(start,x), lb_ctg=L2(x,best_goal),
    // init fwd_g=INF, rev_g=INF.
    __global__ void init_state_kernel(
        const float* d_nodes,       // [n_total * dim]
        float* d_fwd_g,
        float* d_rev_g,
        int* d_fwd_parent,
        int* d_rev_parent,
        float* d_lb_ctc,            // L2(start, x)
        float* d_lb_ctg,            // L2(x, best_goal)
        int n_start_idx,            // first node to init
        int n_end_idx,              // one past last node
        const float* d_start,       // [dim]
        const float* d_goal,        // [dim] best goal
        int dim
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x + n_start_idx;
        if (idx >= n_end_idx) return;

        float dist_start = 0.0f, dist_goal = 0.0f;
        for (int d = 0; d < dim; d++) {
            float ds = d_nodes[idx * dim + d] - d_start[d];
            float dg = d_nodes[idx * dim + d] - d_goal[d];
            dist_start += ds * ds;
            dist_goal += dg * dg;
        }
        d_lb_ctc[idx] = sqrtf(dist_start);
        d_lb_ctg[idx] = sqrtf(dist_goal);
        d_fwd_g[idx] = INF_COST;
        d_rev_g[idx] = INF_COST;
        d_fwd_parent[idx] = idx;
        d_rev_parent[idx] = idx;
    }

    // ========================================================================
    // Kernel 6: knn_kernel — k-nearest neighbors via tiled scan + max-heap
    // ========================================================================
    // 1 block per query node, 256 threads. Each thread maintains a register-
    // based max-heap of size k, scans ALL nodes via shared-memory tiles.
    // After scan, thread 0 collects the global top-k via shared reduction.
    //
    // Processes nodes [query_start .. query_start + n_queries - 1].
    // Each query searches ALL n_total nodes (except itself).
    constexpr int KNN_TILE_SIZE = 64;

    __global__ void knn_kernel(
        const float* d_nodes,
        int query_start,            // first query node index
        int n_queries,
        int n_total,
        int dim,
        int k,                      // desired neighbor count
        int max_neighbors,
        int* d_nn_counts,           // [n_queries] output
        int* d_nn_indices           // [n_queries * max_neighbors] output
    ) {
        int qidx = blockIdx.x;
        if (qidx >= n_queries) return;

        int node_idx = query_start + qidx;
        int tid = threadIdx.x;
        constexpr int LOCAL_K = 32;
        int effective_k = min(k, min(max_neighbors, LOCAL_K)); // clamp to LOCAL_K to prevent register/smem overflow

        // Shared memory layout:
        // [0 .. dim-1]: query node config
        // [dim .. dim + KNN_TILE_SIZE * dim - 1]: tiled node data
        extern __shared__ float smem_knn[];
        float* s_query = smem_knn;
        float* s_tile = s_query + dim;

        // Load query node
        for (int d = tid; d < dim; d += blockDim.x) {
            s_query[d] = d_nodes[node_idx * dim + d];
        }
        __syncthreads();

        // Per-thread max-heap: store (dist_sq, index) pairs
        // Each thread keeps its own local top-k candidates
        // We use a simple insertion sort (k≤32, so this is fast in registers)
        float local_dist[LOCAL_K];
        int local_idx[LOCAL_K];
        int local_count = 0;

        // Initialize with INF
        for (int i = 0; i < LOCAL_K; i++) {
            local_dist[i] = INF_COST;
            local_idx[i] = -1;
        }

        float max_dist = INF_COST; // current k-th largest distance in local heap

        // Scan all nodes in tiles
        for (int tile_base = 0; tile_base < n_total; tile_base += KNN_TILE_SIZE) {
            // Cooperative tile loading
            int tile_count = min(KNN_TILE_SIZE, n_total - tile_base);
            for (int offset = tid; offset < tile_count * dim; offset += blockDim.x) {
                int local_node = offset / dim;
                int local_dim = offset % dim;
                s_tile[local_node * dim + local_dim] = d_nodes[(tile_base + local_node) * dim + local_dim];
            }
            __syncthreads();

            // Each thread processes a subset of tile nodes
            for (int t = tid; t < tile_count; t += blockDim.x) {
                int candidate = tile_base + t;
                if (candidate == node_idx) continue; // skip self

                float dist_sq = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = s_query[d] - s_tile[t * dim + d];
                    dist_sq += diff * diff;
                }

                // Insert into local sorted list if better than worst
                if (dist_sq < max_dist || local_count < effective_k) {
                    // Find insertion point (keep sorted ascending by dist_sq)
                    if (local_count < effective_k) {
                        // Still filling: insert at end, then bubble up
                        int pos = local_count;
                        local_dist[pos] = dist_sq;
                        local_idx[pos] = candidate;
                        local_count++;
                        // Bubble up to maintain sorted order (ascending)
                        while (pos > 0 && local_dist[pos] < local_dist[pos-1]) {
                            float td = local_dist[pos]; local_dist[pos] = local_dist[pos-1]; local_dist[pos-1] = td;
                            int ti = local_idx[pos]; local_idx[pos] = local_idx[pos-1]; local_idx[pos-1] = ti;
                            pos--;
                        }
                    } else {
                        // Replace the last (largest) element
                        local_dist[effective_k - 1] = dist_sq;
                        local_idx[effective_k - 1] = candidate;
                        // Bubble up
                        int pos = effective_k - 1;
                        while (pos > 0 && local_dist[pos] < local_dist[pos-1]) {
                            float td = local_dist[pos]; local_dist[pos] = local_dist[pos-1]; local_dist[pos-1] = td;
                            int ti = local_idx[pos]; local_idx[pos] = local_idx[pos-1]; local_idx[pos-1] = ti;
                            pos--;
                        }
                    }
                    max_dist = local_dist[min(local_count, effective_k) - 1];
                }
            }
            __syncthreads();
        }

        // Reduction: collect best k across all threads
        // Use warp shuffle for in-warp reduction (no syncthreads needed).
        // Only 1 syncthreads per cross-warp step, much faster than
        // the naive log2(blockDim) syncthreads per round.

        __shared__ float s_warp_dist[8]; // one per warp (up to 256 threads)
        __shared__ int s_warp_idx[8];
        __shared__ int s_warp_tid[8];
        __shared__ int s_output_count;
        __shared__ int s_output_indices[32]; // max_neighbors capped at 32
        __shared__ int s_winner_tid; // which thread won this round

        int warp_id = tid >> 5;
        int lane = tid & 31;
        int n_warps = blockDim.x >> 5;

        if (tid == 0) s_output_count = 0;
        __syncthreads();

        int my_cursor = 0; // next local entry to offer

        for (int round = 0; round < effective_k; round++) {
            // Each thread offers its next-best unconsumed candidate
            float my_dist = (my_cursor < local_count) ? local_dist[my_cursor] : INF_COST;
            int my_idx = (my_cursor < local_count) ? local_idx[my_cursor] : -1;
            int my_tid_val = tid;

            // Warp-level reduction using shuffles (no syncthreads!)
            unsigned mask = 0xFFFFFFFF;
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other_dist = __shfl_xor_sync(mask, my_dist, offset);
                int other_idx = __shfl_xor_sync(mask, my_idx, offset);
                int other_tid_val = __shfl_xor_sync(mask, my_tid_val, offset);
                if (other_dist < my_dist) {
                    my_dist = other_dist;
                    my_idx = other_idx;
                    my_tid_val = other_tid_val;
                }
            }

            // Lane 0 of each warp writes to shared
            if (lane == 0) {
                s_warp_dist[warp_id] = my_dist;
                s_warp_idx[warp_id] = my_idx;
                s_warp_tid[warp_id] = my_tid_val;
            }
            __syncthreads(); // single sync for cross-warp

            // Thread 0 reduces across warps (n_warps ≤ 8, so trivial loop)
            if (tid == 0) {
                float best_dist = s_warp_dist[0];
                int best_idx = s_warp_idx[0];
                int best_tid = s_warp_tid[0];
                for (int w = 1; w < n_warps; w++) {
                    if (s_warp_dist[w] < best_dist) {
                        best_dist = s_warp_dist[w];
                        best_idx = s_warp_idx[w];
                        best_tid = s_warp_tid[w];
                    }
                }
                if (best_idx >= 0) {
                    s_output_indices[s_output_count++] = best_idx;
                }
                s_winner_tid = best_tid;
            }
            __syncthreads(); // single sync for broadcast

            // The winning thread advances its cursor
            if (tid == s_winner_tid && my_cursor < local_count) {
                my_cursor++;
            }
            // NOTE: no sync needed here — only the winner modifies its own cursor
        }

        // Write output
        int out_count = s_output_count;
        if (tid == 0) {
            d_nn_counts[qidx] = out_count;
        }
        for (int i = tid; i < out_count; i += blockDim.x) {
            d_nn_indices[qidx * max_neighbors + i] = s_output_indices[i];
        }
    }

    // ========================================================================
    // Kernel 6b: radius_nn_kernel — radius-based nearest neighbors
    // ========================================================================
    // 1 block per query node, 256 threads. Finds all neighbors within
    // radius_sq using tiled shared-memory scan. O(n) with early exit per node.
    // Much faster than k-NN for large batches since no reduction needed.
    constexpr int RNN_TILE_SIZE = 64;

    __global__ void radius_nn_kernel(
        const float* d_nodes,
        int query_start,            // first query node index
        int n_queries,
        int n_total,
        int dim,
        float radius_sq,
        int max_neighbors,
        int* d_nn_counts,           // [n_queries] output
        int* d_nn_indices           // [n_queries * max_neighbors] output
    ) {
        int qidx = blockIdx.x;
        if (qidx >= n_queries) return;

        int node_idx = query_start + qidx;
        int tid = threadIdx.x;

        extern __shared__ float smem_rnn[];
        float* s_query = smem_rnn;
        float* s_tile = s_query + dim;

        // Load query node
        for (int d = tid; d < dim; d += blockDim.x) {
            s_query[d] = d_nodes[node_idx * dim + d];
        }

        __shared__ int s_count;
        if (tid == 0) s_count = 0;
        __syncthreads();

        // Scan all nodes in tiles
        for (int tile_base = 0; tile_base < n_total; tile_base += RNN_TILE_SIZE) {
            int tile_count = min(RNN_TILE_SIZE, n_total - tile_base);

            // Cooperative tile loading
            for (int offset = tid; offset < tile_count * dim; offset += blockDim.x) {
                int local_node = offset / dim;
                int local_dim = offset % dim;
                s_tile[local_node * dim + local_dim] =
                    d_nodes[(tile_base + local_node) * dim + local_dim];
            }
            __syncthreads();

            // Each thread processes a subset of tile nodes
            for (int t = tid; t < tile_count; t += blockDim.x) {
                int candidate = tile_base + t;
                if (candidate == node_idx) continue;

                float dist_sq = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = s_query[d] - s_tile[t * dim + d];
                    dist_sq += diff * diff;
                }

                if (dist_sq < radius_sq) {
                    int pos = atomicAdd(&s_count, 1);
                    if (pos < max_neighbors) {
                        d_nn_indices[qidx * max_neighbors + pos] = candidate;
                    }
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_nn_counts[qidx] = min(s_count, max_neighbors);
        }
    }

    // ========================================================================
    // Kernel 7: insert_edges_kernel
    // ========================================================================
    // Bidirectional edge insertion. Inits cc_status=UNKNOWN, checks_done=0.
    // When an existing node's adj list is full, replaces the longest UNKNOWN
    // edge to maintain shortest-neighbor connectivity (critical for graph
    // connectivity across batches).
    __global__ void insert_edges_kernel(
        const int* d_nn_counts,
        const int* d_nn_indices,
        int n_existing,
        int n_new,
        int max_neighbors,
        int max_epn,
        int* d_adj_targets,
        float* d_adj_weights,
        uint8_t* d_adj_cc_status,
        int* d_adj_checks_done,
        int* d_adj_count,
        const float* d_nodes,
        int dim,
        // Clearance cost parameters
        const float* d_clearance,      // [max_nodes] per-node ESDF clearance (NULL when disabled)
        float clearance_weight,        // alpha; 0 = pure L2
        float clearance_epsilon,       // floor to prevent division by zero
        float max_clearance_penalty,   // cap on penalty multiplier
        float* d_adj_l2_dist           // [adj_total] pure L2 distance backup (NULL when disabled)
    ) {
        int new_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (new_idx >= n_new) return;

        int node_idx = n_existing + new_idx;
        int count = d_nn_counts[new_idx];

        for (int i = 0; i < count; i++) {
            int neighbor = d_nn_indices[new_idx * max_neighbors + i];

            float sum_sq = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = d_nodes[node_idx * dim + d] - d_nodes[neighbor * dim + d];
                sum_sq += diff * diff;
            }
            float l2_dist = sqrtf(sum_sq);

            // Clearance-weighted cost (with capped penalty)
            float dist = l2_dist;
            if (clearance_weight > 0.0f && d_clearance != nullptr) {
                float mc = fmaxf(fminf(d_clearance[node_idx], d_clearance[neighbor]), clearance_epsilon);
                float penalty = fminf(clearance_weight / mc, max_clearance_penalty);
                dist = l2_dist * (1.0f + penalty);
            }

            // Forward edge: node_idx -> neighbor (new node always has room)
            {
                int slot = atomicAdd(&d_adj_count[node_idx], 1);
                if (slot < max_epn) {
                    int base = node_idx * max_epn + slot;
                    d_adj_targets[base] = neighbor;
                    d_adj_weights[base] = dist;
                    d_adj_cc_status[base] = EDGE_UNKNOWN;
                    d_adj_checks_done[base] = 0;
                    if (d_adj_l2_dist) d_adj_l2_dist[base] = l2_dist;
                }
            }
            // Reverse edge: neighbor -> node_idx
            // If neighbor's adj list is full, replace the longest UNKNOWN edge
            // to maintain connectivity to closer nodes from later batches.
            {
                int slot = atomicAdd(&d_adj_count[neighbor], 1);
                if (slot < max_epn) {
                    // Room available — direct insert
                    int base = neighbor * max_epn + slot;
                    d_adj_targets[base] = node_idx;
                    d_adj_weights[base] = dist;
                    d_adj_cc_status[base] = EDGE_UNKNOWN;
                    d_adj_checks_done[base] = 0;
                    if (d_adj_l2_dist) d_adj_l2_dist[base] = l2_dist;
                } else {
                    // Adj list full — find longest UNKNOWN edge to replace.
                    // Compare using L2 distance (not clearance-weighted) so that
                    // nodes near obstacles don't lose short-L2 neighbors due to
                    // inflated clearance costs.
                    int worst_slot = -1;
                    float worst_l2 = l2_dist; // only replace if new edge is shorter in L2
                    for (int e = 0; e < max_epn; e++) {
                        int base = neighbor * max_epn + e;
                        if (d_adj_cc_status[base] == EDGE_UNKNOWN) {
                            float cmp = d_adj_l2_dist ? d_adj_l2_dist[base] : d_adj_weights[base];
                            if (cmp > worst_l2) {
                                worst_l2 = cmp;
                                worst_slot = e;
                            }
                        }
                    }
                    if (worst_slot >= 0) {
                        int base = neighbor * max_epn + worst_slot;
                        d_adj_targets[base] = node_idx;
                        d_adj_weights[base] = dist;
                        d_adj_cc_status[base] = EDGE_UNKNOWN;
                        d_adj_checks_done[base] = 0;
                        if (d_adj_l2_dist) d_adj_l2_dist[base] = l2_dist;
                    }
                }
            }
        }
    }


    // ========================================================================
    // Kernel 8: build_reverse_queue_kernel
    // ========================================================================
    // Expand source nodes (with rev_g < INF) into reverse queue.
    // Only enqueues edges that are not BLACKLIST and where the destination
    // could be improved: rev_g[src] + weight < rev_g[dst].
    __global__ void build_reverse_queue_kernel(
        const int* d_expand_nodes,  // [n_expand] nodes to expand
        int n_expand,
        const float* d_rev_g,
        const float* d_lb_ctc,
        const int* d_adj_targets,
        const float* d_adj_weights,
        const uint8_t* d_adj_cc_status,
        const int* d_adj_checks_done,
        const int* d_adj_count,
        int max_epn,
        float valid_segment_length,
        int num_sparse_checks,      // current sparse resolution
        const uint8_t* d_pruned,    // skip edges to pruned nodes
        float solution_cost,        // P7: prune edges that can't improve solution
        const float* d_adj_l2_dist, // pure L2 for CC segments (NULL = use d_adj_weights)
        // Output
        int* d_rq_src,
        int* d_rq_dst,
        float* d_rq_key_cost,       // rev_g[src] + weight + lb_ctc[dst]
        float* d_rq_key_effort,     // number of remaining sparse checks
        int* d_rq_size,             // [1] atomic
        int max_rq
    ) {
        int expand_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (expand_idx >= n_expand) return;

        int src = d_expand_nodes[expand_idx];
        float g_src = d_rev_g[src];
        if (g_src >= INF_COST) return;

        int n_edges = d_adj_count[src];
        if (n_edges > max_epn) n_edges = max_epn;

        for (int e = 0; e < n_edges; e++) {
            int adj_idx = src * max_epn + e;
            if (d_adj_cc_status[adj_idx] == EDGE_BLACKLIST) continue;

            int dst = d_adj_targets[adj_idx];
            if (d_pruned[dst]) continue;  // skip edges to pruned dead-end nodes
            float w = d_adj_weights[adj_idx];

            // Could improve?
            float new_g = g_src + w;
            if (new_g >= d_rev_g[dst]) continue;

            // Compute effort: use pure L2 distance for CC segment count
            float l2 = (d_adj_l2_dist != nullptr) ? d_adj_l2_dist[adj_idx] : w;
            int total_seg = max(1, (int)ceilf(l2 / valid_segment_length));
            int target_checks = min(num_sparse_checks, total_seg - 1);
            int done = d_adj_checks_done[adj_idx];
            int effort = max(0, target_checks - done);

            float key_cost = new_g + d_lb_ctc[dst];

            // P7: Prune reverse edges that can't improve the solution
            // (OMPL iterateReverseSearch lines 818-822)
            if (solution_cost < INF_COST && key_cost >= solution_cost) continue;

            int pos = atomicAdd(d_rq_size, 1);
            if (pos < max_rq) {
                d_rq_src[pos] = src;
                d_rq_dst[pos] = dst;
                d_rq_key_cost[pos] = key_cost;
                d_rq_key_effort[pos] = (float)effort;
            }
        }
    }

    // ========================================================================
    // Kernel 9: sparse_cc_kernel
    // ========================================================================
    // BFS midpoint sparse CC at current sparse resolution.
    // 4 threads per edge (cooperative FK).
    // Checks midpoints in BFS order: 1/2, 1/4, 3/4, 1/8, 3/8, 5/8, 7/8, ...
    // Incremental: skips first d_adj_checks_done[edge] midpoints.
    __global__ void sparse_cc_kernel(
        const int* d_rq_src,
        const int* d_rq_dst,
        int n_eval,                 // number of edges to evaluate
        const float* d_nodes,
        int dim,
        float valid_segment_length,
        int num_sparse_checks,      // current sparse check count
        const RobotModel model,
        const SceneCollisionData scene,
        float collision_margin,     // inflate sphere radii during edge CC
        // Edge adjacency for incremental tracking
        const int* d_adj_targets,
        int* d_adj_checks_done,
        const int* d_adj_count,
        int max_epn,
        // P3: reverse freeby check
        const int* d_rev_parent,
        const float* d_rev_g,
        // P8: whitelist early-exit
        const uint8_t* d_adj_cc_status,
        // Diagnostics (FIX_B3): [0]=midpoints_checked, [1]=self_coll_hits, [2]=scene_coll_hits
        int* d_diag,
        // Output
        uint8_t* d_cc_results       // [n_eval]: 0=free, 1=collision
    ) {
        int edge_idx = blockIdx.x;
        if (edge_idx >= n_eval) return;
        int tid = threadIdx.x;
        // 4 lanes cooperate on one midpoint, BATCH_SIZE midpoints in flight.
        const int batch_ind = tid / 4;

        int src = d_rq_src[edge_idx];
        int dst = d_rq_dst[edge_idx];

        // P3: Reverse tree freeby — if dst is already child of src in reverse tree,
        // the edge is already validated. Skip CC. (OMPL isInReverseTree, line 1619-1627)
        if (d_rev_parent[dst] == src && d_rev_g[dst] < INF_COST) {
            if (tid == 0) d_cc_results[edge_idx] = 0;  // free (already in tree)
            return;
        }

        extern __shared__ float smem[];
        float* q_src = smem;
        float* q_dst = q_src + dim;
        int n_spheres = model.n_spheres;
        int n_approx = model.n_approx_spheres;
        float* sphere_pos = q_dst + 2 * dim;
        float* approx_sphere_pos = sphere_pos + n_spheres * BATCH_SIZE * 3;
        float* T = approx_sphere_pos + n_approx * BATCH_SIZE * 3;
        volatile int* joint_in_collision = (volatile int*)(T + BATCH_SIZE * ppln::FK_T_SLOTS * 16);

        // Load configs
        if (tid == 0) {
            for (int i = 0; i < dim; i++) {
                q_src[i] = d_nodes[src * dim + i];
                q_dst[i] = d_nodes[dst * dim + i];
            }
        }
        __syncthreads();

        // Compute edge length and segment count
        __shared__ float s_edge_len;
        __shared__ int s_total_seg;
        __shared__ int s_target_checks;
        if (tid == 0) {
            float len = 0.0f;
            for (int i = 0; i < dim; i++) {
                float diff = q_src[i] - q_dst[i];
                len += diff * diff;
            }
            s_edge_len = sqrtf(len);
            s_total_seg = max(1, (int)ceilf(s_edge_len / valid_segment_length));
            s_target_checks = min(num_sparse_checks, s_total_seg - 1);
        }
        __syncthreads();

        int target_checks = s_target_checks;

        // Find the adjacency slot for src->dst to read/update checks_done
        __shared__ int s_adj_slot;
        __shared__ int s_checks_done;
        if (tid == 0) {
            s_adj_slot = -1;
            int n_edges = d_adj_count[src];
            if (n_edges > max_epn) n_edges = max_epn;
            for (int e = 0; e < n_edges; e++) {
                if (d_adj_targets[src * max_epn + e] == dst) {
                    s_adj_slot = src * max_epn + e;
                    break;
                }
            }
            s_checks_done = (s_adj_slot >= 0) ? d_adj_checks_done[s_adj_slot] : 0;
        }
        __syncthreads();

        // P8: Whitelist early-exit — if edge is already fully validated, skip CC
        if (s_adj_slot >= 0 && d_adj_cc_status[s_adj_slot] == EDGE_WHITELIST) {
            if (tid == 0) d_cc_results[edge_idx] = 0;
            return;
        }

        int checks_done = s_checks_done;

        // BFS midpoint generation: generate sequence of fractional positions
        // We generate midpoints in BFS order and skip 'checks_done' already completed
        __shared__ int s_collision;
        __shared__ ppln::collision::TwoPhaseFlags s_flags;
        if (tid == 0) s_collision = 0;
        __syncthreads();

        // BFS midpoint order:
        // Level 0: 1/2
        // Level 1: 1/4, 3/4
        // Level 2: 1/8, 3/8, 5/8, 7/8
        // ...
        // Total midpoints at level L = 2^L, cumulative = 2^(L+1) - 1.
        int new_checks = 0;

        // BATCH_SIZE midpoints per pass, laid out like
        // ppln::search::evaluate_edges_batch: 4 lanes cooperate on one slot,
        // BATCH_SIZE slots per block. The shared buffers were always sized for
        // BATCH_SIZE concurrent slots; the old one-midpoint-at-a-time loop left
        // all but slot 0 idle, which capped this kernel at 4 live threads per
        // block against 16-26 KB of shared memory.
        for (int base = checks_done; base < target_checks; base += BATCH_SIZE) {
            if (s_collision) break;

            // Midpoint k in closed form: level L = floor(log2(k+1)), offset
            // within that level i = (k+1) - 2^L, t = (2i+1)/2^(L+1). Emits the
            // same sequence as the old denominator-doubling loop, but is
            // addressable directly so the slots can be filled independently.
            const int k = base + batch_ind;
            const bool inrange = (k < target_checks);
            float t = 0.0f;
            if (inrange) {
                const int kk = k + 1;
                const int lvl = 31 - __clz(kk);
                t = (float)(2 * (kk - (1 << lvl)) + 1) / (float)(1 << (lvl + 1));
            }
            // Padding slots collapse onto q_src, which is a graph node already
            // known to be collision-free, so they never trip the warp-collective
            // early exits inside the CC primitives.
            float q_interp[ppln::MAX_DIM];
            for (int i = 0; i < dim; i++)
                q_interp[i] = q_src[i] * (1.0f - t) + q_dst[i] * t;

            // The shared anchor: same routine pRRTC's edge eval and MIT*'s own
            // node check call, so this edge's verdict is feasibility-identical
            // with every other paradigm and correct on branching robots, where
            // joint i's parent is not joint i-1.
            ppln::collision::TwoPhaseResult res = ppln::collision::two_phase_cc(
                model, scene, q_interp, sphere_pos, approx_sphere_pos, T,
                joint_in_collision, &s_flags, tid, collision_margin,
                /*check_self=*/true);

            const int group = min(BATCH_SIZE, target_checks - base);

            // Diagnostics
            if (tid == 0 && d_diag) {
                atomicAdd(&d_diag[0], group);              // midpoints checked
                if (res.collision && !res.scene_hit) atomicAdd(&d_diag[1], 1);  // self-collision hits
                if (res.scene_hit) atomicAdd(&d_diag[2], 1);                    // scene-collision hits
            }

            if (res.collision) {
                if (tid == 0) s_collision = 1;
                __syncthreads();
                break;
            }
            new_checks += group;
        }

        if (tid == 0) {
            d_cc_results[edge_idx] = s_collision ? 1 : 0;
            // FIX_V7: Only accumulate checks_done for verified-free midpoints.
            // When collision is detected, do NOT add to checks_done — otherwise
            // repeated evaluations of a colliding edge will overflow checks_done
            // past total_checks, causing all midpoints to be skipped and the
            // edge to be falsely declared FREE.
            if (s_adj_slot >= 0 && !s_collision) {
                atomicAdd(&d_adj_checks_done[s_adj_slot], new_checks);
                // D12: Symmetric checks_done — OMPL sets performedChecks for both
                // directions (isValidAtResolution lines 1561-1562).
                int n_edges_dst = d_adj_count[dst];
                if (n_edges_dst > max_epn) n_edges_dst = max_epn;
                for (int e = 0; e < n_edges_dst; e++) {
                    int adj_idx = dst * max_epn + e;
                    if (d_adj_targets[adj_idx] == src) {
                        atomicAdd(&d_adj_checks_done[adj_idx], new_checks);
                        break;
                    }
                }
            }
        }
    }

    // ========================================================================
    // Kernel 10: update_reverse_tree_kernel
    // ========================================================================
    // For evaluated reverse edges: update rev_g/rev_parent for passed edges,
    // blacklist failed edges. Also blacklist the reverse direction.
    __global__ void update_reverse_tree_kernel(
        const int* d_rq_src,
        const int* d_rq_dst,
        const uint8_t* d_cc_results,
        const float* d_adj_weights,  // adjacency weights
        int n_eval,
        float* d_rev_g,
        int* d_rev_parent,
        // Adjacency for status update
        int* d_adj_targets_arr,
        uint8_t* d_adj_cc_status,
        const int* d_adj_count,
        const float* d_adj_weights_arr,
        int max_epn,
        // Output: newly reached nodes for further expansion
        int* d_newly_reached,
        int* d_n_newly_reached,       // [1] atomic
        int max_newly_reached         // bounds for d_newly_reached writes
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_eval) return;

        int src = d_rq_src[idx];
        int dst = d_rq_dst[idx];
        bool collision = (d_cc_results[idx] != 0);

        // Find edge src->dst in adjacency to get weight and update status
        int n_edges_src = d_adj_count[src];
        if (n_edges_src > max_epn) n_edges_src = max_epn;

        float edge_weight = INF_COST;
        for (int e = 0; e < n_edges_src; e++) {
            int adj_idx = src * max_epn + e;
            if (d_adj_targets_arr[adj_idx] == dst) {
                edge_weight = d_adj_weights_arr[adj_idx];
                if (collision) {
                    d_adj_cc_status[adj_idx] = EDGE_BLACKLIST;
                }
                break;
            }
        }

        // Also update reverse direction dst->src
        int n_edges_dst = d_adj_count[dst];
        if (n_edges_dst > max_epn) n_edges_dst = max_epn;
        for (int e = 0; e < n_edges_dst; e++) {
            int adj_idx = dst * max_epn + e;
            if (d_adj_targets_arr[adj_idx] == src) {
                if (collision) {
                    d_adj_cc_status[adj_idx] = EDGE_BLACKLIST;
                }
                break;
            }
        }

        if (collision) return;

        // Relaxation: if rev_g[src] + w < rev_g[dst], update
        float new_g = d_rev_g[src] + edge_weight;
        float old_g = d_rev_g[dst];
        atomicMinFloat(&d_rev_g[dst], new_g);

        // Check if we actually improved
        float current_g = d_rev_g[dst];
        if (fabsf(current_g - new_g) < 1e-6f && new_g < old_g - 1e-6f) {
            d_rev_parent[dst] = src;
            // Mark as newly reached (with bounds check).
            // Note: race can produce duplicate entries — benign, just causes
            // redundant queue expansion. Bounds check prevents OOB.
            int pos = atomicAdd(d_n_newly_reached, 1);
            if (pos < max_newly_reached) {
                d_newly_reached[pos] = dst;
            }
        }
    }

    // ========================================================================
    // Kernel 11: build_forward_queue_kernel
    // ========================================================================
    // Expand source nodes into forward queue with 3 keys.
    __global__ void build_forward_queue_kernel(
        const int* d_expand_nodes,  // [n_expand] nodes to expand
        int n_expand,
        const float* d_fwd_g,
        const float* d_rev_g,
        const float* d_lb_ctg,
        const int* d_adj_targets,
        const float* d_adj_weights,
        const uint8_t* d_adj_cc_status,
        const int* d_adj_checks_done,
        const int* d_adj_count,
        int max_epn,
        float valid_segment_length,
        float solution_cost,
        const int* d_fwd_expand_tag,  // per-node expand version tag
        const uint8_t* d_pruned,      // skip edges to pruned nodes
        const float* d_adj_l2_dist,   // pure L2 for CC segments (NULL = use d_adj_weights)
        // Output
        int* d_fq_src,
        int* d_fq_dst,
        float* d_fq_key_lb_cost,    // fwd_g[src] + w + rev_g[dst]  (lower bound)
        float* d_fq_key_est_cost,   // fwd_g[src] + w + lb_ctg[dst] (estimate)
        float* d_fq_key_est_effort, // remaining checks
        int* d_fq_tag,              // expansion tag snapshot
        int* d_fq_size,             // [1] atomic
        int max_fq
    ) {
        int expand_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (expand_idx >= n_expand) return;

        int src = d_expand_nodes[expand_idx];
        float g_src = d_fwd_g[src];
        if (g_src >= INF_COST) return;

        int src_tag = d_fwd_expand_tag[src];

        int n_edges = d_adj_count[src];
        if (n_edges > max_epn) n_edges = max_epn;

        for (int e = 0; e < n_edges; e++) {
            int adj_idx = src * max_epn + e;
            if (d_adj_cc_status[adj_idx] == EDGE_BLACKLIST) continue;

            int dst = d_adj_targets[adj_idx];
            if (d_pruned[dst]) continue;  // skip edges to pruned dead-end nodes
            float w = d_adj_weights[adj_idx];
            float new_fwd_g = g_src + w;
            if (new_fwd_g >= d_fwd_g[dst]) continue;

            // Lower bound cost: use rev_g if available, otherwise L2 heuristic
            float ctg = d_rev_g[dst];
            if (ctg >= INF_COST) ctg = d_lb_ctg[dst]; // fallback to L2
            float lb_cost = new_fwd_g + ctg;
            if (lb_cost >= solution_cost) continue;

            // P1: Use rev_g (= admissible/estimated cost-to-go in L2 space) with L2 fallback
            float est_ctg = d_rev_g[dst];
            if (est_ctg >= INF_COST) est_ctg = d_lb_ctg[dst];
            float est_cost = g_src + w + est_ctg;

            // Effort: use pure L2 distance for CC segment count
            float l2 = (d_adj_l2_dist != nullptr) ? d_adj_l2_dist[adj_idx] : w;
            int total_seg = max(1, (int)ceilf(l2 / valid_segment_length));
            int done = d_adj_checks_done[adj_idx];
            int effort = max(0, total_seg - 1 - done);

            int pos = atomicAdd(d_fq_size, 1);
            if (pos < max_fq) {
                d_fq_src[pos] = src;
                d_fq_dst[pos] = dst;
                d_fq_key_lb_cost[pos] = lb_cost;
                d_fq_key_est_cost[pos] = est_cost;
                d_fq_key_est_effort[pos] = (float)effort;
                d_fq_tag[pos] = src_tag;
            }
        }
    }

    // ========================================================================
    // Kernel 12: full_cc_kernel
    // ========================================================================
    // Full-resolution CC in BFS midpoint order, incremental.
    // 4 threads per edge (cooperative FK).
    __global__ void full_cc_kernel(
        const int* d_fq_src,
        const int* d_fq_dst,
        int n_eval,
        const float* d_nodes,
        int dim,
        float valid_segment_length,
        const RobotModel model,
        const SceneCollisionData scene,
        float collision_margin,     // inflate sphere radii during edge CC
        // Adjacency for incremental tracking
        const int* d_adj_targets,
        int* d_adj_checks_done,
        const int* d_adj_count,
        int max_epn,
        // Tag-based stale filtering
        const int* d_fq_tag,            // tag when edge was enqueued
        const int* d_fwd_expand_tag,    // current node tag
        // P3: freeby check
        const int* d_fwd_parent,
        const float* d_fwd_g,
        // P8: whitelist early-exit
        const uint8_t* d_adj_cc_status,
        // Diagnostics (FIX_B3): [3]=midpoints_checked, [4]=self_coll_hits, [5]=scene_coll_hits
        int* d_diag,
        // Output
        uint8_t* d_cc_results       // [n_eval]: 0=free, 1=collision, 2=stale (skipped)
    ) {
        int edge_idx = blockIdx.x;
        if (edge_idx >= n_eval) return;
        int tid = threadIdx.x;
        // 4 lanes cooperate on one midpoint, BATCH_SIZE midpoints in flight.
        const int batch_ind = tid / 4;

        int src = d_fq_src[edge_idx];
        int dst = d_fq_dst[edge_idx];

        // Tag-based stale-edge filtering: if src's tag changed since this
        // edge was enqueued, fwd_g[src] was improved and this edge is stale.
        if (d_fq_tag != nullptr && d_fwd_expand_tag != nullptr) {
            if (d_fq_tag[edge_idx] != d_fwd_expand_tag[src]) {
                if (tid == 0) d_cc_results[edge_idx] = 2; // stale
                return;
            }
        }

        // P3: Forward tree freeby — if dst is already child of src in forward tree,
        // the edge is already validated. Skip CC. (OMPL isInForwardTree, line 1609-1617)
        if (d_fwd_parent[dst] == src && d_fwd_g[dst] < INF_COST) {
            if (tid == 0) d_cc_results[edge_idx] = 0;  // free (already in tree)
            return;
        }

        extern __shared__ float smem[];
        float* q_src = smem;
        float* q_dst = q_src + dim;
        int n_spheres = model.n_spheres;
        int n_approx = model.n_approx_spheres;
        float* sphere_pos = q_dst + 2 * dim;
        float* approx_sphere_pos = sphere_pos + n_spheres * BATCH_SIZE * 3;
        float* T = approx_sphere_pos + n_approx * BATCH_SIZE * 3;
        volatile int* joint_in_collision = (volatile int*)(T + BATCH_SIZE * ppln::FK_T_SLOTS * 16);

        if (tid == 0) {
            for (int i = 0; i < dim; i++) {
                q_src[i] = d_nodes[src * dim + i];
                q_dst[i] = d_nodes[dst * dim + i];
            }
        }
        __syncthreads();

        __shared__ float s_edge_len;
        __shared__ int s_total_seg;
        if (tid == 0) {
            float len = 0.0f;
            for (int i = 0; i < dim; i++) {
                float diff = q_src[i] - q_dst[i];
                len += diff * diff;
            }
            s_edge_len = sqrtf(len);
            s_total_seg = max(1, (int)ceilf(s_edge_len / valid_segment_length));
        }
        __syncthreads();

        int total_checks = s_total_seg - 1; // interior midpoints only

        // Find adjacency slot
        __shared__ int s_adj_slot;
        __shared__ int s_checks_done;
        if (tid == 0) {
            s_adj_slot = -1;
            int n_edges = d_adj_count[src];
            if (n_edges > max_epn) n_edges = max_epn;
            for (int e = 0; e < n_edges; e++) {
                if (d_adj_targets[src * max_epn + e] == dst) {
                    s_adj_slot = src * max_epn + e;
                    break;
                }
            }
            s_checks_done = (s_adj_slot >= 0) ? d_adj_checks_done[s_adj_slot] : 0;
        }
        __syncthreads();

        int checks_done = s_checks_done;

        // P8: Whitelist early-exit — if edge is already fully validated, skip CC
        if (s_adj_slot >= 0 && d_adj_cc_status[s_adj_slot] == EDGE_WHITELIST) {
            if (tid == 0) d_cc_results[edge_idx] = 0;
            return;
        }

        __shared__ int s_collision;
        __shared__ ppln::collision::TwoPhaseFlags s_flags;
        if (tid == 0) s_collision = 0;
        __syncthreads();

        int new_checks = 0;

        // BFS midpoint order, BATCH_SIZE midpoints per pass — same layout and
        // same rationale as sparse_cc_kernel above, but with no sparse cap so
        // every interior midpoint is checked.
        for (int base = checks_done; base < total_checks; base += BATCH_SIZE) {
            if (s_collision) break;

            const int k = base + batch_ind;
            float t = 0.0f;
            if (k < total_checks) {
                const int kk = k + 1;
                const int lvl = 31 - __clz(kk);
                t = (float)(2 * (kk - (1 << lvl)) + 1) / (float)(1 << (lvl + 1));
            }
            float q_interp[ppln::MAX_DIM];
            for (int i = 0; i < dim; i++)
                q_interp[i] = q_src[i] * (1.0f - t) + q_dst[i] * t;

            ppln::collision::TwoPhaseResult res = ppln::collision::two_phase_cc(
                model, scene, q_interp, sphere_pos, approx_sphere_pos, T,
                joint_in_collision, &s_flags, tid, collision_margin,
                /*check_self=*/true);

            const int group = min(BATCH_SIZE, total_checks - base);

            // Diagnostics
            if (tid == 0 && d_diag) {
                atomicAdd(&d_diag[3], group);                                    // midpoints checked
                if (res.collision && !res.scene_hit) atomicAdd(&d_diag[4], 1);   // self-collision hits
                if (res.scene_hit) atomicAdd(&d_diag[5], 1);                     // scene-collision hits
            }

            if (res.collision) {
                if (tid == 0) s_collision = 1;
                __syncthreads();
                break;
            }
            new_checks += group;
        }

        if (tid == 0) {
            d_cc_results[edge_idx] = s_collision ? 1 : 0;
            // FIX_V7: Only accumulate checks_done for verified-free midpoints.
            // (Same fix as sparse_cc_kernel above.)
            if (s_adj_slot >= 0 && !s_collision) {
                atomicAdd(&d_adj_checks_done[s_adj_slot], new_checks);
                // D12: Symmetric checks_done — OMPL sets performedChecks for both
                // directions (isValidAtResolution lines 1561-1562).
                int n_edges_dst = d_adj_count[dst];
                if (n_edges_dst > max_epn) n_edges_dst = max_epn;
                for (int e = 0; e < n_edges_dst; e++) {
                    int adj_idx = dst * max_epn + e;
                    if (d_adj_targets[adj_idx] == src) {
                        atomicAdd(&d_adj_checks_done[adj_idx], new_checks);
                        break;
                    }
                }
            }
        }
    }

    // ========================================================================
    // Kernel 13: update_forward_tree_kernel
    // ========================================================================
    // Update fwd_g/fwd_parent for free edges, whitelist/blacklist in adjacency.
    __global__ void update_forward_tree_kernel(
        const int* d_fq_src,
        const int* d_fq_dst,
        const uint8_t* d_cc_results,
        int n_eval,
        float* d_fwd_g,
        int* d_fwd_parent,
        // Adjacency
        int* d_adj_targets_arr,
        float* d_adj_weights_arr,
        uint8_t* d_adj_cc_status,
        const int* d_adj_count,
        int max_epn,
        // Reverse tree (for checking if edge is in reverse tree)
        const float* d_rev_g,
        const int* d_rev_parent,
        // Output: newly reached for expansion + invalidated reverse edges
        int* d_newly_reached,
        int* d_n_newly_reached,      // [1] atomic
        int* d_rev_invalidated_nodes,
        int* d_n_rev_invalidated,    // [1] atomic
        int max_newly_reached,       // bounds for d_newly_reached writes
        int max_rev_invalidated,     // bounds for d_rev_invalidated_nodes writes
        // Tag-based stale filtering
        int* d_fwd_expand_tag,       // [max_nodes] incremented when fwd_g improves
        // P1b: solution cost pruning
        float solution_cost,
        const float* d_lb_ctg        // L2 heuristic fallback for ctg
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_eval) return;

        int src = d_fq_src[idx];
        int dst = d_fq_dst[idx];

        // Skip stale edges (result == 2 from tag-based filtering)
        if (d_cc_results[idx] == 2) return;

        bool collision = (d_cc_results[idx] != 0);

        // Find edge in adjacency
        int n_edges_src = d_adj_count[src];
        if (n_edges_src > max_epn) n_edges_src = max_epn;

        float edge_weight = INF_COST;
        for (int e = 0; e < n_edges_src; e++) {
            int adj_idx = src * max_epn + e;
            if (d_adj_targets_arr[adj_idx] == dst) {
                edge_weight = d_adj_weights_arr[adj_idx];
                d_adj_cc_status[adj_idx] = collision ? EDGE_BLACKLIST : EDGE_WHITELIST;
                break;
            }
        }

        // D12: Symmetric status propagation — OMPL whitelists/blacklists both directions
        // (isValidAtResolution lines 1533-1534, 1568-1569)
        {
            int n_edges_dst = d_adj_count[dst];
            if (n_edges_dst > max_epn) n_edges_dst = max_epn;
            for (int e = 0; e < n_edges_dst; e++) {
                int adj_idx = dst * max_epn + e;
                if (d_adj_targets_arr[adj_idx] == src) {
                    d_adj_cc_status[adj_idx] = collision ? EDGE_BLACKLIST : EDGE_WHITELIST;
                    break;
                }
            }
        }

        if (collision) {
            // Check if this edge was in the reverse tree
            // If rev_parent[dst] == src or rev_parent[src] == dst, reverse tree invalidated
            if (d_rev_parent[dst] == src || d_rev_parent[src] == dst) {
                int node_to_invalidate = (d_rev_parent[dst] == src) ? dst : src;
                int pos = atomicAdd(d_n_rev_invalidated, 1);
                if (pos < max_rev_invalidated) {
                    d_rev_invalidated_nodes[pos] = node_to_invalidate;
                }
            }
            return;
        }

        // Relaxation in forward tree
        float new_g = d_fwd_g[src] + edge_weight;

        // P1b: OMPL-faithful solution cost pruning (MITstar.cpp line 635)
        // Skip if this edge can't possibly improve the solution
        float ctg_dst = d_rev_g[dst];
        if (ctg_dst >= INF_COST) ctg_dst = d_lb_ctg[dst];
        if (new_g + ctg_dst >= solution_cost) return;

        float old_g = d_fwd_g[dst];
        atomicMinFloat(&d_fwd_g[dst], new_g);

        float current_g = d_fwd_g[dst];
        if (fabsf(current_g - new_g) < 1e-6f && new_g < old_g - 1e-6f) {
            d_fwd_parent[dst] = src;
            atomicAdd(&d_fwd_expand_tag[dst], 1);
            // Note: race can produce duplicate entries — benign, just causes
            // redundant queue expansion. Bounds check prevents OOB.
            int pos = atomicAdd(d_n_newly_reached, 1);
            if (pos < max_newly_reached) {
                d_newly_reached[pos] = dst;
            }
        }
    }

    // ========================================================================
    // Kernel 14: propagate_cost_kernel
    // ========================================================================
    // BFS cost propagation in forward tree. Each thread checks if its parent's
    // cost + edge weight improves its current cost. Iterative (run multiple times).
    __global__ void propagate_cost_kernel(
        float* d_fwd_g,
        const int* d_fwd_parent,
        const float* d_nodes,
        int n_total,
        int dim,
        int* d_changed,               // [1] atomic: set to 1 if any update
        // Adjacency lookup for clearance-weighted costs
        const int* d_adj_targets,
        const float* d_adj_weights,
        const int* d_adj_count,
        int max_epn,
        // Clearance fallback (when edge evicted from adjacency)
        const float* d_clearance,
        float clearance_weight,
        float clearance_epsilon,
        float max_clearance_penalty
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_total) return;

        int parent = d_fwd_parent[idx];
        if (parent == idx) return; // root or unconnected

        float parent_g = d_fwd_g[parent];
        if (parent_g >= INF_COST) return;

        // Look up edge weight from parent's adjacency list.
        // If clearance_weight > 0, d_adj_weights contains the clearance-augmented
        // cost and we must use it (not recompute L2). Falls back to clearance-
        // weighted L2 if edge not found (rare — only when adjacency was evicted).
        float edge_weight = INF_COST;
        int n_edges = d_adj_count[parent];
        if (n_edges > max_epn) n_edges = max_epn;
        for (int e = 0; e < n_edges; e++) {
            if (d_adj_targets[parent * max_epn + e] == idx) {
                edge_weight = d_adj_weights[parent * max_epn + e];
                break;
            }
        }
        // Fallback: recompute (with clearance penalty if enabled)
        if (edge_weight >= INF_COST) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = d_nodes[idx * dim + d] - d_nodes[parent * dim + d];
                dist += diff * diff;
            }
            float l2 = sqrtf(dist);
            edge_weight = l2;
            if (clearance_weight > 0.0f) {
                float mc = fmaxf(fminf(d_clearance[idx], d_clearance[parent]), clearance_epsilon);
                float penalty = fminf(clearance_weight / mc, max_clearance_penalty);
                edge_weight = l2 * (1.0f + penalty);
            }
        }

        float new_g = parent_g + edge_weight;
        if (new_g < d_fwd_g[idx] - 1e-6f) {
            atomicMinFloat(&d_fwd_g[idx], new_g);
            atomicExch(d_changed, 1);
        }
    }

    // ========================================================================
    // Kernel 15: prune_kernel
    // ========================================================================
    // Prune nodes where lb_ctc + lb_ctg > informed_bound.
    __global__ void prune_kernel(
        const float* d_lb_ctc,
        const float* d_lb_ctg,
        float* d_fwd_g,
        float* d_rev_g,
        int* d_fwd_parent,
        int* d_rev_parent,
        // Adjacency cleanup
        uint8_t* d_adj_cc_status,
        int* d_adj_count,
        int max_epn,
        int n_total,
        float informed_bound,
        int num_goals,
        int start_node_idx,
        // Output
        uint8_t* d_pruned               // [n_total]: 1 if pruned
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_total) return;

        // Never prune goals or start
        if (idx < num_goals || idx == start_node_idx) return;

        float f_hat = d_lb_ctc[idx] + d_lb_ctg[idx];
        if (f_hat > informed_bound) {
            d_pruned[idx] = 1;
            d_fwd_g[idx] = INF_COST;
            d_rev_g[idx] = INF_COST;
            d_fwd_parent[idx] = idx;
            d_rev_parent[idx] = idx;
            // Zero adjacency so other kernels skip edges to/from this node
            d_adj_count[idx] = 0;
        }
    }

    // ========================================================================
    // Kernel 15b: prune_cascade_kernel
    // ========================================================================
    // After pruning, cascade invalidation: if a node's fwd_parent was pruned
    // (fwd_g[parent] == INF), reset this node's fwd_g and parent too.
    // Run iteratively until no more changes (like Bellman-Ford).
    __global__ void prune_cascade_kernel(
        float* d_fwd_g,
        int* d_fwd_parent,
        float* d_rev_g,
        int* d_rev_parent,
        const uint8_t* d_pruned,
        int n_total,
        int num_goals,
        int start_node_idx,
        int* d_changed              // [1] atomic: set to 1 if any update
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_total) return;
        if (idx < num_goals || idx == start_node_idx) return;
        if (d_pruned[idx]) return;  // already pruned, skip

        // Forward tree: if parent is pruned or unreachable, disconnect
        int fwd_par = d_fwd_parent[idx];
        if (fwd_par != idx && d_fwd_g[fwd_par] >= INF_COST) {
            d_fwd_g[idx] = INF_COST;
            d_fwd_parent[idx] = idx;
            atomicExch(d_changed, 1);
        }

        // Note: reverse tree cascade is not needed here because
        // reset_reverse_heuristics_kernel (called after prune) resets all
        // rev_g to INF and rebuilds from goals. Only forward cascade matters.
    }

    // ========================================================================
    // Kernel 16: reset_reverse_heuristics_kernel
    // ========================================================================
    // Reset rev_g=INF, rev_parent=self for non-goal nodes.
    __global__ void reset_reverse_heuristics_kernel(
        float* d_rev_g,
        int* d_rev_parent,
        int n_total,
        int num_goals       // goals are nodes 0..num_goals-1, keep rev_g=0
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_total) return;
        if (idx < num_goals) return; // keep goals at rev_g=0

        d_rev_g[idx] = INF_COST;
        d_rev_parent[idx] = idx;
    }

    // ========================================================================
    // Host: compute_CL_matrix
    // ========================================================================
    static void compute_CL_matrix(
        const std::vector<float>& start,
        const std::vector<float>& goal,
        std::vector<float>& CL,
        int dim
    ) {
        CL.resize(dim * dim, 0.0f);

        std::vector<float> a1(dim);
        float norm = 0.0f;
        for (int i = 0; i < dim; i++) {
            a1[i] = goal[i] - start[i];
            norm += a1[i] * a1[i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-10f) {
            for (int i = 0; i < dim; i++) CL[i * dim + i] = 1.0f;
            return;
        }
        for (int i = 0; i < dim; i++) a1[i] /= norm;

        std::vector<std::vector<float>> basis(dim, std::vector<float>(dim, 0.0f));
        basis[0] = a1;

        int col = 1;
        for (int e = 0; e < dim && col < dim; e++) {
            std::vector<float> v(dim, 0.0f);
            v[e] = 1.0f;

            for (int b = 0; b < col; b++) {
                float dot = 0.0f;
                for (int i = 0; i < dim; i++) dot += v[i] * basis[b][i];
                for (int i = 0; i < dim; i++) v[i] -= dot * basis[b][i];
            }

            float v_norm = 0.0f;
            for (int i = 0; i < dim; i++) v_norm += v[i] * v[i];
            v_norm = std::sqrt(v_norm);

            if (v_norm > 1e-8f) {
                for (int i = 0; i < dim; i++) v[i] /= v_norm;
                basis[col] = v;
                col++;
            }
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                CL[i * dim + j] = basis[j][i];
            }
        }
    }

    // ========================================================================
    // Host: extract_path (forward tree: start→goal via fwd_parent)
    // ========================================================================
    static bool extract_path_forward(
        const int* h_fwd_parent,
        const float* h_nodes,
        int goal_node_idx,
        int start_node_idx,
        int n_total,
        int dim,
        float fwd_g_goal,
        std::vector<std::vector<float>>& path,
        float& actual_cost,
        std::vector<int>* path_node_indices = nullptr
    ) {
        if (fwd_g_goal >= INF_COST) return false;

        // Trace from goal back to start via fwd_parent
        path.clear();
        std::vector<int> indices;
        int current = goal_node_idx;
        int max_steps = n_total + 1;

        while (max_steps-- > 0) {
            std::vector<float> cfg(dim);
            for (int i = 0; i < dim; i++) {
                cfg[i] = h_nodes[current * dim + i];
            }
            path.push_back(cfg);
            indices.push_back(current);

            if (h_fwd_parent[current] == current) break; // reached root
            current = h_fwd_parent[current];
        }

        // Verify the chain actually reached the start node
        if (current != start_node_idx) return false;

        // Reverse to get start→goal order
        std::reverse(path.begin(), path.end());
        std::reverse(indices.begin(), indices.end());
        if (path.size() < 2) return false;

        // Recompute actual path cost from edge lengths (avoids parent-cost
        // race where atomicMinFloat wins but a different parent is written)
        actual_cost = 0.0f;
        for (size_t i = 1; i < path.size(); i++) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = path[i][d] - path[i-1][d];
                dist += diff * diff;
            }
            actual_cost += sqrtf(dist);
        }

        if (path_node_indices) *path_node_indices = std::move(indices);
        return true;
    }

    // ========================================================================
    // Host: solve_runtime_scene — Dual-search MIT* orchestration
    // ========================================================================
    MITStarResult solve_runtime_scene(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        SceneCollisionData& scene,
        MITStar_settings& settings,
        RobotModel& model,
        MITStarBuffers* bufs
    ) {
        auto wall_start = std::chrono::steady_clock::now();
        const int dim = model.n_dof;
        const int num_goals = static_cast<int>(goals.size());
        MITStarResult result;
        const bool use_bufs = (bufs != nullptr && bufs->owns_memory);

        const int max_nodes = settings.max_nodes;
        const int max_epn = settings.max_edges_per_node;
        const size_t adj_total = (size_t)max_nodes * max_epn;

        // Max queue sizes
        const int max_rq = settings.m_reverse_eval * 8;
        const int max_fq = settings.m_forward_eval * 8;
        const int max_newly_reached = max_nodes;
        const int max_rev_invalidated = max_nodes;

        // Compute joint-space volume for RGG radius (radius-NN)
        std::vector<float> h_joint_range(dim);
        cudaMemcpy(h_joint_range.data(), model.joint_range,
                   dim * sizeof(float), cudaMemcpyDeviceToHost);
        float joint_vol = 1.0f;
        for (int d = 0; d < dim; d++) joint_vol *= h_joint_range[d];
        // Unit-ball volume in d dimensions: V_d = pi^(d/2) / Gamma(d/2+1)
        float zeta_d = std::pow(M_PI, dim / 2.0f) / std::tgamma(dim / 2.0f + 1.0f);

        // Lambda: compute RGG radius for given n
        auto compute_rgg_radius_sq = [&](int n) -> float {
            if (n <= 1) return 1e10f;
            float r = settings.gamma_rgg
                    * std::pow(joint_vol / zeta_d, 1.0f / dim)
                    * std::pow(std::log(static_cast<float>(n)) / n, 1.0f / dim);
            return r * r;
        };

        // Lambda: compute k for k-NN (OMPL formula, used only for initial nodes)
        auto compute_knn_k = [&](int n) -> int {
            if (n <= 1) return 1;
            float log_n = std::log(static_cast<float>(n));
            int k = static_cast<int>(std::ceil(
                settings.eta_knn * std::exp(1.0f) * (1.0f + 1.0f / dim) * log_n));
            k = std::min(k, settings.max_neighbors);
            return std::max(k, 1);
        };

        // ====================================================================
        // GPU memory allocation
        // ====================================================================
        // Per-node arrays
        float* d_nodes;           // [max_nodes * dim]
        float* d_fwd_g;          // forward cost-to-come
        float* d_rev_g;          // reverse cost-to-come (= admissible cost-to-go)
        int* d_fwd_parent;
        int* d_rev_parent;
        float* d_lb_ctc;         // L2(start, x) — precomputed, immutable
        float* d_lb_ctg;         // L2(x, best_goal) — precomputed, immutable
        uint8_t* d_pruned;       // per-node pruned flag
        int* d_fwd_expand_tag;   // per-node expansion version tag (Phase 2.1)
        float* d_clearance;      // [max_nodes] per-node min ESDF clearance

        // Edge adjacency (fixed-stride)
        int* d_adj_targets;
        float* d_adj_weights;
        uint8_t* d_adj_cc_status;
        int* d_adj_checks_done;
        int* d_adj_count;
        float* d_adj_l2_dist;    // [adj_total] pure L2 distance (only when clearance_weight > 0)
        float* d_sample_clearance = nullptr;
        const bool use_clearance = settings.clearance_weight > 0.0f;

        // Reverse queue
        int* d_rq_src;
        int* d_rq_dst;
        float* d_rq_key_cost;
        float* d_rq_key_effort;
        int* d_rq_size;

        // Forward queue
        int* d_fq_src;
        int* d_fq_dst;
        float* d_fq_key_lb_cost;
        float* d_fq_key_est_cost;
        float* d_fq_key_est_effort;
        int* d_fq_size;
        int* d_fq_tag;           // per-edge expansion tag (Phase 2.1)

        // Sort temp buffers for reverse queue
        int* d_rq_src_sorted;
        int* d_rq_dst_sorted;
        float* d_rq_key_cost_sorted;
        float* d_rq_key_effort_sorted;

        // Sort temp buffers for forward queue
        int* d_fq_src_sorted;
        int* d_fq_dst_sorted;
        float* d_fq_key_lb_cost_sorted;
        float* d_fq_key_est_cost_sorted;
        float* d_fq_key_est_effort_sorted;
        int* d_fq_tag_sorted;

        // Sampling buffers
        float* d_new_samples;
        uint8_t* d_cc_results_sample;

        // NN buffers
        int* d_nn_counts;
        int* d_nn_indices;

        // Edge CC results
        int max_eval = std::max({max_rq, max_fq, settings.m_reverse_eval, settings.m_forward_eval});
        uint8_t* d_edge_cc_results;

        // Newly reached / invalidated / expand
        int* d_newly_reached;
        int* d_n_newly_reached;
        int* d_rev_invalidated;
        int* d_n_rev_invalidated;
        int* d_expand_nodes;

        // Propagation changed flag
        int* d_changed;

        // Start/goal on device
        float* d_start_cfg;
        float* d_best_goal_cfg;
        float* d_CL_matrix;

        // Halton + RNG
        HaltonState_rt* d_halton_states;
        curandState* d_rng_states;

        // Diagnostics
        int* d_diag;

        if (use_bufs) {
            // Use pre-allocated buffers — no cudaMalloc
            d_nodes = bufs->d_nodes;
            d_fwd_g = bufs->d_fwd_g;
            d_rev_g = bufs->d_rev_g;
            d_fwd_parent = bufs->d_fwd_parent;
            d_rev_parent = bufs->d_rev_parent;
            d_lb_ctc = bufs->d_lb_ctc;
            d_lb_ctg = bufs->d_lb_ctg;
            d_pruned = bufs->d_pruned;
            d_fwd_expand_tag = bufs->d_fwd_expand_tag;
            d_clearance = bufs->d_clearance;
            d_adj_targets = bufs->d_adj_targets;
            d_adj_weights = bufs->d_adj_weights;
            d_adj_cc_status = bufs->d_adj_cc_status;
            d_adj_checks_done = bufs->d_adj_checks_done;
            d_adj_count = bufs->d_adj_count;
            d_adj_l2_dist = bufs->d_adj_l2_dist;
            d_sample_clearance = bufs->d_sample_clearance;
            d_rq_src = bufs->d_rq_src;
            d_rq_dst = bufs->d_rq_dst;
            d_rq_key_cost = bufs->d_rq_key_cost;
            d_rq_key_effort = bufs->d_rq_key_effort;
            d_rq_size = bufs->d_rq_size;
            d_fq_src = bufs->d_fq_src;
            d_fq_dst = bufs->d_fq_dst;
            d_fq_key_lb_cost = bufs->d_fq_key_lb_cost;
            d_fq_key_est_cost = bufs->d_fq_key_est_cost;
            d_fq_key_est_effort = bufs->d_fq_key_est_effort;
            d_fq_size = bufs->d_fq_size;
            d_fq_tag = bufs->d_fq_tag;
            d_rq_src_sorted = bufs->d_rq_src_sorted;
            d_rq_dst_sorted = bufs->d_rq_dst_sorted;
            d_rq_key_cost_sorted = bufs->d_rq_key_cost_sorted;
            d_rq_key_effort_sorted = bufs->d_rq_key_effort_sorted;
            d_fq_src_sorted = bufs->d_fq_src_sorted;
            d_fq_dst_sorted = bufs->d_fq_dst_sorted;
            d_fq_key_lb_cost_sorted = bufs->d_fq_key_lb_cost_sorted;
            d_fq_key_est_cost_sorted = bufs->d_fq_key_est_cost_sorted;
            d_fq_key_est_effort_sorted = bufs->d_fq_key_est_effort_sorted;
            d_fq_tag_sorted = bufs->d_fq_tag_sorted;
            d_new_samples = bufs->d_new_samples;
            d_cc_results_sample = bufs->d_cc_results_sample;
            d_nn_counts = bufs->d_nn_counts;
            d_nn_indices = bufs->d_nn_indices;
            d_edge_cc_results = bufs->d_edge_cc_results;
            d_newly_reached = bufs->d_newly_reached;
            d_n_newly_reached = bufs->d_n_newly_reached;
            d_rev_invalidated = bufs->d_rev_invalidated;
            d_n_rev_invalidated = bufs->d_n_rev_invalidated;
            d_expand_nodes = bufs->d_expand_nodes;
            d_changed = bufs->d_changed;
            d_start_cfg = bufs->d_start_cfg;
            d_best_goal_cfg = bufs->d_best_goal_cfg;
            d_CL_matrix = bufs->d_CL_matrix;
            d_halton_states = (HaltonState_rt*)bufs->d_halton_states;
            d_rng_states = bufs->d_rng_states;
            d_diag = bufs->d_diag;
        } else {
            // Original per-call allocation
            cudaMalloc(&d_nodes,      (size_t)max_nodes * dim * sizeof(float));
            cudaMalloc(&d_fwd_g,      (size_t)max_nodes * sizeof(float));
            cudaMalloc(&d_rev_g,      (size_t)max_nodes * sizeof(float));
            cudaMalloc(&d_fwd_parent, (size_t)max_nodes * sizeof(int));
            cudaMalloc(&d_rev_parent, (size_t)max_nodes * sizeof(int));
            cudaMalloc(&d_lb_ctc,     (size_t)max_nodes * sizeof(float));
            cudaMalloc(&d_lb_ctg,     (size_t)max_nodes * sizeof(float));
            cudaMalloc(&d_pruned,     (size_t)max_nodes * sizeof(uint8_t));
            cudaMalloc(&d_fwd_expand_tag, (size_t)max_nodes * sizeof(int));
            cudaMalloc(&d_clearance,  (size_t)max_nodes * sizeof(float));

            cudaMalloc(&d_adj_targets,     adj_total * sizeof(int));
            cudaMalloc(&d_adj_weights,     adj_total * sizeof(float));
            cudaMalloc(&d_adj_cc_status,   adj_total * sizeof(uint8_t));
            cudaMalloc(&d_adj_checks_done, adj_total * sizeof(int));
            cudaMalloc(&d_adj_count,       (size_t)max_nodes * sizeof(int));
            if (use_clearance) {
                cudaMalloc(&d_adj_l2_dist, adj_total * sizeof(float));
                cudaMalloc(&d_sample_clearance, (size_t)settings.batch_size * sizeof(float));
            } else {
                d_adj_l2_dist = nullptr;
            }

            cudaMalloc(&d_rq_src,        (size_t)max_rq * sizeof(int));
            cudaMalloc(&d_rq_dst,        (size_t)max_rq * sizeof(int));
            cudaMalloc(&d_rq_key_cost,   (size_t)max_rq * sizeof(float));
            cudaMalloc(&d_rq_key_effort, (size_t)max_rq * sizeof(float));
            cudaMalloc(&d_rq_size,       sizeof(int));

            cudaMalloc(&d_fq_src,            (size_t)max_fq * sizeof(int));
            cudaMalloc(&d_fq_dst,            (size_t)max_fq * sizeof(int));
            cudaMalloc(&d_fq_key_lb_cost,    (size_t)max_fq * sizeof(float));
            cudaMalloc(&d_fq_key_est_cost,   (size_t)max_fq * sizeof(float));
            cudaMalloc(&d_fq_key_est_effort, (size_t)max_fq * sizeof(float));
            cudaMalloc(&d_fq_size,           sizeof(int));
            cudaMalloc(&d_fq_tag,            (size_t)max_fq * sizeof(int));

            cudaMalloc(&d_rq_src_sorted,         (size_t)max_rq * sizeof(int));
            cudaMalloc(&d_rq_dst_sorted,         (size_t)max_rq * sizeof(int));
            cudaMalloc(&d_rq_key_cost_sorted,    (size_t)max_rq * sizeof(float));
            cudaMalloc(&d_rq_key_effort_sorted,  (size_t)max_rq * sizeof(float));

            cudaMalloc(&d_fq_src_sorted,            (size_t)max_fq * sizeof(int));
            cudaMalloc(&d_fq_dst_sorted,            (size_t)max_fq * sizeof(int));
            cudaMalloc(&d_fq_key_lb_cost_sorted,    (size_t)max_fq * sizeof(float));
            cudaMalloc(&d_fq_key_est_cost_sorted,   (size_t)max_fq * sizeof(float));
            cudaMalloc(&d_fq_key_est_effort_sorted, (size_t)max_fq * sizeof(float));
            cudaMalloc(&d_fq_tag_sorted,            (size_t)max_fq * sizeof(int));

            cudaMalloc(&d_new_samples,       (size_t)settings.batch_size * dim * sizeof(float));
            cudaMalloc(&d_cc_results_sample, (size_t)settings.batch_size * sizeof(uint8_t));

            cudaMalloc(&d_nn_counts,  (size_t)settings.batch_size * sizeof(int));
            cudaMalloc(&d_nn_indices, (size_t)settings.batch_size * settings.max_neighbors * sizeof(int));

            cudaMalloc(&d_edge_cc_results, (size_t)max_eval * sizeof(uint8_t));

            cudaMalloc(&d_newly_reached,     (size_t)max_newly_reached * sizeof(int));
            cudaMalloc(&d_n_newly_reached,   sizeof(int));
            cudaMalloc(&d_rev_invalidated,   (size_t)max_nodes * sizeof(int));
            cudaMalloc(&d_n_rev_invalidated, sizeof(int));
            cudaMalloc(&d_expand_nodes,      (size_t)max_nodes * sizeof(int));

            cudaMalloc(&d_changed, sizeof(int));

            cudaMalloc(&d_start_cfg,     dim * sizeof(float));
            cudaMalloc(&d_best_goal_cfg, dim * sizeof(float));
            cudaMalloc(&d_CL_matrix,     dim * dim * sizeof(float));

            cudaMalloc(&d_halton_states, settings.batch_size * sizeof(HaltonState_rt));
            cudaMalloc(&d_rng_states,    settings.batch_size * sizeof(curandState));

            cudaMalloc(&d_diag, 6 * sizeof(int));
        }
        // FIX_B3 diagnostics: always reset
        cudaMemset(d_diag, 0, 6 * sizeof(int));

        cudaCheckError(cudaGetLastError());

        // ====================================================================
        // Initialize
        // ====================================================================
        // Init RNG and Halton
        {
            int nblk = (settings.batch_size + MITSTAR_BLOCK_SIZE - 1) / MITSTAR_BLOCK_SIZE;
            init_rng_mitstar<<<nblk, MITSTAR_BLOCK_SIZE>>>(d_rng_states, 42, settings.batch_size);
            init_halton_mitstar<<<nblk, MITSTAR_BLOCK_SIZE>>>(
                d_halton_states, d_rng_states, settings.batch_size, dim);
            cudaDeviceSynchronize();
        }

        // Compute c_min and best goal BEFORE init_state_kernel (P3: lb_ctg needs correct goal)
        float c_min = INF_COST;
        int best_goal_idx = 0;
        for (int i = 0; i < num_goals; i++) {
            float d = 0.0f;
            for (int j = 0; j < dim; j++) {
                float diff = start[j] - goals[i][j];
                d += diff * diff;
            }
            d = std::sqrt(d);
            if (d < c_min) {
                c_min = d;
                best_goal_idx = i;
            }
        }

        // Init all arrays
        {
            std::vector<float> h_inf(max_nodes, INF_COST);
            cudaMemcpy(d_fwd_g, h_inf.data(), max_nodes * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rev_g, h_inf.data(), max_nodes * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_lb_ctc, h_inf.data(), max_nodes * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_lb_ctg, h_inf.data(), max_nodes * sizeof(float), cudaMemcpyHostToDevice);

            // Init clearance to large value (= far from obstacles)
            std::vector<float> h_large(max_nodes, 1000.0f);
            cudaMemcpy(d_clearance, h_large.data(), max_nodes * sizeof(float), cudaMemcpyHostToDevice);

            std::vector<int> h_self(max_nodes);
            std::iota(h_self.begin(), h_self.end(), 0);
            cudaMemcpy(d_fwd_parent, h_self.data(), max_nodes * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rev_parent, h_self.data(), max_nodes * sizeof(int), cudaMemcpyHostToDevice);

            cudaMemset(d_adj_count, 0, max_nodes * sizeof(int));
            cudaMemset(d_adj_checks_done, 0, adj_total * sizeof(int));
            cudaMemset(d_adj_cc_status, EDGE_UNKNOWN, adj_total * sizeof(uint8_t));
            cudaMemset(d_pruned, 0, max_nodes * sizeof(uint8_t));
            cudaMemset(d_fwd_expand_tag, 0, max_nodes * sizeof(int));
        }

        // Place goals (nodes 0..num_goals-1) and start (node num_goals)
        int n_total = num_goals + 1;
        const int start_node_idx = num_goals;

        {
            std::vector<float> goals_flat(num_goals * dim);
            for (int i = 0; i < num_goals; i++) {
                std::copy_n(goals[i].data(), dim, goals_flat.data() + i * dim);
            }
            cudaMemcpy(d_nodes, goals_flat.data(), num_goals * dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_nodes + start_node_idx * dim, start.data(), dim * sizeof(float), cudaMemcpyHostToDevice);

            // Goals: rev_g = 0 (sources of reverse search)
            std::vector<float> h_zero(num_goals, 0.0f);
            cudaMemcpy(d_rev_g, h_zero.data(), num_goals * sizeof(float), cudaMemcpyHostToDevice);

            // Start: fwd_g = 0 (source of forward search)
            float zero = 0.0f;
            cudaMemcpy(d_fwd_g + start_node_idx, &zero, sizeof(float), cudaMemcpyHostToDevice);

            cudaMemcpy(d_start_cfg, start.data(), dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_best_goal_cfg, goals[best_goal_idx].data(), dim * sizeof(float), cudaMemcpyHostToDevice);
        }

        // Compute lb_ctc, lb_ctg for initial nodes
        {
            int nblk = (n_total + MITSTAR_BLOCK_SIZE - 1) / MITSTAR_BLOCK_SIZE;
            init_state_kernel<<<nblk, MITSTAR_BLOCK_SIZE>>>(
                d_nodes, d_fwd_g, d_rev_g, d_fwd_parent, d_rev_parent,
                d_lb_ctc, d_lb_ctg, 0, n_total,
                d_start_cfg, d_best_goal_cfg, dim);
            cudaDeviceSynchronize();

            // Restore fwd_g[start]=0 and rev_g[goals]=0 (init_state sets them to INF)
            float zero = 0.0f;
            cudaMemcpy(d_fwd_g + start_node_idx, &zero, sizeof(float), cudaMemcpyHostToDevice);
            std::vector<float> h_zero(num_goals, 0.0f);
            cudaMemcpy(d_rev_g, h_zero.data(), num_goals * sizeof(float), cudaMemcpyHostToDevice);

            // Restore parents
            std::vector<int> h_self_init(n_total);
            std::iota(h_self_init.begin(), h_self_init.end(), 0);
            cudaMemcpy(d_fwd_parent, h_self_init.data(), n_total * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rev_parent, h_self_init.data(), n_total * sizeof(int), cudaMemcpyHostToDevice);
        }

        // Compute ESDF clearance for initial nodes (goals + start)
        if (use_clearance) {
            int smem_clearance = (model.n_spheres * BATCH_SIZE * 3 +
                                  BATCH_SIZE * ppln::FK_T_SLOTS * 16 +
                                  dim) * sizeof(float);
            if (smem_clearance > 48 * 1024) {
                cudaFuncSetAttribute(compute_node_clearance_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_clearance);
            }
            compute_node_clearance_kernel<<<n_total, 4, smem_clearance>>>(
                d_nodes, d_clearance, nullptr, 0, n_total,
                model, scene, dim, settings.collision_margin);
            cudaDeviceSynchronize();
        }

        // State variables
        float solution_cost = INF_COST;
        float eis_cost = INF_COST;
        bool has_solution = false;
        bool has_eis = false;
        int best_goal_found = -1;
        int total_forward_evals = 0;
        int total_reverse_evals = 0;
        int reverse_restarts = 0;
        int total_iterations = 0;
        float last_improvement_ms = 0.0f;  // elapsed_ms when cost last improved

        // Sparse CC resolution tracking
        int num_sparse_checks = std::max(1, (int)std::round(
            1.0f / (settings.valid_segment_length * settings.initial_sparse_factor)));
        bool fq_needs_rebuild = false;
        bool queue_sizes_dirty = true; // must be before lambdas that use it
        bool queue_min_keys_dirty = true;

        // CL matrix for informed sampling
        std::vector<float> h_CL;

        // Shared memory sizes
        int smem_batch_cc = (model.n_spheres * BATCH_SIZE * 3 +
                             model.n_approx_spheres * BATCH_SIZE * 3 +
                             BATCH_SIZE * ppln::FK_T_SLOTS * 16 +
                             BATCH_SIZE * 20 +
                             dim) * sizeof(float);

        int smem_edge_cc = (dim * 3 +
                            model.n_spheres * BATCH_SIZE * 3 +
                            model.n_approx_spheres * BATCH_SIZE * 3 +
                            BATCH_SIZE * ppln::FK_T_SLOTS * 16 +
                            BATCH_SIZE * 20) * sizeof(float);

        // Set extended shared memory for kernels if needed
        if (smem_batch_cc > 48 * 1024) {
            cudaFuncSetAttribute(batch_cc_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_batch_cc);
        }
        if (smem_edge_cc > 48 * 1024) {
            cudaFuncSetAttribute(sparse_cc_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_edge_cc);
            cudaFuncSetAttribute(full_cc_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_edge_cc);
        }

        // CUB sort temp storage
        size_t sort_temp_bytes_rq = 0;
        size_t sort_temp_bytes_fq = 0;
        void* d_sort_temp_rq;
        void* d_sort_temp_fq;
        void* d_reduce_temp;
        size_t max_reduce_temp = 0;
        float* d_min_rq_cost;
        float* d_min_fq_lb_cost;
        size_t cosort_temp_bytes = 0;
        void* d_cosort_temp;

        if (use_bufs) {
            d_sort_temp_rq = bufs->d_sort_temp_rq;
            sort_temp_bytes_rq = bufs->sort_temp_bytes_rq;
            d_sort_temp_fq = bufs->d_sort_temp_fq;
            sort_temp_bytes_fq = bufs->sort_temp_bytes_fq;
            d_reduce_temp = bufs->d_reduce_temp;
            max_reduce_temp = bufs->reduce_temp_bytes;
            d_min_rq_cost = bufs->d_min_rq_cost;
            d_min_fq_lb_cost = bufs->d_min_fq_lb_cost;
            d_cosort_temp = bufs->d_cosort_temp;
            cosort_temp_bytes = bufs->cosort_temp_bytes;
        } else {
            cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_bytes_rq,
                d_rq_key_cost, d_rq_key_cost_sorted,
                d_rq_src, d_rq_src_sorted, max_rq);
            cudaMalloc(&d_sort_temp_rq, std::max(sort_temp_bytes_rq, (size_t)64));

            cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_bytes_fq,
                d_fq_key_lb_cost, d_fq_key_lb_cost_sorted,
                d_fq_src, d_fq_src_sorted, max_fq);
            cudaMalloc(&d_sort_temp_fq, std::max(sort_temp_bytes_fq, (size_t)64));

            size_t reduce_temp_bytes = 0;
            cub::DeviceReduce::Min(nullptr, reduce_temp_bytes,
                d_rq_key_cost, d_rq_key_cost_sorted, max_rq);
            size_t reduce_temp_bytes_fq = 0;
            cub::DeviceReduce::Min(nullptr, reduce_temp_bytes_fq,
                d_fq_key_lb_cost, d_fq_key_lb_cost_sorted, max_fq);
            max_reduce_temp = std::max(reduce_temp_bytes, reduce_temp_bytes_fq);
            cudaMalloc(&d_reduce_temp, std::max(max_reduce_temp, (size_t)64));

            cudaMalloc(&d_min_rq_cost,    sizeof(float));
            cudaMalloc(&d_min_fq_lb_cost, sizeof(float));

            size_t cosort_temp_bytes_float = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, cosort_temp_bytes_float,
                d_fq_key_lb_cost, d_fq_key_lb_cost_sorted,
                d_fq_key_est_cost, d_fq_key_est_cost_sorted, max_fq);
            size_t cosort_temp_bytes_int = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, cosort_temp_bytes_int,
                d_fq_key_lb_cost, d_fq_key_lb_cost_sorted,
                d_fq_dst, d_fq_dst_sorted, max_fq);
            cosort_temp_bytes = std::max({cosort_temp_bytes_float, cosort_temp_bytes_int, (size_t)64});
            cudaMalloc(&d_cosort_temp, cosort_temp_bytes);
        }

        cudaCheckError(cudaGetLastError());

        // ====================================================================
        // Helper lambdas
        // ====================================================================
        auto elapsed_ms = [&]() -> float {
            return std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - wall_start).count();
        };

        auto nblk = [](int n, int bs) { return (n + bs - 1) / bs; };

        // Helper: co-sort an int array by float keys
        auto cosort_int_by_keys = [&d_cosort_temp, cosort_temp_bytes](
            float* d_keys_in, float* d_keys_out,
            int* d_vals_in, int* d_vals_out, int n
        ) {
            size_t temp_bytes = cosort_temp_bytes;
            cub::DeviceRadixSort::SortPairs(d_cosort_temp, temp_bytes,
                d_keys_in, d_keys_out, d_vals_in, d_vals_out, n);
            cudaDeviceSynchronize();
        };

        // Helper: co-sort a float array by float keys
        auto cosort_float_by_keys = [&d_cosort_temp, cosort_temp_bytes](
            float* d_keys_in, float* d_keys_out,
            float* d_vals_in, float* d_vals_out, int n
        ) {
            // Reinterpret floats as unsigned ints for SortPairs (preserves order for non-negative floats)
            // Use a gather approach: sort indices by keys, then gather values
            // Actually, CUB SortPairs works with float values too — it sorts keys and permutes values.
            // But CUB SortPairs requires keys+values of same or compatible types.
            // Simpler approach: use the sorted key order to build a permutation.
            // Actually the simplest: just do another SortPairs with the same keys but float values.
            // CUB treats values as opaque — just needs same count. float and int are both 4 bytes.
            size_t temp_bytes = cosort_temp_bytes;
            cub::DeviceRadixSort::SortPairs(d_cosort_temp, temp_bytes,
                d_keys_in, d_keys_out,
                reinterpret_cast<int*>(d_vals_in), reinterpret_cast<int*>(d_vals_out), n);
            cudaDeviceSynchronize();
        };

        // ====================================================================
        // AdaptiveBatchSize (the "A" in MIT*) — LOG/sigmoid decay.
        // Mirrors ompl::geometric::mitstar::AdaptiveBatchSize::adjustBatchSizeLog.
        // Before the first solution (isinf cost) returns the full batch_size (max).
        // As the informed-set ellipse area S=pi*a*b shrinks with an improving
        // solution, the per-batch sample count decays toward min_batch_size.
        // Returned value is always in [min_batch_size, batch_size], so downstream
        // buffers (allocated for batch_size) are never overrun.
        // ====================================================================
        float S_max_adaptive = -1.0f;
        auto adaptive_batch_size = [&]() -> int {
            if (!settings.adaptive_batch) return settings.batch_size;
            if (!(solution_cost < INF_COST)) return settings.batch_size;  // isinf => max
            double a = (double)solution_cost / 2.0;
            double c = (double)c_min / 2.0;
            double b2 = a * a - c * c;
            if (b2 <= 0.0) return settings.batch_size;
            double b = std::sqrt(b2);
            double S = M_PI * a * b;
            if (S_max_adaptive < 0.0f) S_max_adaptive = (float)S;  // fix S_max on first finite call
            double ratio = (S_max_adaptive > 0.0) ? (S / (double)S_max_adaptive) : 1.0;
            int minS = settings.min_batch_size;
            int maxS = settings.batch_size;
            double lambda = (double)(minS + maxS) / (double)dim;
            double smoothed = 1.0 / (1.0 + std::exp(-10.0 * (ratio - 0.5)));
            double decay = std::log(1.0 + lambda * smoothed) / std::log(1.0 + lambda);
            int bs = minS + (int)((double)(maxS - minS) * decay);
            return std::max(minS, std::min(maxS, bs));
        };

        // ====================================================================
        // improve_approximation — sample, NN, insert edges, prune, restart reverse
        // ====================================================================
        auto improve_approximation = [&]() {
            result.total_batches++;
            int cur_batch_size = adaptive_batch_size();
            // FIX_V6: Reset sparse resolution to initial level on new batch,
            // matching OMPL MIT* improveApproximation() behavior.
            num_sparse_checks = std::max(1, (int)std::round(
                1.0f / (settings.valid_segment_length * settings.initial_sparse_factor)));
            float informed_bound = std::min(solution_cost, eis_cost);

            // Sample batch
            sample_batch_kernel<<<nblk(cur_batch_size, 128), 128>>>(
                d_new_samples, d_halton_states, d_rng_states, model,
                cur_batch_size, dim,
                (has_solution || has_eis), informed_bound,
                d_start_cfg, d_best_goal_cfg,
                (has_solution || has_eis) ? d_CL_matrix : nullptr,
                c_min
            );
            cudaDeviceSynchronize();

            // Batch CC
            batch_cc_kernel<<<cur_batch_size, 4, smem_batch_cc>>>(
                d_new_samples, d_cc_results_sample,
                cur_batch_size, model, scene, dim,
                settings.collision_margin,
                d_sample_clearance, settings.clearance_weight
            );
            cudaDeviceSynchronize();

            // Host-side compaction
            std::vector<uint8_t> h_cc(cur_batch_size);
            cudaMemcpy(h_cc.data(), d_cc_results_sample, cur_batch_size, cudaMemcpyDeviceToHost);

            std::vector<int> valid_indices;
            for (int i = 0; i < cur_batch_size; i++) {
                if (h_cc[i] == 0) valid_indices.push_back(i);
            }
            int n_valid = static_cast<int>(valid_indices.size());
            if (n_valid == 0) return;

            if (n_total + n_valid > max_nodes) {
                n_valid = max_nodes - n_total;
                if (n_valid <= 0) return;
            }

            // Compact and upload
            std::vector<float> h_samples(cur_batch_size * dim);
            cudaMemcpy(h_samples.data(), d_new_samples,
                       cur_batch_size * dim * sizeof(float), cudaMemcpyDeviceToHost);

            std::vector<float> h_valid(n_valid * dim);
            for (int i = 0; i < n_valid; i++) {
                std::copy_n(h_samples.data() + valid_indices[i] * dim, dim,
                           h_valid.data() + i * dim);
            }
            cudaMemcpy(d_nodes + n_total * dim, h_valid.data(),
                       n_valid * dim * sizeof(float), cudaMemcpyHostToDevice);

            int n_existing = n_total;
            n_total += n_valid;

            // Compress and upload clearance for valid samples
            if (use_clearance) {
                std::vector<float> h_sample_clearance(cur_batch_size);
                cudaMemcpy(h_sample_clearance.data(), d_sample_clearance,
                           cur_batch_size * sizeof(float), cudaMemcpyDeviceToHost);
                std::vector<float> h_valid_clearance(n_valid);
                for (int i = 0; i < n_valid; i++) {
                    h_valid_clearance[i] = h_sample_clearance[valid_indices[i]];
                }
                cudaMemcpy(d_clearance + n_existing, h_valid_clearance.data(),
                           n_valid * sizeof(float), cudaMemcpyHostToDevice);
            }

            // Init state for new nodes
            {
                int nb = nblk(n_valid, MITSTAR_BLOCK_SIZE);
                init_state_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                    d_nodes, d_fwd_g, d_rev_g, d_fwd_parent, d_rev_parent,
                    d_lb_ctc, d_lb_ctg, n_existing, n_total,
                    d_start_cfg, d_best_goal_cfg, dim);
                cudaDeviceSynchronize();
            }

            // Radius-NN for new batch nodes (fast: O(n) with early exit)
            float rnn_radius_sq = compute_rgg_radius_sq(n_total);
            int smem_rnn = (dim + RNN_TILE_SIZE * dim) * sizeof(float) + sizeof(int);
            radius_nn_kernel<<<n_valid, MITSTAR_BLOCK_SIZE, smem_rnn>>>(
                d_nodes, n_existing, n_valid, n_total, dim,
                rnn_radius_sq, settings.max_neighbors,
                d_nn_counts, d_nn_indices
            );
            cudaDeviceSynchronize();

            // Insert edges (bidirectional)
            {
                int nb = nblk(n_valid, MITSTAR_BLOCK_SIZE);
                insert_edges_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                    d_nn_counts, d_nn_indices,
                    n_existing, n_valid, settings.max_neighbors,
                    max_epn,
                    d_adj_targets, d_adj_weights, d_adj_cc_status,
                    d_adj_checks_done, d_adj_count,
                    d_nodes, dim,
                    d_clearance, settings.clearance_weight,
                    settings.clearance_epsilon, settings.max_clearance_penalty,
                    d_adj_l2_dist
                );
                cudaDeviceSynchronize();
            }

            // Note: k-NN for initial nodes is only needed on the first batch
            // (handled in the initial setup before the main loop). On subsequent
            // batches, initial nodes already have edges and new nodes connect to
            // them via bidirectional radius-NN. Re-running k-NN here would just
            // insert duplicate edges, wasting d_adj_count slots.

            // Prune if we have a bound
            if (informed_bound < INF_COST) {
                int nb = nblk(n_total, MITSTAR_BLOCK_SIZE);
                prune_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                    d_lb_ctc, d_lb_ctg, d_fwd_g, d_rev_g,
                    d_fwd_parent, d_rev_parent,
                    d_adj_cc_status, d_adj_count, max_epn,
                    n_total, informed_bound,
                    num_goals, start_node_idx, d_pruned
                );
                cudaDeviceSynchronize();

                // Cascade: disconnect children of pruned nodes (iterative)
                // Loop until convergence — tree depth can exceed 5 hops
                for (int cascade_iter = 0; cascade_iter < n_total; cascade_iter++) {
                    cudaMemset(d_changed, 0, sizeof(int));
                    prune_cascade_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                        d_fwd_g, d_fwd_parent, d_rev_g, d_rev_parent,
                        d_pruned, n_total, num_goals, start_node_idx, d_changed
                    );
                    cudaDeviceSynchronize();
                    int h_changed;
                    cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
                    if (!h_changed) break;
                }
            }

            // Restart reverse search
            reverse_restarts++;
            // OMPL: improveApproximation resets sparse to initial level
            // (doubling is handled in iterate_forward_search on each invalidation event)

            // Reset reverse heuristics
            {
                int nb = nblk(n_total, MITSTAR_BLOCK_SIZE);
                reset_reverse_heuristics_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                    d_rev_g, d_rev_parent, n_total, num_goals);
                cudaDeviceSynchronize();
            }

            // FIX_V6: CC cache (blacklist, whitelist, checks_done) is PERMANENT
            // across reverse restarts, matching OMPL MIT* reference implementation.
            // Blacklisted edges stay blacklisted; partially-checked edges retain
            // their checks_done progress. Only the reverse tree (rev_g, rev_parent,
            // queue) is reset above.

            // Rebuild reverse queue from goals
            cudaMemset(d_rq_size, 0, sizeof(int));
            {
                // Upload goal indices as expand nodes
                std::vector<int> h_goals(num_goals);
                std::iota(h_goals.begin(), h_goals.end(), 0);
                cudaMemcpy(d_expand_nodes, h_goals.data(), num_goals * sizeof(int), cudaMemcpyHostToDevice);

                int nb = nblk(num_goals, MITSTAR_BLOCK_SIZE);
                build_reverse_queue_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                    d_expand_nodes, num_goals,
                    d_rev_g, d_lb_ctc,
                    d_adj_targets, d_adj_weights, d_adj_cc_status,
                    d_adj_checks_done, d_adj_count, max_epn,
                    settings.valid_segment_length, num_sparse_checks,
                    d_pruned, solution_cost, d_adj_l2_dist,
                    d_rq_src, d_rq_dst, d_rq_key_cost, d_rq_key_effort,
                    d_rq_size, max_rq
                );
                cudaDeviceSynchronize();
            }

            // Rebuild forward queue from all forward-reached nodes (fwd_g < INF).
            // Note: OMPL uses expandStartVerticesIntoForwardQueue() (start only),
            // but GPU batched processing benefits from seeding fq with the full
            // forward tree to avoid starvation when forward search processes many
            // edges per iteration.
            cudaMemset(d_fq_size, 0, sizeof(int));
            {
                std::vector<float> h_fwd_g_all(n_total);
                cudaMemcpy(h_fwd_g_all.data(), d_fwd_g, n_total * sizeof(float), cudaMemcpyDeviceToHost);

                std::vector<int> h_fwd_reached;
                for (int i = 0; i < n_total; i++) {
                    if (h_fwd_g_all[i] < INF_COST) {
                        h_fwd_reached.push_back(i);
                    }
                }
                int n_fwd_reached = static_cast<int>(h_fwd_reached.size());
                if (n_fwd_reached > 0) {
                    cudaMemcpy(d_expand_nodes, h_fwd_reached.data(),
                               n_fwd_reached * sizeof(int), cudaMemcpyHostToDevice);

                    int nb = nblk(n_fwd_reached, MITSTAR_BLOCK_SIZE);
                    build_forward_queue_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                        d_expand_nodes, n_fwd_reached,
                        d_fwd_g, d_rev_g, d_lb_ctg,
                        d_adj_targets, d_adj_weights, d_adj_cc_status,
                        d_adj_checks_done, d_adj_count, max_epn,
                        settings.valid_segment_length, solution_cost,
                        d_fwd_expand_tag, d_pruned, d_adj_l2_dist,
                        d_fq_src, d_fq_dst, d_fq_key_lb_cost,
                        d_fq_key_est_cost, d_fq_key_est_effort,
                        d_fq_tag, d_fq_size, max_fq
                    );
                    cudaDeviceSynchronize();
                }
            }

            queue_sizes_dirty = true; queue_min_keys_dirty = true;
        };

        // ====================================================================
        // iterate_reverse_search — batched BFS waves with periodic interleave check
        // ====================================================================
        // S3: Only 1 reverse wave per check — prevents draining rq in one call,
        // allows forward/reverse to interleave more like OMPL's per-edge alternation.
        constexpr int MAX_REVERSE_WAVES_PER_CHECK = 1;

        auto iterate_reverse_search = [&]() {
            for (int wave = 0; wave < MAX_REVERSE_WAVES_PER_CHECK; wave++) {
                int h_rq_size;
                cudaMemcpy(&h_rq_size, d_rq_size, sizeof(int), cudaMemcpyDeviceToHost);
                if (h_rq_size == 0) break;
                h_rq_size = std::min(h_rq_size, max_rq);

                // FIX_B1: Batch reverse search like forward search.
                // Previously processed ALL edges in one wave, starving
                // the forward search.  Now respects m_reverse_eval.
                int m = std::min(settings.m_reverse_eval, h_rq_size);

                // H1 FIX: Sort reverse queue by cost key (ascending) so the
                // top-m edges have the lowest cost-to-come estimates.
                // OMPL MIT* processes reverse edges in priority order; without
                // sort, GPU batched processing picked arbitrary FIFO edges,
                // producing suboptimal rev_g that degraded forward queue keys.
                {
                    size_t tmp = sort_temp_bytes_rq;
                    cub::DeviceRadixSort::SortPairs(d_sort_temp_rq, tmp,
                        d_rq_key_cost, d_rq_key_cost_sorted,
                        d_rq_src, d_rq_src_sorted, h_rq_size);
                    cudaDeviceSynchronize();
                    cosort_int_by_keys(d_rq_key_cost, d_rq_key_cost_sorted,
                                       d_rq_dst, d_rq_dst_sorted, h_rq_size);
                    cosort_float_by_keys(d_rq_key_cost, d_rq_key_cost_sorted,
                                         d_rq_key_effort, d_rq_key_effort_sorted, h_rq_size);
                }

                // Copy sorted results back to the primary arrays so all
                // downstream kernels use the same d_rq_src/d_rq_dst pointers.
                cudaMemcpy(d_rq_src, d_rq_src_sorted, h_rq_size * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_rq_dst, d_rq_dst_sorted, h_rq_size * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_rq_key_cost, d_rq_key_cost_sorted, h_rq_size * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_rq_key_effort, d_rq_key_effort_sorted, h_rq_size * sizeof(float), cudaMemcpyDeviceToDevice);

                // Sparse CC on top-m edges
                sparse_cc_kernel<<<m, 64, smem_edge_cc>>>(
                    d_rq_src, d_rq_dst,
                    m, d_nodes, dim,
                    settings.valid_segment_length, num_sparse_checks,
                    model, scene, settings.collision_margin,
                    d_adj_targets, d_adj_checks_done, d_adj_count, max_epn,
                    d_rev_parent, d_rev_g,
                    d_adj_cc_status,
                    d_diag,
                    d_edge_cc_results
                );
                cudaDeviceSynchronize();
                total_reverse_evals += m;

                // Update reverse tree
                cudaMemset(d_n_newly_reached, 0, sizeof(int));
                {
                    int nb = nblk(m, MITSTAR_BLOCK_SIZE);
                    update_reverse_tree_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                        d_rq_src, d_rq_dst, d_edge_cc_results,
                        d_adj_weights, m,
                        d_rev_g, d_rev_parent,
                        d_adj_targets, d_adj_cc_status, d_adj_count, d_adj_weights,
                        max_epn,
                        d_newly_reached, d_n_newly_reached, max_newly_reached
                    );
                    cudaDeviceSynchronize();
                }

                // Expand newly reached nodes → append to rq (preserve unprocessed edges)
                int h_n_newly;
                cudaMemcpy(&h_n_newly, d_n_newly_reached, sizeof(int), cudaMemcpyDeviceToHost);

                // C2 FIX: Compact unprocessed edges to front instead of discarding them.
                // Previously cudaMemset(d_rq_size, 0) lost all h_rq_size - m remaining edges.
                // Now: shift remaining sorted edges down, then append new edges from newly_reached.
                // Copy from d_rq_*_sorted (still intact) to avoid overlapping D2D memcpy.
                int remaining = h_rq_size - m;
                if (remaining > 0) {
                    cudaMemcpy(d_rq_src, d_rq_src_sorted + m, remaining * sizeof(int), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_rq_dst, d_rq_dst_sorted + m, remaining * sizeof(int), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_rq_key_cost, d_rq_key_cost_sorted + m, remaining * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_rq_key_effort, d_rq_key_effort_sorted + m, remaining * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                // Set queue size to remaining (build_reverse_queue_kernel will atomicAdd from here)
                cudaMemcpy(d_rq_size, &remaining, sizeof(int), cudaMemcpyHostToDevice);

                if (h_n_newly > 0) {
                    h_n_newly = std::min(h_n_newly, max_newly_reached);
                    int nb = nblk(h_n_newly, MITSTAR_BLOCK_SIZE);
                    build_reverse_queue_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                        d_newly_reached, h_n_newly,
                        d_rev_g, d_lb_ctc,
                        d_adj_targets, d_adj_weights, d_adj_cc_status,
                        d_adj_checks_done, d_adj_count, max_epn,
                        settings.valid_segment_length, num_sparse_checks,
                        d_pruned, solution_cost, d_adj_l2_dist,
                        d_rq_src, d_rq_dst, d_rq_key_cost, d_rq_key_effort,
                        d_rq_size, max_rq
                    );
                    cudaDeviceSynchronize();
                } else if (remaining == 0) {
                    // No newly reached nodes AND no remaining edges → reverse done
                    break;
                }

                // Time check within batched reverse
                if (elapsed_ms() > settings.time_limit_ms) break;
            }

            // S2: Don't rebuild fq on every reverse search wave.
            // P6 was too aggressive — triggered full fq rebuild after every reverse
            // iteration, causing forward starvation. fq keys may use stale rev_g for
            // priority ordering, but forward CC correctness doesn't depend on key accuracy.
            // fq is rebuilt on improve_approximation and forward-CC invalidation anyway.

            queue_sizes_dirty = true; queue_min_keys_dirty = true;
        };

        // ====================================================================
        // iterate_forward_search
        // ====================================================================
        auto iterate_forward_search = [&]() {
            // Check fq rebuild FIRST — may populate an empty fq
            bool did_fq_rebuild = false;
            if (fq_needs_rebuild) {
                fq_needs_rebuild = false;
                did_fq_rebuild = true;
                cudaMemset(d_fq_size, 0, sizeof(int));

                std::vector<float> h_fwd_g_rebuild(n_total);
                cudaMemcpy(h_fwd_g_rebuild.data(), d_fwd_g, n_total * sizeof(float), cudaMemcpyDeviceToHost);

                std::vector<int> h_fwd_reached_rebuild;
                for (int i = 0; i < n_total; i++) {
                    if (h_fwd_g_rebuild[i] < INF_COST) {
                        h_fwd_reached_rebuild.push_back(i);
                    }
                }
                int n_fwd_rebuild = static_cast<int>(h_fwd_reached_rebuild.size());
                if (n_fwd_rebuild > 0) {
                    cudaMemcpy(d_expand_nodes, h_fwd_reached_rebuild.data(),
                               n_fwd_rebuild * sizeof(int), cudaMemcpyHostToDevice);
                    int nb = nblk(n_fwd_rebuild, MITSTAR_BLOCK_SIZE);
                    build_forward_queue_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                        d_expand_nodes, n_fwd_rebuild,
                        d_fwd_g, d_rev_g, d_lb_ctg,
                        d_adj_targets, d_adj_weights, d_adj_cc_status,
                        d_adj_checks_done, d_adj_count, max_epn,
                        settings.valid_segment_length, solution_cost,
                        d_fwd_expand_tag, d_pruned, d_adj_l2_dist,
                        d_fq_src, d_fq_dst, d_fq_key_lb_cost,
                        d_fq_key_est_cost, d_fq_key_est_effort,
                        d_fq_tag, d_fq_size, max_fq
                    );
                    cudaDeviceSynchronize();
                }
                queue_sizes_dirty = true; queue_min_keys_dirty = true;
            }

            int h_fq_size;
            cudaMemcpy(&h_fq_size, d_fq_size, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_fq_size == 0) return;
            h_fq_size = std::min(h_fq_size, max_fq);

            // P2: Sort key selection — OMPL getFrontIter priority:
            //   Pre-solution + infinite suboptimality: effort-first (find feasible path fast)
            //   Post-solution or finite suboptimality: lb_cost-first (optimality)
            float* sort_key = d_fq_key_lb_cost;
            float* sort_key_sorted = d_fq_key_lb_cost_sorted;

            if (!has_solution && settings.initial_suboptimality > 10.0f) {
                // Pre-solution + infinite suboptimality: effort-first
                sort_key = d_fq_key_est_effort;
                sort_key_sorted = d_fq_key_est_effort_sorted;
            }

            cub::DeviceRadixSort::SortPairs(d_sort_temp_fq, sort_temp_bytes_fq,
                sort_key, sort_key_sorted,
                d_fq_src, d_fq_src_sorted, h_fq_size);
            cudaDeviceSynchronize();
            cosort_int_by_keys(sort_key, sort_key_sorted,
                               d_fq_dst, d_fq_dst_sorted, h_fq_size);
            cosort_int_by_keys(sort_key, sort_key_sorted,
                               d_fq_tag, d_fq_tag_sorted, h_fq_size);
            // Co-sort ALL float key arrays so compaction reads correct values
            if (sort_key != d_fq_key_lb_cost)
                cosort_float_by_keys(sort_key, sort_key_sorted,
                                     d_fq_key_lb_cost, d_fq_key_lb_cost_sorted, h_fq_size);
            if (sort_key != d_fq_key_est_cost)
                cosort_float_by_keys(sort_key, sort_key_sorted,
                                     d_fq_key_est_cost, d_fq_key_est_cost_sorted, h_fq_size);
            if (sort_key != d_fq_key_est_effort)
                cosort_float_by_keys(sort_key, sort_key_sorted,
                                     d_fq_key_est_effort, d_fq_key_est_effort_sorted, h_fq_size);

            int m = std::min(settings.m_forward_eval, h_fq_size);

            // Full CC on top-m edges
            full_cc_kernel<<<m, 64, smem_edge_cc>>>(
                d_fq_src_sorted, d_fq_dst_sorted,
                m, d_nodes, dim,
                settings.valid_segment_length,
                model, scene, settings.collision_margin,
                d_adj_targets, d_adj_checks_done, d_adj_count, max_epn,
                d_fq_tag_sorted, d_fwd_expand_tag,
                d_fwd_parent, d_fwd_g,
                d_adj_cc_status,
                d_diag,
                d_edge_cc_results
            );
            cudaDeviceSynchronize();
            total_forward_evals += m;

            // Update forward tree
            cudaMemset(d_n_newly_reached, 0, sizeof(int));
            cudaMemset(d_n_rev_invalidated, 0, sizeof(int));
            {
                int nb = nblk(m, MITSTAR_BLOCK_SIZE);
                update_forward_tree_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                    d_fq_src_sorted, d_fq_dst_sorted, d_edge_cc_results,
                    m, d_fwd_g, d_fwd_parent,
                    d_adj_targets, d_adj_weights, d_adj_cc_status, d_adj_count,
                    max_epn,
                    d_rev_g, d_rev_parent,
                    d_newly_reached, d_n_newly_reached,
                    d_rev_invalidated, d_n_rev_invalidated,
                    max_newly_reached, max_rev_invalidated,
                    d_fwd_expand_tag,
                    solution_cost, d_lb_ctg
                );
                cudaDeviceSynchronize();
            }

            // P9: Cost propagation — convergence loop (max 10 rounds)
            for (int prop_iter = 0; prop_iter < 10; prop_iter++) {
                cudaMemset(d_changed, 0, sizeof(int));
                int nb = nblk(n_total, MITSTAR_BLOCK_SIZE);
                propagate_cost_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                    d_fwd_g, d_fwd_parent, d_nodes, n_total, dim, d_changed,
                    d_adj_targets, d_adj_weights, d_adj_count, max_epn,
                    d_clearance, settings.clearance_weight,
                    settings.clearance_epsilon, settings.max_clearance_penalty);
                cudaDeviceSynchronize();

                int h_changed;
                cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
                if (!h_changed) break;
            }

            // Check goals
            {
                std::vector<float> h_goal_g(num_goals);
                cudaMemcpy(h_goal_g.data(), d_fwd_g, num_goals * sizeof(float), cudaMemcpyDeviceToHost);

                for (int gi = 0; gi < num_goals; gi++) {
                    if (h_goal_g[gi] < solution_cost) {
                        solution_cost = h_goal_g[gi];
                        has_solution = true;
                        best_goal_found = gi;
                        last_improvement_ms = elapsed_ms();

                        result.cost_history.push_back(solution_cost);
                        result.time_history_ms.push_back(elapsed_ms());

                        // C1 FIX: Eagerly extract path on each solution improvement.
                        // Continued optimization may corrupt fwd_parent chain later
                        // (parent-cost race in update_forward_tree_kernel), so save
                        // the path while the chain is still valid.
                        {
                            std::vector<int> h_fwd_parent_snap(n_total);
                            std::vector<float> h_nodes_snap(n_total * dim);
                            std::vector<float> h_fwd_g_snap(n_total);
                            cudaMemcpy(h_fwd_parent_snap.data(), d_fwd_parent,
                                       n_total * sizeof(int), cudaMemcpyDeviceToHost);
                            cudaMemcpy(h_nodes_snap.data(), d_nodes,
                                       n_total * dim * sizeof(float), cudaMemcpyDeviceToHost);
                            cudaMemcpy(h_fwd_g_snap.data(), d_fwd_g,
                                       n_total * sizeof(float), cudaMemcpyDeviceToHost);

                            std::vector<std::vector<float>> snap_path;
                            float snap_cost = INF_COST;
                            if (extract_path_forward(h_fwd_parent_snap.data(),
                                                     h_nodes_snap.data(),
                                                     gi, start_node_idx, n_total, dim,
                                                     h_fwd_g_snap[gi], snap_path, snap_cost)) {
                                result.path = snap_path;
                                result.cost = snap_cost;
                            }
                        }

                        compute_CL_matrix(start, goals[gi], h_CL, dim);
                        cudaMemcpy(d_CL_matrix, h_CL.data(),
                                   dim * dim * sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_best_goal_cfg, goals[gi].data(),
                                   dim * sizeof(float), cudaMemcpyHostToDevice);

                        // P5: OMPL does NOT rebuild fq on solution improvement.
                        // Stale edges are filtered by tag mechanism and lb_cost >= solution_cost.
                        // fq rebuild only happens on reverse-tree invalidation (D7).
                    }
                }
            }

            // (fq_needs_rebuild logic moved to top of iterate_forward_search)

            // Handle EIS: if collision on reverse-tree edge, no solution yet
            if (!has_solution && !has_eis && settings.use_eis) {
                int h_n_inv;
                cudaMemcpy(&h_n_inv, d_n_rev_invalidated, sizeof(int), cudaMemcpyDeviceToHost);
                if (h_n_inv > 0) {
                    // Compute EIS cost
                    // Download fwd_g and lb_ctg for the forward frontier
                    // Use the best forward node: min(fwd_g[v] + lb_ctg[v])
                    std::vector<float> h_fwd_g(n_total), h_lb_ctg(n_total);
                    cudaMemcpy(h_fwd_g.data(), d_fwd_g, n_total * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_lb_ctg.data(), d_lb_ctg, n_total * sizeof(float), cudaMemcpyDeviceToHost);

                    float best_s_adms = INF_COST;
                    float best_fwd_g_src = INF_COST;
                    for (int v = 0; v < n_total; v++) {
                        if (h_fwd_g[v] < INF_COST) {
                            float s = h_fwd_g[v] + h_lb_ctg[v];
                            if (s < best_s_adms) {
                                best_s_adms = s;
                                best_fwd_g_src = h_fwd_g[v];
                            }
                        }
                    }

                    if (best_s_adms < INF_COST && best_s_adms > 1e-6f) {
                        float gamma = best_fwd_g_src / best_s_adms;
                        // OMPL EstimatedInitialSolution: reliability = gamma,
                        // expand_factor = pow(1 + (1 - reliability), 0.5)
                        //               = sqrt(1 + (1 - gamma)).
                        // (The (1-gamma) term is NOT squared — matches the paper.)
                        float e_gamma = std::sqrt(1.0f + (1.0f - gamma));
                        eis_cost = best_s_adms * e_gamma;
                        has_eis = true;

                        // Compute CL for informed sampling with eis bound
                        if (h_CL.empty()) {
                            compute_CL_matrix(start, goals[best_goal_idx], h_CL, dim);
                            cudaMemcpy(d_CL_matrix, h_CL.data(),
                                       dim * dim * sizeof(float), cudaMemcpyHostToDevice);
                        }
                    }
                }
            }

            // Handle reverse tree invalidation — OMPL-faithful:
            // When forward CC finds collision on a reverse-tree edge,
            // immediately (1) double sparse resolution, (2) full reverse restart,
            // (3) rebuild forward queue. (OMPL MITstar.cpp lines 699-712)
            {
                int h_n_inv;
                cudaMemcpy(&h_n_inv, d_n_rev_invalidated, sizeof(int), cudaMemcpyDeviceToHost);
                h_n_inv = std::min(h_n_inv, max_rev_invalidated);
                if (h_n_inv > 0) {
                    // D4+D5+D3: Immediate sparse doubling with OMPL formula (2n+1)
                    num_sparse_checks = 2 * num_sparse_checks + 1;

                    // D6: Full reverse restart (reset ALL non-goal nodes)
                    reverse_restarts++;
                    {
                        int nb = nblk(n_total, MITSTAR_BLOCK_SIZE);
                        reset_reverse_heuristics_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                            d_rev_g, d_rev_parent, n_total, num_goals);
                        cudaDeviceSynchronize();
                    }

                    // Rebuild reverse queue from goals
                    cudaMemset(d_rq_size, 0, sizeof(int));
                    {
                        std::vector<int> h_goals(num_goals);
                        std::iota(h_goals.begin(), h_goals.end(), 0);
                        cudaMemcpy(d_expand_nodes, h_goals.data(), num_goals * sizeof(int), cudaMemcpyHostToDevice);

                        int nb = nblk(num_goals, MITSTAR_BLOCK_SIZE);
                        build_reverse_queue_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                            d_expand_nodes, num_goals,
                            d_rev_g, d_lb_ctc,
                            d_adj_targets, d_adj_weights, d_adj_cc_status,
                            d_adj_checks_done, d_adj_count, max_epn,
                            settings.valid_segment_length, num_sparse_checks,
                            d_pruned, solution_cost, d_adj_l2_dist,
                            d_rq_src, d_rq_dst, d_rq_key_cost, d_rq_key_effort,
                            d_rq_size, max_rq
                        );
                        cudaDeviceSynchronize();
                    }

                    // D7: Rebuild forward queue (recompute keys with new rev_g)
                    fq_needs_rebuild = true;
                }
            }

            // P10: Skip compact + newly_reached expand when fq was rebuilt this iteration.
            // Rebuild already expanded ALL forward-reached nodes, so compact (which uses
            // stale h_fq_size and sorted buffers from before rebuild) would corrupt the queue.
            if (!did_fq_rebuild) {
                // Compact fq: remove top-m, keep rest
                int remaining = h_fq_size - m;
                if (remaining > 0) {
                    cudaMemcpy(d_fq_src, d_fq_src_sorted + m, remaining * sizeof(int), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_fq_dst, d_fq_dst_sorted + m, remaining * sizeof(int), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_fq_key_lb_cost, d_fq_key_lb_cost_sorted + m,
                               remaining * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_fq_key_est_cost, d_fq_key_est_cost_sorted + m,
                               remaining * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_fq_key_est_effort, d_fq_key_est_effort_sorted + m,
                               remaining * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_fq_tag, d_fq_tag_sorted + m,
                               remaining * sizeof(int), cudaMemcpyDeviceToDevice);
                }

                // Expand newly reached nodes into fq
                int h_n_newly;
                cudaMemcpy(&h_n_newly, d_n_newly_reached, sizeof(int), cudaMemcpyDeviceToHost);

                if (h_n_newly > 0) {
                    h_n_newly = std::min(h_n_newly, max_newly_reached);
                    cudaMemcpy(d_fq_size, &remaining, sizeof(int), cudaMemcpyHostToDevice);

                    int nb = nblk(h_n_newly, MITSTAR_BLOCK_SIZE);
                    build_forward_queue_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                        d_newly_reached, h_n_newly,
                        d_fwd_g, d_rev_g, d_lb_ctg,
                        d_adj_targets, d_adj_weights, d_adj_cc_status,
                        d_adj_checks_done, d_adj_count, max_epn,
                        settings.valid_segment_length, solution_cost,
                        d_fwd_expand_tag, d_pruned, d_adj_l2_dist,
                        d_fq_src, d_fq_dst, d_fq_key_lb_cost,
                        d_fq_key_est_cost, d_fq_key_est_effort,
                        d_fq_tag, d_fq_size, max_fq
                    );
                    cudaDeviceSynchronize();
                } else {
                    cudaMemcpy(d_fq_size, &remaining, sizeof(int), cudaMemcpyHostToDevice);
                }
            }

            queue_sizes_dirty = true; queue_min_keys_dirty = true;
        };

        // OMPL-faithful continue conditions using cached queue state.
        // These are called frequently, so we cache rq_size/fq_size
        // and only re-fetch when queues change. Min-key computation
        // is deferred since it's only needed for interleave decisions
        // and the size check covers most cases.
        int h_rq_size_cached = 0;
        int h_fq_size_cached = 0;
        float h_min_rq_cost_cached = INF_COST;
        float h_min_fq_lb_cost_cached = INF_COST;

        auto refresh_queue_sizes = [&]() {
            if (!queue_sizes_dirty) return;
            cudaMemcpy(&h_rq_size_cached, d_rq_size, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_fq_size_cached, d_fq_size, sizeof(int), cudaMemcpyDeviceToHost);
            h_rq_size_cached = std::min(h_rq_size_cached, max_rq);
            h_fq_size_cached = std::min(h_fq_size_cached, max_fq);
            queue_sizes_dirty = false;
            queue_min_keys_dirty = true; // sizes changed, min keys stale
        };

        auto refresh_queue_min_keys = [&]() {
            if (!queue_min_keys_dirty) return;
            refresh_queue_sizes();

            if (h_rq_size_cached > 0) {
                size_t tb = max_reduce_temp;
                cub::DeviceReduce::Min(d_reduce_temp, tb,
                    d_rq_key_cost, d_min_rq_cost, h_rq_size_cached);
                cudaMemcpy(&h_min_rq_cost_cached, d_min_rq_cost, sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                h_min_rq_cost_cached = INF_COST;
            }

            if (h_fq_size_cached > 0) {
                size_t tb = max_reduce_temp;
                cub::DeviceReduce::Min(d_reduce_temp, tb,
                    d_fq_key_lb_cost, d_min_fq_lb_cost, h_fq_size_cached);
                cudaMemcpy(&h_min_fq_lb_cost_cached, d_min_fq_lb_cost, sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                h_min_fq_lb_cost_cached = INF_COST;
            }

            queue_min_keys_dirty = false;
        };

        // P4: Counter to prevent infinite reverse loops when fq is empty
        int reverse_without_forward_count = 0;

        // OMPL interleaving (simplified for GPU batched architecture):
        // Continue reverse search if rq has edges AND either fq is empty (need to
        // build reverse tree for forward search) or rq has better min-key than fq.
        // Note: OMPL strict "both non-empty" check (line 984) doesn't directly apply
        // to GPU because GPU forward queue needs reverse tree to compute heuristics.
        auto continue_reverse_search = [&]() -> bool {
            refresh_queue_sizes();
            if (h_rq_size_cached == 0) return false;
            if (h_fq_size_cached == 0) {
                // P4: fq empty — reverse can build tree, but cap iterations
                // to avoid infinite reverse loops when no forward edges can be generated
                reverse_without_forward_count++;
                if (reverse_without_forward_count > 20) {
                    reverse_without_forward_count = 0;
                    return false;  // force improve_approximation
                }
                return true;
            }
            reverse_without_forward_count = 0;
            // Both queues non-empty: continue reverse if rq has better min-key
            refresh_queue_min_keys();
            return h_min_rq_cost_cached < h_min_fq_lb_cost_cached;
        };

        auto continue_forward_search = [&]() -> bool {
            refresh_queue_sizes();
            if (h_fq_size_cached == 0) return false;
            // Need fq min lb_cost for comparison with solution_cost
            refresh_queue_min_keys();
            return h_min_fq_lb_cost_cached < solution_cost;
        };

        // ====================================================================
        // Initial: sample first batch, NN, insert edges, then build queues
        // ====================================================================
        // We need edges between initial nodes (goals + start) before building
        // queues. Run improve_approximation's sample→NN→insert pipeline once.
        {
            result.total_batches++;
            float informed_bound = std::min(solution_cost, eis_cost);

            // Sample batch
            sample_batch_kernel<<<nblk(settings.batch_size, 128), 128>>>(
                d_new_samples, d_halton_states, d_rng_states, model,
                settings.batch_size, dim,
                false, informed_bound,
                d_start_cfg, d_best_goal_cfg,
                nullptr,
                c_min
            );
            cudaDeviceSynchronize();

            // Batch CC
            batch_cc_kernel<<<settings.batch_size, 4, smem_batch_cc>>>(
                d_new_samples, d_cc_results_sample,
                settings.batch_size, model, scene, dim,
                settings.collision_margin,
                d_sample_clearance, settings.clearance_weight
            );
            cudaDeviceSynchronize();

            // Host-side compaction
            std::vector<uint8_t> h_cc(settings.batch_size);
            cudaMemcpy(h_cc.data(), d_cc_results_sample, settings.batch_size, cudaMemcpyDeviceToHost);

            std::vector<int> valid_indices;
            for (int i = 0; i < settings.batch_size; i++) {
                if (h_cc[i] == 0) valid_indices.push_back(i);
            }
            int n_valid = static_cast<int>(valid_indices.size());
            if (n_valid > max_nodes - n_total) n_valid = max_nodes - n_total;

            if (n_valid > 0) {
                std::vector<float> h_samples(settings.batch_size * dim);
                cudaMemcpy(h_samples.data(), d_new_samples,
                           settings.batch_size * dim * sizeof(float), cudaMemcpyDeviceToHost);

                std::vector<float> h_valid(n_valid * dim);
                for (int i = 0; i < n_valid; i++) {
                    std::copy_n(h_samples.data() + valid_indices[i] * dim, dim,
                               h_valid.data() + i * dim);
                }
                cudaMemcpy(d_nodes + n_total * dim, h_valid.data(),
                           n_valid * dim * sizeof(float), cudaMemcpyHostToDevice);

                int n_existing = n_total;
                n_total += n_valid;

                // Compress and upload clearance for valid samples
                if (use_clearance) {
                    std::vector<float> h_sample_clearance(settings.batch_size);
                    cudaMemcpy(h_sample_clearance.data(), d_sample_clearance,
                               settings.batch_size * sizeof(float), cudaMemcpyDeviceToHost);
                    std::vector<float> h_valid_clearance(n_valid);
                    for (int i = 0; i < n_valid; i++) {
                        h_valid_clearance[i] = h_sample_clearance[valid_indices[i]];
                    }
                    cudaMemcpy(d_clearance + n_existing, h_valid_clearance.data(),
                               n_valid * sizeof(float), cudaMemcpyHostToDevice);
                }

                // Init state for new nodes
                {
                    int nb = nblk(n_valid, MITSTAR_BLOCK_SIZE);
                    init_state_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                        d_nodes, d_fwd_g, d_rev_g, d_fwd_parent, d_rev_parent,
                        d_lb_ctc, d_lb_ctg, n_existing, n_total,
                        d_start_cfg, d_best_goal_cfg, dim);
                    cudaDeviceSynchronize();
                }

                // Radius-NN for new batch nodes (fast)
                float rnn_radius_sq = compute_rgg_radius_sq(n_total);
                int smem_rnn = (dim + RNN_TILE_SIZE * dim) * sizeof(float) + sizeof(int);
                radius_nn_kernel<<<n_valid, MITSTAR_BLOCK_SIZE, smem_rnn>>>(
                    d_nodes, n_existing, n_valid, n_total, dim,
                    rnn_radius_sq, settings.max_neighbors,
                    d_nn_counts, d_nn_indices
                );
                cudaDeviceSynchronize();

                // Insert edges (bidirectional: new↔existing and new↔new)
                {
                    int nb = nblk(n_valid, MITSTAR_BLOCK_SIZE);
                    insert_edges_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                        d_nn_counts, d_nn_indices,
                        n_existing, n_valid, settings.max_neighbors,
                        max_epn,
                        d_adj_targets, d_adj_weights, d_adj_cc_status,
                        d_adj_checks_done, d_adj_count,
                        d_nodes, dim,
                        d_clearance, settings.clearance_weight,
                        settings.clearance_epsilon, settings.max_clearance_penalty,
                    d_adj_l2_dist
                    );
                    cudaDeviceSynchronize();
                }

                // k-NN for initial nodes (goals + start) — guaranteed connectivity
                {
                    int n_initial = n_existing; // goals + start before new nodes
                    int knn = compute_knn_k(n_total);
                    int smem_knn = (dim + KNN_TILE_SIZE * dim) * sizeof(float);
                    knn_kernel<<<n_initial, MITSTAR_BLOCK_SIZE, smem_knn>>>(
                        d_nodes, 0, n_initial, n_total, dim,
                        knn, settings.max_neighbors,
                        d_nn_counts, d_nn_indices
                    );
                    cudaDeviceSynchronize();

                    int nb2 = nblk(n_initial, MITSTAR_BLOCK_SIZE);
                    insert_edges_kernel<<<nb2, MITSTAR_BLOCK_SIZE>>>(
                        d_nn_counts, d_nn_indices,
                        0, n_initial, settings.max_neighbors,
                        max_epn,
                        d_adj_targets, d_adj_weights, d_adj_cc_status,
                        d_adj_checks_done, d_adj_count,
                        d_nodes, dim,
                        d_clearance, settings.clearance_weight,
                        settings.clearance_epsilon, settings.max_clearance_penalty,
                    d_adj_l2_dist
                    );
                    cudaDeviceSynchronize();
                }
            }
        }

        // ====================================================================

        // Build reverse queue from goals
        cudaMemset(d_rq_size, 0, sizeof(int));
        {
            std::vector<int> h_goals(num_goals);
            std::iota(h_goals.begin(), h_goals.end(), 0);
            cudaMemcpy(d_expand_nodes, h_goals.data(), num_goals * sizeof(int), cudaMemcpyHostToDevice);

            int nb = nblk(num_goals, MITSTAR_BLOCK_SIZE);
            build_reverse_queue_kernel<<<nb, MITSTAR_BLOCK_SIZE>>>(
                d_expand_nodes, num_goals,
                d_rev_g, d_lb_ctc,
                d_adj_targets, d_adj_weights, d_adj_cc_status,
                d_adj_checks_done, d_adj_count, max_epn,
                settings.valid_segment_length, num_sparse_checks,
                d_pruned, solution_cost, d_adj_l2_dist,
                d_rq_src, d_rq_dst, d_rq_key_cost, d_rq_key_effort,
                d_rq_size, max_rq
            );
            cudaDeviceSynchronize();
        }

        // Build forward queue from start
        cudaMemset(d_fq_size, 0, sizeof(int));
        {
            cudaMemcpy(d_expand_nodes, &start_node_idx, sizeof(int), cudaMemcpyHostToDevice);

            build_forward_queue_kernel<<<1, MITSTAR_BLOCK_SIZE>>>(
                d_expand_nodes, 1,
                d_fwd_g, d_rev_g, d_lb_ctg,
                d_adj_targets, d_adj_weights, d_adj_cc_status,
                d_adj_checks_done, d_adj_count, max_epn,
                settings.valid_segment_length, solution_cost,
                d_fwd_expand_tag, d_pruned, d_adj_l2_dist,
                d_fq_src, d_fq_dst, d_fq_key_lb_cost,
                d_fq_key_est_cost, d_fq_key_est_effort,
                d_fq_tag, d_fq_size, max_fq
            );
            cudaDeviceSynchronize();
        }

        queue_sizes_dirty = true; queue_min_keys_dirty = true;

        // ====================================================================
        // Main loop
        // ====================================================================
        // S4v2: Forced round-robin alternation between reverse and forward.
        // The old priority-based interleaving (continue_reverse_search) always
        // picked reverse because reverse BFS costs are lower than forward costs.
        // This starved forward search — it only ran when rq was completely empty.
        //
        // New strategy: each iteration does 1 reverse wave + 1 forward batch.
        // When fq empties after forward (few newly-reached), reload fq from
        // all forward-reached nodes with updated rev_g from recent reverse waves.
        // ====================================================================
        // Time is the only budget. An iteration cap would let a robot whose
        // batches are cheap (Fetch: ~180 batches in 80 ms) abandon most of
        // time_limit_ms; the loop still terminates via the max_nodes break.
        while (elapsed_ms() <= settings.time_limit_ms) {
            // Early exit: solution found and no cost improvement for early_exit_ms
            if (has_solution && settings.early_exit_ms > 0.0f &&
                (elapsed_ms() - last_improvement_ms) > settings.early_exit_ms) break;
            total_iterations++;

            // Check if either queue has work
            refresh_queue_sizes();
            bool has_rq = h_rq_size_cached > 0;
            bool has_fq = h_fq_size_cached > 0;

            if (has_rq || has_fq) {
                // Round-robin interleave: reverse and forward alternate
                while (elapsed_ms() < settings.time_limit_ms) {
                    // Early exit check inside inner loop
                    if (has_solution && settings.early_exit_ms > 0.0f &&
                        (elapsed_ms() - last_improvement_ms) > settings.early_exit_ms) break;
                    bool did_work = false;

                    // One reverse wave (1 BFS level)
                    refresh_queue_sizes();
                    if (h_rq_size_cached > 0) {
                        iterate_reverse_search();
                        did_work = true;
                    }

                    // One forward batch
                    refresh_queue_sizes();
                    if (h_fq_size_cached > 0) {
                        // C1 FIX: When a solution exists, only run forward search if
                        // the best-possible edge can improve it (OMPL: min lb_cost < solution_cost).
                        // This avoids wasting time on stale edges that can't improve cost.
                        bool should_run_forward = true;
                        if (has_solution) {
                            refresh_queue_min_keys();
                            should_run_forward = (h_min_fq_lb_cost_cached < solution_cost);
                        }
                        if (should_run_forward) {
                            iterate_forward_search();
                            did_work = true;
                        } else {
                            // All remaining fq edges are >= solution_cost → drain fq
                            cudaMemset(d_fq_size, 0, sizeof(int));
                            queue_sizes_dirty = true; queue_min_keys_dirty = true;
                        }
                    } else if (did_work) {
                        // fq empty but reverse still running — reload fq from
                        // forward-reached nodes with updated rev_g heuristics.
                        fq_needs_rebuild = true;
                    }

                    if (!did_work) break;
                    // C1 FIX: Do NOT break on has_solution. OMPL MIT* continues
                    // searching after the first feasible path to find shorter ones
                    // (asymptotic optimality). The forward queue naturally narrows
                    // via lb_cost >= solution_cost pruning (line 1215) and the
                    // both-queues-empty check triggers improve_approximation for
                    // new samples in the informed subset.
                }
            } else {
                // Both queues exhausted → improve approximation
                // S5: If max_nodes reached, no point calling improve_approximation
                if (n_total >= max_nodes) break;
                improve_approximation();
            }
        }

        // D3: Final state binary dump — only on failure
        if (settings.debug_dump && !has_solution) {
            FILE* fp = fopen("/tmp/mitstar_dump.bin", "wb");
            if (fp) {
                // Header: n_total, dim, max_epn, num_goals, start_node_idx
                int header[5] = { n_total, dim, max_epn, num_goals, start_node_idx };
                fwrite(header, sizeof(int), 5, fp);

                std::vector<float> h_nodes_d(n_total * dim);
                std::vector<float> h_fwd_g_d(n_total);
                std::vector<float> h_rev_g_d(n_total);
                std::vector<int>   h_fwd_parent_d(n_total);
                std::vector<int>   h_rev_parent_d(n_total);
                std::vector<float> h_lb_ctc_d(n_total);
                std::vector<float> h_lb_ctg_d(n_total);

                cudaMemcpy(h_nodes_d.data(), d_nodes, n_total * dim * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_fwd_g_d.data(), d_fwd_g, n_total * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rev_g_d.data(), d_rev_g, n_total * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_fwd_parent_d.data(), d_fwd_parent, n_total * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rev_parent_d.data(), d_rev_parent, n_total * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_lb_ctc_d.data(), d_lb_ctc, n_total * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_lb_ctg_d.data(), d_lb_ctg, n_total * sizeof(float), cudaMemcpyDeviceToHost);

                fwrite(h_nodes_d.data(), sizeof(float), n_total * dim, fp);
                fwrite(h_fwd_g_d.data(), sizeof(float), n_total, fp);
                fwrite(h_rev_g_d.data(), sizeof(float), n_total, fp);
                fwrite(h_fwd_parent_d.data(), sizeof(int), n_total, fp);
                fwrite(h_rev_parent_d.data(), sizeof(int), n_total, fp);
                fwrite(h_lb_ctc_d.data(), sizeof(float), n_total, fp);
                fwrite(h_lb_ctg_d.data(), sizeof(float), n_total, fp);

                // Edge adjacency
                std::vector<int> h_adj_count_d(n_total);
                cudaMemcpy(h_adj_count_d.data(), d_adj_count, n_total * sizeof(int), cudaMemcpyDeviceToHost);
                fwrite(h_adj_count_d.data(), sizeof(int), n_total, fp);

                std::vector<int> h_adj_targets_d(n_total * max_epn);
                std::vector<uint8_t> h_adj_cc_d(n_total * max_epn);
                cudaMemcpy(h_adj_targets_d.data(), d_adj_targets, n_total * max_epn * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_adj_cc_d.data(), d_adj_cc_status, n_total * max_epn * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                fwrite(h_adj_targets_d.data(), sizeof(int), n_total * max_epn, fp);
                fwrite(h_adj_cc_d.data(), sizeof(uint8_t), n_total * max_epn, fp);

                fclose(fp);
            }
        }

        // ====================================================================
        // Extract result
        // ====================================================================
        result.total_nodes = n_total;
        result.solved = has_solution;
        result.total_forward_edges_evaluated = total_forward_evals;
        result.total_reverse_edges_evaluated = total_reverse_evals;
        result.reverse_restarts = reverse_restarts;
        result.eis_cost = eis_cost;
        result.total_iterations = total_iterations;

        // FIX_B3 diagnostics: read back edge CC counters
        {
            int h_diag[6] = {0};
            cudaMemcpy(h_diag, d_diag, 6 * sizeof(int), cudaMemcpyDeviceToHost);
            result.sparse_midpoints_checked = h_diag[0];
            result.sparse_self_coll_hits    = h_diag[1];
            result.sparse_scene_coll_hits   = h_diag[2];
            result.full_midpoints_checked   = h_diag[3];
            result.full_self_coll_hits      = h_diag[4];
            result.full_scene_coll_hits      = h_diag[5];
        }

        if (has_solution) {
            // Download forward tree data
            std::vector<int> h_fwd_parent(n_total);
            std::vector<float> h_nodes(n_total * dim);
            std::vector<float> h_fwd_g(n_total);
            cudaMemcpy(h_fwd_parent.data(), d_fwd_parent, n_total * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_nodes.data(), d_nodes, n_total * dim * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fwd_g.data(), d_fwd_g, n_total * sizeof(float), cudaMemcpyDeviceToHost);

            // Find best goal
            int goal_idx = best_goal_found;
            if (goal_idx < 0) {
                float best = INF_COST;
                for (int gi = 0; gi < num_goals; gi++) {
                    if (h_fwd_g[gi] < best) {
                        best = h_fwd_g[gi];
                        goal_idx = gi;
                    }
                }
            }

            std::vector<int> path_indices;  // node indices for shortcutting

            if (goal_idx >= 0) {
                std::vector<std::vector<float>> raw_path;
                float actual_cost = INF_COST;
                if (extract_path_forward(h_fwd_parent.data(), h_nodes.data(),
                                         goal_idx, start_node_idx, n_total, dim,
                                         h_fwd_g[goal_idx], raw_path, actual_cost,
                                         &path_indices)) {
                    result.path = raw_path;
                    result.cost = actual_cost;
                } else {
                    // Parent chain corrupted — try all goals as fallback
                    bool found_any = false;
                    for (int gi = 0; gi < num_goals; gi++) {
                        if (h_fwd_g[gi] >= INF_COST) continue;
                        raw_path.clear();
                        path_indices.clear();
                        actual_cost = INF_COST;
                        if (extract_path_forward(h_fwd_parent.data(), h_nodes.data(),
                                                 gi, start_node_idx, n_total, dim,
                                                 h_fwd_g[gi], raw_path, actual_cost,
                                                 &path_indices)) {
                            result.path = raw_path;
                            result.cost = actual_cost;
                            found_any = true;
                            printf("[MIT*] path extraction: primary goal %d failed, "
                                   "fallback to goal %d succeeded (cost=%.4f)\n",
                                   goal_idx, gi, actual_cost);
                            break;
                        }
                    }
                    if (!found_any) {
                        // C1 FIX: If eager extraction saved a path, keep it
                        if (!result.path.empty()) {
                            // result.path and result.cost already set by eager extraction
                            fprintf(stderr, "[MIT*] path extraction at end failed, "
                                    "using eagerly-extracted path (cost=%.4f, %zu waypoints)\n",
                                    result.cost, result.path.size());
                        } else {
                            fprintf(stderr, "[MIT*] path extraction FAILED for all goals "
                                   "(best_goal=%d, fwd_g=%.4f, n_total=%d). "
                                   "Parent chain corrupted after optimization.\n",
                                   goal_idx, h_fwd_g[goal_idx], n_total);
                            result.solved = false;
                            result.cost = solution_cost;
                        }
                    }
                }
            } else {
                result.solved = false;
                result.cost = solution_cost;
            }

            // ================================================================
            // Path shortcutting (post-processing, inside has_solution block)
            // ================================================================
            if (settings.shortcut_path && result.solved && result.path.size() > 2) {
                if (!path_indices.empty()) {
                    // Fast path: read configs directly from d_nodes via indices
                    result.path = ppln::shortcut::shortcut_path(
                        d_nodes, path_indices, result.path, model, scene,
                        settings.valid_segment_length,
                        settings.collision_margin + settings.shortcut_collision_margin,
                        dim);
                } else {
                    // Fallback: upload path configs to GPU temp buffer
                    // (used when eagerly-extracted path has no indices)
                    result.path = ppln::shortcut::shortcut_path(
                        result.path, model, scene,
                        settings.valid_segment_length,
                        settings.collision_margin + settings.shortcut_collision_margin,
                        dim);
                }
                // Recompute L2 cost for the simplified path
                float sc = 0.0f;
                for (size_t i = 1; i < result.path.size(); i++) {
                    float d2 = 0.0f;
                    for (int d = 0; d < dim; d++) {
                        float diff = result.path[i][d] - result.path[i-1][d];
                        d2 += diff * diff;
                    }
                    sc += sqrtf(d2);
                }
                result.cost = sc;
            }
        }

        // ====================================================================
        // Cleanup
        // ====================================================================
        if (!use_bufs) {
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
            cudaFree(d_cosort_temp);
            cudaFree(d_sort_temp_fq);
            cudaFree(d_reduce_temp);
            cudaFree(d_min_rq_cost);
            cudaFree(d_min_fq_lb_cost);
        }

        result.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wall_start).count();

        return result;
    }

} // namespace MITStar
