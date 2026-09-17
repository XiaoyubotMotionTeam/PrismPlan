#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "src/planning/robot_model.cuh"
#include "src/planning/runtime_kinematics.cuh"
#include "src/collision/scene_collision.cuh"

namespace ppln::shortcut {

using ppln::RobotModel;
using ppln::collision::SceneCollisionData;

// ========================================================================
// Kernel: shortcut_cc_kernel
// ========================================================================
// Check collision for skip-edges between path waypoints.
// Each block checks one edge (src_wp -> dst_wp), 4 threads per block.
// FK is the shared tree walk (fk_joint_transforms_runtime), so a shortcut
// waypoint gets the same link poses every planner's edge check sees.
// Reads configs from d_nodes via d_path_indices indirection.

__global__ void shortcut_cc_kernel(
    const float* d_nodes,       // [max_nodes * dim] all nodes (already on GPU)
    const int* d_path_indices,  // [N] node indices that form the path
    const int* d_pairs,         // [n_edges * 2] (src_wp_idx, dst_wp_idx) into path_indices
    int n_edges,
    int dim,
    float valid_segment_length,
    const RobotModel model,
    const SceneCollisionData scene,
    float collision_margin,
    uint8_t* d_results          // [n_edges] 0=collision, 1=free
) {
    int eidx = blockIdx.x;
    if (eidx >= n_edges) return;
    int tid = threadIdx.x;
    if (tid >= 4) return;

    int src_wp = d_pairs[eidx * 2];
    int dst_wp = d_pairs[eidx * 2 + 1];
    int src_node = d_path_indices[src_wp];
    int dst_node = d_path_indices[dst_wp];

    extern __shared__ float smem[];
    float* q_src = smem;
    float* q_dst = q_src + dim;
    float* q_interp = q_dst + dim;
    int n_spheres = model.n_spheres;
    float* sphere_pos = q_interp + dim;
    float* jt_store = sphere_pos + n_spheres * 3;   // [n_joints * 16] world frames

    // Load src and dst configs directly from d_nodes
    if (tid == 0) {
        for (int i = 0; i < dim; i++) {
            q_src[i] = d_nodes[src_node * dim + i];
            q_dst[i] = d_nodes[dst_node * dim + i];
        }
    }
    __syncthreads();

    // Compute edge length and number of segments
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

    __shared__ int s_collision;
    if (tid == 0) s_collision = 0;
    __syncthreads();

    int total_seg = s_total_seg;

    // Check interior midpoints sequentially
    for (int seg = 1; seg < total_seg; seg++) {
        if (s_collision) break;

        float t = (float)seg / (float)total_seg;

        if (tid == 0) {
            for (int i = 0; i < dim; i++) {
                q_interp[i] = q_src[i] * (1.0f - t) + q_dst[i] * t;
            }
        }
        __syncthreads();

        // Single-thread tree FK — same walk every planner's edge check uses
        __shared__ int s_midpoint_coll;
        __shared__ int s_ns;
        if (tid == 0) {
            s_midpoint_coll = 0;

            ppln::collision::fk_joint_transforms_runtime(model, q_interp, jt_store);

            // Compute sphere world positions into shared memory
            int ns = n_spheres;
            s_ns = ns;
            for (int si = 0; si < ns; si++) {
                float4 sph = model.spheres[si];
                int ji = model.sphere_to_joint[si];
                float* jt = &jt_store[ji * 16];
                sphere_pos[si * 3 + 0] = jt[0]*sph.x + jt[4]*sph.y + jt[8]*sph.z + jt[12];
                sphere_pos[si * 3 + 1] = jt[1]*sph.x + jt[5]*sph.y + jt[9]*sph.z + jt[13];
                sphere_pos[si * 3 + 2] = jt[2]*sph.x + jt[6]*sph.y + jt[10]*sph.z + jt[14];
            }
        }
        __syncthreads();

        // 4-thread parallel scene collision check
        {
            int ns = s_ns;
            bool my_hit = false;
            for (int si = tid; si < ns; si += 4) {
                float sr = model.spheres[si].w;
                if (sr <= 0.0f) continue;  // disabled sphere
                if (ppln::collision::sphere_scene_in_collision(
                        scene,
                        sphere_pos[si * 3 + 0],
                        sphere_pos[si * 3 + 1],
                        sphere_pos[si * 3 + 2],
                        sr + collision_margin,
                        si,
                        scene.sphere_obb_acm_mask)) {
                    my_hit = true;
                    break;
                }
            }
            if (__any_sync(0xf, my_hit)) s_midpoint_coll = 1;
        }
        __syncthreads();

        // 4-thread parallel self-collision check
        if (!s_midpoint_coll) {
            bool my_self_hit = false;
            for (int i = tid; i < model.n_self_cc_ranges; i += 4) {
                if (my_self_hit) break;
                int s1 = model.self_cc_ranges[i * 3 + 0];
                float s1r = model.spheres[s1].w;
                if (s1r <= 0.0f) continue;
                float s1x = sphere_pos[s1 * 3 + 0];
                float s1y = sphere_pos[s1 * 3 + 1];
                float s1z = sphere_pos[s1 * 3 + 2];
                int rng_start = model.self_cc_ranges[i * 3 + 1];
                int rng_end   = model.self_cc_ranges[i * 3 + 2];
                for (int j = rng_start; j <= rng_end; j++) {
                    float jr = model.spheres[j].w;
                    if (jr <= 0.0f) continue;
                    float dx = s1x - sphere_pos[j * 3 + 0];
                    float dy = s1y - sphere_pos[j * 3 + 1];
                    float dz = s1z - sphere_pos[j * 3 + 2];
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    float r_sum = s1r + jr;
                    if (dist_sq < r_sum * r_sum) {
                        my_self_hit = true;
                        break;
                    }
                }
            }
            if (__any_sync(0xf, my_self_hit)) s_midpoint_coll = 1;
        }
        __syncthreads();

        if (s_midpoint_coll) {
            if (tid == 0) s_collision = 1;
            __syncthreads();
            break;
        }
    }

    if (tid == 0) d_results[eidx] = s_collision ? 0 : 1;
}

// ========================================================================
// Host: shortcut_path — GPU-accelerated path simplification
// ========================================================================
// Checks all O(N^2) skip-edges in parallel on GPU, then greedily picks the
// farthest collision-free jump from each waypoint.
// Reads configs directly from d_nodes (already on GPU) via path_indices.

inline std::vector<std::vector<float>> shortcut_path(
    const float* d_nodes,
    const std::vector<int>& path_indices,
    const std::vector<std::vector<float>>& path,
    RobotModel& model,
    SceneCollisionData& scene,
    float valid_segment_length,
    float collision_margin,
    int dim
) {
    int N = (int)path_indices.size();
    if (N <= 2) return path;

    // Build all skip-edge pairs: (i, j) where j > i + 1
    std::vector<int> h_pairs;
    for (int i = 0; i < N - 2; i++)
        for (int j = i + 2; j < N; j++) {
            h_pairs.push_back(i);
            h_pairs.push_back(j);
        }
    int n_edges = (int)(h_pairs.size() / 2);
    if (n_edges == 0) return path;

    int* d_path_idx;
    int* d_pairs;
    uint8_t* d_results;
    cudaMalloc(&d_path_idx, N * sizeof(int));
    cudaMalloc(&d_pairs, n_edges * 2 * sizeof(int));
    cudaMalloc(&d_results, n_edges * sizeof(uint8_t));
    cudaMemset(d_results, 0, n_edges * sizeof(uint8_t));  // 0 = collision (safe default)
    cudaMemcpy(d_path_idx, path_indices.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs, h_pairs.data(), n_edges * 2 * sizeof(int), cudaMemcpyHostToDevice);

    int smem_shortcut = (dim * 3 + model.n_spheres * 3 + model.n_joints * 16) * sizeof(float);
    if (smem_shortcut > 48 * 1024) {
        cudaFuncSetAttribute(shortcut_cc_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_shortcut);
    }

    shortcut_cc_kernel<<<n_edges, 4, smem_shortcut>>>(
        d_nodes, d_path_idx, d_pairs, n_edges, dim,
        valid_segment_length, model, scene, collision_margin,
        d_results);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[shortcut] kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_path_idx);
        cudaFree(d_pairs);
        cudaFree(d_results);
        return path;
    }

    std::vector<uint8_t> h_results(n_edges);
    cudaMemcpy(h_results.data(), d_results, n_edges * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_path_idx);
    cudaFree(d_pairs);
    cudaFree(d_results);

    // Build edge_free lookup
    int idx = 0;
    std::vector<std::vector<bool>> edge_free(N, std::vector<bool>(N, false));
    for (int i = 0; i < N - 2; i++)
        for (int j = i + 2; j < N; j++)
            edge_free[i][j] = (h_results[idx++] == 1);

    // Greedy forward: from waypoint 0, jump to farthest reachable
    std::vector<std::vector<float>> simplified;
    int i = 0;
    simplified.push_back(path[0]);
    while (i < N - 1) {
        int farthest = i + 1;
        for (int j = N - 1; j > i + 1; j--) {
            if (edge_free[i][j]) {
                farthest = j;
                break;
            }
        }
        simplified.push_back(path[farthest]);
        i = farthest;
    }

    return simplified;
}

// Overload: shortcut from path configs directly (uploads to GPU temp buffer).
inline std::vector<std::vector<float>> shortcut_path(
    const std::vector<std::vector<float>>& path,
    RobotModel& model,
    SceneCollisionData& scene,
    float valid_segment_length,
    float collision_margin,
    int dim
) {
    int N = (int)path.size();
    if (N <= 2) return path;

    std::vector<float> h_nodes(N * dim);
    for (int i = 0; i < N; i++)
        for (int d = 0; d < dim; d++)
            h_nodes[i * dim + d] = path[i][d];

    float* d_tmp_nodes;
    cudaMalloc(&d_tmp_nodes, N * dim * sizeof(float));
    cudaMemcpy(d_tmp_nodes, h_nodes.data(), N * dim * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<int> trivial_indices(N);
    for (int i = 0; i < N; i++) trivial_indices[i] = i;

    auto result = shortcut_path(d_tmp_nodes, trivial_indices, path, model, scene,
                                valid_segment_length, collision_margin, dim);
    cudaFree(d_tmp_nodes);
    return result;
}

} // namespace ppln::shortcut
