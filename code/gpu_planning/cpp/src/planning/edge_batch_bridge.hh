#pragma once

// ============================================================================
// Host-side bridge for the GPU batched edge evaluator (edge_eval.cuh).
//
// Owns persistent device + pinned-host buffers so the CPU MHA* search can
// flush a batch of lazy edges, launch evaluate_edges_batch<<<num_edges,64>>>,
// and read back {valid, cost, clearance} without per-call cudaMalloc.
//
// SceneCollisionData / RobotModel are held BY VALUE and passed by value to the
// kernel exactly like the pRRTC / MITStar solvers — they already carry device pointers
// uploaded once at planner setup, so copying the struct is cheap and correct.
// ============================================================================

#include "src/planning/edge_eval.cuh"
#include "src/collision/scene_collision.cuh"
#include "src/planning/robot_model.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace ppln::search {

class EdgeEvalGpu {
public:
    EdgeEvalGpu(int dim, int max_edges, ppln::collision::SceneCollisionData scene, RobotModel model)
        : dim_(dim), max_edges_(max_edges), scene_(scene), model_(model)
    {
        const size_t cfg_elems = (size_t)max_edges_ * dim_;
        cudaMalloc(&d_from_,      cfg_elems * sizeof(float));
        cudaMalloc(&d_to_,        cfg_elems * sizeof(float));
        cudaMalloc(&d_valid_,     (size_t)max_edges_ * sizeof(uint8_t));
        cudaMalloc(&d_cost_,      (size_t)max_edges_ * sizeof(float));
        cudaMalloc(&d_clearance_, (size_t)max_edges_ * sizeof(float));
        // pinned host mirrors for fast async copies
        cudaHostAlloc(&h_from_,      cfg_elems * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&h_to_,        cfg_elems * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&h_valid_,     (size_t)max_edges_ * sizeof(uint8_t), cudaHostAllocDefault);
        cudaHostAlloc(&h_cost_,      (size_t)max_edges_ * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&h_clearance_, (size_t)max_edges_ * sizeof(float), cudaHostAllocDefault);
        cudaStreamCreate(&stream_);
    }

    ~EdgeEvalGpu() {
        cudaFree(d_from_); cudaFree(d_to_); cudaFree(d_valid_);
        cudaFree(d_cost_); cudaFree(d_clearance_);
        cudaFreeHost(h_from_); cudaFreeHost(h_to_); cudaFreeHost(h_valid_);
        cudaFreeHost(h_cost_); cudaFreeHost(h_clearance_);
        cudaStreamDestroy(stream_);
    }

    EdgeEvalGpu(const EdgeEvalGpu&) = delete;
    EdgeEvalGpu& operator=(const EdgeEvalGpu&) = delete;

    // Evaluate `num_edges` edges. from_flat / to_flat are row-major [num_edges*dim].
    // Results are written into the caller-provided out vectors (resized here).
    void evaluate(const float* from_flat, const float* to_flat, int num_edges,
                  const EdgeEvalParams& params,
                  std::vector<uint8_t>& out_valid,
                  std::vector<float>&   out_cost,
                  std::vector<float>&   out_clearance)
    {
        if (num_edges <= 0) { out_valid.clear(); out_cost.clear(); out_clearance.clear(); return; }
        if (num_edges > max_edges_)
            throw std::runtime_error("EdgeEvalGpu: num_edges exceeds max_edges_");

        const size_t cfg_bytes = (size_t)num_edges * dim_ * sizeof(float);
        std::copy(from_flat, from_flat + (size_t)num_edges * dim_, h_from_);
        std::copy(to_flat,   to_flat   + (size_t)num_edges * dim_, h_to_);

        cudaMemcpyAsync(d_from_, h_from_, cfg_bytes, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_to_,   h_to_,   cfg_bytes, cudaMemcpyHostToDevice, stream_);

        evaluate_edges_batch<<<num_edges, 64, 0, stream_>>>(
            d_from_, d_to_, num_edges, dim_, scene_, model_, params,
            d_valid_, d_cost_, d_clearance_);

        cudaMemcpyAsync(h_valid_, d_valid_, (size_t)num_edges * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_cost_,  d_cost_,  (size_t)num_edges * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        if (params.compute_clearance)
            cudaMemcpyAsync(h_clearance_, d_clearance_, (size_t)num_edges * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        out_valid.assign(h_valid_, h_valid_ + num_edges);
        out_cost.assign(h_cost_, h_cost_ + num_edges);
        if (params.compute_clearance)
            out_clearance.assign(h_clearance_, h_clearance_ + num_edges);
        else
            out_clearance.clear();
    }

    int dim() const { return dim_; }
    int max_edges() const { return max_edges_; }

private:
    int dim_;
    int max_edges_;
    ppln::collision::SceneCollisionData scene_;
    RobotModel model_;
    cudaStream_t stream_{};
    float*   d_from_{};   float*   d_to_{};
    uint8_t* d_valid_{};  float*   d_cost_{};  float* d_clearance_{};
    float*   h_from_{};   float*   h_to_{};
    uint8_t* h_valid_{};  float*   h_cost_{};  float* h_clearance_{};
};

} // namespace ppln::search
