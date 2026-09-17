#include "chomp_buffers.hh"

CHOMPBuffers CHOMPBuffers::create(int n_dof, int num_timesteps)
{
    CHOMPBuffers b;
    b.n_dof = n_dof;
    b.num_timesteps = num_timesteps;

    const int D = n_dof;
    const int T = num_timesteps;
    const size_t dt = (size_t)D * T;

    // Smoothness metric R and its inverse A^{-1}.
    cudaMalloc(&b.d_R,    (size_t)T * T * sizeof(float));
    cudaMalloc(&b.d_Ainv, (size_t)T * T * sizeof(float));

    // Working + best trajectory.
    cudaMalloc(&b.d_theta, dt * sizeof(float));
    cudaMalloc(&b.d_best,  dt * sizeof(float));

    // Obstacle gradient.
    cudaMalloc(&b.d_obs_grad, dt * sizeof(float));

    // Cost breakdown.
    cudaMalloc(&b.d_state_costs,   dt * sizeof(float));
    cudaMalloc(&b.d_control_costs, dt * sizeof(float));
    cudaMalloc(&b.d_state_cost,    sizeof(float));
    cudaMalloc(&b.d_control_cost,  sizeof(float));
    cudaMalloc(&b.d_total_cost,    sizeof(float));

    // Endpoints.
    cudaMalloc(&b.d_start, (size_t)D * sizeof(float));
    cudaMalloc(&b.d_goal,  (size_t)D * sizeof(float));

    // Validity flag.
    cudaMalloc(&b.d_collision, sizeof(int));

    // Pinned host mirrors.
    cudaHostAlloc(&b.h_valid,      sizeof(int),   cudaHostAllocDefault);
    cudaHostAlloc(&b.h_total_cost, sizeof(float), cudaHostAllocDefault);

    b.owns_memory = true;
    return b;
}

void CHOMPBuffers::destroy() {
    if (!owns_memory) return;

    cudaFree(d_R);
    cudaFree(d_Ainv);
    cudaFree(d_theta);
    cudaFree(d_best);
    cudaFree(d_obs_grad);
    cudaFree(d_state_costs);
    cudaFree(d_control_costs);
    cudaFree(d_state_cost);
    cudaFree(d_control_cost);
    cudaFree(d_total_cost);
    cudaFree(d_start);
    cudaFree(d_goal);
    cudaFree(d_collision);
    cudaFreeHost(h_valid);
    cudaFreeHost(h_total_cost);

    owns_memory = false;
}
