#include "stomp_buffers.hh"

// Note: unlike mitstar_buffers.cu, STOMP needs no CUB workspaces, so there is
// no cub.cuh vs utils.cuh (#define M 4) include-order constraint here.

STOMPBuffers STOMPBuffers::create(
    int n_dof, int num_timesteps,
    int num_rollouts_new, int num_rollouts_old,
    int num_batch)
{
    STOMPBuffers b;
    b.n_dof = n_dof;
    b.num_timesteps = num_timesteps;
    b.num_rollouts_new = num_rollouts_new;
    b.num_rollouts_old = num_rollouts_old;
    b.num_rollouts_all = num_rollouts_new + num_rollouts_old + 1;
    b.num_batch = num_batch;

    const int D  = n_dof;
    const int T  = num_timesteps;
    const int Rn = num_rollouts_new;
    const int R  = b.num_rollouts_all;
    const int Bt = num_batch;

    const size_t bdt  = (size_t)Bt * D * T;
    const size_t brdt = (size_t)Bt * R * D * T;
    const size_t br   = (size_t)Bt * R;
    const size_t bdtr = (size_t)Bt * D * T * R;

    // --- Cholesky factor of the control-cost covariance ---
    cudaMalloc(&b.d_L, (size_t)T * T * sizeof(float));

    // --- Update-smoothing projection matrix M ---
    cudaMalloc(&b.d_M, (size_t)T * T * sizeof(float));

    // --- Raw control-cost quadratic form R (theta^T R theta) ---
    cudaMalloc(&b.d_R, (size_t)T * T * sizeof(float));

    // --- Optimum + candidate trajectories ---
    cudaMalloc(&b.d_parameters_optimized, bdt * sizeof(float));
    cudaMalloc(&b.d_candidate,            bdt * sizeof(float));
    cudaMalloc(&b.d_delta,                bdt * sizeof(float));
    cudaMalloc(&b.d_best_optimum,         bdt * sizeof(float));

    // --- Rollout population ---
    cudaMalloc(&b.d_stored_rollouts, brdt * sizeof(float));
    cudaMalloc(&b.d_noise,           brdt * sizeof(float));
    cudaMalloc(&b.d_control_costs,   brdt * sizeof(float));
    cudaMalloc(&b.d_state_costs,     brdt * sizeof(float));
    cudaMalloc(&b.d_total_costs,     brdt * sizeof(float));

    // --- Previous-iteration snapshots (aliasing-free top-k reuse source) ---
    cudaMalloc(&b.d_prev_stored_rollouts, brdt * sizeof(float));
    cudaMalloc(&b.d_prev_noise,           brdt * sizeof(float));
    cudaMalloc(&b.d_prev_control_costs,   brdt * sizeof(float));
    cudaMalloc(&b.d_prev_state_costs,     brdt * sizeof(float));
    cudaMalloc(&b.d_prev_total_cost,      br   * sizeof(float));

    // --- Per-rollout scalar total cost (top-k key) ---
    cudaMalloc(&b.d_total_cost, br * sizeof(float));

    // --- Local per-(d,t) rollout weights ---
    cudaMalloc(&b.d_probabilities, bdtr * sizeof(float));

    // --- Cached optimum cost breakdown ---
    cudaMalloc(&b.d_parameters_state_costs,   bdt * sizeof(float));
    cudaMalloc(&b.d_parameters_control_costs, bdt * sizeof(float));
    cudaMalloc(&b.d_parameters_state_cost,    (size_t)Bt * sizeof(float));
    cudaMalloc(&b.d_parameters_control_cost,  (size_t)Bt * sizeof(float));
    cudaMalloc(&b.d_parameters_total_cost,    (size_t)Bt * sizeof(float));

    // --- Candidate cost breakdown ---
    cudaMalloc(&b.d_candidate_state_costs,   bdt * sizeof(float));
    cudaMalloc(&b.d_candidate_control_costs, bdt * sizeof(float));
    cudaMalloc(&b.d_candidate_total_cost,    (size_t)Bt * sizeof(float));

    // --- Top-k reuse indices ---
    cudaMalloc(&b.d_topk_idx, (size_t)R * sizeof(int));

    // --- Endpoints ---
    cudaMalloc(&b.d_start, (size_t)D * sizeof(float));
    cudaMalloc(&b.d_goal,  (size_t)D * sizeof(float));

    // --- RNG states: one per thread of the P1 sampling kernel, grid (Rn, D), block T ---
    cudaMalloc(&b.d_rng_states, (size_t)Rn * D * T * sizeof(curandState));

    // --- Validity flag ---
    cudaMalloc(&b.d_collision, (size_t)Bt * sizeof(int));

    // --- Pinned host mirrors for per-iteration loop control ---
    cudaHostAlloc(&b.h_valid,      (size_t)Bt * sizeof(int),   cudaHostAllocDefault);
    cudaHostAlloc(&b.h_total_cost, (size_t)Bt * sizeof(float), cudaHostAllocDefault);

    b.owns_memory = true;
    return b;
}

void STOMPBuffers::destroy() {
    if (!owns_memory) return;

    cudaFree(d_L);
    cudaFree(d_M);
    cudaFree(d_R);

    cudaFree(d_parameters_optimized);
    cudaFree(d_candidate);
    cudaFree(d_delta);
    cudaFree(d_best_optimum);

    cudaFree(d_stored_rollouts);
    cudaFree(d_noise);
    cudaFree(d_control_costs);
    cudaFree(d_state_costs);
    cudaFree(d_total_costs);

    cudaFree(d_prev_stored_rollouts);
    cudaFree(d_prev_noise);
    cudaFree(d_prev_control_costs);
    cudaFree(d_prev_state_costs);
    cudaFree(d_prev_total_cost);

    cudaFree(d_total_cost);

    cudaFree(d_probabilities);

    cudaFree(d_parameters_state_costs);
    cudaFree(d_parameters_control_costs);
    cudaFree(d_parameters_state_cost);
    cudaFree(d_parameters_control_cost);
    cudaFree(d_parameters_total_cost);

    cudaFree(d_candidate_state_costs);
    cudaFree(d_candidate_control_costs);
    cudaFree(d_candidate_total_cost);

    cudaFree(d_topk_idx);

    cudaFree(d_start);
    cudaFree(d_goal);

    cudaFree(d_rng_states);

    cudaFree(d_collision);

    cudaFreeHost(h_valid);
    cudaFreeHost(h_total_cost);

    owns_memory = false;
}
