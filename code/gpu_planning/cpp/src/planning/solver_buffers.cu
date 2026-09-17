#include "solver_buffers.hh"
#include <cstdio>

SolverBuffers SolverBuffers::create(
    int max_samples, int n_dof,
    int num_new_configs, int granularity,
    int n_joints, bool enable_mesh)
{
    SolverBuffers b;
    b.max_samples = max_samples;
    b.n_dof = n_dof;
    b.num_new_configs = num_new_configs;
    b.granularity = granularity;
    b.n_joints = n_joints;
    b.mesh_enabled = enable_mesh;

    const size_t config_size = n_dof * sizeof(float);

    // Pointer-to-pointer arrays (kernel reads these to find tree storage)
    cudaMalloc(&b.d_nodes,   2 * sizeof(float*));
    cudaMalloc(&b.d_parents, 2 * sizeof(int*));
    cudaMalloc(&b.d_radii,   2 * sizeof(float*));

    // Tree node/parent/radii storage (2 trees)
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&b.d_nodes_ptrs[i],   (size_t)max_samples * config_size);
        cudaMalloc(&b.d_parents_ptrs[i], (size_t)max_samples * sizeof(int));
        cudaMalloc(&b.d_radii_ptrs[i],   (size_t)max_samples * sizeof(float));
    }

    // Upload tree pointers to device
    cudaMemcpy(b.d_nodes,   b.d_nodes_ptrs,   2 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(b.d_parents, b.d_parents_ptrs,  2 * sizeof(int*),   cudaMemcpyHostToDevice);
    cudaMemcpy(b.d_radii,   b.d_radii_ptrs,    2 * sizeof(float*), cudaMemcpyHostToDevice);

    // RNG states
    int num_rng = num_new_configs * n_dof;
    cudaMalloc(&b.rng_states,    num_rng * sizeof(curandState));
    cudaMalloc(&b.halton_states, num_new_configs * sizeof(HaltonState_runtime));

    // Mesh collision transforms (conditional)
    if (enable_mesh && n_joints > 0) {
        size_t mesh_buf_size = (size_t)num_new_configs * granularity
                               * n_joints * 16 * sizeof(float);
        cudaMalloc(&b.d_mesh_transforms, mesh_buf_size);
    }

    // Host-pinned solved flag
    cudaMallocHost(&b.h_solved, sizeof(int));

    b.owns_memory = true;
    return b;
}

void SolverBuffers::destroy() {
    if (!owns_memory) return;

    cudaFree(d_nodes_ptrs[0]);
    cudaFree(d_nodes_ptrs[1]);
    cudaFree(d_parents_ptrs[0]);
    cudaFree(d_parents_ptrs[1]);
    cudaFree(d_radii_ptrs[0]);
    cudaFree(d_radii_ptrs[1]);
    cudaFree(d_nodes);
    cudaFree(d_parents);
    cudaFree(d_radii);
    cudaFree(rng_states);
    cudaFree(halton_states);
    if (d_mesh_transforms) cudaFree(d_mesh_transforms);
    if (h_solved) cudaFreeHost(h_solved);

    // Reset all pointers
    d_nodes_ptrs[0] = d_nodes_ptrs[1] = nullptr;
    d_parents_ptrs[0] = d_parents_ptrs[1] = nullptr;
    d_radii_ptrs[0] = d_radii_ptrs[1] = nullptr;
    d_nodes = nullptr;
    d_parents = nullptr;
    d_radii = nullptr;
    rng_states = nullptr;
    halton_states = nullptr;
    d_mesh_transforms = nullptr;
    h_solved = nullptr;
    owns_memory = false;
}
