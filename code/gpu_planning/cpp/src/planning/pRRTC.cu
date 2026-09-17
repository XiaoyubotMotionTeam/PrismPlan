#include "Planners.hh"
#include "Robots.hh"
#include "utils.cuh"
#include "robot_model.cuh"
#include "pRRTC_settings.hh"
#include "halton_state.hh"
#include "solver_buffers.hh"
#include "src/collision/environment.hh"
#include "src/robots/panda.cuh"
#include "src/collision/scene_collision.cuh"
#include "src/collision/mesh_collision.cuh"
#include "src/collision/two_phase_cc.cuh"
#include "src/planning/shortcut.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>



/*
Parallelized RRTC: Each block works to add a config to the tree (either start or goal depending on balance)
*/


namespace pRRTC {
    using namespace ppln;
    __device__ volatile int solved = 0;
    __device__ volatile int atomic_free_index[2]; // separate for tree_a and tree_b
    __device__ volatile int nodes_size[2];
    __device__ volatile int completed_nodes[2]; // track completed nodes for each tree
    constexpr int MAX_PATH_SIZE = 5000;
    __device__ float path[2][MAX_PATH_SIZE]; // solution path segments for tree_a, and tree_b
    __device__ int path_size[2] = {0, 0};
    __device__ float cost = 0.0;
    __device__ int reached_goal_idx = 0;
    __device__ int solved_iters = 0; // value of iters in the block that solves the problem
    __constant__ pRRTC_settings d_settings;

    constexpr int MAX_GRANULARITY = 32;
    constexpr int MAX_THREADS_PER_BLOCK = 4*MAX_GRANULARITY;

    constexpr int BLOCK_SIZE = 64;
    constexpr float UNWRITTEN_VAL = -9999.0f;

    template<typename Robot>
    struct HaltonState {
        float b[Robot::dimension];   // bases
        float n[Robot::dimension];   // numerators
        float d[Robot::dimension];   // denominators
    };

    void __device__ shuffle_array(float *array, int n, curandState &state) {
        for (int i = n - 1; i > 0; i--) {
            int j = curand(&state) % (i + 1);
            float temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    template<typename Robot>
    __device__ void halton_initialize(HaltonState<Robot>& state, size_t skip_iterations, curandState& rng_state, int idx) {
        
        float primes[16] = {
            3.f, 5.f, 7.f, 11.f, 13.f, 17.f, 19.f, 23.f,
            29.f, 31.f, 37.f, 41.f, 43.f, 47.f, 53.f, 59.f
        };
        if (idx != 0) shuffle_array(primes, 16, rng_state);
        
        // Initialize bases from primes
        for (size_t i = 0; i < Robot::dimension; i++) {
            state.b[i] = primes[i];
            state.n[i] = 0.0f;
            state.d[i] = 1.0f;
        }
        
        // Skip iterations if requested
        volatile float temp_result[Robot::dimension];
        for (size_t i = 0; i < skip_iterations; i++) {
            halton_next(state, (float *)temp_result);
        }
    }

    template<typename Robot>
    __device__ void halton_next(HaltonState<Robot>& state, float* result) {
        for (size_t i = 0; i < Robot::dimension; i++) {
            float xf = state.d[i] - state.n[i];
            bool x_eq_1 = (xf == 1.0f);
            
            if (x_eq_1) {
                // x == 1 case
                state.d[i] = floorf(state.d[i] * state.b[i]);
                state.n[i] = 1.0f;
            } else {
                // x != 1 case
                float y = floorf(state.d[i] / state.b[i]);
                
                // Continue dividing by b until we find the right digit position
                while (xf <= y) {
                    y = floorf(y / state.b[i]);
                }
                
                state.n[i] = floorf((state.b[i] + 1.0f) * y) - xf;
            }
            
            result[i] = state.n[i] / state.d[i];
        }
    }

    __global__ void init_rng(curandState* states, unsigned long seed, int num_rng_states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_rng_states) return;
        curand_init(seed + idx, idx, 0, &states[idx]);
    }

    template <typename Robot>
    __global__ void init_halton(HaltonState<Robot>* states, curandState* cr_states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= d_settings.num_new_configs) return;
        // int skip = (curand_uniform(&cr_states[idx]) * 50000.0f);
        int skip = 0;
        if (idx == 0) skip = 0;
        halton_initialize(states[idx], skip, cr_states[idx], idx);
    }

    __device__ inline void print_config(volatile float *config, int dim) {
        for (int i = 0; i < dim; i++) {
            printf("%f ,", config[i]);
        }
        printf("\n");
    }

    inline void setup_environment_on_device(ppln::collision::Environment<float> *&d_env, 
                                      const ppln::collision::Environment<float> &h_env) {
        // allocate the environment struct
        cudaMalloc(&d_env, sizeof(ppln::collision::Environment<float>));
        
        // Initialize struct to zeros first
        cudaMemset(d_env, 0, sizeof(ppln::collision::Environment<float>));

        // Handle each primitive type separately
        if (h_env.num_spheres > 0) {
            // Allocate and copy spheres array
            ppln::collision::Sphere<float> *d_spheres;
            cudaMalloc(&d_spheres, sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres);
            cudaMemcpy(d_spheres, h_env.spheres, 
                    sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres, 
                    cudaMemcpyHostToDevice);
            
            // Update the struct fields directly
            cudaMemcpy(&(d_env->spheres), &d_spheres, sizeof(ppln::collision::Sphere<float>*), 
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_spheres), &h_env.num_spheres, sizeof(unsigned int), 
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_capsules > 0) {
            ppln::collision::Capsule<float> *d_capsules;
            cudaMalloc(&d_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules);
            cudaMemcpy(d_capsules, h_env.capsules,
                    sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->capsules), &d_capsules, sizeof(ppln::collision::Capsule<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_capsules), &h_env.num_capsules, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        // Repeat for each primitive type...
        if (h_env.num_z_aligned_capsules > 0) {
            ppln::collision::Capsule<float> *d_z_capsules;
            cudaMalloc(&d_z_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules);
            cudaMemcpy(d_z_capsules, h_env.z_aligned_capsules,
                    sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_capsules), &d_z_capsules, sizeof(ppln::collision::Capsule<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_capsules), &h_env.num_z_aligned_capsules, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_cylinders > 0) {
            ppln::collision::Cylinder<float> *d_cylinders;
            cudaMalloc(&d_cylinders, sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders);
            cudaMemcpy(d_cylinders, h_env.cylinders,
                    sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->cylinders), &d_cylinders, sizeof(ppln::collision::Cylinder<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cylinders), &h_env.num_cylinders, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_cuboids;
            cudaMalloc(&d_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids);
            cudaMemcpy(d_cuboids, h_env.cuboids,
                    sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->cuboids), &d_cuboids, sizeof(ppln::collision::Cuboid<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cuboids), &h_env.num_cuboids, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_z_aligned_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_z_cuboids;
            cudaMalloc(&d_z_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids);
            cudaMemcpy(d_z_cuboids, h_env.z_aligned_cuboids,
                    sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_cuboids), &d_z_cuboids, sizeof(ppln::collision::Cuboid<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_cuboids), &h_env.num_z_aligned_cuboids, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }
    }


    inline void cleanup_environment_on_device(ppln::collision::Environment<float> *d_env, 
                                        const ppln::collision::Environment<float> &h_env) {
        // Get the pointers from device struct before freeing
        ppln::collision::Sphere<float> *d_spheres = nullptr;
        ppln::collision::Capsule<float> *d_capsules = nullptr;
        ppln::collision::Capsule<float> *d_z_capsules = nullptr;
        ppln::collision::Cylinder<float> *d_cylinders = nullptr;
        ppln::collision::Cuboid<float> *d_cuboids = nullptr;
        ppln::collision::Cuboid<float> *d_z_cuboids = nullptr;

        // Copy each pointer from device memory
        if (h_env.num_spheres > 0) {
            cudaMemcpy(&d_spheres, &(d_env->spheres), sizeof(ppln::collision::Sphere<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_spheres);
        }
        
        if (h_env.num_capsules > 0) {
            cudaMemcpy(&d_capsules, &(d_env->capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_capsules);
        }
        
        if (h_env.num_z_aligned_capsules > 0) {
            cudaMemcpy(&d_z_capsules, &(d_env->z_aligned_capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_capsules);
        }
        
        if (h_env.num_cylinders > 0) {
            cudaMemcpy(&d_cylinders, &(d_env->cylinders), sizeof(ppln::collision::Cylinder<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cylinders);
        }
        
        if (h_env.num_cuboids > 0) {
            cudaMemcpy(&d_cuboids, &(d_env->cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cuboids);
        }
        
        if (h_env.num_z_aligned_cuboids > 0) {
            cudaMemcpy(&d_z_cuboids, &(d_env->z_aligned_cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_cuboids);
        }

        // Finally free the environment struct itself
        cudaFree(d_env);
    }

    __global__ void reset_device_variables_kernel() {
        solved = 0;
        
        atomic_free_index[0] = 0;
        atomic_free_index[1] = 0;
        nodes_size[0] = 0;
        nodes_size[1] = 0;
        completed_nodes[0] = 0;
        completed_nodes[1] = 0;
        
        path_size[0] = 0;
        path_size[1] = 0;
        
        for (int tree = 0; tree < 2; tree++) {
            for (int i = 0; i < MAX_PATH_SIZE; i++) {
                path[tree][i] = 0.0f;
            }
        }
        
        cost = 0.0f;
        reached_goal_idx = 0;
    }

    void reset_device_variables() {
        reset_device_variables_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
    }

    __device__ __forceinline__ void reset_to_unwritten_state(volatile float *buffer, int size, int tid) {
        if (tid == 0) {
            for (int i = 0; i < size; i++) {
                buffer[i] = UNWRITTEN_VAL;
            }
        }
        __syncthreads();
    }
    
    template <typename Robot>
    __global__ void
    // __launch_bounds__(128, 8)
    rrtc(
        float **nodes,
        int **parents,
        float **radii,
        HaltonState<Robot> *halton_states,
        curandState *rng_states,
        ppln::collision::Environment<float> *env
    )
    {
        static constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x; // 0 ... NUM_NEW_CONFIGS
        __shared__ int t_tree_id; // this tree
        __shared__ int o_tree_id; // the other tree
        __shared__ float config[dim];
        __shared__ float sdata[MAX_THREADS_PER_BLOCK];
        __shared__ int sindex[MAX_THREADS_PER_BLOCK];
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ float *t_nodes;
        __shared__ float *o_nodes;
        __shared__ int *t_parents;
        __shared__ int *o_parents;
        __shared__ float scale;
        __shared__ float *nearest_node;
        __shared__ float delta[dim];
        __shared__ int index;
        __shared__ float vec[dim];
        __shared__ unsigned int n_extensions;
        __shared__ bool should_skip;
        __align__(16) __shared__ volatile float sphere_pos[6000]; // ~assuming max 120 spheres with granularity 32, each has x y z coordinates
        __align__(16) __shared__ volatile float sphere_pos_approx[2500]; // ~assuming 50 spheres with granularity 32, each has x y z coordinates
        __align__(16) __shared__ volatile int link_CC[640]; //assuming max granularity 32, max number of links 20
        __align__(16) __shared__ float T[16 * 2 * 16];

        int iter = 0;
        const long long clock_start = clock64();
        const long long time_limit_clocks = (d_settings.time_limit_ms > 0.0f && d_settings.gpu_clock_rate_khz > 0)
            ? (long long)(d_settings.time_limit_ms * d_settings.gpu_clock_rate_khz) : 0;

        while (true) {
            if (tid == 0) {
                // printf("iter: %d\n", iter);
                // printf("tree size: %d\n", atomic_free_index[0]);
                iter++;
                if (iter > d_settings.max_iters) {
                    atomicCAS((int *)&solved, 0, -1);
                }
                if (time_limit_clocks > 0 && (clock64() - clock_start) > time_limit_clocks) {
                    atomicCAS((int *)&solved, 0, -1);
                }

                if (d_settings.balance == 0 || iter == 1) {
                    t_tree_id = (bid < (d_settings.num_new_configs / 2))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1 && abs(atomic_free_index[0]-atomic_free_index[1]) < 1.5 * d_settings.num_new_configs) { // dynamic balance
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0]+atomic_free_index[1]);
                    float balance_factor = 1 - ratio;
                    t_tree_id = (bid < (d_settings.num_new_configs * balance_factor))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1) {
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0] + atomic_free_index[1]);
                    if (ratio < d_settings.tree_ratio) t_tree_id = 0;
                    else t_tree_id = 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 2) { // vamp balance
                    float ratio = abs(atomic_free_index[t_tree_id] - atomic_free_index[o_tree_id]) / (float) atomic_free_index[t_tree_id];
                    if (ratio < d_settings.tree_ratio)
                    {
                        t_tree_id = 1 - t_tree_id;
                        o_tree_id = 1 - t_tree_id;
                    }
                }

                t_nodes = nodes[t_tree_id];
                o_nodes = nodes[o_tree_id];
                t_parents = parents[t_tree_id];
                o_parents = parents[o_tree_id];
                
                halton_next(halton_states[bid], (float *)config);
                Robot::scale_cfg((float *)config);
                local_cc_result[0] = 0;
                // printf("config: %f %f %f %f %f %f %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6]);
                // 14 dim config for baxter
                // printf("config: %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13]);
                // // print out first 3 configs for testing
                // for (int i = 0; i < 4; i++) {
                //     float temp_config[dim];
                //     halton_next(halton_states[bid], (float *)temp_config);
                //     Robot::scale_cfg((float *)temp_config);
                //     printf("test q: %f %f %f %f %f %f %f\n", temp_config[0], temp_config[1], temp_config[2], temp_config[3], temp_config[4], temp_config[5], temp_config[6]);
                // }

            }

            // reset link_CC every iteration
            for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                link_CC[r]=0;
            }

            __syncthreads();

            // parallelized nearest neighbor search
            float local_min_dist = FLT_MAX;
            int local_near_idx = 0;
            float dist;
            int size = min(atomic_free_index[t_tree_id], completed_nodes[t_tree_id]);
            for (int i = tid; i < size; i += blockDim.x) {
                dist = device_utils::sq_l2_dist((float *)&t_nodes[i * dim], (float *) config, dim);
                if (dist < local_min_dist) {
                    local_min_dist = dist;
                    local_near_idx = i;
                }
            }
            sdata[tid] = local_min_dist;
            sindex[tid] = local_near_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                float sdata_tid = sdata[tid];
                float sdata_tid_s = sdata[tid + s];
                __syncthreads();
                if (tid < s){
                    if (sdata_tid_s < sdata_tid) {
                        sdata[tid] = sdata[tid + s];
                        sindex[tid] = sindex[tid + s];
                    }
                }
                __syncthreads();
            }

            // nn index is in sindex[0], distance in sdata[0]
            if (tid == 0) {
                sdata[0] = sqrt(sdata[0]);
                scale = min(1.0f, d_settings.range / (sdata[0]));
                nearest_node = &t_nodes[sindex[0] * dim];

                should_skip = (d_settings.dynamic_domain && radii[t_tree_id][sindex[0]] < sdata[0]);
            }
            __syncthreads();

            if (should_skip) {
                // if (tid == 0) printf("skipping\n");
                continue;
            }
            __syncthreads();

            if (tid < dim) {
                config[tid] = nearest_node[tid] + ((config[tid] - nearest_node[tid]) * scale);
                delta[tid] = (config[tid] - nearest_node[tid]) / (float) d_settings.granularity;
            }
            __syncthreads();

            // validate edge
            float interp_cfg[dim];
            for (int i = 0; i < dim; i++) {
                interp_cfg[i] = nearest_node[i] + (int(tid/4 + 1) * delta[i]);
            }
            
            //approximate FK & CC first, if collision found then detailed FK & CC
            int detailed_FK=0;
            // if (tid == 0) {
            //     printf("q: %f %f %f %f %f %f %f\n", interp_cfg[0], interp_cfg[1], interp_cfg[2], interp_cfg[3], interp_cfg[4], interp_cfg[5], interp_cfg[6]);
            // }
            
            ppln::collision::fk_approx<Robot>(interp_cfg, sphere_pos_approx, T, tid);
            __syncthreads();
            bool config_in_collision2_approx = not ppln::collision::env_collision_check_approx<Robot>(sphere_pos_approx, link_CC, env, tid);
            atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2_approx ? 1u : 0u);
            
            __syncthreads();
            // if collision found in approx env check, proceed to detailed env check
            if (local_cc_result[0]==1){
                // if (tid == 0) printf("approx env collision\n");
                if (tid==0) local_cc_result[0]=0;
                __syncthreads();
                //reset_to_unwritten_state(sphere_pos, 4000, tid);
                ppln::collision::fk<Robot>(interp_cfg, sphere_pos, T, tid);
                detailed_FK=1;
                __syncthreads();
                bool config_in_collision2 = not ppln::collision::env_collision_check<Robot>(sphere_pos, link_CC, env, tid);
                // if (tid == 63) {
                //     printf("config_in_collision2: %d\n", config_in_collision2);
                // }
                atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2 ? 1u : 0u);
                __syncthreads();
            }
            
            for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                link_CC[r]=0;
            }
            __syncthreads();
            // if env check is collision free, proceed to self-collision check
            if (local_cc_result[0]==0){
                
                bool config_in_collision_approx = not ppln::collision::self_collision_check_approx<Robot>(sphere_pos_approx, link_CC, tid);
                atomicOr((unsigned int *)&local_cc_result[0], config_in_collision_approx ? 1u : 0u);
                __syncthreads();
                // if collision found in approx self check, proceed to detailed self check
                if (local_cc_result[0]==1){
                    // if (tid == 0) printf("approx self collision\n");
                    if (tid==0) local_cc_result[0]=0;
                    __syncthreads();
                    if (detailed_FK==0){
                        //reset_to_unwritten_state(sphere_pos, 4000, tid);
                        ppln::collision::fk<Robot>(interp_cfg, sphere_pos, T, tid);
                        detailed_FK=1;
                        __syncthreads();
                    }
                    bool config_in_collision = not ppln::collision::self_collision_check<Robot>(sphere_pos, link_CC, tid);
                    atomicOr((unsigned int *)&local_cc_result[0], config_in_collision ? 1u : 0u);
                    __syncthreads();
                }
                //if(blockIdx.x==0) printf("tid %d: env_collision - %d\n", tid, config_in_collision2);
            }

            bool edge_good = local_cc_result[0] == 0;
            __syncthreads();
            if (edge_good) {
                // grow tree
                if (tid == 0) {
                    // printf("edge good\n");
                    index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                    if (index >= d_settings.max_samples) solved = -1;
                    
                    t_parents[index] = sindex[0];
                    
                    if (d_settings.dynamic_domain) {
                        radii[t_tree_id][index] = FLT_MAX;
                        volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                        float old_radius, new_radius;
                        int expected, desired;
                        do {
                            old_radius = *radius_ptr;
                            if (old_radius == FLT_MAX) break;
                            new_radius = old_radius * (1 + d_settings.dd_alpha);
                            expected = __float_as_int(old_radius);
                            desired = __float_as_int(new_radius);
                        } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
                    }
                    // float last_interp_cfg[dim];
                    // for (int i = 0; i < dim; i++) {
                    //     last_interp_cfg[i] = nearest_node[i] + (32 * delta[i]);
                    // }
                    // printf("last_interp_cfg: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", last_interp_cfg[0], last_interp_cfg[1], last_interp_cfg[2], last_interp_cfg[3], last_interp_cfg[4], last_interp_cfg[5], last_interp_cfg[6], last_interp_cfg[7], last_interp_cfg[8], last_interp_cfg[9], last_interp_cfg[10], last_interp_cfg[11], last_interp_cfg[12], last_interp_cfg[13]);
                    // printf("config added: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13]);
                }
                __syncthreads();

                if (tid < dim) {
                    t_nodes[index * dim + tid] = config[tid];
                }
                if (tid == 0) {
                    atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                    __threadfence();
                }
                __syncthreads();

                // connect
                local_min_dist = FLT_MAX;
                local_near_idx = 0;
                int size = min(atomic_free_index[o_tree_id], completed_nodes[o_tree_id]);
                for (unsigned int i = tid; i < size; i += blockDim.x) {
                    dist = device_utils::sq_l2_dist((float *)&o_nodes[i * dim], (float *)config, dim);
                    if (dist < local_min_dist) {
                        local_min_dist = dist;
                        local_near_idx = i;
                    }
                }
                sdata[tid] = local_min_dist;
                sindex[tid] = local_near_idx;
                __syncthreads();
                
                for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                    float sdata_tid = sdata[tid];
                    float sdata_tid_s = sdata[tid + s];
                    __syncthreads();
                    if (tid < s){
                        if (sdata_tid_s < sdata_tid) {
                            sdata[tid] = sdata[tid + s];
                            sindex[tid] = sindex[tid + s];
                        }
                    }
                    __syncthreads();
                }
                
                
                if (tid == 0) {
                    sdata[0] = sqrt(sdata[0]);
                    nearest_node = &o_nodes[sindex[0] * dim];
                    n_extensions = ceil(sdata[0] / d_settings.range);
                    local_cc_result[0] = 0;
                }
                __syncthreads();

                if (tid < dim) {
                    vec[tid] = (nearest_node[tid] - config[tid]) / (float) n_extensions;
                }
                __syncthreads();

                // validate the edge to the nearest neighbor in opposite tree, go as far as we can
                int i_extensions = 0;
                int extension_parent_idx = index;
                while (i_extensions < n_extensions) {
                    for (int i = 0; i < dim; i++) {
                        interp_cfg[i] = config[i] + (int(tid/4 + 1) * (vec[i] / (float) d_settings.granularity));
                    }
                    __syncthreads();
                    
                    //approximate FK & CC first, if collision found then detailed FK & CC
                    int detailed_FK=0;
                    // if (tid == 0) {
                    //     printf("q: %f %f %f %f %f %f %f\n", interp_cfg[0], interp_cfg[1], interp_cfg[2], interp_cfg[3], interp_cfg[4], interp_cfg[5], interp_cfg[6]);
                    // }
                    // clear link_CC
                    for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                        link_CC[r]=0;
                    }
                    __syncthreads();
                    ppln::collision::fk_approx<Robot>(interp_cfg, sphere_pos_approx, T, tid);
                    __syncthreads();
                    bool config_in_collision2_approx = not ppln::collision::env_collision_check_approx<Robot>(sphere_pos_approx, link_CC, env, tid);
                    atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2_approx ? 1u : 0u);
                    __syncthreads();
                    // if collision found in approx env check, proceed to detailed env check
                    if (local_cc_result[0]==1){
                        // if (tid == 0) printf("approx env collision in extension\n");
                        if (tid==0) local_cc_result[0]=0;
                        __syncthreads();
                        //reset_to_unwritten_state(sphere_pos, 4000, tid);
                        ppln::collision::fk<Robot>(interp_cfg, sphere_pos, T, tid);
                        detailed_FK=1;
                        __syncthreads();
                        bool config_in_collision2 = not ppln::collision::env_collision_check<Robot>(sphere_pos, link_CC, env, tid);
                        atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2 ? 1u : 0u);
                        __syncthreads();
                    }
                    //if (tid==0) {
                        //printf("new round\n");
                        //ppln::collision::fkcc<Robot>(interp_cfg, env, tid);
                    //}
                    // if env check is collision free, proceed to self-collision check
                    for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                        link_CC[r]=0;
                    }
                    __syncthreads();
                    if (local_cc_result[0]==0){
                        bool config_in_collision_approx = not ppln::collision::self_collision_check_approx<Robot>(sphere_pos_approx, link_CC, tid);
                        atomicOr((unsigned int *)&local_cc_result[0], config_in_collision_approx ? 1u : 0u);
                        __syncthreads();
                        // if collision found in approx self check, proceed to detailed self check
                        if (local_cc_result[0]==1){
                            // if (tid == 0) printf("approx self collision in extension\n");
                            if (tid==0) local_cc_result[0]=0;
                            __syncthreads();
                            if (detailed_FK==0){
                                //reset_to_unwritten_state(sphere_pos, 4000, tid);
                                ppln::collision::fk<Robot>(interp_cfg, sphere_pos, T, tid);
                                detailed_FK=1;
                                __syncthreads();
                            }
                            bool config_in_collision = not ppln::collision::self_collision_check<Robot>(sphere_pos, link_CC, tid);
                            atomicOr((unsigned int *)&local_cc_result[0], config_in_collision ? 1u : 0u);
                            __syncthreads();
                        }
                        //if(blockIdx.x==0) printf("tid %d: env_collision - %d\n", tid, config_in_collision2);
                    }

                    bool ext_edge_good = local_cc_result[0] == 0;
                    if (!ext_edge_good) break;
                    if (tid == 0) {
                        index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                        if (index >= d_settings.max_samples) solved = -1;
                        t_parents[index] = extension_parent_idx;
                        radii[t_tree_id][index] = FLT_MAX;
                        extension_parent_idx = index;
                        local_cc_result[0] = 0;
                        // printf("config added (extension): %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13]);
                    }
                    __syncthreads();
                    if (tid < dim) {
                        config[tid] = config[tid] + vec[tid];
                        t_nodes[index * dim + tid] = config[tid];
                    }
                    if (tid == 0) {
                        atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                        __threadfence();
                    }
                    __syncthreads();
                    i_extensions++;
                    __syncthreads();
                }
                if (i_extensions == n_extensions) { // connected
                    if (tid == 0 && atomicCAS((int *)&solved, 0, 1) == 0) {
                        // trace back to the start and goal.
                        int current = index;
                        int parent;
                        int t_path_size = 0;
                        int o_path_size = 0;
                        while (t_parents[current] != current) {
                            parent = t_parents[current];
                            cost += device_utils::l2_dist((float *)&t_nodes[current * dim], (float *)&t_nodes[parent * dim], dim);
                            if ((t_path_size + 1) * dim > MAX_PATH_SIZE) { solved = -1; break; }
                            for (int i = 0; i < dim; i++) path[t_tree_id][t_path_size * dim + i] = t_nodes[current * dim + i];
                            t_path_size++;
                            current = parent;

                        }
                        if (t_tree_id == 1) reached_goal_idx = current;
                        current = sindex[0];
                        while(o_parents[current] != current) {
                            parent = o_parents[current];
                            cost += device_utils::l2_dist((float *)&o_nodes[current * dim], (float *)&o_nodes[parent * dim], dim);
                            if ((o_path_size + 1) * dim > MAX_PATH_SIZE) { solved = -1; break; }
                            for (int i = 0; i < dim; i++) path[o_tree_id][o_path_size * dim + i] = o_nodes[current * dim + i];
                            o_path_size++;
                            current = parent;
                        }
                        if (t_tree_id == 0) reached_goal_idx = current;
                        path_size[t_tree_id] = t_path_size;
                        path_size[o_tree_id] = o_path_size;
                        solved_iters = iter;
                    }
                    __syncthreads();
                }
            }
            else if (d_settings.dynamic_domain && tid == 0) {      
                // printf("no config added\n");
                volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                float old_radius, new_radius;
                int expected, desired;
                do {
                    old_radius = *radius_ptr;
                    if (old_radius == FLT_MAX) {
                        new_radius = d_settings.dd_radius;
                    } else {
                        new_radius = fmaxf(old_radius * (1.f - d_settings.dd_alpha), d_settings.dd_min_radius);
                    }
                    expected = __float_as_int(old_radius);
                    desired = __float_as_int(new_radius);
                } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
            }
            __syncthreads();
            if (solved != 0) return;
        }
    }




    template <typename Robot>
    PlannerResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &h_environment,
        pRRTC_settings &settings
    ) 
    {
        auto start_time = std::chrono::steady_clock::now();
        static constexpr auto dim = Robot::dimension;
        std::size_t start_index = 0;
        PlannerResult<Robot> res;

        // Auto-fill GPU clock rate for kernel-side time limit
        if (settings.gpu_clock_rate_khz == 0 && settings.time_limit_ms > 0.0f) {
            int clock_rate_khz = 0;
            if (cudaDeviceGetAttribute(&clock_rate_khz,
                                       cudaDevAttrClockRate, 0) == cudaSuccess) {
                settings.gpu_clock_rate_khz = clock_rate_khz;
            }
        }

        // copy data to GPU
        cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings));
        int num_goals = goals.size();
        float *nodes[2];
        int *parents[2];
        float *radii[2];
        float **d_nodes;
        int **d_parents;
        float **d_radii;
        cudaMalloc(&d_nodes, 2 * sizeof(float*));
        cudaMalloc(&d_parents, 2 * sizeof(int*));
        cudaMalloc(&d_radii, 2 * sizeof(float*));
        const std::size_t config_size = dim * sizeof(float);

        for (int i = 0; i < 2; i++) {
            cudaMalloc(&nodes[i], settings.max_samples * config_size);
            cudaMalloc(&parents[i], settings.max_samples * sizeof(int));
            cudaMalloc(&radii[i], settings.max_samples * sizeof(float));
        }
        cudaMemcpy(d_nodes, nodes, 2 * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_parents, parents, 2 * sizeof(int*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_radii, radii, 2 * sizeof(float*), cudaMemcpyHostToDevice);

        // set nodes to unitialized
        std::vector<float> nodes_init(settings.max_samples * dim, UNWRITTEN_VAL);
        cudaMemcpy((void *)nodes[0], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)nodes[1], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
        
       

        // initialize radii
        std::vector<float> radii_init(num_goals, FLT_MAX);
        cudaMemcpy((void *)radii[0], radii_init.data(), sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)radii[1], radii_init.data(), sizeof(float) * num_goals, cudaMemcpyHostToDevice);
        
        // create a curandState for each thread
        curandState *rng_states;
        int num_rng_states = settings.num_new_configs * dim;
        cudaMalloc(&rng_states, num_rng_states * sizeof(curandState));
        int numBlocks = (num_rng_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_rng<<<numBlocks, BLOCK_SIZE>>>(rng_states, 1, num_rng_states);

        HaltonState<Robot> *halton_states;
        cudaMalloc(&halton_states, settings.num_new_configs * sizeof(HaltonState<Robot>));
        int numBlocks1 = (settings.num_new_configs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_halton<Robot><<<numBlocks1, BLOCK_SIZE>>>(halton_states, rng_states);

        // free index for next available position in tree_a and tree_b
        int h_free_index[2] = {1, num_goals};
        cudaMemcpyToSymbol(atomic_free_index, &h_free_index, sizeof(int) * 2);
        cudaMemcpyToSymbol(nodes_size, &h_free_index, sizeof(int) * 2);
        
        // initialize completed_nodes counter
        int h_completed_nodes[2] = {1, num_goals}; // start and goals are already written
        cudaMemcpyToSymbol(completed_nodes, &h_completed_nodes, sizeof(int) * 2);
        
        // allocate for obstacles
        ppln::collision::Environment<float> *env;
        setup_environment_on_device(env, h_environment);
        cudaCheckError(cudaGetLastError());
        
        // Setup pinned memory for signaling
        int *h_solved;
        int current_samples[2];
        int h_solved_iters = -1;
        cudaMallocHost(&h_solved, sizeof(int));
        *h_solved = -1;

        
        auto copy_start_time = std::chrono::steady_clock::now();
        // add start to tree_a and goals to tree_b
        cudaMemcpy((void *)nodes[0], start.data(), config_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)parents[0], &start_index, sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy((void *)nodes[1], goals.data(), config_size * num_goals, cudaMemcpyHostToDevice);
        std::vector<int> parents_b_init(num_goals);
        iota(parents_b_init.begin(), parents_b_init.end(), 0); // consecutive integers from 0 ... num_goals - 1
        cudaMemcpy((void *)parents[1], parents_b_init.data(), sizeof(int) * num_goals, cudaMemcpyHostToDevice);
        res.copy_ns = get_elapsed_nanoseconds(copy_start_time);

        auto kernel_start_time = std::chrono::steady_clock::now();
        rrtc<Robot><<<settings.num_new_configs, 4*settings.granularity>>> (
            d_nodes,
            d_parents,
            d_radii,
            halton_states,
            rng_states,
            env
        );
        cudaDeviceSynchronize();
        res.kernel_ns = get_elapsed_nanoseconds(kernel_start_time);

        // get data from device
        copy_start_time = std::chrono::steady_clock::now();
        cudaMemcpyFromSymbol(current_samples, atomic_free_index, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(h_solved, solved, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&h_solved_iters, solved_iters, sizeof(int), 0, cudaMemcpyDeviceToHost);
        res.copy_ns += get_elapsed_nanoseconds(copy_start_time);

        cudaCheckError(cudaGetLastError());

        // add data to result struct
        if (*h_solved!=1) *h_solved=0;
        res.start_tree_size = current_samples[0];
        res.goal_tree_size = current_samples[1];
        if (*h_solved) {
            int h_path_size[2];
            float h_paths[2][MAX_PATH_SIZE];
            float h_cost;
            int h_reached_goal_idx;
            cudaMemcpyFromSymbol(h_path_size, path_size, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(h_paths, path, sizeof(float) * 2 * MAX_PATH_SIZE, 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&h_cost, cost, sizeof(float), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&h_reached_goal_idx, reached_goal_idx, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaCheckError(cudaGetLastError());
            res.path.emplace_back(goals[h_reached_goal_idx]);
            typename Robot::Configuration config;
            for (int i = h_path_size[1] - 1; i >= 0; i--) {
                std::copy_n(h_paths[1] + i * dim, dim, config.begin());
                res.path.emplace_back(config);
            }
            for (int i = 0; i < h_path_size[0]; i++) {
                std::copy_n(h_paths[0] + i * dim, dim, config.begin());
                res.path.emplace_back(config);
            }
            res.path.emplace_back(start);
            res.cost = h_cost;
            res.path_length = (h_path_size[0] + h_path_size[1]);
        }
        res.solved = (*h_solved) != 0;
        res.iters = h_solved_iters;
        
        cleanup_environment_on_device(env, h_environment);
        reset_device_variables();
        cudaFree((void *)nodes[0]);
        cudaFree((void *)nodes[1]);
        cudaFree((void *)parents[0]);
        cudaFree((void *)parents[1]);
        cudaFree((void *)radii[0]);
        cudaFree((void *)radii[1]);
        cudaFree(rng_states);
        cudaFree(halton_states);
        cudaFree(d_nodes);
        cudaFree(d_parents);
        cudaFree(d_radii);
        cudaFreeHost(h_solved);
        cudaCheckError(cudaGetLastError());
        res.wall_ns = get_elapsed_nanoseconds(start_time);
        return res;
    }

    //template PlannerResult<typename ppln::robots::Sphere> solve<ppln::robots::Sphere>(std::array<float, 3>&, std::vector<std::array<float, 3>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::Panda> solve<ppln::robots::Panda>(std::array<float, 7>&, std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, pRRTC_settings&);


    // ======================================================================
    // Runtime (data-driven) pRRTC — uses RobotModel instead of template Robot
    // ======================================================================

    // HaltonState_runtime is now in halton_state.hh

    void __device__ halton_initialize_runtime(HaltonState_runtime& state, int dim, size_t skip_iterations, curandState& rng_state, int idx) {
        float primes[16] = {
            3.f, 5.f, 7.f, 11.f, 13.f, 17.f, 19.f, 23.f,
            29.f, 31.f, 37.f, 41.f, 43.f, 47.f, 53.f, 59.f
        };
        if (idx != 0) shuffle_array(primes, 16, rng_state);
        for (int i = 0; i < dim; i++) {
            state.b[i] = primes[i];
            state.n[i] = 0.0f;
            state.d[i] = 1.0f;
        }
        for (size_t s = 0; s < skip_iterations; s++) {
            for (int i = 0; i < dim; i++) {
                float xf = state.d[i] - state.n[i];
                if (xf == 1.0f) {
                    state.d[i] = floorf(state.d[i] * state.b[i]);
                    state.n[i] = 1.0f;
                } else {
                    float y = floorf(state.d[i] / state.b[i]);
                    while (xf <= y) y = floorf(y / state.b[i]);
                    state.n[i] = floorf((state.b[i] + 1.0f) * y) - xf;
                }
            }
        }
    }

    __device__ void halton_next_runtime(HaltonState_runtime& state, float* result, int dim) {
        for (int i = 0; i < dim; i++) {
            float xf = state.d[i] - state.n[i];
            if (xf == 1.0f) {
                state.d[i] = floorf(state.d[i] * state.b[i]);
                state.n[i] = 1.0f;
            } else {
                float y = floorf(state.d[i] / state.b[i]);
                while (xf <= y) y = floorf(y / state.b[i]);
                state.n[i] = floorf((state.b[i] + 1.0f) * y) - xf;
            }
            result[i] = state.n[i] / state.d[i];
        }
    }

    __global__ void init_halton_runtime(HaltonState_runtime* states, curandState* cr_states, int dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= d_settings.num_new_configs) return;
        halton_initialize_runtime(states[idx], dim, 0, cr_states[idx], idx);
    }

    // Runtime rrtc kernel — RobotModel passed by value (contains device pointers)
    __global__ void
    rrtc_runtime(
        float **nodes,
        int **parents,
        float **radii,
        HaltonState_runtime *halton_states,
        curandState *rng_states,
        ppln::collision::Environment<float> *env,
        ppln::RobotModel model,
        int dim
    )
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        __shared__ int t_tree_id;
        __shared__ int o_tree_id;
        __shared__ float config[ppln::MAX_DIM];
        __shared__ float sdata[MAX_THREADS_PER_BLOCK];
        __shared__ int sindex[MAX_THREADS_PER_BLOCK];
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ float *t_nodes;
        __shared__ float *o_nodes;
        __shared__ int *t_parents;
        __shared__ int *o_parents;
        __shared__ float scale;
        __shared__ float *nearest_node;
        __shared__ float delta[ppln::MAX_DIM];
        __shared__ int index;
        __shared__ float vec[ppln::MAX_DIM];
        __shared__ unsigned int n_extensions;
        __shared__ bool should_skip;
        __align__(16) __shared__ volatile float sphere_pos[6000];
        __align__(16) __shared__ volatile float sphere_pos_approx[2500];
        __align__(16) __shared__ volatile int link_CC[640];
        __align__(16) __shared__ float T[16 * 2 * 16];

        int iter = 0;
        const long long clock_start = clock64();
        const long long time_limit_clocks = (d_settings.time_limit_ms > 0.0f && d_settings.gpu_clock_rate_khz > 0)
            ? (long long)(d_settings.time_limit_ms * d_settings.gpu_clock_rate_khz) : 0;

        while (true) {
            if (tid == 0) {
                iter++;
                if (iter > d_settings.max_iters) {
                    atomicCAS((int *)&solved, 0, -1);
                }
                if (time_limit_clocks > 0 && (clock64() - clock_start) > time_limit_clocks) {
                    atomicCAS((int *)&solved, 0, -1);
                }

                if (d_settings.balance == 0 || iter == 1) {
                    t_tree_id = (bid < (d_settings.num_new_configs / 2))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1 && abs(atomic_free_index[0]-atomic_free_index[1]) < 1.5 * d_settings.num_new_configs) {
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0]+atomic_free_index[1]);
                    float balance_factor = 1 - ratio;
                    t_tree_id = (bid < (d_settings.num_new_configs * balance_factor))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1) {
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0] + atomic_free_index[1]);
                    if (ratio < d_settings.tree_ratio) t_tree_id = 0;
                    else t_tree_id = 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 2) {
                    float ratio = abs(atomic_free_index[t_tree_id] - atomic_free_index[o_tree_id]) / (float) atomic_free_index[t_tree_id];
                    if (ratio < d_settings.tree_ratio) {
                        t_tree_id = 1 - t_tree_id;
                        o_tree_id = 1 - t_tree_id;
                    }
                }

                t_nodes = nodes[t_tree_id];
                o_nodes = nodes[o_tree_id];
                t_parents = parents[t_tree_id];
                o_parents = parents[o_tree_id];

                halton_next_runtime(halton_states[bid], (float *)config, dim);
                ppln::collision::scale_cfg_runtime(model, (float *)config);
                local_cc_result[0] = 0;
            }

            for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                link_CC[r]=0;
            }
            __syncthreads();

            // parallelized nearest neighbor search
            float local_min_dist = FLT_MAX;
            int local_near_idx = 0;
            float dist;
            int size = min(atomic_free_index[t_tree_id], completed_nodes[t_tree_id]);
            for (int i = tid; i < size; i += blockDim.x) {
                dist = device_utils::sq_l2_dist((float *)&t_nodes[i * dim], (float *) config, dim);
                if (dist < local_min_dist) {
                    local_min_dist = dist;
                    local_near_idx = i;
                }
            }
            sdata[tid] = local_min_dist;
            sindex[tid] = local_near_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                float sdata_tid = sdata[tid];
                float sdata_tid_s = sdata[tid + s];
                __syncthreads();
                if (tid < s){
                    if (sdata_tid_s < sdata_tid) {
                        sdata[tid] = sdata[tid + s];
                        sindex[tid] = sindex[tid + s];
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                sdata[0] = sqrt(sdata[0]);
                scale = min(1.0f, d_settings.range / (sdata[0]));
                nearest_node = &t_nodes[sindex[0] * dim];
                should_skip = (d_settings.dynamic_domain && radii[t_tree_id][sindex[0]] < sdata[0]);
            }
            __syncthreads();

            if (should_skip) continue;
            __syncthreads();

            if (tid < dim) {
                config[tid] = nearest_node[tid] + ((config[tid] - nearest_node[tid]) * scale);
                delta[tid] = (config[tid] - nearest_node[tid]) / (float) d_settings.granularity;
            }
            __syncthreads();

            // validate edge
            float interp_cfg[ppln::MAX_DIM];
            for (int i = 0; i < dim; i++) {
                interp_cfg[i] = nearest_node[i] + (int(tid/4 + 1) * delta[i]);
            }

            int detailed_FK = 0;

            ppln::collision::fk_approx_runtime(model, interp_cfg, sphere_pos_approx, T, tid);
            __syncthreads();
            bool config_in_collision2_approx = !ppln::collision::env_collision_check_approx_runtime(model, sphere_pos_approx, link_CC, env, tid);
            atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2_approx ? 1u : 0u);
            __syncthreads();

            if (local_cc_result[0] == 1) {
                if (tid == 0) local_cc_result[0] = 0;
                __syncthreads();
                ppln::collision::fk_runtime(model, interp_cfg, sphere_pos, T, tid);
                detailed_FK = 1;
                __syncthreads();
                bool config_in_collision2 = !ppln::collision::env_collision_check_runtime(model, sphere_pos, link_CC, env, tid);
                atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2 ? 1u : 0u);
                __syncthreads();
            }

            for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                link_CC[r]=0;
            }
            __syncthreads();

            if (local_cc_result[0] == 0) {
                bool config_in_collision_approx = !ppln::collision::self_collision_check_approx_runtime(model, sphere_pos_approx, link_CC, tid);
                atomicOr((unsigned int *)&local_cc_result[0], config_in_collision_approx ? 1u : 0u);
                __syncthreads();
                if (local_cc_result[0] == 1) {
                    if (tid == 0) local_cc_result[0] = 0;
                    __syncthreads();
                    if (detailed_FK == 0) {
                        ppln::collision::fk_runtime(model, interp_cfg, sphere_pos, T, tid);
                        detailed_FK = 1;
                        __syncthreads();
                    }
                    bool config_in_collision = !ppln::collision::self_collision_check_runtime(model, sphere_pos, link_CC, tid);
                    atomicOr((unsigned int *)&local_cc_result[0], config_in_collision ? 1u : 0u);
                    __syncthreads();
                }
            }

            bool edge_good = local_cc_result[0] == 0;
            __syncthreads();
            if (edge_good) {
                if (tid == 0) {
                    index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                    if (index >= d_settings.max_samples) solved = -1;
                    t_parents[index] = sindex[0];
                    if (d_settings.dynamic_domain) {
                        radii[t_tree_id][index] = FLT_MAX;
                        volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                        float old_radius, new_radius;
                        int expected, desired;
                        do {
                            old_radius = *radius_ptr;
                            if (old_radius == FLT_MAX) break;
                            new_radius = old_radius * (1 + d_settings.dd_alpha);
                            expected = __float_as_int(old_radius);
                            desired = __float_as_int(new_radius);
                        } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
                    }
                }
                __syncthreads();

                if (tid < dim) {
                    t_nodes[index * dim + tid] = config[tid];
                }
                if (tid == 0) {
                    atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                    __threadfence();
                }
                __syncthreads();

                // connect
                local_min_dist = FLT_MAX;
                local_near_idx = 0;
                size = min(atomic_free_index[o_tree_id], completed_nodes[o_tree_id]);
                for (unsigned int i = tid; i < size; i += blockDim.x) {
                    dist = device_utils::sq_l2_dist((float *)&o_nodes[i * dim], (float *)config, dim);
                    if (dist < local_min_dist) {
                        local_min_dist = dist;
                        local_near_idx = i;
                    }
                }
                sdata[tid] = local_min_dist;
                sindex[tid] = local_near_idx;
                __syncthreads();

                for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                    float sdata_tid = sdata[tid];
                    float sdata_tid_s = sdata[tid + s];
                    __syncthreads();
                    if (tid < s) {
                        if (sdata_tid_s < sdata_tid) {
                            sdata[tid] = sdata[tid + s];
                            sindex[tid] = sindex[tid + s];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    sdata[0] = sqrt(sdata[0]);
                    nearest_node = &o_nodes[sindex[0] * dim];
                    n_extensions = ceil(sdata[0] / d_settings.range);
                    local_cc_result[0] = 0;
                }
                __syncthreads();

                if (tid < dim) {
                    vec[tid] = (nearest_node[tid] - config[tid]) / (float) n_extensions;
                }
                __syncthreads();

                int i_extensions = 0;
                int extension_parent_idx = index;
                while (i_extensions < n_extensions) {
                    for (int i = 0; i < dim; i++) {
                        interp_cfg[i] = config[i] + (int(tid/4 + 1) * (vec[i] / (float) d_settings.granularity));
                    }
                    __syncthreads();

                    detailed_FK = 0;
                    for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                        link_CC[r]=0;
                    }
                    __syncthreads();
                    ppln::collision::fk_approx_runtime(model, interp_cfg, sphere_pos_approx, T, tid);
                    __syncthreads();
                    config_in_collision2_approx = !ppln::collision::env_collision_check_approx_runtime(model, sphere_pos_approx, link_CC, env, tid);
                    atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2_approx ? 1u : 0u);
                    __syncthreads();

                    if (local_cc_result[0] == 1) {
                        if (tid == 0) local_cc_result[0] = 0;
                        __syncthreads();
                        ppln::collision::fk_runtime(model, interp_cfg, sphere_pos, T, tid);
                        detailed_FK = 1;
                        __syncthreads();
                        bool config_in_collision2 = !ppln::collision::env_collision_check_runtime(model, sphere_pos, link_CC, env, tid);
                        atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2 ? 1u : 0u);
                        __syncthreads();
                    }

                    for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                        link_CC[r]=0;
                    }
                    __syncthreads();
                    if (local_cc_result[0] == 0) {
                        bool config_in_collision_approx = !ppln::collision::self_collision_check_approx_runtime(model, sphere_pos_approx, link_CC, tid);
                        atomicOr((unsigned int *)&local_cc_result[0], config_in_collision_approx ? 1u : 0u);
                        __syncthreads();
                        if (local_cc_result[0] == 1) {
                            if (tid == 0) local_cc_result[0] = 0;
                            __syncthreads();
                            if (detailed_FK == 0) {
                                ppln::collision::fk_runtime(model, interp_cfg, sphere_pos, T, tid);
                                detailed_FK = 1;
                                __syncthreads();
                            }
                            bool config_in_collision = !ppln::collision::self_collision_check_runtime(model, sphere_pos, link_CC, tid);
                            atomicOr((unsigned int *)&local_cc_result[0], config_in_collision ? 1u : 0u);
                            __syncthreads();
                        }
                    }

                    bool ext_edge_good = local_cc_result[0] == 0;
                    if (!ext_edge_good) break;
                    if (tid == 0) {
                        index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                        if (index >= d_settings.max_samples) solved = -1;
                        t_parents[index] = extension_parent_idx;
                        radii[t_tree_id][index] = FLT_MAX;
                        extension_parent_idx = index;
                        local_cc_result[0] = 0;
                    }
                    __syncthreads();
                    if (tid < dim) {
                        config[tid] = config[tid] + vec[tid];
                        t_nodes[index * dim + tid] = config[tid];
                    }
                    if (tid == 0) {
                        atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                        __threadfence();
                    }
                    __syncthreads();
                    i_extensions++;
                    __syncthreads();
                }
                if (i_extensions == n_extensions) {
                    if (tid == 0 && atomicCAS((int *)&solved, 0, 1) == 0) {
                        int current = index;
                        int parent;
                        int t_path_size = 0;
                        int o_path_size = 0;
                        while (t_parents[current] != current) {
                            parent = t_parents[current];
                            cost += device_utils::l2_dist((float *)&t_nodes[current * dim], (float *)&t_nodes[parent * dim], dim);
                            if ((t_path_size + 1) * dim > MAX_PATH_SIZE) { solved = -1; break; }
                            for (int i = 0; i < dim; i++) path[t_tree_id][t_path_size * dim + i] = t_nodes[current * dim + i];
                            t_path_size++;
                            current = parent;
                        }
                        if (t_tree_id == 1) reached_goal_idx = current;
                        current = sindex[0];
                        while(o_parents[current] != current) {
                            parent = o_parents[current];
                            cost += device_utils::l2_dist((float *)&o_nodes[current * dim], (float *)&o_nodes[parent * dim], dim);
                            if ((o_path_size + 1) * dim > MAX_PATH_SIZE) { solved = -1; break; }
                            for (int i = 0; i < dim; i++) path[o_tree_id][o_path_size * dim + i] = o_nodes[current * dim + i];
                            o_path_size++;
                            current = parent;
                        }
                        if (t_tree_id == 0) reached_goal_idx = current;
                        path_size[t_tree_id] = t_path_size;
                        path_size[o_tree_id] = o_path_size;
                        solved_iters = iter;
                    }
                    __syncthreads();
                }
            }
            else if (d_settings.dynamic_domain && tid == 0) {
                volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                float old_radius, new_radius;
                int expected, desired;
                do {
                    old_radius = *radius_ptr;
                    if (old_radius == FLT_MAX) {
                        new_radius = d_settings.dd_radius;
                    } else {
                        new_radius = fmaxf(old_radius * (1.f - d_settings.dd_alpha), d_settings.dd_min_radius);
                    }
                    expected = __float_as_int(old_radius);
                    desired = __float_as_int(new_radius);
                } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
            }
            __syncthreads();
            if (solved != 0) return;
        }
    }

    // Runtime planner result uses RuntimePlannerResult from Planners.hh

    RuntimePlannerResult solve_runtime(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        ppln::collision::Environment<float>& h_environment,
        pRRTC_settings& settings,
        ppln::RobotModel& model
    )
    {
        auto start_time = std::chrono::steady_clock::now();
        const int dim = model.n_dof;
        std::size_t start_index = 0;
        RuntimePlannerResult res;

        // Auto-fill GPU clock rate for kernel-side time limit
        if (settings.gpu_clock_rate_khz == 0 && settings.time_limit_ms > 0.0f) {
            int clock_rate_khz = 0;
            if (cudaDeviceGetAttribute(&clock_rate_khz,
                                       cudaDevAttrClockRate, 0) == cudaSuccess) {
                settings.gpu_clock_rate_khz = clock_rate_khz;
            }
        }

        cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings));
        int num_goals = static_cast<int>(goals.size());
        float *nodes[2];
        int *parents[2];
        float *radii_arr[2];
        float **d_nodes;
        int **d_parents;
        float **d_radii;
        cudaMalloc(&d_nodes, 2 * sizeof(float*));
        cudaMalloc(&d_parents, 2 * sizeof(int*));
        cudaMalloc(&d_radii, 2 * sizeof(float*));
        const std::size_t config_size = dim * sizeof(float);

        for (int i = 0; i < 2; i++) {
            cudaMalloc(&nodes[i], settings.max_samples * config_size);
            cudaMalloc(&parents[i], settings.max_samples * sizeof(int));
            cudaMalloc(&radii_arr[i], settings.max_samples * sizeof(float));
        }
        cudaMemcpy(d_nodes, nodes, 2 * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_parents, parents, 2 * sizeof(int*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_radii, radii_arr, 2 * sizeof(float*), cudaMemcpyHostToDevice);

        std::vector<float> nodes_init(settings.max_samples * dim, UNWRITTEN_VAL);
        cudaMemcpy((void *)nodes[0], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)nodes[1], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);

        std::vector<float> radii_init(num_goals, FLT_MAX);
        cudaMemcpy((void *)radii_arr[0], radii_init.data(), sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)radii_arr[1], radii_init.data(), sizeof(float) * num_goals, cudaMemcpyHostToDevice);

        curandState *rng_states;
        int num_rng_states = settings.num_new_configs * dim;
        cudaMalloc(&rng_states, num_rng_states * sizeof(curandState));
        int numBlocks = (num_rng_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_rng<<<numBlocks, BLOCK_SIZE>>>(rng_states, 1, num_rng_states);

        HaltonState_runtime *halton_states;
        cudaMalloc(&halton_states, settings.num_new_configs * sizeof(HaltonState_runtime));
        int numBlocks1 = (settings.num_new_configs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_halton_runtime<<<numBlocks1, BLOCK_SIZE>>>(halton_states, rng_states, dim);

        int h_free_index[2] = {1, num_goals};
        cudaMemcpyToSymbol(atomic_free_index, &h_free_index, sizeof(int) * 2);
        cudaMemcpyToSymbol(nodes_size, &h_free_index, sizeof(int) * 2);
        int h_completed_nodes[2] = {1, num_goals};
        cudaMemcpyToSymbol(completed_nodes, &h_completed_nodes, sizeof(int) * 2);

        ppln::collision::Environment<float> *env;
        setup_environment_on_device(env, h_environment);
        cudaCheckError(cudaGetLastError());

        int *h_solved;
        int current_samples[2];
        int h_solved_iters = -1;
        cudaMallocHost(&h_solved, sizeof(int));
        *h_solved = -1;

        auto copy_start_time = std::chrono::steady_clock::now();
        cudaMemcpy((void *)nodes[0], start.data(), config_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)parents[0], &start_index, sizeof(int), cudaMemcpyHostToDevice);

        // Flatten goals into contiguous buffer
        std::vector<float> goals_flat(num_goals * dim);
        for (int i = 0; i < num_goals; i++) {
            std::copy_n(goals[i].data(), dim, goals_flat.data() + i * dim);
        }
        cudaMemcpy((void *)nodes[1], goals_flat.data(), config_size * num_goals, cudaMemcpyHostToDevice);
        std::vector<int> parents_b_init(num_goals);
        std::iota(parents_b_init.begin(), parents_b_init.end(), 0);
        cudaMemcpy((void *)parents[1], parents_b_init.data(), sizeof(int) * num_goals, cudaMemcpyHostToDevice);
        res.copy_ns = get_elapsed_nanoseconds(copy_start_time);

        auto kernel_start_time = std::chrono::steady_clock::now();
        rrtc_runtime<<<settings.num_new_configs, 4*settings.granularity>>>(
            d_nodes, d_parents, d_radii,
            halton_states, rng_states, env,
            model, dim
        );
        cudaDeviceSynchronize();
        res.kernel_ns = get_elapsed_nanoseconds(kernel_start_time);

        copy_start_time = std::chrono::steady_clock::now();
        cudaMemcpyFromSymbol(current_samples, atomic_free_index, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(h_solved, solved, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&h_solved_iters, solved_iters, sizeof(int), 0, cudaMemcpyDeviceToHost);
        res.copy_ns += get_elapsed_nanoseconds(copy_start_time);

        cudaCheckError(cudaGetLastError());

        if (*h_solved != 1) *h_solved = 0;
        res.start_tree_size = current_samples[0];
        res.goal_tree_size = current_samples[1];
        if (*h_solved) {
            int h_path_size[2];
            float h_paths[2][MAX_PATH_SIZE];
            float h_cost;
            int h_reached_goal_idx;
            cudaMemcpyFromSymbol(h_path_size, path_size, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(h_paths, path, sizeof(float) * 2 * MAX_PATH_SIZE, 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&h_cost, cost, sizeof(float), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&h_reached_goal_idx, reached_goal_idx, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaCheckError(cudaGetLastError());

            // goal
            res.path.push_back(goals[h_reached_goal_idx]);
            std::vector<float> cfg(dim);
            for (int i = h_path_size[1] - 1; i >= 0; i--) {
                std::copy_n(h_paths[1] + i * dim, dim, cfg.data());
                res.path.push_back(cfg);
            }
            for (int i = 0; i < h_path_size[0]; i++) {
                std::copy_n(h_paths[0] + i * dim, dim, cfg.data());
                res.path.push_back(cfg);
            }
            res.path.push_back(start);
            res.cost = h_cost;
            res.path_length = (h_path_size[0] + h_path_size[1]);
        }
        res.solved = (*h_solved) != 0;
        res.iters = h_solved_iters;

        cleanup_environment_on_device(env, h_environment);
        reset_device_variables();
        cudaFree((void *)nodes[0]);
        cudaFree((void *)nodes[1]);
        cudaFree((void *)parents[0]);
        cudaFree((void *)parents[1]);
        cudaFree((void *)radii_arr[0]);
        cudaFree((void *)radii_arr[1]);
        cudaFree(rng_states);
        cudaFree(halton_states);
        cudaFree(d_nodes);
        cudaFree(d_parents);
        cudaFree(d_radii);
        cudaFreeHost(h_solved);
        cudaCheckError(cudaGetLastError());
        res.wall_ns = get_elapsed_nanoseconds(start_time);
        return res;
    }

    // =========================================================================
    // Scene-based runtime kernel — reads the collision tensors directly
    // =========================================================================

    __global__ void
    rrtc_runtime_scene(
        float **nodes,
        int **parents,
        float **radii,
        HaltonState_runtime *halton_states,
        curandState *rng_states,
        ppln::collision::SceneCollisionData scene,
        ppln::RobotModel model,
        int dim,
        float *mesh_transforms    // [num_blocks * granularity * n_joints * 16], or nullptr
    )
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        __shared__ int t_tree_id;
        __shared__ int o_tree_id;
        __shared__ float config[ppln::MAX_DIM];
        __shared__ float sdata[MAX_THREADS_PER_BLOCK];
        __shared__ int sindex[MAX_THREADS_PER_BLOCK];
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ float *t_nodes;
        __shared__ float *o_nodes;
        __shared__ int *t_parents;
        __shared__ int *o_parents;
        __shared__ float scale;
        __shared__ float *nearest_node;
        __shared__ float delta[ppln::MAX_DIM];
        __shared__ int index;
        __shared__ float vec[ppln::MAX_DIM];
        __shared__ unsigned int n_extensions;
        __shared__ bool should_skip;
        __align__(16) __shared__ volatile float sphere_pos[6000];
        __align__(16) __shared__ volatile float sphere_pos_approx[2500];
        __align__(16) __shared__ volatile int link_CC[640];
        __align__(16) __shared__ float T[16 * 2 * 16];
        __shared__ ppln::collision::TwoPhaseFlags cc_flags;

        int iter = 0;
        const long long clock_start = clock64();
        const long long time_limit_clocks = (d_settings.time_limit_ms > 0.0f && d_settings.gpu_clock_rate_khz > 0)
            ? (long long)(d_settings.time_limit_ms * d_settings.gpu_clock_rate_khz) : 0;

        while (true) {
            if (tid == 0) {
                iter++;
                if (iter > d_settings.max_iters) {
                    atomicCAS((int *)&solved, 0, -1);
                }
                if (time_limit_clocks > 0 && (clock64() - clock_start) > time_limit_clocks) {
                    atomicCAS((int *)&solved, 0, -1);
                }

                if (d_settings.balance == 0 || iter == 1) {
                    t_tree_id = (bid < (d_settings.num_new_configs / 2))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1 && abs(atomic_free_index[0]-atomic_free_index[1]) < 1.5 * d_settings.num_new_configs) {
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0]+atomic_free_index[1]);
                    float balance_factor = 1 - ratio;
                    t_tree_id = (bid < (d_settings.num_new_configs * balance_factor))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1) {
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0] + atomic_free_index[1]);
                    if (ratio < d_settings.tree_ratio) t_tree_id = 0;
                    else t_tree_id = 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 2) {
                    float ratio = abs(atomic_free_index[t_tree_id] - atomic_free_index[o_tree_id]) / (float) atomic_free_index[t_tree_id];
                    if (ratio < d_settings.tree_ratio) {
                        t_tree_id = 1 - t_tree_id;
                        o_tree_id = 1 - t_tree_id;
                    }
                }

                t_nodes = nodes[t_tree_id];
                o_nodes = nodes[o_tree_id];
                t_parents = parents[t_tree_id];
                o_parents = parents[o_tree_id];

                halton_next_runtime(halton_states[bid], (float *)config, dim);
                ppln::collision::scale_cfg_runtime(model, (float *)config);
                local_cc_result[0] = 0;
            }

            for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                link_CC[r]=0;
            }
            __syncthreads();

            // parallelized nearest neighbor search
            float local_min_dist = FLT_MAX;
            int local_near_idx = 0;
            float dist;
            int size = min(atomic_free_index[t_tree_id], completed_nodes[t_tree_id]);
            for (int i = tid; i < size; i += blockDim.x) {
                dist = device_utils::sq_l2_dist((float *)&t_nodes[i * dim], (float *) config, dim);
                if (dist < local_min_dist) {
                    local_min_dist = dist;
                    local_near_idx = i;
                }
            }
            sdata[tid] = local_min_dist;
            sindex[tid] = local_near_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                float sdata_tid = sdata[tid];
                float sdata_tid_s = sdata[tid + s];
                __syncthreads();
                if (tid < s){
                    if (sdata_tid_s < sdata_tid) {
                        sdata[tid] = sdata[tid + s];
                        sindex[tid] = sindex[tid + s];
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                sdata[0] = sqrt(sdata[0]);
                scale = min(1.0f, d_settings.range / (sdata[0]));
                nearest_node = &t_nodes[sindex[0] * dim];
                should_skip = (d_settings.dynamic_domain && radii[t_tree_id][sindex[0]] < sdata[0]);
            }
            __syncthreads();

            if (should_skip) continue;
            __syncthreads();

            if (tid < dim) {
                config[tid] = nearest_node[tid] + ((config[tid] - nearest_node[tid]) * scale);
                delta[tid] = (config[tid] - nearest_node[tid]) / (float) d_settings.granularity;
            }
            __syncthreads();

            // validate edge
            float interp_cfg[ppln::MAX_DIM];
            for (int i = 0; i < dim; i++) {
                interp_cfg[i] = nearest_node[i] + (int(tid/4 + 1) * delta[i]);
            }

            int detailed_FK = 0;

            // Mesh collision mode uses per-block global memory for joint transforms
            const bool mesh_mode = d_settings.enable_mesh_collision && mesh_transforms != nullptr
                                   && model.mesh.n_total_bvh_nodes > 0;
            const int granularity = d_settings.granularity;
            // Per-block offset into mesh_transforms buffer
            float *block_mesh_xforms = mesh_mode
                ? &mesh_transforms[bid * granularity * model.n_joints * 16]
                : nullptr;

            if (!mesh_mode) {
                // The shared anchor (src/collision/two_phase_cc.cuh) -- the same routine
                // MIT*, MHA*/wPA*SE edge evaluation, and the optimizers' validity check
                // call. The sphere-vs-OBB verdict behind every number we report is one
                // implementation, not several that happen to agree.
                ppln::collision::TwoPhaseResult cc = ppln::collision::two_phase_cc(
                    model, scene, interp_cfg, sphere_pos, sphere_pos_approx, T,
                    link_CC, &cc_flags, tid, d_settings.collision_margin,
                    /*check_self=*/true);
                detailed_FK = cc.did_full_fk ? 1 : 0;
                if (tid == 0 && cc.collision)
                    atomicOr((unsigned int *)&local_cc_result[0], 1u);
                __syncthreads();
            } else {
                ppln::collision::fk_approx_runtime(model, interp_cfg, sphere_pos_approx, T, tid);
                __syncthreads();
                bool config_in_collision2_approx = !ppln::collision::scene_collision_check_approx_runtime(model, sphere_pos_approx, link_CC, scene, tid, d_settings.collision_margin);
                atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2_approx ? 1u : 0u);
                __syncthreads();

                if (local_cc_result[0] == 1) {
                    if (tid == 0) local_cc_result[0] = 0;
                    __syncthreads();
                    if (mesh_mode) {
                        // Mesh precision: FK for joint transforms, then BVH collision
                        ppln::collision::fk_mesh_transforms_runtime(model, interp_cfg, T, block_mesh_xforms, tid);
                        __syncthreads();
                        // Thread 0 of each 4-thread group checks mesh for its config
                        const int thread_ind = tid % 4;
                        const int batch_ind = tid / 4;
                        if (thread_ind == 0) {
                            float *my_xforms = block_mesh_xforms + batch_ind * model.n_joints * 16;
                            volatile int *my_link_CC = &link_CC[20 * batch_ind];
                            bool mesh_hit = ppln::collision::mesh_env_collision_check(
                                model.mesh, my_xforms, my_link_CC, scene, model.n_joints);
                            if (mesh_hit)
                                atomicOr((unsigned int *)&local_cc_result[0], 1u);
                        }
                        __syncthreads();
                    } else {
                        ppln::collision::fk_runtime(model, interp_cfg, sphere_pos, T, tid);
                        detailed_FK = 1;
                        __syncthreads();
                        bool config_in_collision2 = !ppln::collision::scene_collision_check_runtime(model, sphere_pos, link_CC, scene, tid, d_settings.collision_margin);
                        atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2 ? 1u : 0u);
                        __syncthreads();
                    }
                }

                for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                    link_CC[r]=0;
                }
                __syncthreads();

                if (local_cc_result[0] == 0) {
                    bool config_in_collision_approx = !ppln::collision::self_collision_check_approx_runtime(model, sphere_pos_approx, link_CC, tid);
                    atomicOr((unsigned int *)&local_cc_result[0], config_in_collision_approx ? 1u : 0u);
                    __syncthreads();
                    if (local_cc_result[0] == 1) {
                        if (tid == 0) local_cc_result[0] = 0;
                        __syncthreads();
                        if (detailed_FK == 0) {
                            ppln::collision::fk_runtime(model, interp_cfg, sphere_pos, T, tid);
                            detailed_FK = 1;
                            __syncthreads();
                        }
                        bool config_in_collision = !ppln::collision::self_collision_check_runtime(model, sphere_pos, link_CC, tid);
                        atomicOr((unsigned int *)&local_cc_result[0], config_in_collision ? 1u : 0u);
                        __syncthreads();
                    }
                }
            }
            bool edge_good = local_cc_result[0] == 0;
            __syncthreads();
            if (edge_good) {
                if (tid == 0) {
                    index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                    if (index >= d_settings.max_samples) solved = -1;
                    t_parents[index] = sindex[0];
                    if (d_settings.dynamic_domain) {
                        radii[t_tree_id][index] = FLT_MAX;
                        volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                        float old_radius, new_radius;
                        int expected, desired;
                        do {
                            old_radius = *radius_ptr;
                            if (old_radius == FLT_MAX) break;
                            new_radius = old_radius * (1 + d_settings.dd_alpha);
                            expected = __float_as_int(old_radius);
                            desired = __float_as_int(new_radius);
                        } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
                    }
                }
                __syncthreads();

                if (tid < dim) {
                    t_nodes[index * dim + tid] = config[tid];
                }
                if (tid == 0) {
                    atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                    __threadfence();
                }
                __syncthreads();

                // connect
                local_min_dist = FLT_MAX;
                local_near_idx = 0;
                size = min(atomic_free_index[o_tree_id], completed_nodes[o_tree_id]);
                for (unsigned int i = tid; i < size; i += blockDim.x) {
                    dist = device_utils::sq_l2_dist((float *)&o_nodes[i * dim], (float *)config, dim);
                    if (dist < local_min_dist) {
                        local_min_dist = dist;
                        local_near_idx = i;
                    }
                }
                sdata[tid] = local_min_dist;
                sindex[tid] = local_near_idx;
                __syncthreads();

                for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                    float sdata_tid = sdata[tid];
                    float sdata_tid_s = sdata[tid + s];
                    __syncthreads();
                    if (tid < s) {
                        if (sdata_tid_s < sdata_tid) {
                            sdata[tid] = sdata[tid + s];
                            sindex[tid] = sindex[tid + s];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    sdata[0] = sqrt(sdata[0]);
                    nearest_node = &o_nodes[sindex[0] * dim];
                    n_extensions = ceil(sdata[0] / d_settings.range);
                    local_cc_result[0] = 0;
                }
                __syncthreads();

                if (tid < dim) {
                    vec[tid] = (nearest_node[tid] - config[tid]) / (float) n_extensions;
                }
                __syncthreads();

                int i_extensions = 0;
                int extension_parent_idx = index;
                while (i_extensions < n_extensions) {
                    for (int i = 0; i < dim; i++) {
                        interp_cfg[i] = config[i] + (int(tid/4 + 1) * (vec[i] / (float) d_settings.granularity));
                    }
                    __syncthreads();

                    detailed_FK = 0;
                    for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                        link_CC[r]=0;
                    }
                    __syncthreads();
                    if (!mesh_mode) {
                        // The shared anchor (src/collision/two_phase_cc.cuh) -- the same routine
                        // MIT*, MHA*/wPA*SE edge evaluation, and the optimizers' validity check
                        // call. The sphere-vs-OBB verdict behind every number we report is one
                        // implementation, not several that happen to agree.
                        ppln::collision::TwoPhaseResult cc = ppln::collision::two_phase_cc(
                            model, scene, interp_cfg, sphere_pos, sphere_pos_approx, T,
                            link_CC, &cc_flags, tid, d_settings.collision_margin,
                            /*check_self=*/true);
                        detailed_FK = cc.did_full_fk ? 1 : 0;
                        if (tid == 0 && cc.collision)
                            atomicOr((unsigned int *)&local_cc_result[0], 1u);
                        __syncthreads();
                    } else {
                        ppln::collision::fk_approx_runtime(model, interp_cfg, sphere_pos_approx, T, tid);
                        __syncthreads();
                        bool config_in_collision2_approx = !ppln::collision::scene_collision_check_approx_runtime(model, sphere_pos_approx, link_CC, scene, tid, d_settings.collision_margin);
                        atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2_approx ? 1u : 0u);
                        __syncthreads();

                        if (local_cc_result[0] == 1) {
                            if (tid == 0) local_cc_result[0] = 0;
                            __syncthreads();
                            if (mesh_mode) {
                                ppln::collision::fk_mesh_transforms_runtime(model, interp_cfg, T, block_mesh_xforms, tid);
                                __syncthreads();
                                const int thread_ind = tid % 4;
                                const int batch_ind = tid / 4;
                                if (thread_ind == 0) {
                                    float *my_xforms = block_mesh_xforms + batch_ind * model.n_joints * 16;
                                    volatile int *my_link_CC = &link_CC[20 * batch_ind];
                                    bool mesh_hit = ppln::collision::mesh_env_collision_check(
                                        model.mesh, my_xforms, my_link_CC, scene, model.n_joints);
                                    if (mesh_hit)
                                        atomicOr((unsigned int *)&local_cc_result[0], 1u);
                                }
                                __syncthreads();
                            } else {
                                ppln::collision::fk_runtime(model, interp_cfg, sphere_pos, T, tid);
                                detailed_FK = 1;
                                __syncthreads();
                                bool config_in_collision2 = !ppln::collision::scene_collision_check_runtime(model, sphere_pos, link_CC, scene, tid, d_settings.collision_margin);
                                atomicOr((unsigned int *)&local_cc_result[0], config_in_collision2 ? 1u : 0u);
                                __syncthreads();
                            }
                        }

                        for (int r=(tid/4)*20+5*(tid%4); r<(tid/4)*20+5*(tid%4)+5; r++){
                            link_CC[r]=0;
                        }
                        __syncthreads();
                        if (local_cc_result[0] == 0) {
                            bool config_in_collision_approx = !ppln::collision::self_collision_check_approx_runtime(model, sphere_pos_approx, link_CC, tid);
                            atomicOr((unsigned int *)&local_cc_result[0], config_in_collision_approx ? 1u : 0u);
                            __syncthreads();
                            if (local_cc_result[0] == 1) {
                                if (tid == 0) local_cc_result[0] = 0;
                                __syncthreads();
                                if (detailed_FK == 0) {
                                    ppln::collision::fk_runtime(model, interp_cfg, sphere_pos, T, tid);
                                    detailed_FK = 1;
                                    __syncthreads();
                                }
                                bool config_in_collision = !ppln::collision::self_collision_check_runtime(model, sphere_pos, link_CC, tid);
                                atomicOr((unsigned int *)&local_cc_result[0], config_in_collision ? 1u : 0u);
                                __syncthreads();
                            }
                        }
                    }
                    bool ext_edge_good = local_cc_result[0] == 0;
                    if (!ext_edge_good) break;
                    if (tid == 0) {
                        index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                        if (index >= d_settings.max_samples) solved = -1;
                        t_parents[index] = extension_parent_idx;
                        radii[t_tree_id][index] = FLT_MAX;
                        extension_parent_idx = index;
                        local_cc_result[0] = 0;
                    }
                    __syncthreads();
                    if (tid < dim) {
                        config[tid] = config[tid] + vec[tid];
                        t_nodes[index * dim + tid] = config[tid];
                    }
                    if (tid == 0) {
                        atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                        __threadfence();
                    }
                    __syncthreads();
                    i_extensions++;
                    __syncthreads();
                }
                if (i_extensions == n_extensions) {
                    if (tid == 0 && atomicCAS((int *)&solved, 0, 1) == 0) {
                        int current = index;
                        int parent;
                        int t_path_size = 0;
                        int o_path_size = 0;
                        while (t_parents[current] != current) {
                            parent = t_parents[current];
                            cost += device_utils::l2_dist((float *)&t_nodes[current * dim], (float *)&t_nodes[parent * dim], dim);
                            if ((t_path_size + 1) * dim > MAX_PATH_SIZE) { solved = -1; break; }
                            for (int i = 0; i < dim; i++) path[t_tree_id][t_path_size * dim + i] = t_nodes[current * dim + i];
                            t_path_size++;
                            current = parent;
                        }
                        if (t_tree_id == 1) reached_goal_idx = current;
                        current = sindex[0];
                        while(o_parents[current] != current) {
                            parent = o_parents[current];
                            cost += device_utils::l2_dist((float *)&o_nodes[current * dim], (float *)&o_nodes[parent * dim], dim);
                            if ((o_path_size + 1) * dim > MAX_PATH_SIZE) { solved = -1; break; }
                            for (int i = 0; i < dim; i++) path[o_tree_id][o_path_size * dim + i] = o_nodes[current * dim + i];
                            o_path_size++;
                            current = parent;
                        }
                        if (t_tree_id == 0) reached_goal_idx = current;
                        path_size[t_tree_id] = t_path_size;
                        path_size[o_tree_id] = o_path_size;
                        solved_iters = iter;
                    }
                    __syncthreads();
                }
            }
            else if (d_settings.dynamic_domain && tid == 0) {
                volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                float old_radius, new_radius;
                int expected, desired;
                do {
                    old_radius = *radius_ptr;
                    if (old_radius == FLT_MAX) {
                        new_radius = d_settings.dd_radius;
                    } else {
                        new_radius = fmaxf(old_radius * (1.f - d_settings.dd_alpha), d_settings.dd_min_radius);
                    }
                    expected = __float_as_int(old_radius);
                    desired = __float_as_int(new_radius);
                } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
            }
            __syncthreads();
            if (solved != 0) return;
        }
    }

    // =========================================================================
    // solve_runtime_scene — no setup_environment_on_device, uses SceneCollisionData
    // =========================================================================

    RuntimePlannerResult solve_runtime_scene(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        ppln::collision::SceneCollisionData& scene,
        pRRTC_settings& settings,
        ppln::RobotModel& model,
        SolverBuffers* bufs
    )
    {
        auto start_time = std::chrono::steady_clock::now();
        const int dim = model.n_dof;
        std::size_t start_index = 0;
        RuntimePlannerResult res;
        const bool use_bufs = (bufs != nullptr && bufs->owns_memory);

        // Auto-fill GPU clock rate for kernel-side time limit
        if (settings.gpu_clock_rate_khz == 0 && settings.time_limit_ms > 0.0f) {
            int clock_rate_khz = 0;
            if (cudaDeviceGetAttribute(&clock_rate_khz,
                                       cudaDevAttrClockRate, 0) == cudaSuccess) {
                settings.gpu_clock_rate_khz = clock_rate_khz;
            }
        }

        cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings));
        int num_goals = static_cast<int>(goals.size());
        float *nodes[2];
        int *parents[2];
        float *radii_arr[2];
        float **d_nodes;
        int **d_parents;
        float **d_radii;
        const std::size_t config_size = dim * sizeof(float);

        if (use_bufs) {
            // Use pre-allocated buffers — no cudaMalloc
            nodes[0] = bufs->d_nodes_ptrs[0];
            nodes[1] = bufs->d_nodes_ptrs[1];
            parents[0] = bufs->d_parents_ptrs[0];
            parents[1] = bufs->d_parents_ptrs[1];
            radii_arr[0] = bufs->d_radii_ptrs[0];
            radii_arr[1] = bufs->d_radii_ptrs[1];
            d_nodes = bufs->d_nodes;
            d_parents = bufs->d_parents;
            d_radii = bufs->d_radii;
        } else {
            // Original per-call allocation
            cudaMalloc(&d_nodes, 2 * sizeof(float*));
            cudaMalloc(&d_parents, 2 * sizeof(int*));
            cudaMalloc(&d_radii, 2 * sizeof(float*));
            for (int i = 0; i < 2; i++) {
                cudaMalloc(&nodes[i], settings.max_samples * config_size);
                cudaMalloc(&parents[i], settings.max_samples * sizeof(int));
                cudaMalloc(&radii_arr[i], settings.max_samples * sizeof(float));
            }
            cudaMemcpy(d_nodes, nodes, 2 * sizeof(float*), cudaMemcpyHostToDevice);
            cudaMemcpy(d_parents, parents, 2 * sizeof(int*), cudaMemcpyHostToDevice);
            cudaMemcpy(d_radii, radii_arr, 2 * sizeof(float*), cudaMemcpyHostToDevice);
        }

        std::vector<float> nodes_init(settings.max_samples * dim, UNWRITTEN_VAL);
        cudaMemcpy((void *)nodes[0], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)nodes[1], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);

        std::vector<float> radii_init(num_goals, FLT_MAX);
        cudaMemcpy((void *)radii_arr[0], radii_init.data(), sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)radii_arr[1], radii_init.data(), sizeof(float) * num_goals, cudaMemcpyHostToDevice);

        curandState *rng_states;
        HaltonState_runtime *halton_states;
        int num_rng_states = settings.num_new_configs * dim;
        if (use_bufs) {
            rng_states = bufs->rng_states;
            halton_states = bufs->halton_states;
        } else {
            cudaMalloc(&rng_states, num_rng_states * sizeof(curandState));
            cudaMalloc(&halton_states, settings.num_new_configs * sizeof(HaltonState_runtime));
        }
        // RNG must be re-initialized every solve (seed-dependent)
        int numBlocks = (num_rng_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_rng<<<numBlocks, BLOCK_SIZE>>>(rng_states, 1, num_rng_states);
        int numBlocks1 = (settings.num_new_configs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_halton_runtime<<<numBlocks1, BLOCK_SIZE>>>(halton_states, rng_states, dim);

        int h_free_index[2] = {1, num_goals};
        cudaMemcpyToSymbol(atomic_free_index, &h_free_index, sizeof(int) * 2);
        cudaMemcpyToSymbol(nodes_size, &h_free_index, sizeof(int) * 2);
        int h_completed_nodes[2] = {1, num_goals};
        cudaMemcpyToSymbol(completed_nodes, &h_completed_nodes, sizeof(int) * 2);

        // No setup_environment_on_device — scene pointers are already on GPU
        cudaCheckError(cudaGetLastError());

        // Mesh joint transforms buffer
        float *d_mesh_transforms = nullptr;
        if (use_bufs) {
            d_mesh_transforms = bufs->d_mesh_transforms;  // may be nullptr if mesh disabled
        } else if (settings.enable_mesh_collision && model.mesh.n_total_bvh_nodes > 0) {
            size_t mesh_buf_size = (size_t)settings.num_new_configs * settings.granularity
                                   * model.n_joints * 16 * sizeof(float);
            cudaMalloc(&d_mesh_transforms, mesh_buf_size);
        }

        int *h_solved;
        int current_samples[2];
        int h_solved_iters = -1;
        if (use_bufs) {
            h_solved = bufs->h_solved;
        } else {
            cudaMallocHost(&h_solved, sizeof(int));
        }
        *h_solved = -1;

        auto copy_start_time = std::chrono::steady_clock::now();
        cudaMemcpy((void *)nodes[0], start.data(), config_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)parents[0], &start_index, sizeof(int), cudaMemcpyHostToDevice);

        // Flatten goals into contiguous buffer
        std::vector<float> goals_flat(num_goals * dim);
        for (int i = 0; i < num_goals; i++) {
            std::copy_n(goals[i].data(), dim, goals_flat.data() + i * dim);
        }
        cudaMemcpy((void *)nodes[1], goals_flat.data(), config_size * num_goals, cudaMemcpyHostToDevice);
        std::vector<int> parents_b_init(num_goals);
        std::iota(parents_b_init.begin(), parents_b_init.end(), 0);
        cudaMemcpy((void *)parents[1], parents_b_init.data(), sizeof(int) * num_goals, cudaMemcpyHostToDevice);
        res.copy_ns = get_elapsed_nanoseconds(copy_start_time);

        auto kernel_start_time = std::chrono::steady_clock::now();
        rrtc_runtime_scene<<<settings.num_new_configs, 4*settings.granularity>>>(
            d_nodes, d_parents, d_radii,
            halton_states, rng_states, scene,
            model, dim, d_mesh_transforms
        );
        cudaDeviceSynchronize();
        res.kernel_ns = get_elapsed_nanoseconds(kernel_start_time);

        copy_start_time = std::chrono::steady_clock::now();
        cudaMemcpyFromSymbol(current_samples, atomic_free_index, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(h_solved, solved, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&h_solved_iters, solved_iters, sizeof(int), 0, cudaMemcpyDeviceToHost);
        res.copy_ns += get_elapsed_nanoseconds(copy_start_time);

        cudaCheckError(cudaGetLastError());

        if (*h_solved != 1) *h_solved = 0;
        res.start_tree_size = current_samples[0];
        res.goal_tree_size = current_samples[1];
        if (*h_solved) {
            int h_path_size[2];
            float h_paths[2][MAX_PATH_SIZE];
            float h_cost;
            int h_reached_goal_idx;
            cudaMemcpyFromSymbol(h_path_size, path_size, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(h_paths, path, sizeof(float) * 2 * MAX_PATH_SIZE, 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&h_cost, cost, sizeof(float), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&h_reached_goal_idx, reached_goal_idx, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaCheckError(cudaGetLastError());

            // goal
            res.path.push_back(goals[h_reached_goal_idx]);
            std::vector<float> cfg(dim);
            for (int i = h_path_size[1] - 1; i >= 0; i--) {
                std::copy_n(h_paths[1] + i * dim, dim, cfg.data());
                res.path.push_back(cfg);
            }
            for (int i = 0; i < h_path_size[0]; i++) {
                std::copy_n(h_paths[0] + i * dim, dim, cfg.data());
                res.path.push_back(cfg);
            }
            res.path.push_back(start);
            res.cost = h_cost;
            res.path_length = (h_path_size[0] + h_path_size[1]);
        }
        res.solved = (*h_solved) != 0;
        res.iters = h_solved_iters;

        // Path shortcutting (post-processing)
        if (settings.shortcut_path && res.solved && res.path.size() > 2) {
            res.path = ppln::shortcut::shortcut_path(
                res.path, model, scene,
                settings.valid_segment_length,
                settings.collision_margin + settings.shortcut_collision_margin,
                dim);
        }

        // No cleanup_environment_on_device — scene is externally owned
        reset_device_variables();
        if (!use_bufs) {
            cudaFree(d_mesh_transforms);
            cudaFree((void *)nodes[0]);
            cudaFree((void *)nodes[1]);
            cudaFree((void *)parents[0]);
            cudaFree((void *)parents[1]);
            cudaFree((void *)radii_arr[0]);
            cudaFree((void *)radii_arr[1]);
            cudaFree(rng_states);
            cudaFree(halton_states);
            cudaFree(d_nodes);
            cudaFree(d_parents);
            cudaFree(d_radii);
            cudaFreeHost(h_solved);
        }
        cudaCheckError(cudaGetLastError());
        res.wall_ns = get_elapsed_nanoseconds(start_time);
        return res;
    }

    // =========================================================================
    // check_collision_mesh — standalone batch mesh collision check kernel
    // =========================================================================
    // One thread per configuration. Performs:
    //   1. Approx FK + approx env CC (sphere)
    //   2. If approx hit: full FK transforms + BVH mesh env CC
    // Returns per-config collision status.

    __global__ void check_collision_mesh_kernel(
        const float* configs,       // [N * n_dof]
        int N,
        ppln::collision::SceneCollisionData scene,
        ppln::RobotModel model,
        bool* results               // [N] output
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        int n_dof = model.n_dof;
        int n_joints = model.n_joints;
        const float* q = &configs[idx * n_dof];

        // Per-joint world frames via the shared tree FK (runtime_kinematics.cuh).
        float joint_transforms[(ppln::MAX_DIM + 1) * 16];
        ppln::collision::fk_joint_transforms_runtime(model, q, joint_transforms);

        // Approx sphere check (simplified: transform approx spheres, check against scene)
        // NOTE: pass -1 as sphere_idx to disable ESDF ACM lookup — the ACM mask is
        // indexed by full-sphere index, not approx-sphere index (see scene_collision.cuh).
        bool approx_hit = false;
        int link_cc[20] = {};
        for (int si = 0; si < model.n_approx_spheres; si++) {
            float4 sph = model.approx_spheres[si];
            int ji = model.approx_sphere_to_joint[si];
            float* jt = &joint_transforms[ji * 16];
            float wx = jt[0]*sph.x + jt[4]*sph.y + jt[8]*sph.z + jt[12];
            float wy = jt[1]*sph.x + jt[5]*sph.y + jt[9]*sph.z + jt[13];
            float wz = jt[2]*sph.x + jt[6]*sph.y + jt[10]*sph.z + jt[14];
            if (ppln::collision::sphere_scene_in_collision(scene, wx, wy, wz, sph.w,
                    -1, scene.sphere_obb_acm_mask_approx)) {
                approx_hit = true;
                link_cc[ji]++;
            }
        }

        if (!approx_hit) {
            results[idx] = false;
            return;
        }

        // Mesh BVH check for links with approx collision
        if (model.mesh.n_total_bvh_nodes > 0) {
            volatile int link_cc_vol[20];
            for (int i = 0; i < 20; i++) link_cc_vol[i] = link_cc[i];
            bool mesh_hit = ppln::collision::mesh_env_collision_check(
                model.mesh, joint_transforms, link_cc_vol, scene, n_joints);
            results[idx] = mesh_hit;
        } else {
            // Fallback: detailed sphere check
            bool detailed_hit = false;
            for (int si = 0; si < model.n_spheres && !detailed_hit; si++) {
                float4 sph = model.spheres[si];
                int ji = model.sphere_to_joint[si];
                if (link_cc[ji] <= 0) continue;
                float* jt = &joint_transforms[ji * 16];
                float wx = jt[0]*sph.x + jt[4]*sph.y + jt[8]*sph.z + jt[12];
                float wy = jt[1]*sph.x + jt[5]*sph.y + jt[9]*sph.z + jt[13];
                float wz = jt[2]*sph.x + jt[6]*sph.y + jt[10]*sph.z + jt[14];
                if (ppln::collision::sphere_scene_in_collision(scene, wx, wy, wz, sph.w,
                        si, scene.sphere_obb_acm_mask))
                    detailed_hit = true;
            }
            results[idx] = detailed_hit;
        }
    }

    void check_collision_mesh(
        const float* h_configs,
        int N,
        ppln::collision::SceneCollisionData& scene,
        ppln::RobotModel& model,
        bool* h_results
    ) {
        if (N <= 0) return;
        int n_dof = model.n_dof;

        float *d_configs;
        bool *d_results;
        cudaMalloc(&d_configs, (size_t)N * n_dof * sizeof(float));
        cudaMalloc(&d_results, (size_t)N * sizeof(bool));
        cudaMemcpy(d_configs, h_configs, (size_t)N * n_dof * sizeof(float), cudaMemcpyHostToDevice);

        int block_size = 128;
        int num_blocks = (N + block_size - 1) / block_size;
        check_collision_mesh_kernel<<<num_blocks, block_size>>>(
            d_configs, N, scene, model, d_results);
        cudaDeviceSynchronize();

        cudaMemcpy(h_results, d_results, (size_t)N * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(d_configs);
        cudaFree(d_results);
    }

    // =========================================================================
    // check_collision_scene — batch scene collision check.
    //   Runs the same two-phase approximate->detailed check the planners run,
    //   over the same cooperative tree FK (fk_approx_runtime / fk_runtime), so
    //   the boolean verdict here is bit-identical to the one every planner sees
    //   internally. 4 threads per configuration; environment only, no
    //   self-collision (start/goal self-feasibility is the caller's business).
    // =========================================================================

    __global__ void check_collision_scene_kernel(
        const float* configs,       // [N * n_dof]
        int N,
        ppln::collision::SceneCollisionData scene,
        ppln::RobotModel model,
        float collision_margin,
        bool* results               // [N] output
    ) {
        const int idx = blockIdx.x;
        if (idx >= N) return;
        const int tid = threadIdx.x;
        if (tid >= 4) return;

        const int dim = model.n_dof;

        extern __shared__ float smem[];
        float* sphere_pos = smem;
        float* approx_sphere_pos = sphere_pos + model.n_spheres * BATCH_SIZE * 3;
        float* T = approx_sphere_pos + model.n_approx_spheres * BATCH_SIZE * 3;
        volatile int* joint_in_collision =
            (volatile int*)(T + BATCH_SIZE * ppln::FK_T_SLOTS * 16);
        float* q = (float*)(&joint_in_collision[BATCH_SIZE * 20]);

        if (tid == 0) {
            for (int i = 0; i < dim; i++) q[i] = configs[idx * dim + i];
        }
        for (int i = tid; i < 20; i += 4) joint_in_collision[i] = 0;
        __syncthreads();

        // Phase 1: approx spheres enclose the true link geometry, so no approx
        // hit means no collision. A hit only flags which joints to re-test.
        ppln::collision::fk_approx_runtime(model, q, approx_sphere_pos, T, tid);
        __syncthreads();
        bool approx_ok = ppln::collision::scene_collision_check_approx_runtime(
            model, approx_sphere_pos, joint_in_collision, scene, tid, collision_margin);
        __syncthreads();

        if (!__any_sync(0xf, !approx_ok)) {
            if (tid == 0) results[idx] = false;
            return;
        }

        // Phase 2: exact check, gated on the joints phase 1 flagged.
        ppln::collision::fk_runtime(model, q, sphere_pos, T, tid);
        __syncthreads();
        bool env_ok = ppln::collision::scene_collision_check_runtime(
            model, sphere_pos, joint_in_collision, scene, tid, collision_margin);
        bool hit = __any_sync(0xf, !env_ok);
        if (tid == 0) results[idx] = hit;
    }

    void check_collision_scene(
        const float* h_configs,
        int N,
        ppln::collision::SceneCollisionData& scene,
        ppln::RobotModel& model,
        float collision_margin,
        bool* h_results
    ) {
        if (N <= 0) return;
        int n_dof = model.n_dof;

        float *d_configs;
        bool *d_results;
        cudaMalloc(&d_configs, (size_t)N * n_dof * sizeof(float));
        cudaMalloc(&d_results, (size_t)N * sizeof(bool));
        cudaMemcpy(d_configs, h_configs, (size_t)N * n_dof * sizeof(float), cudaMemcpyHostToDevice);

        // One block of 4 cooperating threads per configuration, matching the
        // planners' batch_cc launch so the shared-memory layout lines up.
        int smem = (model.n_spheres * BATCH_SIZE * 3 +
                    model.n_approx_spheres * BATCH_SIZE * 3 +
                    BATCH_SIZE * ppln::FK_T_SLOTS * 16 +
                    BATCH_SIZE * 20 +
                    n_dof) * sizeof(float);
        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(check_collision_scene_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }
        check_collision_scene_kernel<<<N, 4, smem>>>(
            d_configs, N, scene, model, collision_margin, d_results);
        cudaDeviceSynchronize();

        cudaMemcpy(h_results, d_results, (size_t)N * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(d_configs);
        cudaFree(d_results);
    }

}
