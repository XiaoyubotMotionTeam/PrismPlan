// ============================================================================
// chomp.cu — GPU-native CHOMP (Ratliff 2009) gradient trajectory optimizer.
//
// This is the OPTIMIZATION-paradigm, GRADIENT-BASED representative, contrasting
// STOMP (derivative-free / sampled rollouts) on the SAME shared per-sphere
// collision-kinematics substrate. The design deliberately mirrors stomp.cu: the
// host loop launches a fixed sequence of small kernels on the default stream per
// iteration; all trajectory/gradient/cost math stays resident on the GPU. The
// only per-iteration host<->device traffic is (a) a small D*T readback used to
// size the validity-gate densification and (b) the pinned h_valid/h_total_cost
// readback that drives the early-stop state machine.
//
// The obstacle gradient is the NUMERICAL (central finite-difference) derivative
// of the EXACT SAME config cost STOMP evaluates:
//   collision_cost_weight * (scene_sum_esdf_hinge + scene_sum_obb_hinge
//                            + self_min_hinge)  over  fk_runtime.
// It is never a separately-derived ESDF-gradient/Jacobian, so CHOMP's collision
// signal can never diverge from the shared boolean anchor used by the
// sampling/search planners — that bit-identical substrate is the fair-comparison
// property the paper rests on.
//
// The smoothness metric R is the SAME 5-point acceleration control-cost matrix
// STOMP uses (see chomp_precompute / stomp_precompute). The covariant update is
//   theta <- theta - step_size * A^{-1} (collision_cost_weight*obs_grad
//                                        + smoothness_weight * R theta),
// with A^{-1} = (R + 1e-8 I)^{-1}. Because R @ (straight line) ~= 0 between the
// pinned endpoints, the smoothness term pulls the trajectory toward the
// zero-acceleration seed while the obstacle gradient pushes it clear.
// ============================================================================

#include "Planners.hh"
#include "utils.cuh"
#include "robot_model.cuh"
#include "CHOMP_settings.hh"
#include "chomp_buffers.hh"
#include "runtime_kinematics.cuh"
#include "src/collision/scene_collision.cuh"
#include "src/collision/two_phase_cc.cuh"

#include "chomp_precompute.hh"

#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>

namespace CHOMP {
using namespace ppln;
using namespace ppln::collision;

namespace {

// ============================================================================
// Linear-interpolated + clamped initial trajectory (identical to STOMP's seed).
// ============================================================================
__global__ void chomp_init_trajectory_kernel(
    const float* start, const float* goal,
    const RobotModel model,
    float* theta, int D, int Tn)
{
    int d = blockIdx.x;
    int t = threadIdx.x;
    if (d >= D || t >= Tn) return;
    float frac = (Tn > 1) ? (float)t / (float)(Tn - 1) : 0.0f;
    float v = start[d] * (1.0f - frac) + goal[d] * frac;
    v = fmaxf(model.joint_lower[d], fminf(model.joint_upper[d], v));
    theta[d * Tn + t] = v;
}

// ============================================================================
// Obstacle gradient by CENTRAL finite difference of the shared substrate config
// cost. One block == one (timestep t, dim d) partial derivative; 4 cooperating
// threads run the FK+hinge substrate exactly as STOMP's state-cost kernel does.
// Interior timesteps only — endpoints are pinned so their gradient is zero.
//
// For the (t,d) partial we load the full config column q[.] = theta[.,t], then
//   cost_plus  = C(q + eps e_d),   cost_minus = C(q - eps e_d)
//   obs_grad[d,t] = collision_cost_weight * (cost_plus - cost_minus)/(2 eps)
// C(q) is the SAME sum-hinge the sampling/search anchor and STOMP consume, so
// the gradient can only ever point along that shared cost surface.
// ============================================================================
__global__ void chomp_obs_grad_kernel(
    const float* theta,          // [D*Tn]
    float* obs_grad,             // [D*Tn]
    const RobotModel model,
    const SceneCollisionData scene,
    int D, int Tn,
    float collision_cost_weight,
    float world_margin,
    float self_margin,
    float fd_epsilon)
{
    int t = blockIdx.x;
    int dim = blockIdx.y;
    if (t >= Tn || dim >= D) return;
    int tid = threadIdx.x;
    if (tid >= 4) return;

    // Endpoints pinned to start/goal: no gradient.
    if (t == 0 || t == Tn - 1) {
        if (tid == 0) obs_grad[dim * Tn + t] = 0.0f;
        return;
    }

    extern __shared__ float smem[];
    int n_spheres = model.n_spheres;
    float* sphere_pos = smem;
    float* fk_scratch = sphere_pos + n_spheres * BATCH_SIZE * 3;
    float* q = fk_scratch + BATCH_SIZE * ppln::FK_T_SLOTS * 16;

    // q <- theta[:,t] with +eps perturbation on dimension `dim`.
    if (tid == 0) {
        for (int d = 0; d < D; d++) q[d] = theta[d * Tn + t];
        q[dim] += fd_epsilon;
    }
    __syncthreads();

    fk_runtime(model, q, sphere_pos, fk_scratch, tid);
    __syncthreads();

    float cost_plus = scene_sum_esdf_hinge_runtime(model, sphere_pos, scene, tid, world_margin)
                    + scene_sum_obb_hinge_runtime(model, sphere_pos, scene, tid, world_margin)
                    + self_min_hinge_runtime(model, sphere_pos, tid, self_margin);
    __syncthreads();

    // q <- theta[:,t] with -eps perturbation on dimension `dim`.
    if (tid == 0) {
        q[dim] -= 2.0f * fd_epsilon;  // from (+eps) to (-eps)
    }
    __syncthreads();

    fk_runtime(model, q, sphere_pos, fk_scratch, tid);
    __syncthreads();

    float cost_minus = scene_sum_esdf_hinge_runtime(model, sphere_pos, scene, tid, world_margin)
                     + scene_sum_obb_hinge_runtime(model, sphere_pos, scene, tid, world_margin)
                     + self_min_hinge_runtime(model, sphere_pos, tid, self_margin);

    if (tid == 0) {
        obs_grad[dim * Tn + t] =
            collision_cost_weight * (cost_plus - cost_minus) / (2.0f * fd_epsilon);
    }
}

// ============================================================================
// Per-(d,t) obstacle (state) cost, broadcast over D — SAME form as STOMP's
// state-cost kernel, providing a comparable cost signal. One block == one
// timestep; 4 cooperating threads.
// ============================================================================
__global__ void chomp_state_cost_kernel(
    const float* theta,          // [D*Tn]
    float* state_costs,          // [D*Tn] output, broadcast over D
    const RobotModel model,
    const SceneCollisionData scene,
    int D, int Tn,
    float collision_cost_weight,
    float world_margin,
    float self_margin)
{
    int t = blockIdx.x;
    if (t >= Tn) return;
    int tid = threadIdx.x;
    if (tid >= 4) return;

    extern __shared__ float smem[];
    int n_spheres = model.n_spheres;
    float* sphere_pos = smem;
    float* fk_scratch = sphere_pos + n_spheres * BATCH_SIZE * 3;
    float* q = fk_scratch + BATCH_SIZE * ppln::FK_T_SLOTS * 16;

    if (tid == 0) {
        for (int d = 0; d < D; d++) q[d] = theta[d * Tn + t];
    }
    __syncthreads();

    fk_runtime(model, q, sphere_pos, fk_scratch, tid);
    __syncthreads();

    float world_hinge = scene_sum_esdf_hinge_runtime(model, sphere_pos, scene, tid, world_margin)
                      + scene_sum_obb_hinge_runtime(model, sphere_pos, scene, tid, world_margin);
    float self_hinge = self_min_hinge_runtime(model, sphere_pos, tid, self_margin);

    if (tid == 0) {
        float cost = collision_cost_weight * (world_hinge + self_hinge);
        for (int d = 0; d < D; d++) state_costs[d * Tn + t] = cost;
    }
}

// ============================================================================
// Per-(d,t) smoothness (control) cost as the decomposed quadratic form
//   control_costs[d,t] = smoothness_weight * theta[t] * (R @ theta_d)[t]
// so the sum over (d,t) recovers smoothness_weight * theta^T R theta. R is the
// identical 5-point acceleration matrix uploaded to d_R.
// ============================================================================
__global__ void chomp_control_cost_kernel(
    const float* theta,          // [D*Tn]
    float* control_costs,        // [D*Tn]
    const float* R,              // [Tn*Tn] row-major
    int D, int Tn,
    float smoothness_weight)
{
    int d = blockIdx.x;
    int t = threadIdx.x;
    if (d >= D || t >= Tn) return;

    float Rtheta = 0.0f;
    for (int tp = 0; tp < Tn; tp++) Rtheta += R[t * Tn + tp] * theta[d * Tn + tp];
    control_costs[d * Tn + t] = smoothness_weight * theta[d * Tn + t] * Rtheta;
}

// ============================================================================
// Reduce the (D*T) cost breakdown into scalar state/control/total costs. Single
// block, Tn threads. state_costs is D-broadcast so the d=0 slice suffices.
// ============================================================================
__global__ void chomp_reduce_costs_kernel(
    const float* state_costs,    // [D*Tn], D-broadcast
    const float* control_costs,  // [D*Tn]
    float* state_cost, float* control_cost, float* total_cost,
    int D, int Tn)
{
    int t = threadIdx.x;
    extern __shared__ float sh[];   // [2*Tn]: state partials, control partials
    float* sh_state = sh;
    float* sh_ctrl = sh + Tn;

    float sc = (t < Tn) ? state_costs[t] : 0.0f;  // d=0 slice
    float ctrl = 0.0f;
    if (t < Tn) {
        for (int d = 0; d < D; d++) ctrl += control_costs[d * Tn + t];
    }
    sh_state[t] = sc;
    sh_ctrl[t] = ctrl;
    __syncthreads();

    if (t == 0) {
        float s = 0.0f, c = 0.0f;
        for (int i = 0; i < Tn; i++) { s += sh_state[i]; c += sh_ctrl[i]; }
        *state_cost = s;
        *control_cost = c;
        *total_cost = s + c;
    }
}

// ============================================================================
// Covariant CHOMP update. One block == one dim d; Tn threads.
//   Phase A: g[t] = obs_grad[d,t] + smoothness_weight * (R @ theta_d)[t]
//            (obs_grad already carries collision_cost_weight).
//   Phase B: step[t] = (A^{-1} @ g)[t]
//   Phase C: theta[d,t] -= step_size * step[t], endpoints pinned, clamp to
//            limits. step_size is a SCALAR. The per-DoF variant (scale[d] from
//            ppln::make_dof_scale, dof_scale.hh) was implemented and measured to
//            change NOTHING here: success identical at 3/70 Panda, 0/70 Fetch,
//            0/30 Baxter, with only Panda's cost moving over a 3-problem sample.
//            It was reverted rather than shipped on a plausible-sounding
//            argument. See the STOMP noise kernel for why the same idea is
//            actively harmful for a stochastic amplitude.
// The Rtheta read of theta happens before the barrier; the theta write happens
// only after, and each block owns a disjoint d-column, so there is no race.
// ============================================================================
__global__ void chomp_update_kernel(
    float* theta,                // [D*Tn] in/out
    const float* obs_grad,       // [D*Tn]
    const float* R,              // [Tn*Tn]
    const float* Ainv,           // [Tn*Tn]
    const RobotModel model,
    int D, int Tn,
    float step_size,
    float smoothness_weight)
{
    int d = blockIdx.x;
    int t = threadIdx.x;
    if (d >= D || t >= Tn) return;

    extern __shared__ float g[];  // [Tn]

    float Rtheta = 0.0f;
    for (int tp = 0; tp < Tn; tp++) Rtheta += R[t * Tn + tp] * theta[d * Tn + tp];
    g[t] = obs_grad[d * Tn + t] + smoothness_weight * Rtheta;
    __syncthreads();

    float step = 0.0f;
    for (int tp = 0; tp < Tn; tp++) step += Ainv[t * Tn + tp] * g[tp];

    if (t == 0 || t == Tn - 1) return;  // pin endpoints to start/goal

    float v = theta[d * Tn + t] - step_size * step;
    v = fmaxf(model.joint_lower[d], fminf(model.joint_upper[d], v));
    theta[d * Tn + t] = v;
}

// ============================================================================
// Binary validity gate — identical to STOMP's stomp_validity_kernel. Densifies
// each segment of theta on the fly at collision_check_step resolution and
// OR-reduces scene + self collision across all sub-points.
// ============================================================================
__global__ void chomp_validity_kernel(
    const float* theta,
    const RobotModel model,
    const SceneCollisionData scene,
    int D, int Tn,
    int n_sub,
    float collision_margin,
    int* d_valid)
{
    int flat = blockIdx.x;
    int total = (Tn - 1) * n_sub + 1;
    if (flat >= total) return;
    int tid = threadIdx.x;
    if (tid >= 4) return;

    extern __shared__ float smem[];
    int n_spheres = model.n_spheres;
    float* sphere_pos = smem;
    float* approx_sphere_pos = sphere_pos + n_spheres * BATCH_SIZE * 3;
    float* fk_scratch = approx_sphere_pos + model.n_approx_spheres * BATCH_SIZE * 3;
    float* q = fk_scratch + BATCH_SIZE * ppln::FK_T_SLOTS * 16;

    __shared__ int jc[20];
    __shared__ ppln::collision::TwoPhaseFlags cc_flags;

    if (tid == 0) {
        if (flat == total - 1) {
            for (int d = 0; d < D; d++) q[d] = theta[d * Tn + (Tn - 1)];
        } else {
            int seg = flat / n_sub;
            int sub = flat % n_sub;
            float frac = (float)sub / (float)n_sub;
            for (int d = 0; d < D; d++) {
                float a = theta[d * Tn + seg];
                float b = theta[d * Tn + seg + 1];
                q[d] = a + (b - a) * frac;
            }
        }
    }
    __syncthreads();

    // The shared anchor (src/collision/two_phase_cc.cuh). The optimizer's COST
    // is still the summed hinge -- a boolean has no gradient for CHOMP to
    // descend and no spread for STOMP to reweight rollouts by -- but the
    // feasibility VERDICT that sets result.solved is the same routine the
    // sampling and search planners use, so a trajectory this planner reports
    // as collision-free is collision-free for all of them by construction.
    ppln::collision::TwoPhaseResult cc = ppln::collision::two_phase_cc(
        model, scene, q, sphere_pos, approx_sphere_pos, fk_scratch,
        jc, &cc_flags, tid, collision_margin, /*check_self=*/true);

    if (tid == 0 && cc.collision) {
        atomicExch(d_valid, 0);
    }
}

} // namespace (anonymous)

// ============================================================================
// Host: CHOMP::solve_runtime_scene
// ============================================================================
CHOMPResult solve_runtime_scene(
    std::vector<float>& start,
    std::vector<std::vector<float>>& goals,
    ppln::collision::SceneCollisionData& scene,
    CHOMP_settings& settings,
    ppln::RobotModel& model,
    CHOMPBuffers* bufs)
{
    auto wall_start = std::chrono::steady_clock::now();

    const int D = model.n_dof;
    const int Tn = settings.num_timesteps;

    CHOMPResult result;
    const bool use_bufs = (bufs != nullptr && bufs->owns_memory);

    const size_t bdt = (size_t)D * Tn;

    float *d_R, *d_Ainv, *d_theta, *d_best, *d_obs_grad;
    float *d_state_costs, *d_control_costs;
    float *d_state_cost, *d_control_cost, *d_total_cost;
    float *d_start, *d_goal;
    int *d_collision;
    int *h_valid;
    float *h_total_cost;

    if (use_bufs) {
        d_R = bufs->d_R;
        d_Ainv = bufs->d_Ainv;
        d_theta = bufs->d_theta;
        d_best = bufs->d_best;
        d_obs_grad = bufs->d_obs_grad;
        d_state_costs = bufs->d_state_costs;
        d_control_costs = bufs->d_control_costs;
        d_state_cost = bufs->d_state_cost;
        d_control_cost = bufs->d_control_cost;
        d_total_cost = bufs->d_total_cost;
        d_start = bufs->d_start;
        d_goal = bufs->d_goal;
        d_collision = bufs->d_collision;
        h_valid = bufs->h_valid;
        h_total_cost = bufs->h_total_cost;
    } else {
        cudaMalloc(&d_R, (size_t)Tn * Tn * sizeof(float));
        cudaMalloc(&d_Ainv, (size_t)Tn * Tn * sizeof(float));
        cudaMalloc(&d_theta, bdt * sizeof(float));
        cudaMalloc(&d_best, bdt * sizeof(float));
        cudaMalloc(&d_obs_grad, bdt * sizeof(float));
        cudaMalloc(&d_state_costs, bdt * sizeof(float));
        cudaMalloc(&d_control_costs, bdt * sizeof(float));
        cudaMalloc(&d_state_cost, sizeof(float));
        cudaMalloc(&d_control_cost, sizeof(float));
        cudaMalloc(&d_total_cost, sizeof(float));
        cudaMalloc(&d_start, (size_t)D * sizeof(float));
        cudaMalloc(&d_goal, (size_t)D * sizeof(float));
        cudaMalloc(&d_collision, sizeof(int));
        cudaHostAlloc(&h_valid, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_total_cost, sizeof(float), cudaHostAllocDefault);
    }

    // --- Host: smoothness metric R (5-point acceleration control cost, bit-
    // identical to STOMP's) and its covariant preconditioner A^{-1}=(R+eps I)^-1.
    {
        std::vector<float> h_R = compute_control_cost_R(Tn, settings.delta_t);
        cudaMemcpy(d_R, h_R.data(), (size_t)Tn * Tn * sizeof(float), cudaMemcpyHostToDevice);
        std::vector<float> h_Ainv = compute_smoothness_Ainv(Tn, settings.delta_t);
        cudaMemcpy(d_Ainv, h_Ainv.data(), (size_t)Tn * Tn * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_start, start.data(), (size_t)D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goal, goals[0].data(), (size_t)D * sizeof(float), cudaMemcpyHostToDevice);

    // Initial trajectory: device joint-space linear interpolation start->goal.
    chomp_init_trajectory_kernel<<<D, Tn>>>(d_start, d_goal, model, d_theta, D, Tn);

    // Shared-memory sizing for the per-config FK+cost kernels, mirroring STOMP.
    size_t smem_state_bytes = (size_t)(model.n_spheres * BATCH_SIZE * 3 + BATCH_SIZE * ppln::FK_T_SLOTS * 16 + D) * sizeof(float);
    // The validity kernel additionally stages the approximate sphere set,
    // because the shared two-phase anchor gates its exact pass on an
    // approximate pass. Sized separately so the cost kernels keep their
    // original (smaller) footprint and their existing offsets.
    size_t smem_validity_bytes = smem_state_bytes
        + (size_t)(model.n_approx_spheres * BATCH_SIZE * 3) * sizeof(float);
    if (smem_state_bytes > 48 * 1024) {
        cudaFuncSetAttribute(chomp_obs_grad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_state_bytes);
        cudaFuncSetAttribute(chomp_state_cost_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_state_bytes);
        cudaFuncSetAttribute(chomp_validity_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_validity_bytes);
    }

    // Cost signal helper: recompute state + control breakdown and reduce to
    // scalars for the current d_theta.
    auto refresh_costs = [&]() {
        chomp_state_cost_kernel<<<Tn, 4, smem_state_bytes>>>(
            d_theta, d_state_costs, model, scene, D, Tn,
            settings.collision_cost_weight, settings.world_collision_margin, settings.self_collision_margin);
        chomp_control_cost_kernel<<<D, Tn>>>(
            d_theta, d_control_costs, d_R, D, Tn, settings.smoothness_weight);
        chomp_reduce_costs_kernel<<<1, Tn, (size_t)(2 * Tn) * sizeof(float)>>>(
            d_state_costs, d_control_costs, d_state_cost, d_control_cost, d_total_cost, D, Tn);
    };

    auto check_validity = [&]() -> bool {
        std::vector<float> h_opt((size_t)D * Tn);
        cudaMemcpy(h_opt.data(), d_theta, (size_t)D * Tn * sizeof(float), cudaMemcpyDeviceToHost);
        float max_seg = 0.0f;
        for (int t = 0; t < Tn - 1; t++) {
            float d2 = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = h_opt[(size_t)d * Tn + t + 1] - h_opt[(size_t)d * Tn + t];
                d2 += diff * diff;
            }
            max_seg = std::max(max_seg, std::sqrt(d2));
        }
        int n_sub = 1;
        if (settings.collision_check_step > 0.0f && max_seg > settings.collision_check_step) {
            n_sub = (int)std::ceil(max_seg / settings.collision_check_step);
        }
        int total_checks = (Tn - 1) * n_sub + 1;
        int one = 1;
        cudaMemcpy(d_collision, &one, sizeof(int), cudaMemcpyHostToDevice);
        chomp_validity_kernel<<<total_checks, 4, smem_validity_bytes>>>(
            d_theta, model, scene, D, Tn, n_sub, settings.collision_margin, d_collision);
        cudaMemcpy(h_valid, d_collision, sizeof(int), cudaMemcpyDeviceToHost);
        return (*h_valid) != 0;
    };

    // Seed the initial cost breakdown and validity.
    refresh_costs();
    bool valid = check_validity();

    int polish_left = settings.num_iterations_after_valid;
    float prev_total = std::numeric_limits<float>::infinity();
    int converge_streak = 0;
    int iters_run = 0;

    // Best VALID trajectory seen so far. CHOMP descends the working theta each
    // iteration, but collision-freeness can appear/disappear as the trajectory
    // moves, so the returned result is the best valid snapshot (fall back to the
    // working theta if none was ever valid).
    float best_total = std::numeric_limits<float>::infinity();
    float best_state = 0.0f, best_control = 0.0f;
    bool best_valid = false;
    if (valid) {
        cudaMemcpy(d_best, d_theta, bdt * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&best_state, d_state_cost, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&best_control, d_control_cost, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&best_total, d_total_cost, sizeof(float), cudaMemcpyDeviceToHost);
        best_valid = true;
    }

    for (int iter = 0; iter < settings.num_iterations; iter++) {
        if (best_valid && polish_left <= 0) break;

        if (settings.time_limit_ms > 0.0f) {
            float elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wall_start).count() / 1000.0f;
            if (elapsed_ms > settings.time_limit_ms) break;
        }

        // Obstacle gradient = central FD of the shared substrate cost.
        chomp_obs_grad_kernel<<<dim3(Tn, D), 4, smem_state_bytes>>>(
            d_theta, d_obs_grad, model, scene, D, Tn,
            settings.collision_cost_weight, settings.world_collision_margin,
            settings.self_collision_margin, settings.fd_epsilon);

        // Covariant descent step: theta -= step_size * A^{-1} (obs_grad + lambda R theta).
        chomp_update_kernel<<<D, Tn, (size_t)Tn * sizeof(float)>>>(
            d_theta, d_obs_grad, d_R, d_Ainv, model, D, Tn,
            settings.step_size, settings.smoothness_weight);

        // Refresh the comparable cost signal for the updated trajectory.
        refresh_costs();

        iters_run++;

        cudaMemcpy(h_total_cost, d_total_cost, sizeof(float), cudaMemcpyDeviceToHost);
        float cur_total = *h_total_cost;

        valid = check_validity();
        if (valid && cur_total < best_total) {
            cudaMemcpy(d_best, d_theta, bdt * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&best_state, d_state_cost, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&best_control, d_control_cost, sizeof(float), cudaMemcpyDeviceToHost);
            best_total = cur_total;
            best_valid = true;
        }

        // Convergence + polish accounting driven by the BEST valid cost. Once a
        // valid best exists, run at most num_iterations_after_valid more
        // iterations, stopping early if the best has stalled for two in a row.
        if (best_valid) {
            float impr = prev_total - best_total;
            converge_streak = (impr >= 0.0f && impr < settings.convergence_eps) ? converge_streak + 1 : 0;
            prev_total = best_total;
            polish_left--;
            if (converge_streak >= 2) break;
        }
    }

    // Return the best VALID trajectory if one was found; otherwise the working
    // theta (result.solved reflects which).
    const float* d_result = best_valid ? d_best : d_theta;
    std::vector<float> h_opt_final((size_t)D * Tn);
    cudaMemcpy(h_opt_final.data(), d_result, (size_t)D * Tn * sizeof(float), cudaMemcpyDeviceToHost);
    result.path.assign(Tn, std::vector<float>(D));
    for (int t = 0; t < Tn; t++)
        for (int d = 0; d < D; d++)
            result.path[t][d] = h_opt_final[(size_t)d * Tn + t];
    for (int d = 0; d < D; d++) {
        result.path[0][d] = start[d];
        result.path[Tn - 1][d] = goals[0][d];
    }

    float h_final_state, h_final_control, h_final_total;
    if (best_valid) {
        h_final_state = best_state;
        h_final_control = best_control;
        h_final_total = best_total;
    } else {
        h_final_state = 0.0f; h_final_control = 0.0f; h_final_total = 0.0f;
        cudaMemcpy(&h_final_state, d_state_cost, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_final_control, d_control_cost, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_final_total, d_total_cost, sizeof(float), cudaMemcpyDeviceToHost);
    }

    result.solved = best_valid;
    result.iterations_run = iters_run;
    result.final_state_cost = h_final_state;
    result.final_control_cost = h_final_control;
    result.final_total_cost = h_final_total;

    if (!use_bufs) {
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
    }

    result.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - wall_start).count();
    return result;
}

} // namespace CHOMP
