// ============================================================================
// stomp.cu — GPU-native STOMP (Kalakrishnan 2011) trajectory optimizer
// Ported from the Python/torch reference `stomp_core._StompSolver`.
//
// Design mirrors MITStar.cu: host loop launches a fixed sequence of small
// kernels on the default stream per iteration; all rollout/cost/probability
// math stays resident on the GPU. The only per-iteration host<->device
// traffic is: (a) a small D*T readback used to size the validity-gate
// densification, and (b) the pinned h_valid/h_total_cost readback used to
// drive the early-stop state machine — replacing the Python solver's
// per-cost-evaluation syncs.
//
// A SINGLE control-cost metric R is used everywhere (the ros-industrial
// 5-point acceleration matrix; see stomp_precompute.cpp:build_control_cost_R):
//   - compute_cholesky_L / compute_projection_M derive the sampling covariance
//     R_inv/L and the update-smoothing M from R.
//   - stomp_control_cost_kernel evaluates the per-rollout smoothness cost as
//     the quadratic form theta^T R theta under the identical R (uploaded to
//     d_R), so sampling, smoothing and cost-weighting share one metric.
// ============================================================================

#include "Planners.hh"
#include "utils.cuh"
#include "robot_model.cuh"
#include "STOMP_settings.hh"
#include "stomp_buffers.hh"
#include "runtime_kinematics.cuh"
#include "src/collision/scene_collision.cuh"
#include "src/collision/two_phase_cc.cuh"

#include "stomp_precompute.hh"

#include <curand_kernel.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>

namespace STOMP {
using namespace ppln;
using namespace ppln::collision;

namespace {

// ============================================================================
// P0: RNG init — one curandState per (rollout, dim, timestep) thread of P1.
// ============================================================================
__global__ void stomp_init_rng_kernel(curandState* states, unsigned long long seed, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    curand_init(seed + idx, idx, 0, &states[idx]);
}

// ============================================================================
// Linear-interpolated + clamped initial trajectory.
// ============================================================================
__global__ void stomp_init_trajectory_kernel(
    const float* start, const float* goal,
    const RobotModel model,
    float* optimum, int D, int Tn)
{
    int d = blockIdx.x;
    int t = threadIdx.x;
    if (d >= D || t >= Tn) return;
    float frac = (Tn > 1) ? (float)t / (float)(Tn - 1) : 0.0f;
    float v = start[d] * (1.0f - frac) + goal[d] * frac;
    v = fmaxf(model.joint_lower[d], fminf(model.joint_upper[d], v));
    optimum[d * Tn + t] = v;
}

// ============================================================================
// P1: sample noise = L @ z (z ~ N(0,I)), pin endpoints, clamp to limits.
// row_offset lets the very first (init_it) iteration fill [0, Rn+Ro) with two
// launches reusing the Rn-sized RNG state array (curand states are stateful, so
// the second launch simply continues each stream — no aliasing issue).
//
// The exploration amplitude is a SCALAR (noise_scale), deliberately. A per-DoF
// amplitude — each joint's noise scaled by its share of the robot's mean travel,
// the way ppln::make_dof_scale (dof_scale.hh) does it for the lattice planners —
// was implemented here and MEASURED WORSE. Real MBM, 1 s budget, iteration cap
// lifted so time is the binding constraint:
//     per-DoF on   ->  Panda 0.69   Fetch 0.01   Baxter 0.03
//     per-DoF off  ->  Panda 0.69   Fetch 0.06   Baxter 0.07   <- shipped
// The earlier argument for it had the sign backwards: dof_scale SHRINKS the
// 0.386 m prismatic torso's noise ~10x relative to the 6.3 rad revolute joints,
// starving the very joint it was meant to help. The general rule this settles:
// per-DoF range normalisation is correct for a DISCRETISED ACTION SET (a lattice
// step must mean the same fraction of every joint's travel — it helped MHA* and
// wPA*SE) and wrong for a STOCHASTIC EXPLORATION AMPLITUDE, which wants absolute
// configuration-space reach. Do not reintroduce it without a measurement that
// contradicts the table above.
// ============================================================================
__global__ void stomp_sample_noise_kernel(
    curandState* rng_states,
    const float* L,           // [Tn*Tn], lower-triangular
    const float* optimum,     // [D*Tn]
    const RobotModel model,
    float* noise_out,         // [R*D*Tn]
    float* rollout_out,       // [R*D*Tn]
    int D, int Tn, int row_offset)
{
    int r_local = blockIdx.x;
    int d = blockIdx.y;
    int t = threadIdx.x;
    int rng_idx = (r_local * D + d) * Tn + t;

    extern __shared__ float z[]; // [Tn]
    z[t] = curand_normal(&rng_states[rng_idx]);
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k <= t; k++) sum += L[t * Tn + k] * z[k];

    int out_row = row_offset + r_local;
    float opt = optimum[d * Tn + t];
    float mean = opt;
    if (t == 0 || t == Tn - 1) { sum = 0.0f; }  // pin endpoints to start/goal

    float rollout = fmaxf(model.joint_lower[d], fminf(model.joint_upper[d], mean + sum));

    int idx = (out_row * D + d) * Tn + t;
    noise_out[idx] = rollout - opt;   // noise still relative to optimum (update math unchanged)
    rollout_out[idx] = rollout;
}

// ============================================================================
// P1b: select the `Ro` lowest-total_cost rollouts from the PREVIOUS
// iteration's snapshot (single-thread selection sort; R is tiny, ~O(R*Ro)).
// ============================================================================
__global__ void stomp_select_topk_kernel(
    const float* prev_total_cost, // [R]
    int* topk_idx,                // [R] scratch, only [0,Ro) used
    int R, int Ro)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    extern __shared__ float cost_copy[]; // [R]
    for (int i = 0; i < R; i++) cost_copy[i] = prev_total_cost[i];
    for (int k = 0; k < Ro; k++) {
        int best = 0;
        float best_cost = cost_copy[0];
        for (int i = 1; i < R; i++) {
            if (cost_copy[i] < best_cost) { best_cost = cost_copy[i]; best = i; }
        }
        topk_idx[k] = best;
        cost_copy[best] = 3.0e38f;
    }
}

// Gather the Ro selected reuse rollouts into slots [Rn, R-1); noise is
// recomputed relative to the CURRENT optimum (positions/costs carry over
// unchanged since they don't depend on which optimum they're offset from).
__global__ void stomp_gather_reuse_kernel(
    const int* topk_idx,
    const float* prev_stored_rollouts,
    const float* prev_control_costs,
    const float* prev_state_costs,
    const float* optimum,
    float* stored_rollouts, float* control_costs, float* state_costs, float* noise,
    int D, int Tn, int Rn)
{
    int i = blockIdx.x;
    int d = blockIdx.y;
    int t = threadIdx.x;
    int src = topk_idx[i];
    int out_row = Rn + i;
    int src_idx = (src * D + d) * Tn + t;
    int out_idx = (out_row * D + d) * Tn + t;

    float sr = prev_stored_rollouts[src_idx];
    stored_rollouts[out_idx] = sr;
    control_costs[out_idx] = prev_control_costs[src_idx];
    state_costs[out_idx] = prev_state_costs[src_idx];
    noise[out_idx] = sr - optimum[d * Tn + t];
}

// Seed the last rollout slot [R-1] with the current optimum (noise == 0).
__global__ void stomp_seed_last_slot_kernel(
    const float* optimum,
    const float* parameters_control_costs,
    const float* parameters_state_costs,
    float* stored_rollouts, float* control_costs, float* state_costs, float* noise,
    int D, int Tn, int R)
{
    int d = blockIdx.x;
    int t = threadIdx.x;
    int out_row = R - 1;
    int idx = d * Tn + t;
    int out_idx = (out_row * D + d) * Tn + t;
    stored_rollouts[out_idx] = optimum[idx];
    control_costs[out_idx] = parameters_control_costs[idx];
    state_costs[out_idx] = parameters_state_costs[idx];
    noise[out_idx] = 0.0f;
}

// ============================================================================
// P2: per-(row,timestep) state (collision) cost. Generic over n_rows so the
// same kernel serves the main R-row population AND the single-row
// candidate/initial-trajectory seeding path.
// One block == one (row, t) config; 4 cooperating threads (BATCH_SIZE layout,
// batch_ind always 0 since blockDim==4). Mirrors MITStar.cu's
// compute_node_clearance_kernel shared-memory layout exactly.
// ============================================================================
__global__ void stomp_state_cost_kernel(
    const float* rollouts,      // [n_rows*D*Tn]
    float* state_costs,         // [n_rows*D*Tn] output, broadcast over D
    const RobotModel model,
    const SceneCollisionData scene,
    int D, int Tn, int n_rows,
    float collision_cost_weight,
    float world_margin,
    float self_margin)
{
    int flat = blockIdx.x;
    int total = n_rows * Tn;
    if (flat >= total) return;
    int r = flat / Tn;
    int t = flat % Tn;
    int tid = threadIdx.x;
    if (tid >= 4) return;

    extern __shared__ float smem[];
    int n_spheres = model.n_spheres;
    float* sphere_pos = smem;
    float* fk_scratch = sphere_pos + n_spheres * BATCH_SIZE * 3;
    float* q = fk_scratch + BATCH_SIZE * ppln::FK_T_SLOTS * 16;

    if (tid == 0) {
        for (int d = 0; d < D; d++) q[d] = rollouts[(r * D + d) * Tn + t];
    }
    __syncthreads();

    fk_runtime(model, q, sphere_pos, fk_scratch, tid);
    __syncthreads();

    float world_hinge = scene_sum_esdf_hinge_runtime(model, sphere_pos, scene, tid, world_margin)
                       + scene_sum_obb_hinge_runtime(model, sphere_pos, scene, tid, world_margin);
    float self_hinge = self_min_hinge_runtime(model, sphere_pos, tid, self_margin);

    if (tid == 0) {
        float cost = collision_cost_weight * (world_hinge + self_hinge);
        for (int d = 0; d < D; d++) state_costs[(r * D + d) * Tn + t] = cost;
    }
}

// ============================================================================
// P3: per-rollout control (smoothness) cost as the quadratic form theta^T R
// theta, decomposed per timestep so the row-sum recovers it:
//   control_costs[r,d,t] = w * theta[t] * (R @ theta_{r,d})[t]
// R is the SAME 5-point acceleration control-cost matrix used by the sampling
// covariance L and the projection M (uploaded to d_R). This matches
// ros-industrial computeParametersControlCosts (theta .* (R theta)). Generic
// over n_rows, same reuse as P2.
// ============================================================================
__global__ void stomp_control_cost_kernel(
    const float* rollouts,     // [n_rows*D*Tn]
    float* control_costs,      // [n_rows*D*Tn]
    const float* R,            // [Tn*Tn] row-major control-cost matrix
    int D, int Tn, int n_rows,
    float control_cost_weight, float delta_t)
{
    int r = blockIdx.x;
    int d = blockIdx.y;
    int t = threadIdx.x;
    if (r >= n_rows || d >= D || t >= Tn) return;

    int base = (r * D + d) * Tn;
    // (R @ theta)[t] = sum_tp R[t,tp] * theta[tp]
    float Rtheta = 0.0f;
    for (int tp = 0; tp < Tn; tp++) Rtheta += R[t * Tn + tp] * rollouts[base + tp];
    control_costs[base + t] = control_cost_weight * rollouts[base + t] * Rtheta;
}

// ============================================================================
// P4: total_costs[r,d,t] = state + control (elementwise, ALL R rows); and the
// per-rollout scalar total_cost[r] = state_costs[r,0,:].sum() (D-broadcast,
// take one slice) + control_costs[r,:,:].sum() (all D,T).
// One block per rollout; Tn threads; thread 0 does the final tiny serial sum
// (Tn ~= 51, negligible cost, avoids non-power-of-2 tree-reduction bugs).
// ============================================================================
__global__ void stomp_reduce_costs_kernel(
    const float* state_costs, const float* control_costs,
    float* total_costs, float* total_cost,
    int D, int Tn, int R)
{
    int r = blockIdx.x;
    int t = threadIdx.x;
    if (r >= R || t >= Tn) return;

    extern __shared__ float sdata[]; // [Tn]
    float partial = state_costs[(r * D + 0) * Tn + t];
    for (int d = 0; d < D; d++) {
        int idx = (r * D + d) * Tn + t;
        float tc = state_costs[idx] + control_costs[idx];
        total_costs[idx] = tc;
        partial += control_costs[idx];
    }

    sdata[t] = partial;
    __syncthreads();
    if (t == 0) {
        float sum = 0.0f;
        for (int i = 0; i < Tn; i++) sum += sdata[i];
        total_cost[r] = sum;
    }
}

// ============================================================================
// P5: per-(dim,timestep) LOCAL probability weighting over R rollouts.
// importance_weights is always 1.0 in stomp_core.py, so unnorm == exp(...).
// ============================================================================
__global__ void stomp_probabilities_kernel(
    const float* total_costs, float* probabilities,
    int D, int Tn, int R, float h)
{
    int d = blockIdx.x;
    int t = blockIdx.y;
    int r = threadIdx.x;
    if (r >= R) return;

    extern __shared__ float sh[]; // [R + 2]: sh[R]=min, sh[R+1]=max, then reused for unnorm sum
    float c = total_costs[(r * D + d) * Tn + t];
    sh[r] = c;
    __syncthreads();

    if (r == 0) {
        float mn = sh[0], mx = sh[0];
        for (int i = 1; i < R; i++) { mn = fminf(mn, sh[i]); mx = fmaxf(mx, sh[i]); }
        sh[R] = mn;
        sh[R + 1] = mx;
    }
    __syncthreads();

    float mn = sh[R];
    float mx = sh[R + 1];
    float denom = fmaxf(mx - mn, 1.0e-8f); // MIN_COST_DIFFERENCE
    float unnorm = expf(-h * (c - mn) / denom);
    __syncthreads();
    sh[r] = unnorm;
    __syncthreads();

    if (r == 0) {
        float sum = 0.0f;
        for (int i = 0; i < R; i++) sum += sh[i];
        sh[R] = fmaxf(sum, 1.0e-12f);
    }
    __syncthreads();

    probabilities[d * Tn * R + t * R + r] = unnorm / sh[R];
}

// ============================================================================
// P6: delta[d,t] = sum_r noise[r,d,t] * p[d,t,r]; pin endpoints. This is the
// RAW per-timestep weighted noise average — it is NOT yet smooth, because the
// probability weights vary independently per timestep. It must be projected
// through M (P6b) before being applied, exactly as in Kalakrishnan STOMP.
// ============================================================================
__global__ void stomp_update_kernel(
    const float* noise, const float* probabilities,
    float* delta, int D, int Tn, int R)
{
    int d = blockIdx.x;
    int t = blockIdx.y;
    int r = threadIdx.x;
    if (r >= R) return;

    extern __shared__ float sh[]; // [R]
    float n = noise[(r * D + d) * Tn + t];
    float p = probabilities[d * Tn * R + t * R + r];
    sh[r] = n * p;
    __syncthreads();

    if (r == 0) {
        float sum = 0.0f;
        for (int i = 0; i < R; i++) sum += sh[i];
        if (t == 0 || t == Tn - 1) sum = 0.0f;
        delta[d * Tn + t] = sum;
    }
}

// ============================================================================
// P6b: smoothing projection. candidate[d,t] = optimum[d,t] + (M @ delta)[d,t],
// with M = R^-1 (per-column normalised, from compute_projection_M). This is
// the STOMP update-smoothing step: it re-projects the raw weighted noise
// average onto the smooth subspace the noise was sampled from. Endpoints are
// re-pinned so start/goal never drift.
// ============================================================================
__global__ void stomp_project_kernel(
    const float* delta,       // [D*Tn] raw weighted noise average
    const float* Mproj,       // [Tn*Tn] row-major projection matrix (named Mproj:
                              // utils.cuh has `#define M 4`, which would clobber `M`)
    const float* optimum,     // [D*Tn]
    float* candidate,         // [D*Tn]
    int D, int Tn)
{
    int d = blockIdx.x;
    int t = threadIdx.x;
    if (d >= D || t >= Tn) return;

    float acc = 0.0f;
    for (int tp = 0; tp < Tn; tp++) acc += Mproj[t * Tn + tp] * delta[d * Tn + tp];
    if (t == 0 || t == Tn - 1) acc = 0.0f;
    candidate[d * Tn + t] = optimum[d * Tn + t] + acc;
}

// ============================================================================
// P6c: UNCONDITIONALLY apply the candidate as the new working optimum and
// refresh its cached cost breakdown. Canonical STOMP applies the update every
// iteration (it is an expectation-style average, not a greedy line search);
// the "keep the best valid trajectory" concern is handled separately by the
// host-side best-snapshot, so this kernel does no accept/reject.
// ============================================================================
__global__ void stomp_apply_kernel(
    const float* candidate,
    const float* candidate_state_costs,
    const float* candidate_control_costs,
    float* parameters_optimized,
    float* parameters_state_costs,
    float* parameters_control_costs,
    float* parameters_state_cost,
    float* parameters_control_cost,
    float* parameters_total_cost,
    int D, int Tn)
{
    int tid = threadIdx.x;
    extern __shared__ float sh[];      // [Tn] state partials
    float* sh_ctrl = sh + Tn;          // [Tn] control partials
    float* sh_scalar = sh_ctrl + Tn;   // [2]: state_cost, control_sum

    float sc = (tid < Tn) ? candidate_state_costs[tid] : 0.0f; // d=0 slice
    sh[tid] = sc;

    float ctrl = 0.0f;
    if (tid < Tn) {
        for (int d = 0; d < D; d++) ctrl += candidate_control_costs[d * Tn + tid];
    }
    sh_ctrl[tid] = ctrl;
    __syncthreads();

    if (tid == 0) {
        float state_cost = 0.0f, control_sum = 0.0f;
        for (int t = 0; t < Tn; t++) { state_cost += sh[t]; control_sum += sh_ctrl[t]; }
        sh_scalar[0] = state_cost;
        sh_scalar[1] = control_sum;
    }
    __syncthreads();

    float state_cost = sh_scalar[0];
    float control_sum = sh_scalar[1];

    if (tid < Tn) {
        for (int d = 0; d < D; d++) {
            int idx = d * Tn + tid;
            parameters_optimized[idx] = candidate[idx];
            parameters_state_costs[idx] = candidate_state_costs[idx];
            parameters_control_costs[idx] = candidate_control_costs[idx];
        }
    }
    if (tid == 0) {
        *parameters_state_cost = state_cost;
        *parameters_control_cost = control_sum;
        *parameters_total_cost = state_cost + control_sum;
    }
}

// Seed the cached scalar optimum costs on the very first iteration
// (unconditional — no accept/reject comparison needed yet).
__global__ void stomp_seed_scalar_costs_kernel(
    const float* parameters_state_costs,
    const float* parameters_control_costs,
    float* parameters_state_cost, float* parameters_control_cost, float* parameters_total_cost,
    int D, int Tn)
{
    int tid = threadIdx.x;
    extern __shared__ float sh[];    // [Tn]
    float* sh_ctrl = sh + Tn;        // [Tn]

    float sc = (tid < Tn) ? parameters_state_costs[tid] : 0.0f;
    sh[tid] = sc;

    float ctrl = 0.0f;
    if (tid < Tn) {
        for (int d = 0; d < D; d++) ctrl += parameters_control_costs[d * Tn + tid];
    }
    sh_ctrl[tid] = ctrl;
    __syncthreads();

    if (tid == 0) {
        float state_cost = 0.0f, control_sum = 0.0f;
        for (int t = 0; t < Tn; t++) { state_cost += sh[t]; control_sum += sh_ctrl[t]; }
        *parameters_state_cost = state_cost;
        *parameters_control_cost = control_sum;
        *parameters_total_cost = state_cost + control_sum;
    }
}

// ============================================================================
// P7: binary validity gate. Densifies each segment of the optimum on the fly
// (interpolated in-register, no dense buffer needed) at `collision_check_step`
// resolution and OR-reduces scene + self collision across all sub-points.
// ============================================================================
__global__ void stomp_validity_kernel(
    const float* optimum,
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
            for (int d = 0; d < D; d++) q[d] = optimum[d * Tn + (Tn - 1)];
        } else {
            int seg = flat / n_sub;
            int sub = flat % n_sub;
            float frac = (float)sub / (float)n_sub;
            for (int d = 0; d < D; d++) {
                float a = optimum[d * Tn + seg];
                float b = optimum[d * Tn + seg + 1];
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
// Host: STOMP::solve_runtime_scene
// ============================================================================
STOMPResult solve_runtime_scene(
    std::vector<float>& start,
    std::vector<std::vector<float>>& goals,
    ppln::collision::SceneCollisionData& scene,
    STOMP_settings& settings,
    ppln::RobotModel& model,
    STOMPBuffers* bufs)
{
    auto wall_start = std::chrono::steady_clock::now();

    const int D = model.n_dof;
    const int Tn = settings.num_timesteps;
    const int Rn = settings.num_rollouts_new;
    const int Ro = settings.num_rollouts_old;
    const int R = Rn + Ro + 1;

    STOMPResult result;
    const bool use_bufs = (bufs != nullptr && bufs->owns_memory);

    const size_t bdt = (size_t)D * Tn;
    const size_t brdt = (size_t)R * D * Tn;
    const size_t br = (size_t)R;
    const size_t bdtr = (size_t)D * Tn * R;

    float *d_L, *d_M, *d_R, *d_parameters_optimized, *d_candidate, *d_delta, *d_best_optimum;
    float *d_stored_rollouts, *d_noise, *d_control_costs, *d_state_costs, *d_total_costs;
    float *d_prev_stored_rollouts, *d_prev_noise, *d_prev_control_costs, *d_prev_state_costs, *d_prev_total_cost;
    float *d_total_cost;
    float *d_probabilities;
    float *d_parameters_state_costs, *d_parameters_control_costs;
    float *d_parameters_state_cost, *d_parameters_control_cost, *d_parameters_total_cost;
    float *d_candidate_state_costs, *d_candidate_control_costs, *d_candidate_total_cost;
    int *d_topk_idx;
    float *d_start, *d_goal;
    curandState *d_rng_states;
    int *d_collision;
    int *h_valid;
    float *h_total_cost;

    if (use_bufs) {
        d_L = bufs->d_L;
        d_M = bufs->d_M;
        d_R = bufs->d_R;
        d_parameters_optimized = bufs->d_parameters_optimized;
        d_candidate = bufs->d_candidate;
        d_delta = bufs->d_delta;
        d_best_optimum = bufs->d_best_optimum;
        d_stored_rollouts = bufs->d_stored_rollouts;
        d_noise = bufs->d_noise;
        d_control_costs = bufs->d_control_costs;
        d_state_costs = bufs->d_state_costs;
        d_total_costs = bufs->d_total_costs;
        d_prev_stored_rollouts = bufs->d_prev_stored_rollouts;
        d_prev_noise = bufs->d_prev_noise;
        d_prev_control_costs = bufs->d_prev_control_costs;
        d_prev_state_costs = bufs->d_prev_state_costs;
        d_prev_total_cost = bufs->d_prev_total_cost;
        d_total_cost = bufs->d_total_cost;
        d_probabilities = bufs->d_probabilities;
        d_parameters_state_costs = bufs->d_parameters_state_costs;
        d_parameters_control_costs = bufs->d_parameters_control_costs;
        d_parameters_state_cost = bufs->d_parameters_state_cost;
        d_parameters_control_cost = bufs->d_parameters_control_cost;
        d_parameters_total_cost = bufs->d_parameters_total_cost;
        d_candidate_state_costs = bufs->d_candidate_state_costs;
        d_candidate_control_costs = bufs->d_candidate_control_costs;
        d_candidate_total_cost = bufs->d_candidate_total_cost;
        d_topk_idx = bufs->d_topk_idx;
        d_start = bufs->d_start;
        d_goal = bufs->d_goal;
        d_rng_states = bufs->d_rng_states;
        d_collision = bufs->d_collision;
        h_valid = bufs->h_valid;
        h_total_cost = bufs->h_total_cost;
    } else {
        cudaMalloc(&d_L, (size_t)Tn * Tn * sizeof(float));
        cudaMalloc(&d_M, (size_t)Tn * Tn * sizeof(float));
        cudaMalloc(&d_R, (size_t)Tn * Tn * sizeof(float));
        cudaMalloc(&d_parameters_optimized, bdt * sizeof(float));
        cudaMalloc(&d_candidate, bdt * sizeof(float));
        cudaMalloc(&d_delta, bdt * sizeof(float));
        cudaMalloc(&d_best_optimum, bdt * sizeof(float));
        cudaMalloc(&d_stored_rollouts, brdt * sizeof(float));
        cudaMalloc(&d_noise, brdt * sizeof(float));
        cudaMalloc(&d_control_costs, brdt * sizeof(float));
        cudaMalloc(&d_state_costs, brdt * sizeof(float));
        cudaMalloc(&d_total_costs, brdt * sizeof(float));
        cudaMalloc(&d_prev_stored_rollouts, brdt * sizeof(float));
        cudaMalloc(&d_prev_noise, brdt * sizeof(float));
        cudaMalloc(&d_prev_control_costs, brdt * sizeof(float));
        cudaMalloc(&d_prev_state_costs, brdt * sizeof(float));
        cudaMalloc(&d_prev_total_cost, br * sizeof(float));
        cudaMalloc(&d_total_cost, br * sizeof(float));
        cudaMalloc(&d_probabilities, bdtr * sizeof(float));
        cudaMalloc(&d_parameters_state_costs, bdt * sizeof(float));
        cudaMalloc(&d_parameters_control_costs, bdt * sizeof(float));
        cudaMalloc(&d_parameters_state_cost, sizeof(float));
        cudaMalloc(&d_parameters_control_cost, sizeof(float));
        cudaMalloc(&d_parameters_total_cost, sizeof(float));
        cudaMalloc(&d_candidate_state_costs, bdt * sizeof(float));
        cudaMalloc(&d_candidate_control_costs, bdt * sizeof(float));
        cudaMalloc(&d_candidate_total_cost, sizeof(float));
        cudaMalloc(&d_topk_idx, (size_t)R * sizeof(int));
        cudaMalloc(&d_start, (size_t)D * sizeof(float));
        cudaMalloc(&d_goal, (size_t)D * sizeof(float));
        cudaMalloc(&d_rng_states, (size_t)Rn * D * Tn * sizeof(curandState));
        cudaMalloc(&d_collision, sizeof(int));
        cudaHostAlloc(&h_valid, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_total_cost, sizeof(float), cudaHostAllocDefault);
    }

    // --- Host: Cholesky factor of the (rescaled) control-cost covariance ---
    // Mirrors generate_control_cost_matrix exactly: boundary-TRUNCATED
    // acceleration stencil A (T+2,T), R = A^T A + 1e-8 I, R_inv = R^-1
    // rescaled so max entry == 1/T, resymmetrized, then Cholesky L L^T = R_inv.
    {
        std::vector<float> h_L = compute_cholesky_L(Tn, settings.delta_t, settings.noise_scale);
        cudaMemcpy(d_L, h_L.data(), (size_t)Tn * Tn * sizeof(float), cudaMemcpyHostToDevice);
    }

    // --- Host: update-smoothing projection matrix M (per-column normalised
    // R^-1, from compute_projection_M). Distinct from L's global rescale;
    // applied in stomp_project_kernel to smooth the raw weighted noise average. ---
    {
        std::vector<float> h_M = compute_projection_M(Tn, settings.delta_t);
        cudaMemcpy(d_M, h_M.data(), (size_t)Tn * Tn * sizeof(float), cudaMemcpyHostToDevice);
    }

    // --- Host: raw control-cost matrix R (same 5-point metric as L and M).
    // Per-rollout smoothness cost is evaluated on device as theta^T R theta. ---
    {
        std::vector<float> h_R = compute_control_cost_R(Tn, settings.delta_t);
        cudaMemcpy(d_R, h_R.data(), (size_t)Tn * Tn * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_start, start.data(), (size_t)D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goal, goals[0].data(), (size_t)D * sizeof(float), cudaMemcpyHostToDevice);

    {
        int count = Rn * D * Tn;
        int block = 256;
        int nblk = (count + block - 1) / block;
        stomp_init_rng_kernel<<<nblk, block>>>(d_rng_states, 0x5be5ULL, count);
    }

    // Initial optimum: device joint-space linear interpolation between
    // start and goals[0].
    stomp_init_trajectory_kernel<<<D, Tn>>>(d_start, d_goal, model, d_parameters_optimized, D, Tn);

    // Shared-memory sizing for the per-config FK+cost kernels (P2/P7),
    // exactly mirroring MITStar.cu's compute_node_clearance_kernel sizing.
    size_t smem_state_bytes = (size_t)(model.n_spheres * BATCH_SIZE * 3 + BATCH_SIZE * ppln::FK_T_SLOTS * 16 + D) * sizeof(float);
    // The validity kernel additionally stages the approximate sphere set,
    // because the shared two-phase anchor gates its exact pass on an
    // approximate pass. Sized separately so the cost kernels keep their
    // original (smaller) footprint and their existing offsets.
    size_t smem_validity_bytes = smem_state_bytes
        + (size_t)(model.n_approx_spheres * BATCH_SIZE * 3) * sizeof(float);
    if (smem_state_bytes > 48 * 1024) {
        cudaFuncSetAttribute(stomp_state_cost_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_state_bytes);
        cudaFuncSetAttribute(stomp_validity_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_validity_bytes);
    }

    // Seed the initial optimum's cost breakdown (n_rows=1 reuse of P2/P3).
    stomp_state_cost_kernel<<<Tn, 4, smem_state_bytes>>>(
        d_parameters_optimized, d_parameters_state_costs, model, scene, D, Tn, 1,
        settings.collision_cost_weight, settings.world_collision_margin, settings.self_collision_margin);
    stomp_control_cost_kernel<<<dim3(1, D), Tn>>>(
        d_parameters_optimized, d_parameters_control_costs, d_R, D, Tn, 1,
        settings.control_cost_weight, settings.delta_t);
    stomp_seed_scalar_costs_kernel<<<1, Tn, 2 * Tn * sizeof(float)>>>(
        d_parameters_state_costs, d_parameters_control_costs,
        d_parameters_state_cost, d_parameters_control_cost, d_parameters_total_cost, D, Tn);

    auto check_validity = [&]() -> bool {
        std::vector<float> h_opt((size_t)D * Tn);
        cudaMemcpy(h_opt.data(), d_parameters_optimized, (size_t)D * Tn * sizeof(float), cudaMemcpyDeviceToHost);
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
        stomp_validity_kernel<<<total_checks, 4, smem_validity_bytes>>>(
            d_parameters_optimized, model, scene, D, Tn, n_sub, settings.collision_margin, d_collision);
        cudaMemcpy(h_valid, d_collision, sizeof(int), cudaMemcpyDeviceToHost);
        return (*h_valid) != 0;
    };

    bool valid = check_validity();
    int polish_left = settings.num_iterations_after_valid;
    float prev_total = std::numeric_limits<float>::infinity();
    int converge_streak = 0;
    int iters_run = 0;
    bool init_it = true;

    // Best VALID optimum seen so far (fix #2). Because apply is now
    // unconditional, the working optimum drifts freely and may become worse or
    // collision-prone between iterations; this snapshot is what actually gets
    // returned, so STOMP can keep exploring without losing the best result.
    float best_total = std::numeric_limits<float>::infinity();
    float best_state = 0.0f, best_control = 0.0f;
    bool best_valid = false;
    if (valid) {
        cudaMemcpy(d_best_optimum, d_parameters_optimized, bdt * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&best_state, d_parameters_state_cost, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&best_control, d_parameters_control_cost, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&best_total, d_parameters_total_cost, sizeof(float), cudaMemcpyDeviceToHost);
        best_valid = true;
    }

    for (int iter = 0; iter < settings.num_iterations; iter++) {
        if (best_valid && polish_left <= 0) break;

        // time_limit_ms is inherited from pRRTC_settings; unlike pRRTC's single
        // persistent kernel (clock64()-gated), this loop is host-side with a
        // blocking cudaMemcpy every iteration already, so a steady_clock poll
        // here adds no extra synchronization.
        if (settings.time_limit_ms > 0.0f) {
            float elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - wall_start).count() / 1000.0f;
            if (elapsed_ms > settings.time_limit_ms) break;
        }

        size_t smem_p1 = (size_t)Tn * sizeof(float);
        if (init_it) {
            stomp_sample_noise_kernel<<<dim3(Rn, D), Tn, smem_p1>>>(
                d_rng_states, d_L, d_parameters_optimized, model, d_noise, d_stored_rollouts, D, Tn, 0);
            stomp_sample_noise_kernel<<<dim3(Ro, D), Tn, smem_p1>>>(
                d_rng_states, d_L, d_parameters_optimized, model, d_noise, d_stored_rollouts, D, Tn, Rn);
        } else {
            stomp_sample_noise_kernel<<<dim3(Rn, D), Tn, smem_p1>>>(
                d_rng_states, d_L, d_parameters_optimized, model, d_noise, d_stored_rollouts, D, Tn, 0);

            stomp_select_topk_kernel<<<1, 1, (size_t)R * sizeof(float)>>>(d_prev_total_cost, d_topk_idx, R, Ro);
            stomp_gather_reuse_kernel<<<dim3(Ro, D), Tn>>>(
                d_topk_idx, d_prev_stored_rollouts, d_prev_control_costs, d_prev_state_costs,
                d_parameters_optimized, d_stored_rollouts, d_control_costs, d_state_costs, d_noise, D, Tn, Rn);
        }

        stomp_seed_last_slot_kernel<<<D, Tn>>>(
            d_parameters_optimized, d_parameters_control_costs, d_parameters_state_costs,
            d_stored_rollouts, d_control_costs, d_state_costs, d_noise, D, Tn, R);

        int n_fresh = init_it ? (Rn + Ro) : Rn;
        stomp_state_cost_kernel<<<n_fresh * Tn, 4, smem_state_bytes>>>(
            d_stored_rollouts, d_state_costs, model, scene, D, Tn, n_fresh,
            settings.collision_cost_weight, settings.world_collision_margin, settings.self_collision_margin);
        stomp_control_cost_kernel<<<dim3(n_fresh, D), Tn>>>(
            d_stored_rollouts, d_control_costs, d_R, D, Tn, n_fresh, settings.control_cost_weight, settings.delta_t);

        stomp_reduce_costs_kernel<<<R, Tn, (size_t)Tn * sizeof(float)>>>(
            d_state_costs, d_control_costs, d_total_costs, d_total_cost, D, Tn, R);

        cudaMemcpyAsync(d_prev_stored_rollouts, d_stored_rollouts, brdt * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(d_prev_control_costs, d_control_costs, brdt * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(d_prev_state_costs, d_state_costs, brdt * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(d_prev_total_cost, d_total_cost, br * sizeof(float), cudaMemcpyDeviceToDevice);

        stomp_probabilities_kernel<<<dim3(D, Tn), R, (size_t)(R + 2) * sizeof(float)>>>(
            d_total_costs, d_probabilities, D, Tn, R, settings.exponentiated_cost_sensitivity);

        stomp_update_kernel<<<dim3(D, Tn), R, (size_t)R * sizeof(float)>>>(
            d_noise, d_probabilities, d_delta, D, Tn, R);

        // Smooth the raw weighted noise average through M, then form the
        // candidate optimum = current optimum + (M @ delta). This is the STOMP
        // update-smoothing step (fix #1: previously the raw delta was applied
        // directly, with no M projection).
        stomp_project_kernel<<<D, Tn>>>(
            d_delta, d_M, d_parameters_optimized, d_candidate, D, Tn);

        stomp_state_cost_kernel<<<Tn, 4, smem_state_bytes>>>(
            d_candidate, d_candidate_state_costs, model, scene, D, Tn, 1,
            settings.collision_cost_weight, settings.world_collision_margin, settings.self_collision_margin);
        stomp_control_cost_kernel<<<dim3(1, D), Tn>>>(
            d_candidate, d_candidate_control_costs, d_R, D, Tn, 1, settings.control_cost_weight, settings.delta_t);

        // Unconditionally adopt the candidate as the new working optimum (fix
        // #2: the old accept/reject gate froze the optimum at the first valid
        // trajectory, killing STOMP's exploration). The best VALID trajectory
        // seen is preserved separately by the host-side snapshot below.
        stomp_apply_kernel<<<1, Tn, (size_t)(2 * Tn + 2) * sizeof(float)>>>(
            d_candidate, d_candidate_state_costs, d_candidate_control_costs,
            d_parameters_optimized, d_parameters_state_costs, d_parameters_control_costs,
            d_parameters_state_cost, d_parameters_control_cost, d_parameters_total_cost,
            D, Tn);

        init_it = false;
        iters_run++;

        cudaMemcpy(h_total_cost, d_parameters_total_cost, sizeof(float), cudaMemcpyDeviceToHost);
        float cur_total = *h_total_cost;

        valid = check_validity();
        if (valid && cur_total < best_total) {
            cudaMemcpy(d_best_optimum, d_parameters_optimized, bdt * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&best_state, d_parameters_state_cost, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&best_control, d_parameters_control_cost, sizeof(float), cudaMemcpyDeviceToHost);
            best_total = cur_total;
            best_valid = true;
        }

        // Convergence + polish accounting is driven by the BEST valid cost, not
        // the (now freely drifting) working optimum. Once a valid best exists,
        // run at most num_iterations_after_valid more iterations, stopping early
        // if the best has stalled for two consecutive iterations.
        if (best_valid) {
            float impr = prev_total - best_total;
            converge_streak = (impr >= 0.0f && impr < settings.convergence_eps) ? converge_streak + 1 : 0;
            prev_total = best_total;
            polish_left--;
            if (converge_streak >= 2) break;
        }
    }

    // Return the best VALID trajectory if one was ever found; otherwise fall
    // back to the current working optimum (result.solved reflects which).
    const float* d_result = best_valid ? d_best_optimum : d_parameters_optimized;
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
        cudaMemcpy(&h_final_state, d_parameters_state_cost, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_final_control, d_parameters_control_cost, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_final_total, d_parameters_total_cost, sizeof(float), cudaMemcpyDeviceToHost);
    }

    result.solved = best_valid;
    result.iterations_run = iters_run;
    result.final_state_cost = h_final_state;
    result.final_control_cost = h_final_control;
    result.final_total_cost = h_final_total;

    if (!use_bufs) {
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
    }

    result.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - wall_start).count();
    return result;
}

} // namespace STOMP
