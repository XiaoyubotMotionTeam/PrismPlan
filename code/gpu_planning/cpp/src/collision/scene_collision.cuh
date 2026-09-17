#pragma once

#include <cuda_runtime.h>
#include "src/planning/robot_model.cuh"
#include "src/collision/scene_collision_data.hh"

// BATCH_SIZE needed by scene_collision_check_runtime / approx.
// utils.cuh defines it with #ifndef guard so robot headers' prior #define is safe.
#ifndef BATCH_SIZE
#define BATCH_SIZE 16
#endif

namespace ppln::collision {

// ---------------------------------------------------------------------------
// Quaternion transform: world → local via inverse pose
// ---------------------------------------------------------------------------
// pose = [tx, ty, tz, qw, qx, qy, qz, pad]  (the INVERSE pose: b_T_w)
// Transform convention (transform_sphere_quat):
//   p_local = quat_rotate(q, p_world) + t
// This is equivalent to the affine transform  p_local = R * p_world + t
// where R = rotation_matrix(q) and t is the inverse-pose translation.

__device__ __forceinline__ void transform_sphere_to_local(
    const float* pose,  // [8]
    float sx, float sy, float sz,
    float& lx, float& ly, float& lz)
{
    // Quaternion: qw, qx, qy, qz
    float qw = pose[3];
    float qx = pose[4];
    float qy = pose[5];
    float qz = pose[6];

    // Rotate by quaternion: v' = q * v * q_conj
    // Optimized formula (no temp quaternion):
    //   t = 2 * cross(q.xyz, v)
    //   v' = v + qw * t + cross(q.xyz, t)
    float tx = 2.0f * (qy * sz - qz * sy);
    float ty = 2.0f * (qz * sx - qx * sz);
    float tz = 2.0f * (qx * sy - qy * sx);

    float rx = sx + qw * tx + (qy * tz - qz * ty);
    float ry = sy + qw * ty + (qz * tx - qx * tz);
    float rz = sz + qw * tz + (qx * ty - qy * tx);

    // Add inverse-pose translation
    lx = rx + pose[0];
    ly = ry + pose[1];
    lz = rz + pose[2];
}

// ---------------------------------------------------------------------------
// Sphere vs single OBB (in OBB-local frame)
// ---------------------------------------------------------------------------
// half = dims / 2.  Computes squared distance from sphere center to OBB surface.
// Returns true if distance < sphere_radius.

__device__ __forceinline__ bool sphere_single_obb_collision(
    const float* dims,   // [4]: dx, dy, dz, pad (full extents)
    const float* pose,   // [8]: inverse pose
    float sx, float sy, float sz, float sr)
{
    float lx, ly, lz;
    transform_sphere_to_local(pose, sx, sy, sz, lx, ly, lz);

    float hx = dims[0] * 0.5f;
    float hy = dims[1] * 0.5f;
    float hz = dims[2] * 0.5f;

    // Signed distance components (0 if inside)
    float cx = fmaxf(0.0f, fabsf(lx) - hx);
    float cy = fmaxf(0.0f, fabsf(ly) - hy);
    float cz = fmaxf(0.0f, fabsf(lz) - hz);

    float dist_sq = cx * cx + cy * cy + cz * cz;
    return dist_sq < (sr * sr);
}

// ---------------------------------------------------------------------------
// Sphere vs single OBB: signed clearance (analytic box SDF)
// ---------------------------------------------------------------------------
// Standard box signed-distance formula (Inigo Quilez), evaluated in the same
// OBB-local frame as sphere_single_obb_collision.  Positive = free (distance
// to nearest surface), negative = penetration depth — matching the sign
// convention already used by sphere_esdf_clearance.

__device__ __forceinline__ float sphere_single_obb_clearance(
    const float* dims,   // [4]: dx, dy, dz, pad (full extents)
    const float* pose,   // [8]: inverse pose
    float sx, float sy, float sz, float sr)
{
    float lx, ly, lz;
    transform_sphere_to_local(pose, sx, sy, sz, lx, ly, lz);

    float hx = dims[0] * 0.5f;
    float hy = dims[1] * 0.5f;
    float hz = dims[2] * 0.5f;

    float qx = fabsf(lx) - hx;
    float qy = fabsf(ly) - hy;
    float qz = fabsf(lz) - hz;

    float ox = fmaxf(qx, 0.0f);
    float oy = fmaxf(qy, 0.0f);
    float oz = fmaxf(qz, 0.0f);
    float outside = sqrtf(ox * ox + oy * oy + oz * oz);
    // Center-inside-box penetration depth (negative); zero when outside.
    float inside = fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f);

    return (outside + inside) - sr;
}

// ---------------------------------------------------------------------------
// Sphere vs all enabled OBBs
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool sphere_obb_in_collision(
    const SceneCollisionData& scene,
    float sx, float sy, float sz, float sr,
    int sphere_idx = -1,
    const uint8_t* obb_acm = nullptr)
{
    for (int i = 0; i < scene.n_obbs; i++) {
        if (scene.obb_enable[i] == 0) continue;
        // Per-sphere OBB ACM: skip if mask says 0
        if (sphere_idx >= 0 && obb_acm != nullptr &&
            obb_acm[sphere_idx * scene.n_obbs + i] == 0)
            continue;
        if (sphere_single_obb_collision(
                &scene.obb_dims[i * 4],
                &scene.obb_pose[i * 8],
                sx, sy, sz, sr)) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Sphere OBB clearance (all enabled OBBs, MIN reduction)
// ---------------------------------------------------------------------------
// Same ACM-gated iteration as sphere_obb_in_collision, but returns the
// minimum signed clearance across all enabled OBBs instead of a boolean.
// Positive = free, negative = penetration.  Returns large value (1000.0f)
// when no OBB is enabled/relevant, matching sphere_esdf_clearance.

__device__ __forceinline__ float sphere_obb_clearance(
    const SceneCollisionData& scene,
    float sx, float sy, float sz, float sr,
    int sphere_idx = -1,
    const uint8_t* obb_acm = nullptr)
{
    float min_clearance = 1000.0f;
    for (int i = 0; i < scene.n_obbs; i++) {
        if (scene.obb_enable[i] == 0) continue;
        if (sphere_idx >= 0 && obb_acm != nullptr &&
            obb_acm[sphere_idx * scene.n_obbs + i] == 0)
            continue;
        float c = sphere_single_obb_clearance(
            &scene.obb_dims[i * 4],
            &scene.obb_pose[i * 8],
            sx, sy, sz, sr);
        min_clearance = fminf(min_clearance, c);
    }
    return min_clearance;
}

// ---------------------------------------------------------------------------
// Sphere vs ESDF grid (all enabled layers)
// ---------------------------------------------------------------------------
// For each enabled voxel layer:
//   1. Transform sphere to grid-local coords
//   2. Quantize to voxel index
//   3. Read signed distance from features
//   4. Collision when distance + sphere_radius > 0
//
// ESDF sign convention (opposite of standard SDF):
//   - Positive = inside obstacle (penetration depth)
//   - Negative = free space (distance to nearest surface)
//   - -1000.0  = unobserved / out-of-bounds sentinel
//   - Grid origin is at (-dx/2, -dy/2, -dz/2) in local frame
//   - Linear index: ix * ny * nz + iy * nz + iz   (X-major)

// Unobserved voxel sentinel
constexpr float VOXEL_UNOBSERVED_DISTANCE = -1000.0f;

// robust_floor: when a float is within
// threshold of an integer, snap to that integer instead of flooring down.
// Prevents off-by-one grid dimensions from floating-point imprecision
// (e.g. grid_d/vs = 501.9999 → floor gives 501, robust_floor gives 502).
__device__ __forceinline__ int robust_floor_scalar(float x, float threshold = 1e-4f)
{
    float nearest = roundf(x);
    float diff = x - nearest;  // signed, NOT abs — matches the sign convention
    if (diff >= threshold) {
        return __float2int_rd(x);  // clearly above nearest int → floor
    }
    return __float2int_rn(nearest); // close to or below nearest int → round
}

__device__ __forceinline__ bool sphere_esdf_in_collision(
    const SceneCollisionData& scene,
    float sx, float sy, float sz, float sr,
    int sphere_idx = -1)
{
    for (int layer = 0; layer < scene.n_voxel_layers; layer++) {
        if (scene.voxel_enable[layer] == 0) continue;

        // Per-sphere per-layer ACM: skip if mask says 0 (sphere_acm_mask)
        if (sphere_idx >= 0 && scene.sphere_acm_mask != nullptr &&
            scene.sphere_acm_mask[sphere_idx * scene.n_voxel_layers + layer] == 0)
            continue;

        const float* params = &scene.voxel_params[layer * 4];
        const float* pose   = &scene.voxel_pose[layer * 8];

        float grid_dx = params[0];
        float grid_dy = params[1];
        float grid_dz = params[2];
        float vs      = params[3];  // voxel_size

        if (vs <= 0.0f) continue;

        // Transform sphere to grid-local coords
        float lx, ly, lz;
        transform_sphere_to_local(pose, sx, sy, sz, lx, ly, lz);

        float inv_vs = 1.0f / vs;

        // Grid dimensions (robust_floor(grid_d / vs) + 1)
        int nx = robust_floor_scalar(grid_dx * inv_vs) + 1;
        int ny = robust_floor_scalar(grid_dy * inv_vs) + 1;
        int nz = robust_floor_scalar(grid_dz * inv_vs) + 1;

        // Voxel index via C-style truncation (compute_voxel_index)
        // Grid origin at (-dx/2, -dy/2, -dz/2)
        int ix = (int)((lx + grid_dx * 0.5f) * inv_vs);
        int iy = (int)((ly + grid_dy * 0.5f) * inv_vs);
        int iz = (int)((lz + grid_dz * 0.5f) * inv_vs);

        // Bounds check with offset=2 guard (boundary protection; offset=2
        // ensures safe finite-difference gradient lookups
        // on neighbouring voxels, and treats edge voxels as unobserved/free)
        constexpr int BOUNDARY_OFFSET = 2;
        if (ix >= nx - BOUNDARY_OFFSET || iy >= ny - BOUNDARY_OFFSET || iz >= nz - BOUNDARY_OFFSET
            || ix <= BOUNDARY_OFFSET || iy <= BOUNDARY_OFFSET || iz <= BOUNDARY_OFFSET)
            continue;

        int linear_idx = ix * ny * nz + iy * nz + iz;
        if (linear_idx >= scene.max_voxels_per_layer) continue;

        float distance = scene.voxel_features[layer * scene.max_voxels_per_layer + linear_idx];

        // Skip unobserved voxels (sentinel = -1000.0)
        if (distance <= VOXEL_UNOBSERVED_DISTANCE + 1.0f) continue;

        // Sign convention: positive = inside obstacle.
        // Collision when: distance + sphere_radius > 0
        //   i.e. the sphere surface penetrates into the occupied region.
        if (distance > -sr) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Sphere ESDF clearance (distance to nearest obstacle surface)
// ---------------------------------------------------------------------------
// Same grid lookup as sphere_esdf_in_collision but returns the minimum
// signed clearance across all enabled voxel layers.
// Positive = free (distance to nearest surface), negative = penetration.
// Returns large value (1000.0f) for unobserved/out-of-bounds/no layers.

__device__ __forceinline__ float sphere_esdf_clearance(
    const SceneCollisionData& scene,
    float sx, float sy, float sz, float sr,
    int sphere_idx = -1)
{
    float min_clearance = 1000.0f;

    for (int layer = 0; layer < scene.n_voxel_layers; layer++) {
        if (scene.voxel_enable[layer] == 0) continue;

        if (sphere_idx >= 0 && scene.sphere_acm_mask != nullptr &&
            scene.sphere_acm_mask[sphere_idx * scene.n_voxel_layers + layer] == 0)
            continue;

        const float* params = &scene.voxel_params[layer * 4];
        const float* pose   = &scene.voxel_pose[layer * 8];

        float grid_dx = params[0];
        float grid_dy = params[1];
        float grid_dz = params[2];
        float vs      = params[3];

        if (vs <= 0.0f) continue;

        float lx, ly, lz;
        transform_sphere_to_local(pose, sx, sy, sz, lx, ly, lz);

        float inv_vs = 1.0f / vs;

        int nx = robust_floor_scalar(grid_dx * inv_vs) + 1;
        int ny = robust_floor_scalar(grid_dy * inv_vs) + 1;
        int nz = robust_floor_scalar(grid_dz * inv_vs) + 1;

        int ix = (int)((lx + grid_dx * 0.5f) * inv_vs);
        int iy = (int)((ly + grid_dy * 0.5f) * inv_vs);
        int iz = (int)((lz + grid_dz * 0.5f) * inv_vs);

        constexpr int BOUNDARY_OFFSET = 2;
        if (ix >= nx - BOUNDARY_OFFSET || iy >= ny - BOUNDARY_OFFSET || iz >= nz - BOUNDARY_OFFSET
            || ix <= BOUNDARY_OFFSET || iy <= BOUNDARY_OFFSET || iz <= BOUNDARY_OFFSET)
            continue;

        int linear_idx = ix * ny * nz + iy * nz + iz;
        if (linear_idx >= scene.max_voxels_per_layer) continue;

        float distance = scene.voxel_features[layer * scene.max_voxels_per_layer + linear_idx];

        if (distance <= VOXEL_UNOBSERVED_DISTANCE + 1.0f) continue;

        // Sign convention: positive = inside obstacle.
        // clearance = -(distance) - sr; positive when free.
        float clearance = -(distance) - sr;
        min_clearance = fminf(min_clearance, clearance);
    }
    return min_clearance;
}

// ---------------------------------------------------------------------------
// Full robot ESDF clearance (4-thread cooperative, like scene_collision_check_runtime)
// ---------------------------------------------------------------------------
// Returns min clearance across all robot spheres.
// Positive = free, negative = penetration.

__device__ float scene_min_clearance_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos,
    const SceneCollisionData& scene,
    const int tid,
    float collision_margin = 0.0f)
{
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    int ns = model.n_spheres;
    int rem = ns % 4;
    float my_min = 1000.0f;

    // Iterate spheres in 4-way stripes (end to start, matching CC pattern)
    for (int i = ns - 1 - thread_ind; i >= rem; i -= 4) {
        float c = sphere_esdf_clearance(
            scene,
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            model.spheres[i].w + collision_margin,
            i);
        my_min = fminf(my_min, c);
    }

    // Handle remainder
    if (thread_ind < rem) {
        int i = thread_ind;
        float c = sphere_esdf_clearance(
            scene,
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            model.spheres[i].w + collision_margin,
            i);
        my_min = fminf(my_min, c);
    }

    // Reduce across 4 threads via warp shuffle
    for (int offset = 2; offset >= 1; offset >>= 1) {
        float other = __shfl_xor_sync(0xf, my_min, offset);
        my_min = fminf(my_min, other);
    }

    return my_min;
}

// ---------------------------------------------------------------------------
// Full robot ESDF sum-hinge penalty (4-thread cooperative)
// ---------------------------------------------------------------------------
// Sum over spheres of the hinge penalty relu(world_margin - clearance).
// scene_min_clearance_runtime is MIN-only and cannot reproduce STOMP's
// smooth per-sphere world-collision cost, which sums hinge penalties across
// all spheres rather than taking their minimum.

__device__ float scene_sum_esdf_hinge_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos,
    const SceneCollisionData& scene,
    const int tid,
    float world_margin)
{
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    int ns = model.n_spheres;
    int rem = ns % 4;
    float my_sum = 0.0f;

    // Iterate spheres in 4-way stripes (end to start, matching CC pattern)
    for (int i = ns - 1 - thread_ind; i >= rem; i -= 4) {
        float c = sphere_esdf_clearance(
            scene,
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            model.spheres[i].w,
            i);
        my_sum += fmaxf(0.0f, world_margin - c);
    }

    // Handle remainder
    if (thread_ind < rem) {
        int i = thread_ind;
        float c = sphere_esdf_clearance(
            scene,
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            model.spheres[i].w,
            i);
        my_sum += fmaxf(0.0f, world_margin - c);
    }

    // Reduce across 4 threads via warp shuffle (sum, not min)
    for (int offset = 2; offset >= 1; offset >>= 1) {
        my_sum += __shfl_xor_sync(0xf, my_sum, offset);
    }

    return my_sum;
}

// ---------------------------------------------------------------------------
// Full robot OBB sum-hinge penalty (4-thread cooperative)
// ---------------------------------------------------------------------------
// Sum over spheres of the hinge penalty relu(world_margin - obb_clearance).
// Mirrors scene_sum_esdf_hinge_runtime exactly, but against
// sphere_obb_clearance instead of sphere_esdf_clearance — gives STOMP's
// smooth per-timestep world-collision cost a gradient signal for OBB
// (cuboid) obstacles, which previously only had a binary validity gate.

__device__ float scene_sum_obb_hinge_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos,
    const SceneCollisionData& scene,
    const int tid,
    float world_margin)
{
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    int ns = model.n_spheres;
    int rem = ns % 4;
    float my_sum = 0.0f;

    // Iterate spheres in 4-way stripes (end to start, matching CC pattern)
    for (int i = ns - 1 - thread_ind; i >= rem; i -= 4) {
        float c = sphere_obb_clearance(
            scene,
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            model.spheres[i].w,
            i,
            scene.sphere_obb_acm_mask);
        my_sum += fmaxf(0.0f, world_margin - c);
    }

    // Handle remainder
    if (thread_ind < rem) {
        int i = thread_ind;
        float c = sphere_obb_clearance(
            scene,
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            model.spheres[i].w,
            i,
            scene.sphere_obb_acm_mask);
        my_sum += fmaxf(0.0f, world_margin - c);
    }

    // Reduce across 4 threads via warp shuffle (sum, not min)
    for (int offset = 2; offset >= 1; offset >>= 1) {
        my_sum += __shfl_xor_sync(0xf, my_sum, offset);
    }

    return my_sum;
}

// ---------------------------------------------------------------------------
// Unified: sphere vs scene (OBBs + ESDF)
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool sphere_scene_in_collision(
    const SceneCollisionData& scene,
    float sx, float sy, float sz, float sr,
    int sphere_idx = -1,
    const uint8_t* obb_acm = nullptr)
{
    if (sphere_obb_in_collision(scene, sx, sy, sz, sr, sphere_idx, obb_acm)) return true;
    if (sphere_esdf_in_collision(scene, sx, sy, sz, sr, sphere_idx)) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Full environment collision check with joint-level early exit (scene version)
// ---------------------------------------------------------------------------
// Same structure as env_collision_check_runtime in panda.cuh, but uses
// sphere_scene_in_collision instead of sphere_environment_in_collision.
// 4 threads cooperate per configuration.

__device__ bool scene_collision_check_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos,
    volatile int* joint_in_collision,
    const SceneCollisionData& scene,
    const int tid,
    float collision_margin = 0.0f)
{
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision = false;
    int ns = model.n_spheres;
    int rem = ns % 4;

    // Iterate spheres from end to start (4-way striped), matching existing pattern
    for (int i = ns - 1 - thread_ind; i >= rem; i -= 4) {
        if (joint_in_collision[20 * batch_ind + model.sphere_to_joint[i]] > 0 &&
            sphere_scene_in_collision(
                scene,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                model.spheres[i].w + collision_margin,
                i,
                scene.sphere_obb_acm_mask))
        {
            has_collision = true;
        }
        if (__any_sync(0xffffffff, has_collision)) return false;
    }

    // Handle remainder
    if (thread_ind < rem) {
        int i = thread_ind;
        if (joint_in_collision[20 * batch_ind + model.sphere_to_joint[i]] > 0 &&
            sphere_scene_in_collision(
                scene,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                model.spheres[i].w + collision_margin,
                i,
                scene.sphere_obb_acm_mask))
        {
            has_collision = true;
        }
    }
    return !has_collision;
}

// ---------------------------------------------------------------------------
// Approximate environment collision check (scene version)
// ---------------------------------------------------------------------------
// Same structure as env_collision_check_approx_runtime in panda.cuh.
//
// NOTE: sphere_idx is passed as -1 to disable per-sphere ESDF ACM lookup.
// The ESDF ACM mask (scene.sphere_acm_mask) is indexed by full-sphere index,
// NOT approx-sphere index.  Passing the approx index `i` would read the
// wrong ACM entry (e.g. approx sphere 6 ≠ full sphere 6) and could
// incorrectly exempt entire joints from ESDF layers.  Since the approx
// check only gates the detailed full-sphere check (which has correct ACM),
// disabling ESDF ACM here is conservative and safe.
// OBB ACM (sphere_obb_acm_mask_approx) IS correctly sized for approx
// spheres and is passed as-is.

__device__ bool scene_collision_check_approx_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos_approx,
    volatile int* joint_in_collision,
    const SceneCollisionData& scene,
    const int tid,
    float collision_margin = 0.0f)
{
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;
    int na = model.n_approx_spheres;
    int chunk = na / 4;

    for (int i = chunk * thread_ind; i < chunk * (thread_ind + 1) && i < na; i++) {
        if (sphere_scene_in_collision(
                scene,
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                model.approx_spheres[i].w + collision_margin,
                -1,
                scene.sphere_obb_acm_mask_approx))
        {
            atomicAdd((int*)&joint_in_collision[20 * batch_ind + model.approx_sphere_to_joint[i]], 1);
            out = false;
        }
    }

    // Handle remaining spheres
    int handled = chunk * 4;
    if (handled + thread_ind < na) {
        int i = handled + thread_ind;
        if (sphere_scene_in_collision(
                scene,
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                model.approx_spheres[i].w + collision_margin,
                -1,
                scene.sphere_obb_acm_mask_approx))
        {
            atomicAdd((int*)&joint_in_collision[20 * batch_ind + model.approx_sphere_to_joint[i]], 1);
            out = false;
        }
    }
    return out;
}

} // namespace ppln::collision
