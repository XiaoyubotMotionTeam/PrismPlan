#pragma once

#include <cstdint>

// Host-compatible struct holding raw GPU pointers into the collision tensors.
// Can be included from both .cu and .cpp files.

namespace ppln::collision {

struct SceneCollisionData {
    // --- OBB primitives (one entry per cuboid obstacle) ---
    // dims[i*4..i*4+3]   = [dx, dy, dz, pad]  (full extents, NOT half)
    // pose[i*8..i*8+7]   = [x, y, z, qw, qx, qy, qz, pad]  (INVERSE pose: world->OBB local)
    // enable[i]           = 0 or 1
    float*   obb_dims    = nullptr;
    float*   obb_pose    = nullptr;
    uint8_t* obb_enable  = nullptr;
    int      n_obbs      = 0;

    // --- ESDF voxel grids (one entry per voxel layer) ---
    // params[layer*4..layer*4+3]  = [dx, dy, dz, voxel_size]
    // pose[layer*8..layer*8+7]    = inverse pose (world->grid local)
    // enable[layer]               = 0 or 1
    // features[layer*max_voxels + idx] = signed distance (positive inside obstacle)
    float*   voxel_params    = nullptr;
    float*   voxel_pose      = nullptr;
    uint8_t* voxel_enable    = nullptr;
    float*   voxel_features  = nullptr;
    int      n_voxel_layers       = 0;
    int      max_voxels_per_layer = 0;

    // --- Per-sphere ACM mask for ESDF voxel layers (optional) ---
    // mask[sphere_idx * n_voxel_layers + layer_idx]:  0 = skip, 1 = check.
    // nullptr means no ACM (all spheres check all layers).
    // Indexed by pRRTC sphere order (remapped from tensor order in Python).
    uint8_t* sphere_acm_mask = nullptr;

    // --- Per-sphere ACM mask for OBB obstacles (optional) ---
    // mask[sphere_idx * n_obbs + obb_idx]:  0 = skip, 1 = check.
    // nullptr means no ACM (all spheres check all OBBs).
    uint8_t* sphere_obb_acm_mask = nullptr;

    // --- Per-approx-sphere ACM mask for OBB obstacles (optional) ---
    // mask[approx_sphere_idx * n_obbs + obb_idx]:  0 = skip, 1 = check.
    // nullptr means no ACM (all approx spheres check all OBBs).
    uint8_t* sphere_obb_acm_mask_approx = nullptr;
};

} // namespace ppln::collision
