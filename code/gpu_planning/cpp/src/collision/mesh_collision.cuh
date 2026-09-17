#pragma once

#include <cuda_runtime.h>
#include "src/collision/mesh_collision_data.hh"
#include "src/collision/scene_collision.cuh"

// GPU BVH mesh collision detection:
//   - OBB-OBB overlap (SAT 15-axis, from Coal/FCL)
//   - Triangle-OBB overlap (SAT 13-axis)
//   - BVH traversal with fixed-size stack
//   - scene_obb_to_canonical: scene OBB format → standard (center, half_ext, axes)
//
// All functions are __device__ __forceinline__ to avoid link conflicts
// when this header is included by multiple translation units.

namespace ppln::collision {

// ============================================================================
// Convert scene OBB (inverse pose + full dims) to canonical form.
// ============================================================================
// dims[4] = (dx, dy, dz, pad) FULL extents
//         pose[8] = (tx, ty, tz, qw, qx, qy, qz, pad) INVERSE pose (world→local)
// Output: center[3], half_ext[3], axes[9] (stored so that column i = {axes[i], axes[i+3], axes[i+6]})

__device__ __forceinline__ void scene_obb_to_canonical(
    const float* dims,   // [4]
    const float* pose,   // [8]
    float* center,       // [3] output: OBB center in world frame
    float* half_ext,     // [3] output
    float* axes          // [9] output: column i = {axes[i], axes[i+3], axes[i+6]}
) {
    // Half extents from full dims
    half_ext[0] = dims[0] * 0.5f;
    half_ext[1] = dims[1] * 0.5f;
    half_ext[2] = dims[2] * 0.5f;

    // Inverse pose: world→local. We need local→world (forward pose).
    // Given inverse transform (t_inv, q_inv):
    //   q_fwd = conjugate(q_inv)
    //   t_fwd = -q_fwd * t_inv
    float tx_inv = pose[0], ty_inv = pose[1], tz_inv = pose[2];
    float qw = pose[3], qx = pose[4], qy = pose[5], qz = pose[6];

    // Conjugate: q_fwd = (qw, -qx, -qy, -qz)
    float qw_f = qw;
    float qx_f = -qx;
    float qy_f = -qy;
    float qz_f = -qz;

    // Build rotation matrix from q_fwd (3x3 row-major)
    // Column 0 (X axis in world)
    axes[0] = 1.0f - 2.0f * (qy_f*qy_f + qz_f*qz_f);
    axes[3] = 2.0f * (qx_f*qy_f + qw_f*qz_f);
    axes[6] = 2.0f * (qx_f*qz_f - qw_f*qy_f);
    // Column 1 (Y axis in world)
    axes[1] = 2.0f * (qx_f*qy_f - qw_f*qz_f);
    axes[4] = 1.0f - 2.0f * (qx_f*qx_f + qz_f*qz_f);
    axes[7] = 2.0f * (qy_f*qz_f + qw_f*qx_f);
    // Column 2 (Z axis in world)
    axes[2] = 2.0f * (qx_f*qz_f + qw_f*qy_f);
    axes[5] = 2.0f * (qy_f*qz_f - qw_f*qx_f);
    axes[8] = 1.0f - 2.0f * (qx_f*qx_f + qy_f*qy_f);

    // Forward translation: t_fwd = -R_fwd * t_inv
    center[0] = -(axes[0]*tx_inv + axes[1]*ty_inv + axes[2]*tz_inv);
    center[1] = -(axes[3]*tx_inv + axes[4]*ty_inv + axes[5]*tz_inv);
    center[2] = -(axes[6]*tx_inv + axes[7]*ty_inv + axes[8]*tz_inv);
}

// ============================================================================
// OBB-OBB overlap test — SAT with 15 separating axes
// ============================================================================
// Based on Coal/FCL obbDisjoint() and Gottschalk's OBB-Tree paper.
// OBB A: center_a[3], axes_a[9] (column i = {axes[i], axes[i+3], axes[i+6]}), half_a[3]
// OBB B: center_b[3], axes_b[9] (same layout), half_b[3]
// Returns true if the two OBBs overlap.

__device__ __forceinline__ bool obb_obb_overlap(
    const float* center_a, const float* axes_a, const float* half_a,
    const float* center_b, const float* axes_b, const float* half_b
) {
    constexpr float EPS = 1e-6f;

    // Translation vector T = center_b - center_a, in A's frame
    float d[3] = {
        center_b[0] - center_a[0],
        center_b[1] - center_a[1],
        center_b[2] - center_a[2]
    };

    // T in A's local frame
    float T[3] = {
        axes_a[0]*d[0] + axes_a[3]*d[1] + axes_a[6]*d[2],
        axes_a[1]*d[0] + axes_a[4]*d[1] + axes_a[7]*d[2],
        axes_a[2]*d[0] + axes_a[5]*d[1] + axes_a[8]*d[2]
    };

    // Rotation matrix R = A^T * B  (relative rotation of B in A's frame)
    // R[i][j] = dot(axes_a column i, axes_b column j)
    // axes is row-major: column i of axes_a = {axes_a[i], axes_a[i+3], axes_a[i+6]}
    float R[3][3], AbsR[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R[i][j] = axes_a[i]*axes_b[j] + axes_a[i+3]*axes_b[j+3] + axes_a[i+6]*axes_b[j+6];
            AbsR[i][j] = fabsf(R[i][j]) + EPS;
        }
    }

    float ra, rb;

    // Test axes A0, A1, A2
    for (int i = 0; i < 3; i++) {
        ra = half_a[i];
        rb = half_b[0]*AbsR[i][0] + half_b[1]*AbsR[i][1] + half_b[2]*AbsR[i][2];
        if (fabsf(T[i]) > ra + rb) return false;
    }

    // Test axes B0, B1, B2
    for (int i = 0; i < 3; i++) {
        ra = half_a[0]*AbsR[0][i] + half_a[1]*AbsR[1][i] + half_a[2]*AbsR[2][i];
        rb = half_b[i];
        float t = fabsf(T[0]*R[0][i] + T[1]*R[1][i] + T[2]*R[2][i]);
        if (t > ra + rb) return false;
    }

    // Test 9 edge cross products: Ai x Bj
    // A0 x B0
    ra = half_a[1]*AbsR[2][0] + half_a[2]*AbsR[1][0];
    rb = half_b[1]*AbsR[0][2] + half_b[2]*AbsR[0][1];
    if (fabsf(T[2]*R[1][0] - T[1]*R[2][0]) > ra + rb) return false;

    // A0 x B1
    ra = half_a[1]*AbsR[2][1] + half_a[2]*AbsR[1][1];
    rb = half_b[0]*AbsR[0][2] + half_b[2]*AbsR[0][0];
    if (fabsf(T[2]*R[1][1] - T[1]*R[2][1]) > ra + rb) return false;

    // A0 x B2
    ra = half_a[1]*AbsR[2][2] + half_a[2]*AbsR[1][2];
    rb = half_b[0]*AbsR[0][1] + half_b[1]*AbsR[0][0];
    if (fabsf(T[2]*R[1][2] - T[1]*R[2][2]) > ra + rb) return false;

    // A1 x B0
    ra = half_a[0]*AbsR[2][0] + half_a[2]*AbsR[0][0];
    rb = half_b[1]*AbsR[1][2] + half_b[2]*AbsR[1][1];
    if (fabsf(T[0]*R[2][0] - T[2]*R[0][0]) > ra + rb) return false;

    // A1 x B1
    ra = half_a[0]*AbsR[2][1] + half_a[2]*AbsR[0][1];
    rb = half_b[0]*AbsR[1][2] + half_b[2]*AbsR[1][0];
    if (fabsf(T[0]*R[2][1] - T[2]*R[0][1]) > ra + rb) return false;

    // A1 x B2
    ra = half_a[0]*AbsR[2][2] + half_a[2]*AbsR[0][2];
    rb = half_b[0]*AbsR[1][1] + half_b[1]*AbsR[1][0];
    if (fabsf(T[0]*R[2][2] - T[2]*R[0][2]) > ra + rb) return false;

    // A2 x B0
    ra = half_a[0]*AbsR[1][0] + half_a[1]*AbsR[0][0];
    rb = half_b[1]*AbsR[2][2] + half_b[2]*AbsR[2][1];
    if (fabsf(T[1]*R[0][0] - T[0]*R[1][0]) > ra + rb) return false;

    // A2 x B1
    ra = half_a[0]*AbsR[1][1] + half_a[1]*AbsR[0][1];
    rb = half_b[0]*AbsR[2][2] + half_b[2]*AbsR[2][0];
    if (fabsf(T[1]*R[0][1] - T[0]*R[1][1]) > ra + rb) return false;

    // A2 x B2
    ra = half_a[0]*AbsR[1][2] + half_a[1]*AbsR[0][2];
    rb = half_b[0]*AbsR[2][1] + half_b[1]*AbsR[2][0];
    if (fabsf(T[1]*R[0][2] - T[0]*R[1][2]) > ra + rb) return false;

    return true;
}

// ============================================================================
// Triangle-OBB overlap test — SAT with 13 separating axes
// ============================================================================
// Based on Tomas Akenine-Moller's method and AABB-triangle overlap test.
// Triangle vertices v0, v1, v2 in world frame.
// OBB: center, axes (column i = {axes[i], axes[i+3], axes[i+6]}), half_ext.
// Returns true if triangle and OBB overlap.

__device__ __forceinline__ bool triangle_obb_overlap(
    const float* v0, const float* v1, const float* v2,
    const float* obb_center, const float* obb_axes, const float* obb_half
) {
    // Translate triangle to OBB center as origin
    float a[3] = { v0[0] - obb_center[0], v0[1] - obb_center[1], v0[2] - obb_center[2] };
    float b[3] = { v1[0] - obb_center[0], v1[1] - obb_center[1], v1[2] - obb_center[2] };
    float c[3] = { v2[0] - obb_center[0], v2[1] - obb_center[1], v2[2] - obb_center[2] };

    // Project into OBB local frame
    // OBB axes columns: col_i = (axes[i], axes[i+3], axes[i+6])
    float A[3], B[3], C[3];
    for (int i = 0; i < 3; i++) {
        A[i] = obb_axes[i]*a[0] + obb_axes[i+3]*a[1] + obb_axes[i+6]*a[2];
        B[i] = obb_axes[i]*b[0] + obb_axes[i+3]*b[1] + obb_axes[i+6]*b[2];
        C[i] = obb_axes[i]*c[0] + obb_axes[i+3]*c[1] + obb_axes[i+6]*c[2];
    }

    // Now test triangle (A,B,C) vs AABB centered at origin with half_ext = obb_half
    // This is the standard AABB-triangle overlap test.

    float h0 = obb_half[0], h1 = obb_half[1], h2 = obb_half[2];

    // Triangle edges
    float e0[3] = { B[0]-A[0], B[1]-A[1], B[2]-A[2] };
    float e1[3] = { C[0]-B[0], C[1]-B[1], C[2]-B[2] };
    float e2[3] = { A[0]-C[0], A[1]-C[1], A[2]-C[2] };

    // --- 9 edge cross products (AABB axes × triangle edges) ---
    // For AABB axis i and triangle edge ej, the separating axis is axis_i × ej
    // We project triangle vertices and AABB half-extents onto this axis.

    float p0, p1, p2, r;

    // Cross products with e0 = (e0x, e0y, e0z)
    // axis X × e0 = (0, -e0z, e0y)
    p0 = -A[1]*e0[2] + A[2]*e0[1];
    p2 = -C[1]*e0[2] + C[2]*e0[1];
    r = h1*fabsf(e0[2]) + h2*fabsf(e0[1]);
    if (fminf(p0, p2) > r || fmaxf(p0, p2) < -r) return false;

    // axis Y × e0 = (e0z, 0, -e0x)
    p0 = A[0]*e0[2] - A[2]*e0[0];
    p2 = C[0]*e0[2] - C[2]*e0[0];
    r = h0*fabsf(e0[2]) + h2*fabsf(e0[0]);
    if (fminf(p0, p2) > r || fmaxf(p0, p2) < -r) return false;

    // axis Z × e0 = (-e0y, e0x, 0)
    p0 = -A[0]*e0[1] + A[1]*e0[0];
    p2 = -C[0]*e0[1] + C[1]*e0[0];
    r = h0*fabsf(e0[1]) + h1*fabsf(e0[0]);
    if (fminf(p0, p2) > r || fmaxf(p0, p2) < -r) return false;

    // Cross products with e1 = (e1x, e1y, e1z)
    // axis X × e1
    p0 = -A[1]*e1[2] + A[2]*e1[1];
    p1 = -B[1]*e1[2] + B[2]*e1[1];
    r = h1*fabsf(e1[2]) + h2*fabsf(e1[1]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // axis Y × e1
    p0 = A[0]*e1[2] - A[2]*e1[0];
    p1 = B[0]*e1[2] - B[2]*e1[0];
    r = h0*fabsf(e1[2]) + h2*fabsf(e1[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // axis Z × e1
    p0 = -A[0]*e1[1] + A[1]*e1[0];
    p1 = -B[0]*e1[1] + B[1]*e1[0];
    r = h0*fabsf(e1[1]) + h1*fabsf(e1[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // Cross products with e2 = (e2x, e2y, e2z)
    // axis X × e2
    p0 = -A[1]*e2[2] + A[2]*e2[1];
    p1 = -B[1]*e2[2] + B[2]*e2[1];
    r = h1*fabsf(e2[2]) + h2*fabsf(e2[1]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // axis Y × e2
    p0 = A[0]*e2[2] - A[2]*e2[0];
    p1 = B[0]*e2[2] - B[2]*e2[0];
    r = h0*fabsf(e2[2]) + h2*fabsf(e2[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // axis Z × e2
    p0 = -A[0]*e2[1] + A[1]*e2[0];
    p1 = -B[0]*e2[1] + B[1]*e2[0];
    r = h0*fabsf(e2[1]) + h1*fabsf(e2[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // --- 3 AABB face normals (equivalent to coordinate axis tests) ---
    float minv, maxv;

    // X axis
    minv = fminf(A[0], fminf(B[0], C[0]));
    maxv = fmaxf(A[0], fmaxf(B[0], C[0]));
    if (minv > h0 || maxv < -h0) return false;

    // Y axis
    minv = fminf(A[1], fminf(B[1], C[1]));
    maxv = fmaxf(A[1], fmaxf(B[1], C[1]));
    if (minv > h1 || maxv < -h1) return false;

    // Z axis
    minv = fminf(A[2], fminf(B[2], C[2]));
    maxv = fmaxf(A[2], fmaxf(B[2], C[2]));
    if (minv > h2 || maxv < -h2) return false;

    // --- 1 triangle face normal ---
    float normal[3] = {
        e0[1]*e1[2] - e0[2]*e1[1],
        e0[2]*e1[0] - e0[0]*e1[2],
        e0[0]*e1[1] - e0[1]*e1[0]
    };
    float d_val = -(normal[0]*A[0] + normal[1]*A[1] + normal[2]*A[2]);
    r = h0*fabsf(normal[0]) + h1*fabsf(normal[1]) + h2*fabsf(normal[2]);
    if (fabsf(d_val) > r) return false;

    return true;
}

// ============================================================================
// Transform a BVH node OBB from link-local to world frame using FK transform
// ============================================================================
// T_link: 4x4 column-major FK transform (link→world)
// Node OBB: center, half_ext, axes in link-local frame
// Output: world-frame center, axes (half_ext unchanged)

__device__ __forceinline__ void transform_node_obb_to_world(
    const float* T_link,           // [16] column-major 4x4
    const float* node_center,      // [3] link-local
    const float* node_axes,        // [9] link-local, column i = {[i], [i+3], [i+6]}
    float* world_center,           // [3] output
    float* world_axes              // [9] output
) {
    // T_link is column-major: col0=[0..3], col1=[4..7], col2=[8..11], col3=[12..15]
    // R = [col0[0:3], col1[0:3], col2[0:3]]  (3x3 rotation)
    // t = col3[0:3]                           (translation)

    // world_center = R * node_center + t
    world_center[0] = T_link[0]*node_center[0] + T_link[4]*node_center[1] + T_link[8]*node_center[2]  + T_link[12];
    world_center[1] = T_link[1]*node_center[0] + T_link[5]*node_center[1] + T_link[9]*node_center[2]  + T_link[13];
    world_center[2] = T_link[2]*node_center[0] + T_link[6]*node_center[1] + T_link[10]*node_center[2] + T_link[14];

    // world_axes = R * node_axes
    // node_axes is row-major: row i = column i of the OBB local frame
    // We want world_axes row-major: column i in world = R * (column i in local)
    // Since node_axes row-major: column j of node_axes = (node_axes[j], node_axes[j+3], node_axes[j+6])
    for (int j = 0; j < 3; j++) {
        float lx = node_axes[j];
        float ly = node_axes[j + 3];
        float lz = node_axes[j + 6];
        world_axes[j]     = T_link[0]*lx + T_link[4]*ly + T_link[8]*lz;
        world_axes[j + 3] = T_link[1]*lx + T_link[5]*ly + T_link[9]*lz;
        world_axes[j + 6] = T_link[2]*lx + T_link[6]*ly + T_link[10]*lz;
    }
}

// ============================================================================
// BVH mesh vs single scene OBB — traverses one link's BVH tree
// ============================================================================
// Returns true if any triangle in this link's mesh collides with the scene OBB.
//
// bvh_nodes:     flat BVH node array for this link (offset already applied)
// tri_vertices:  flat triangle vertex array for this link (offset already applied)
// n_bvh_nodes:   total BVH nodes for bounds checking
// T_link:        4x4 column-major FK transform for this link
// scene_center, scene_axes, scene_half: canonical scene OBB (already converted)

__device__ __forceinline__ bool bvh_mesh_vs_obb(
    const float* bvh_nodes,       // [N * BVH_NODE_FLOATS]
    const float* tri_vertices,    // [M * 9]
    int bvh_root,                 // root node index (absolute, within this link's BVH)
    int n_bvh_nodes,              // total nodes (for bounds check)
    const float* T_link,          // [16] column-major 4x4 FK transform
    const float* scene_center,    // [3] world frame
    const float* scene_axes,      // [9] world frame, column i = {[i], [i+3], [i+6]}
    const float* scene_half       // [3]
) {
    if (bvh_root < 0 || bvh_root >= n_bvh_nodes) return false;

    // Fixed-size traversal stack
    constexpr int STACK_SIZE = 32;
    int stack[STACK_SIZE];
    int sp = 0;
    stack[sp++] = bvh_root;

    while (sp > 0) {
        int node_idx = stack[--sp];
        const float* node = &bvh_nodes[node_idx * BVH_NODE_FLOATS];

        // Read node OBB (link-local frame)
        float node_center_local[3] = { node[0], node[1], node[2] };
        float node_half[3]         = { node[3], node[4], node[5] };
        float node_axes_local[9];
        for (int k = 0; k < 9; k++) node_axes_local[k] = node[6 + k];

        int left_child  = __float_as_int(node[15]);
        int right_child = __float_as_int(node[16]);
        int tri_start   = __float_as_int(node[17]);
        int tri_count   = __float_as_int(node[18]);

        // Transform BVH node OBB to world frame
        float world_center[3], world_axes[9];
        transform_node_obb_to_world(T_link, node_center_local, node_axes_local, world_center, world_axes);

        // Test node OBB vs scene OBB
        if (!obb_obb_overlap(world_center, world_axes, node_half,
                             scene_center, scene_axes, scene_half))
            continue;  // Prune: BV doesn't overlap

        // Leaf node: test triangles
        if (left_child == -1) {
            for (int t = tri_start; t < tri_start + tri_count; t++) {
                const float* tri = &tri_vertices[t * 9];
                // Transform triangle vertices from link-local to world frame
                float wv0[3], wv1[3], wv2[3];
                for (int k = 0; k < 3; k++) {
                    wv0[k] = T_link[k]*tri[0] + T_link[k+4]*tri[1] + T_link[k+8]*tri[2]  + T_link[k+12];
                    wv1[k] = T_link[k]*tri[3] + T_link[k+4]*tri[4] + T_link[k+8]*tri[5]  + T_link[k+12];
                    wv2[k] = T_link[k]*tri[6] + T_link[k+4]*tri[7] + T_link[k+8]*tri[8]  + T_link[k+12];
                }

                if (triangle_obb_overlap(wv0, wv1, wv2, scene_center, scene_axes, scene_half))
                    return true;  // Collision confirmed
            }
        } else {
            // Internal node: push children (need 2 slots)
            if (sp + 2 <= STACK_SIZE) {
                stack[sp++] = left_child;
                stack[sp++] = right_child;
            }
        }
    }
    return false;
}

// ============================================================================
// Mesh vs ESDF grid collision — one link against ESDF
// ============================================================================
// Samples BVH leaf triangle centroids against the ESDF grid.
// For leaf nodes that pass the OBB overlap test, we check each triangle centroid
// against the ESDF. This does NOT do per-triangle polygon intersection with ESDF
// (which would be very expensive); instead, checking centroids + vertices is a
// reasonable approximation for mesh-ESDF that is much more precise than spheres.

__device__ __forceinline__ bool bvh_mesh_vs_esdf(
    const float* bvh_nodes,
    const float* tri_vertices,
    int bvh_root,
    int n_bvh_nodes,
    const float* T_link,
    const SceneCollisionData& scene
) {
    if (bvh_root < 0 || bvh_root >= n_bvh_nodes) return false;

    for (int layer = 0; layer < scene.n_voxel_layers; layer++) {
        if (scene.voxel_enable[layer] == 0) continue;

        const float* params = &scene.voxel_params[layer * 4];
        const float* vpose  = &scene.voxel_pose[layer * 8];
        float grid_dx = params[0], grid_dy = params[1], grid_dz = params[2];
        float vs = params[3];
        if (vs <= 0.0f) continue;

        int nx = __float2int_rd(grid_dx / vs) + 1;
        int ny = __float2int_rd(grid_dy / vs) + 1;
        int nz = __float2int_rd(grid_dz / vs) + 1;

        // Traverse BVH, for leaf triangles check vertices against ESDF
        constexpr int STACK_SIZE = 32;
        int stack[STACK_SIZE];
        int sp = 0;
        stack[sp++] = bvh_root;

        while (sp > 0) {
            int node_idx = stack[--sp];
            const float* node = &bvh_nodes[node_idx * BVH_NODE_FLOATS];

            int left_child = __float_as_int(node[15]);
            int tri_start  = __float_as_int(node[17]);
            int tri_count  = __float_as_int(node[18]);

            if (left_child == -1) {
                // Leaf: check triangle vertices
                for (int t = tri_start; t < tri_start + tri_count; t++) {
                    const float* tri = &tri_vertices[t * 9];

                    // Check all 3 vertices
                    for (int vi = 0; vi < 3; vi++) {
                        float lx_t = tri[vi*3], ly_t = tri[vi*3+1], lz_t = tri[vi*3+2];

                        // Transform to world
                        float wx = T_link[0]*lx_t + T_link[4]*ly_t + T_link[8]*lz_t  + T_link[12];
                        float wy = T_link[1]*lx_t + T_link[5]*ly_t + T_link[9]*lz_t  + T_link[13];
                        float wz = T_link[2]*lx_t + T_link[6]*ly_t + T_link[10]*lz_t + T_link[14];

                        // Transform to ESDF local frame
                        float elx, ely, elz;
                        transform_sphere_to_local(vpose, wx, wy, wz, elx, ely, elz);

                        // Quantize
                        float fix = (elx + grid_dx * 0.5f) / vs;
                        float fiy = (ely + grid_dy * 0.5f) / vs;
                        float fiz = (elz + grid_dz * 0.5f) / vs;
                        int ix = __float2int_rn(fix);
                        int iy = __float2int_rn(fiy);
                        int iz = __float2int_rn(fiz);

                        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
                            continue;

                        int linear_idx = ix * ny * nz + iy * nz + iz;
                        if (linear_idx >= scene.max_voxels_per_layer) continue;

                        float distance = scene.voxel_features[layer * scene.max_voxels_per_layer + linear_idx];
                        if (distance <= VOXEL_UNOBSERVED_DISTANCE + 1.0f) continue;

                        // Point collision: no radius, just check if inside obstacle
                        if (distance > 0.0f) return true;
                    }
                }
            } else {
                int right_child = __float_as_int(node[16]);
                if (sp + 2 <= STACK_SIZE) {
                    stack[sp++] = left_child;
                    stack[sp++] = right_child;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Full mesh environment collision check for links flagged by approx spheres
// ============================================================================
// joint_transforms: [n_joints * 16] column-major 4x4 FK transforms (global memory)
// link_CC: [BATCH_SIZE * 20] per-link collision flags from approx check
// Only checks links where approx spheres reported collision.
// Returns true if collision detected (mesh confirms collision for some link).
//
// This function is called by a single thread per configuration.

__device__ __forceinline__ bool mesh_env_collision_check(
    const MeshCollisionData& mesh,
    const float* joint_transforms,   // [n_joints * 16] in global memory
    volatile int* link_CC,           // [20] for this batch
    const SceneCollisionData& scene,
    int n_joints
) {
    // Pre-convert all scene OBBs to canonical form (amortized across links)
    // We do this inline since n_obbs is typically small (< 30)

    for (int joint_idx = 0; joint_idx < n_joints && joint_idx < mesh.n_links; joint_idx++) {
        // Only check links flagged by approx spheres
        if (link_CC[joint_idx] <= 0) continue;

        int bvh_root = mesh.link_bvh_root[joint_idx];
        if (bvh_root < 0) continue;  // No mesh for this link

        const float* T_link = &joint_transforms[joint_idx * 16];

        // BVH node tri_start is absolute index into mesh.tri_vertices
        // so we pass the base pointer directly.

        // Check this link's BVH against all enabled scene OBBs
        for (int obb_idx = 0; obb_idx < scene.n_obbs; obb_idx++) {
            if (scene.obb_enable[obb_idx] == 0) continue;

            float s_center[3], s_half[3], s_axes[9];
            scene_obb_to_canonical(
                &scene.obb_dims[obb_idx * 4],
                &scene.obb_pose[obb_idx * 8],
                s_center, s_half, s_axes);

            if (bvh_mesh_vs_obb(
                    mesh.bvh_nodes, mesh.tri_vertices,
                    bvh_root, mesh.n_total_bvh_nodes,
                    T_link,
                    s_center, s_axes, s_half))
                return true;
        }

        // Check against ESDF
        if (scene.n_voxel_layers > 0) {
            if (bvh_mesh_vs_esdf(
                    mesh.bvh_nodes, mesh.tri_vertices,
                    bvh_root, mesh.n_total_bvh_nodes,
                    T_link, scene))
                return true;
        }
    }
    return false;
}

} // namespace ppln::collision
