#pragma once

#include <cstdint>

// Host-compatible struct holding GPU pointers for BVH mesh collision data.
// Can be included from both .cu and .cpp files (no CUDA device annotations).

namespace ppln::collision {

// GPU BVH node: OBB bounding volume in link-local frame.
// Stored as flat float array: [n_nodes x 21]
// Layout per node (21 floats):
//   [0-2]   obb_center (link-local)
//   [3-5]   obb_half_ext
//   [6-14]  obb_axes (3x3, column i = {[6+i], [9+i], [12+i]}, i.e. OBB axis i in local frame)
//   [15]    left_child index (-1 if leaf)
//   [16]    right_child index
//   [17]    tri_start (leaf: first triangle index)
//   [18]    tri_count (leaf: number of triangles)
//   [19-20] padding
constexpr int BVH_NODE_FLOATS = 21;

struct MeshCollisionData {
    // BVH nodes (all links' BVH trees stored contiguously)
    float* bvh_nodes    = nullptr;  // [n_total_bvh_nodes * BVH_NODE_FLOATS]
    int    n_total_bvh_nodes = 0;

    // Triangle vertices in link-local frame
    float* tri_vertices = nullptr;  // [n_total_triangles * 9]  (v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z)
    int    n_total_triangles = 0;

    // Per-link BVH info
    int*   link_bvh_root   = nullptr;  // [n_links] root node index in bvh_nodes (-1 if no mesh)
    int*   link_tri_offset = nullptr;  // [n_links] start triangle index in tri_vertices
    int*   link_tri_count  = nullptr;  // [n_links] number of triangles
    int    n_links = 0;
};

} // namespace ppln::collision
