#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/collision/mesh_collision_data.hh"

namespace ppln {

// Maximum DOF supported at compile time (Baxter = 14).
// Used for fixed-size shared-memory arrays in kernels.
constexpr int MAX_DIM = 14;

// Number of 4x4 scratch slots the FK keeps per batch element. A serial chain
// needs 2 (root + rolling accumulator); a branching tree needs one slot per
// live branch point along the DFS. Baxter -- two arms hanging off a single
// base -- also needs exactly 2, because the DFS finishes the left arm before
// starting the right and the right arm's parent is the root. Raise this if a
// robot with nested branches is ever added, and keep every kernel's T buffer
// sized BATCH_SIZE * FK_T_SLOTS * 16.
constexpr int FK_T_SLOTS = 2;

// Runtime robot model — all pointers are device global memory.
// Passed by value to kernels (struct is small, pointers are device-side).
struct RobotModel {
    int n_dof;                    // degrees of freedom (e.g. 7 for Panda)
    int n_joints;                 // FK chain length = n_dof + 1 (includes base fixed joint)
    int n_spheres;                // full collision spheres
    int n_approx_spheres;         // approximate collision spheres
    int n_self_cc_ranges;         // self-collision check pairs (full)
    int n_approx_self_cc_ranges;  // self-collision check pairs (approx)

    // --- Device pointers (global memory) ---

    // FK chain: [n_joints * 16] row-major 4x4 per joint
    float* fixed_transforms;
    // [n_joints] joint type: -1=FIXED, 0=X_PRISM, ..., 5=Z_ROT
    int* joint_types;

    // Full collision model
    float4* spheres;              // [n_spheres] (x,y,z,r) in link-local frame
    int* sphere_to_joint;         // [n_spheres] -> joint index (0..n_joints-1)
    int* self_cc_ranges;          // [n_self_cc_ranges * 3] flat: {sphere_i, start, end}

    // Approximate collision model
    float4* approx_spheres;       // [n_approx_spheres]
    int* approx_sphere_to_joint;  // [n_approx_spheres]
    int* approx_self_cc_ranges;   // [n_approx_self_cc_ranges * 3] flat

    // Sphere-emission index, CSR over joints: the spheres attached to joint i
    // are sphere_order[sphere_joint_offsets[i] .. sphere_joint_offsets[i+1]).
    // Derived from sphere_to_joint, so FK does not depend on the asset happening
    // to list spheres in ascending, contiguous joint order (Baxter's second arm
    // reuses joint 0 partway through the list, which breaks that assumption).
    int* sphere_joint_offsets;        // [n_joints + 1]
    int* sphere_order;                // [n_spheres]
    int* approx_sphere_joint_offsets; // [n_joints + 1]
    int* approx_sphere_order;         // [n_approx_spheres]

    // Approx FK chain (may differ from full — e.g. same transforms but different spheres)
    float* approx_fixed_transforms; // [n_joints * 16] (often same data as fixed_transforms)
    int* approx_joint_types;        // [n_joints]

    // Kinematic tree topology. A serial chain is the degenerate case
    // (joint_parents[i] == i-1, joint_id_to_dof[i] == i-1, dfs_order == identity),
    // for which the tree walk reduces exactly to the old serial loop.
    int n_t_slots;                // scratch 4x4 slots needed per batch element
    int* joint_parents;           // [n_joints] parent joint (root points at itself)
    int* joint_id_to_dof;         // [n_joints] index into q (-1 for the fixed base)
    int* t_memory_idx;            // [n_joints] which scratch slot holds joint i's transform
    int* dfs_order;               // [n_joints] visit order; parents precede children

    // Joint limits
    float* joint_lower;           // [n_dof]
    float* joint_upper;           // [n_dof]
    float* joint_range;           // [n_dof] = upper - lower (for Halton scaling)

    // Mesh collision data (GPU BVH, optional)
    ppln::collision::MeshCollisionData mesh;
};

// Host-side RAII helper for building and destroying a RobotModel on device.
struct RobotModelDevice {
    RobotModel model{};
    bool owns_memory = false;

    // Allocate device memory and copy host data.
    // All host vectors must be sized correctly before calling.
    static RobotModelDevice create(
        int n_dof,
        int n_joints,
        // Full model
        const std::vector<float>& h_fixed_transforms,      // [n_joints * 16]
        const std::vector<int>& h_joint_types,              // [n_joints]
        const std::vector<float>& h_spheres_flat,           // [n_spheres * 4]
        const std::vector<int>& h_sphere_to_joint,          // [n_spheres]
        const std::vector<int>& h_self_cc_ranges_flat,      // [n_self_cc * 3]
        // Approx model
        const std::vector<float>& h_approx_spheres_flat,    // [n_approx * 4]
        const std::vector<int>& h_approx_sphere_to_joint,   // [n_approx]
        const std::vector<int>& h_approx_self_cc_ranges_flat, // [n_approx_self * 3]
        const std::vector<float>& h_approx_fixed_transforms, // [n_joints * 16]
        const std::vector<int>& h_approx_joint_types,        // [n_joints]
        // Joint limits
        const std::vector<float>& h_joint_lower,            // [n_dof]
        const std::vector<float>& h_joint_upper,            // [n_dof]
        // Mesh collision (optional)
        const std::vector<float>& h_mesh_bvh_nodes = {},    // [n_nodes * BVH_NODE_FLOATS]
        const std::vector<float>& h_mesh_tri_vertices = {}, // [n_tris * 9]
        const std::vector<int>& h_mesh_link_bvh_root = {},  // [n_links]
        const std::vector<int>& h_mesh_link_tri_offset = {},// [n_links]
        const std::vector<int>& h_mesh_link_tri_count = {}, // [n_links]
        // Kinematic tree topology. Leave empty for a serial chain: the serial
        // defaults are synthesised below, so existing callers keep working and
        // keep producing bit-identical results.
        const std::vector<int>& h_joint_parents = {},       // [n_joints]
        const std::vector<int>& h_joint_id_to_dof = {},     // [n_joints]
        const std::vector<int>& h_t_memory_idx = {},        // [n_joints]
        const std::vector<int>& h_dfs_order = {}            // [n_joints]
    ) {
        RobotModelDevice rmd;
        rmd.owns_memory = true;
        RobotModel& m = rmd.model;

        m.n_dof = n_dof;
        m.n_joints = n_joints;
        m.n_spheres = static_cast<int>(h_sphere_to_joint.size());
        m.n_approx_spheres = static_cast<int>(h_approx_sphere_to_joint.size());
        m.n_self_cc_ranges = static_cast<int>(h_self_cc_ranges_flat.size()) / 3;
        m.n_approx_self_cc_ranges = static_cast<int>(h_approx_self_cc_ranges_flat.size()) / 3;

        // Helper lambda for cuda alloc + copy
        auto alloc_copy = [](auto*& d_ptr, const auto& h_vec, size_t elem_size) {
            size_t bytes = h_vec.size() * elem_size;
            if (bytes == 0) { d_ptr = nullptr; return; }
            cudaMalloc(&d_ptr, bytes);
            cudaMemcpy(d_ptr, h_vec.data(), bytes, cudaMemcpyHostToDevice);
        };

        // Fixed transforms
        alloc_copy(m.fixed_transforms, h_fixed_transforms, sizeof(float));
        alloc_copy(m.joint_types, h_joint_types, sizeof(int));

        // Full spheres — convert from flat float[N*4] to float4[N]
        {
            int ns = m.n_spheres;
            std::vector<float4> f4(ns);
            for (int i = 0; i < ns; i++) {
                f4[i] = make_float4(
                    h_spheres_flat[i*4+0], h_spheres_flat[i*4+1],
                    h_spheres_flat[i*4+2], h_spheres_flat[i*4+3]);
            }
            cudaMalloc(&m.spheres, ns * sizeof(float4));
            cudaMemcpy(m.spheres, f4.data(), ns * sizeof(float4), cudaMemcpyHostToDevice);
        }

        alloc_copy(m.sphere_to_joint, h_sphere_to_joint, sizeof(int));
        alloc_copy(m.self_cc_ranges, h_self_cc_ranges_flat, sizeof(int));

        // Approx spheres
        {
            int na = m.n_approx_spheres;
            std::vector<float4> f4(na);
            for (int i = 0; i < na; i++) {
                f4[i] = make_float4(
                    h_approx_spheres_flat[i*4+0], h_approx_spheres_flat[i*4+1],
                    h_approx_spheres_flat[i*4+2], h_approx_spheres_flat[i*4+3]);
            }
            cudaMalloc(&m.approx_spheres, na * sizeof(float4));
            cudaMemcpy(m.approx_spheres, f4.data(), na * sizeof(float4), cudaMemcpyHostToDevice);
        }

        alloc_copy(m.approx_sphere_to_joint, h_approx_sphere_to_joint, sizeof(int));
        alloc_copy(m.approx_self_cc_ranges, h_approx_self_cc_ranges_flat, sizeof(int));
        alloc_copy(m.approx_fixed_transforms, h_approx_fixed_transforms, sizeof(float));
        alloc_copy(m.approx_joint_types, h_approx_joint_types, sizeof(int));

        // Sphere-emission CSR index, derived from sphere_to_joint.
        {
            auto build_csr = [&](const std::vector<int>& to_joint,
                                 int*& d_offsets, int*& d_order) {
                std::vector<int> offsets(n_joints + 1, 0);
                for (int j : to_joint) offsets[j + 1]++;
                for (int i = 0; i < n_joints; i++) offsets[i + 1] += offsets[i];
                std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
                std::vector<int> order(to_joint.size());
                for (int s = 0; s < static_cast<int>(to_joint.size()); s++) {
                    order[cursor[to_joint[s]]++] = s;
                }
                alloc_copy(d_offsets, offsets, sizeof(int));
                alloc_copy(d_order, order, sizeof(int));
            };
            build_csr(h_sphere_to_joint, m.sphere_joint_offsets, m.sphere_order);
            build_csr(h_approx_sphere_to_joint, m.approx_sphere_joint_offsets,
                      m.approx_sphere_order);
        }

        // Joint limits + range
        alloc_copy(m.joint_lower, h_joint_lower, sizeof(float));
        alloc_copy(m.joint_upper, h_joint_upper, sizeof(float));

        // Kinematic tree topology — synthesise the serial chain when absent.
        {
            std::vector<int> parents = h_joint_parents;
            std::vector<int> id_to_dof = h_joint_id_to_dof;
            std::vector<int> t_mem = h_t_memory_idx;
            std::vector<int> dfs = h_dfs_order;

            if (static_cast<int>(parents.size()) != n_joints) {
                parents.resize(n_joints);
                for (int i = 0; i < n_joints; i++) parents[i] = (i > 0) ? i - 1 : 0;
            }
            if (static_cast<int>(id_to_dof.size()) != n_joints) {
                id_to_dof.resize(n_joints);
                for (int i = 0; i < n_joints; i++) id_to_dof[i] = i - 1;
            }
            if (static_cast<int>(t_mem.size()) != n_joints) {
                t_mem.assign(n_joints, 1);
                if (n_joints > 0) t_mem[0] = 0;
            }
            if (static_cast<int>(dfs.size()) != n_joints) {
                dfs.resize(n_joints);
                for (int i = 0; i < n_joints; i++) dfs[i] = i;
            }

            int slots = 1;
            for (int i = 0; i < n_joints; i++) slots = std::max(slots, t_mem[i] + 1);
            m.n_t_slots = slots;

            alloc_copy(m.joint_parents, parents, sizeof(int));
            alloc_copy(m.joint_id_to_dof, id_to_dof, sizeof(int));
            alloc_copy(m.t_memory_idx, t_mem, sizeof(int));
            alloc_copy(m.dfs_order, dfs, sizeof(int));
        }

        // Compute range = upper - lower
        std::vector<float> h_range(n_dof);
        for (int i = 0; i < n_dof; i++) {
            h_range[i] = h_joint_upper[i] - h_joint_lower[i];
        }
        alloc_copy(m.joint_range, h_range, sizeof(float));

        // Mesh collision data (optional)
        if (!h_mesh_bvh_nodes.empty() && !h_mesh_link_bvh_root.empty()) {
            auto& mesh = m.mesh;
            mesh.n_total_bvh_nodes = static_cast<int>(h_mesh_bvh_nodes.size()) / collision::BVH_NODE_FLOATS;
            mesh.n_total_triangles = static_cast<int>(h_mesh_tri_vertices.size()) / 9;
            mesh.n_links = static_cast<int>(h_mesh_link_bvh_root.size());

            alloc_copy(mesh.bvh_nodes, h_mesh_bvh_nodes, sizeof(float));
            alloc_copy(mesh.tri_vertices, h_mesh_tri_vertices, sizeof(float));
            alloc_copy(mesh.link_bvh_root, h_mesh_link_bvh_root, sizeof(int));
            alloc_copy(mesh.link_tri_offset, h_mesh_link_tri_offset, sizeof(int));
            alloc_copy(mesh.link_tri_count, h_mesh_link_tri_count, sizeof(int));
        }

        return rmd;
    }

    void destroy() {
        if (!owns_memory) return;
        cudaFree(model.fixed_transforms);
        cudaFree(model.joint_types);
        cudaFree(model.spheres);
        cudaFree(model.sphere_to_joint);
        cudaFree(model.self_cc_ranges);
        cudaFree(model.approx_spheres);
        cudaFree(model.approx_sphere_to_joint);
        cudaFree(model.approx_self_cc_ranges);
        cudaFree(model.sphere_joint_offsets);
        cudaFree(model.sphere_order);
        cudaFree(model.approx_sphere_joint_offsets);
        cudaFree(model.approx_sphere_order);
        cudaFree(model.approx_fixed_transforms);
        cudaFree(model.approx_joint_types);
        cudaFree(model.joint_parents);
        cudaFree(model.joint_id_to_dof);
        cudaFree(model.t_memory_idx);
        cudaFree(model.dfs_order);
        cudaFree(model.joint_lower);
        cudaFree(model.joint_upper);
        cudaFree(model.joint_range);
        // Mesh collision data
        cudaFree(model.mesh.bvh_nodes);
        cudaFree(model.mesh.tri_vertices);
        cudaFree(model.mesh.link_bvh_root);
        cudaFree(model.mesh.link_tri_offset);
        cudaFree(model.mesh.link_tri_count);
        owns_memory = false;
        model = RobotModel{};
    }

    ~RobotModelDevice() {
        // Note: destructor does NOT auto-free — user must call destroy()
        // explicitly, because this struct may be copied by value.
    }
};

} // namespace ppln
