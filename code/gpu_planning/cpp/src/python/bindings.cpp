#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include "src/planning/Planners.hh"
#include "src/planning/robot_model.cuh"
#include "src/planning/solver_buffers.hh"
#include "src/planning/mitstar_buffers.hh"
#include "src/planning/STOMP_settings.hh"
#include "src/planning/stomp_buffers.hh"
#include "src/planning/CHOMP_settings.hh"
#include "src/planning/chomp_buffers.hh"
#include "src/collision/factory.hh"
#include "src/collision/scene_collision_data.hh"
#include "src/collision/mesh_collision_data.hh"

// BATCH_SIZE is defined in utils.cuh (CUDA-only), but we need it here for
// host-side validation that settings.granularity does not exceed it.
#ifndef BATCH_SIZE
#define BATCH_SIZE 16
#endif

namespace py = pybind11;

// ---------------------------------------------------------------------------
// EnvironmentBuilder — accumulates shapes, then produces an Environment<float>
// ---------------------------------------------------------------------------
struct EnvironmentBuilder {
    std::vector<ppln::collision::Sphere<float>>   spheres;
    std::vector<ppln::collision::Cuboid<float>>   cuboids;
    std::vector<ppln::collision::Capsule<float>>  capsules;  // Capsule = Cylinder; used for cylinder obstacles

    void add_sphere(float cx, float cy, float cz, float r) {
        spheres.push_back(ppln::collision::factory::sphere::flat(cx, cy, cz, r));
    }

    void add_cuboid(float cx, float cy, float cz,
                    float rho, float theta, float phi,
                    float hx, float hy, float hz) {
        cuboids.push_back(
            ppln::collision::factory::cuboid::flat(cx, cy, cz, rho, theta, phi, hx, hy, hz));
    }

    // Quaternion version: pose = [x, y, z, qw, qx, qy, qz], dims = [sx, sy, sz] (full extents)
    void add_cuboid_quat(float cx, float cy, float cz,
                         float qw, float qx, float qy, float qz,
                         float sx, float sy, float sz) {
        Eigen::Quaternionf q(qw, qx, qy, qz);
        q.normalize();
        Eigen::Vector3f center(cx, cy, cz);
        Eigen::Vector3f half_extents(sx / 2.0f, sy / 2.0f, sz / 2.0f);
        cuboids.push_back(
            ppln::collision::factory::cuboid::eigen_rot(center, q, half_extents));
    }

    void add_cylinder(float cx, float cy, float cz,
                      float rho, float theta, float phi,
                      float radius, float length) {
        capsules.push_back(
            ppln::collision::factory::cylinder::center::flat(
                cx, cy, cz, rho, theta, phi, radius, length));
    }

    // Build a fresh Environment<float>.
    // The Environment destructor will delete[] the arrays.
    ppln::collision::Environment<float> build() const {
        ppln::collision::Environment<float> env{};

        if (!spheres.empty()) {
            env.num_spheres = static_cast<unsigned int>(spheres.size());
            env.spheres = new ppln::collision::Sphere<float>[env.num_spheres];
            std::copy(spheres.begin(), spheres.end(), env.spheres);
        } else {
            env.num_spheres = 0;
            env.spheres = nullptr;
        }

        if (!cuboids.empty()) {
            env.num_cuboids = static_cast<unsigned int>(cuboids.size());
            env.cuboids = new ppln::collision::Cuboid<float>[env.num_cuboids];
            std::copy(cuboids.begin(), cuboids.end(), env.cuboids);
        } else {
            env.num_cuboids = 0;
            env.cuboids = nullptr;
        }

        if (!capsules.empty()) {
            env.num_capsules = static_cast<unsigned int>(capsules.size());
            env.capsules = new ppln::collision::Capsule<float>[env.num_capsules];
            std::copy(capsules.begin(), capsules.end(), env.capsules);
        } else {
            env.num_capsules = 0;
            env.capsules = nullptr;
        }

        // Unused shape types — initialize to safe defaults
        env.num_cylinders = 0;
        env.cylinders = nullptr;
        env.num_z_aligned_capsules = 0;
        env.z_aligned_capsules = nullptr;
        env.num_z_aligned_cuboids = 0;
        env.z_aligned_cuboids = nullptr;

        return env;
    }
};

// ---------------------------------------------------------------------------
// RobotModelBuilder — Python-side builder for ppln::RobotModel on device
// ---------------------------------------------------------------------------
struct RobotModelBuilder {
    int n_dof = 0;
    int n_joints = 0;

    std::vector<float> fixed_transforms;        // [n_joints * 16] row-major
    std::vector<int>   joint_types;             // [n_joints]
    std::vector<float> spheres_flat;            // [n_spheres * 4] (x,y,z,r)
    std::vector<int>   sphere_to_joint;         // [n_spheres]
    std::vector<int>   self_cc_ranges_flat;     // [n_self_cc * 3]

    std::vector<float> approx_spheres_flat;     // [n_approx * 4]
    std::vector<int>   approx_sphere_to_joint;  // [n_approx]
    std::vector<int>   approx_self_cc_ranges_flat; // [n_approx_self * 3]
    std::vector<float> approx_fixed_transforms; // [n_joints * 16]
    std::vector<int>   approx_joint_types;      // [n_joints]

    std::vector<float> joint_lower;             // [n_dof]
    std::vector<float> joint_upper;             // [n_dof]

    // Kinematic tree topology (optional). Leave empty for a serial chain —
    // RobotModelDevice::create synthesises the degenerate serial values, so
    // serial robots behave exactly as before. Required for branching robots
    // such as the 14-DoF dual-arm Baxter, whose right arm hangs off the base.
    std::vector<int>   joint_parents;           // [n_joints]
    std::vector<int>   joint_id_to_dof;         // [n_joints]
    std::vector<int>   t_memory_idx;            // [n_joints]
    std::vector<int>   dfs_order;               // [n_joints]

    // Mesh collision (optional)
    std::vector<float> mesh_bvh_nodes;          // [n_nodes * BVH_NODE_FLOATS]
    std::vector<float> mesh_tri_vertices;       // [n_tris * 9]
    std::vector<int>   mesh_link_bvh_root;      // [n_links]
    std::vector<int>   mesh_link_tri_offset;    // [n_links]
    std::vector<int>   mesh_link_tri_count;     // [n_links]

    // Device-side model (built lazily)
    ppln::RobotModelDevice device_model;
    bool built = false;

    ppln::RobotModel& build_on_device() {
        if (built) return device_model.model;

        // If approx transforms not set, use same as full
        if (approx_fixed_transforms.empty()) {
            approx_fixed_transforms = fixed_transforms;
            approx_joint_types = joint_types;
        }

        device_model = ppln::RobotModelDevice::create(
            n_dof, n_joints,
            fixed_transforms, joint_types,
            spheres_flat, sphere_to_joint, self_cc_ranges_flat,
            approx_spheres_flat, approx_sphere_to_joint, approx_self_cc_ranges_flat,
            approx_fixed_transforms, approx_joint_types,
            joint_lower, joint_upper,
            mesh_bvh_nodes, mesh_tri_vertices,
            mesh_link_bvh_root, mesh_link_tri_offset, mesh_link_tri_count,
            joint_parents, joint_id_to_dof, t_memory_idx, dfs_order
        );
        built = true;
        return device_model.model;
    }

    // Update joint limits on device WITHOUT full rebuild.
    // Recomputes joint_range = upper - lower and uploads all three arrays.
    void update_joint_limits() {
        if (!built) {
            build_on_device();
            return;
        }
        if (static_cast<int>(joint_lower.size()) != n_dof ||
            static_cast<int>(joint_upper.size()) != n_dof)
            throw std::runtime_error("joint_lower/joint_upper size != n_dof");

        auto& m = device_model.model;
        size_t bytes = n_dof * sizeof(float);
        cudaError_t err;
        err = cudaMemcpy(m.joint_lower, joint_lower.data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaMemcpy joint_lower failed: ") + cudaGetErrorString(err));
        err = cudaMemcpy(m.joint_upper, joint_upper.data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaMemcpy joint_upper failed: ") + cudaGetErrorString(err));

        // Recompute range on host then upload
        std::vector<float> h_range(n_dof);
        for (int i = 0; i < n_dof; i++)
            h_range[i] = joint_upper[i] - joint_lower[i];
        err = cudaMemcpy(m.joint_range, h_range.data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaMemcpy joint_range failed: ") + cudaGetErrorString(err));
    }

    void destroy() {
        if (built) {
            device_model.destroy();
            built = false;
        }
    }

    // Ensure GPU memory is freed when Python GC collects this object.
    ~RobotModelBuilder() {
        destroy();
    }
};

// ---------------------------------------------------------------------------
// Python-facing PlannerResult (robot-agnostic, numpy path)
// ---------------------------------------------------------------------------
struct PyPlannerResult {
    bool   solved;
    py::array_t<float> path;     // shape (N, dim), start→goal order
    int    iters;
    float  cost;
    double wall_time_ms;
};

// Convert C++ PlannerResult to PyPlannerResult, reversing path to start→goal
template <typename Robot>
PyPlannerResult convert_result(const PlannerResult<Robot>& res) {
    constexpr int dim = Robot::dimension;
    const int n = static_cast<int>(res.path.size());

    py::array_t<float> path({n, dim});
    auto buf = path.mutable_unchecked<2>();

    // Reverse: pRRTC returns goal→start, we want start→goal
    for (int i = 0; i < n; ++i) {
        const auto& cfg = res.path[n - 1 - i];
        for (int j = 0; j < dim; ++j)
            buf(i, j) = cfg[j];
    }

    return PyPlannerResult{
        res.solved,
        std::move(path),
        res.iters,
        res.cost,
        static_cast<double>(res.wall_ns) / 1e6
    };
}

// Convert RuntimePlannerResult to PyPlannerResult, reversing path to start→goal
static PyPlannerResult convert_runtime_result(const RuntimePlannerResult& res, int dim) {
    const int n = static_cast<int>(res.path.size());

    py::array_t<float> path({n, dim});
    auto buf = path.mutable_unchecked<2>();

    // Reverse: runtime solve returns goal→start, we want start→goal
    for (int i = 0; i < n; ++i) {
        const auto& cfg = res.path[n - 1 - i];
        for (int j = 0; j < dim; ++j)
            buf(i, j) = cfg[j];
    }

    return PyPlannerResult{
        res.solved,
        std::move(path),
        res.iters,
        res.cost,
        static_cast<double>(res.wall_ns) / 1e6
    };
}

// ---------------------------------------------------------------------------
// solve_panda  — unconstrained pRRTC for Panda 7-DOF (backward compat)
// ---------------------------------------------------------------------------
static PyPlannerResult solve_panda(
    py::array_t<float> start_np,
    py::array_t<float> goals_np,
    EnvironmentBuilder& env_builder,
    pRRTC_settings& settings)
{
    if (settings.granularity > BATCH_SIZE)
        throw std::runtime_error(
            "settings.granularity (" + std::to_string(settings.granularity) +
            ") exceeds BATCH_SIZE (" + std::to_string(BATCH_SIZE) +
            "). Shared memory layout requires granularity <= BATCH_SIZE.");

    using Robot = ppln::robots::Panda;
    constexpr int dim = Robot::dimension;

    // Parse start
    auto s = start_np.unchecked<1>();
    if (s.shape(0) != dim)
        throw std::runtime_error("start must have 7 elements");
    std::array<float, dim> start;
    for (int i = 0; i < dim; ++i) start[i] = s(i);

    // Parse goals — shape (M, 7)
    auto g = goals_np.unchecked<2>();
    if (g.shape(1) != dim)
        throw std::runtime_error("goals must have shape (M, 7)");
    std::vector<std::array<float, dim>> goals(g.shape(0));
    for (int i = 0; i < g.shape(0); ++i)
        for (int j = 0; j < dim; ++j)
            goals[i][j] = g(i, j);

    auto env = env_builder.build();
    auto res = pRRTC::solve<Robot>(start, goals, env, settings);
    return convert_result<Robot>(res);
}

// ---------------------------------------------------------------------------
// solve — runtime unconstrained pRRTC for any robot
// ---------------------------------------------------------------------------
static PyPlannerResult solve_runtime(
    py::array_t<float> start_np,
    py::array_t<float> goals_np,
    EnvironmentBuilder& env_builder,
    pRRTC_settings& settings,
    RobotModelBuilder& robot)
{
    if (settings.granularity > BATCH_SIZE)
        throw std::runtime_error(
            "settings.granularity (" + std::to_string(settings.granularity) +
            ") exceeds BATCH_SIZE (" + std::to_string(BATCH_SIZE) +
            "). Shared memory layout requires granularity <= BATCH_SIZE.");

    auto& model = robot.build_on_device();
    int dim = model.n_dof;

    auto s = start_np.unchecked<1>();
    if (s.shape(0) != dim)
        throw std::runtime_error("start size mismatch with robot n_dof");
    std::vector<float> start(dim);
    for (int i = 0; i < dim; ++i) start[i] = s(i);

    auto g = goals_np.unchecked<2>();
    if (g.shape(1) != dim)
        throw std::runtime_error("goals columns mismatch with robot n_dof");
    std::vector<std::vector<float>> goals(g.shape(0), std::vector<float>(dim));
    for (int i = 0; i < g.shape(0); ++i)
        for (int j = 0; j < dim; ++j)
            goals[i][j] = g(i, j);

    auto env = env_builder.build();
    auto res = pRRTC::solve_runtime(start, goals, env, settings, model);
    return convert_runtime_result(res, dim);
}

// ---------------------------------------------------------------------------
// SceneCollisionDataBuilder — zero-copy GPU pointers from collision tensors
// ---------------------------------------------------------------------------
struct SceneCollisionDataBuilder {
    // OBB (cube tensor list)
    int64_t obb_dims_ptr = 0;     // tensor.data_ptr()
    int64_t obb_pose_ptr = 0;
    int64_t obb_enable_ptr = 0;
    int n_obbs = 0;

    // ESDF voxels (voxel tensor list)
    int64_t voxel_params_ptr = 0;
    int64_t voxel_pose_ptr = 0;
    int64_t voxel_enable_ptr = 0;
    int64_t voxel_features_ptr = 0;
    int n_voxel_layers = 0;
    int max_voxels_per_layer = 0;

    // ACM mask (optional, sphere_acm_mask remapped to pRRTC sphere order)
    int64_t sphere_acm_mask_ptr = 0;

    // OBB ACM masks (optional, for per-sphere OBB collision filtering)
    int64_t sphere_obb_acm_mask_ptr = 0;         // (n_spheres, n_obbs)
    int64_t sphere_obb_acm_mask_approx_ptr = 0;  // (n_approx_spheres, n_obbs)

    ppln::collision::SceneCollisionData build() const {
        ppln::collision::SceneCollisionData d{};
        d.obb_dims   = reinterpret_cast<float*>(obb_dims_ptr);
        d.obb_pose   = reinterpret_cast<float*>(obb_pose_ptr);
        d.obb_enable = reinterpret_cast<uint8_t*>(obb_enable_ptr);
        d.n_obbs     = n_obbs;
        d.voxel_params   = reinterpret_cast<float*>(voxel_params_ptr);
        d.voxel_pose     = reinterpret_cast<float*>(voxel_pose_ptr);
        d.voxel_enable   = reinterpret_cast<uint8_t*>(voxel_enable_ptr);
        d.voxel_features = reinterpret_cast<float*>(voxel_features_ptr);
        d.n_voxel_layers      = n_voxel_layers;
        d.max_voxels_per_layer = max_voxels_per_layer;
        d.sphere_acm_mask = reinterpret_cast<uint8_t*>(sphere_acm_mask_ptr);
        d.sphere_obb_acm_mask = reinterpret_cast<uint8_t*>(sphere_obb_acm_mask_ptr);
        d.sphere_obb_acm_mask_approx = reinterpret_cast<uint8_t*>(sphere_obb_acm_mask_approx_ptr);
        return d;
    }
};

// ---------------------------------------------------------------------------
// solve_scene — runtime pRRTC with scene collision data
// ---------------------------------------------------------------------------
static PyPlannerResult solve_runtime_scene(
    py::array_t<float> start_np,
    py::array_t<float> goals_np,
    SceneCollisionDataBuilder& scene_builder,
    pRRTC_settings& settings,
    RobotModelBuilder& robot,
    py::object bufs_obj)
{
    if (settings.granularity > BATCH_SIZE)
        throw std::runtime_error(
            "settings.granularity (" + std::to_string(settings.granularity) +
            ") exceeds BATCH_SIZE (" + std::to_string(BATCH_SIZE) +
            "). Shared memory layout requires granularity <= BATCH_SIZE.");

    auto& model = robot.build_on_device();
    int dim = model.n_dof;

    auto s = start_np.unchecked<1>();
    if (s.shape(0) != dim)
        throw std::runtime_error("start size mismatch with robot n_dof");
    std::vector<float> start(dim);
    for (int i = 0; i < dim; ++i) start[i] = s(i);

    auto g = goals_np.unchecked<2>();
    if (g.shape(1) != dim)
        throw std::runtime_error("goals columns mismatch with robot n_dof");
    std::vector<std::vector<float>> goals(g.shape(0), std::vector<float>(dim));
    for (int i = 0; i < g.shape(0); ++i)
        for (int j = 0; j < dim; ++j)
            goals[i][j] = g(i, j);

    SolverBuffers* bufs_ptr = nullptr;
    if (!bufs_obj.is_none())
        bufs_ptr = bufs_obj.cast<SolverBuffers*>();

    auto scene = scene_builder.build();
    auto res = pRRTC::solve_runtime_scene(start, goals, scene, settings, model, bufs_ptr);
    return convert_runtime_result(res, dim);
}

// ---------------------------------------------------------------------------
// check_collision_mesh — standalone GPU mesh collision check
// ---------------------------------------------------------------------------
static py::array_t<bool> check_collision_mesh_py(
    py::array_t<float> configs_np,
    SceneCollisionDataBuilder& scene_builder,
    RobotModelBuilder& robot)
{
    auto& model = robot.build_on_device();
    int dim = model.n_dof;

    auto c = configs_np.unchecked<2>();
    int N = static_cast<int>(c.shape(0));
    if (c.shape(1) != dim)
        throw std::runtime_error("configs columns mismatch with robot n_dof");

    // Flatten to contiguous
    std::vector<float> h_configs(N * dim);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < dim; j++)
            h_configs[i * dim + j] = c(i, j);

    auto scene = scene_builder.build();
    // Use raw bool array for CUDA interface
    std::vector<uint8_t> h_results_raw(N);

    // Call CUDA
    {
        bool* h_results_bool = new bool[N];
        pRRTC::check_collision_mesh(h_configs.data(), N, scene, model, h_results_bool);
        for (int i = 0; i < N; i++)
            h_results_raw[i] = h_results_bool[i] ? 1 : 0;
        delete[] h_results_bool;
    }

    py::array_t<bool> result(N);
    auto buf = result.mutable_unchecked<1>();
    for (int i = 0; i < N; i++)
        buf(i) = (h_results_raw[i] != 0);

    return result;
}

// ---------------------------------------------------------------------------
// Scene collision check using planning-consistent FK (__sinf/__cosf)
// ---------------------------------------------------------------------------
static py::array_t<bool> check_collision_scene_py(
    py::array_t<float> configs_np,
    SceneCollisionDataBuilder& scene_builder,
    RobotModelBuilder& robot,
    float collision_margin = 0.0f)
{
    auto& model = robot.build_on_device();
    int dim = model.n_dof;

    auto c = configs_np.unchecked<2>();
    int N = static_cast<int>(c.shape(0));
    if (c.shape(1) != dim)
        throw std::runtime_error("configs columns mismatch with robot n_dof");

    std::vector<float> h_configs(N * dim);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < dim; j++)
            h_configs[i * dim + j] = c(i, j);

    auto scene = scene_builder.build();
    std::vector<uint8_t> h_results_raw(N);

    {
        bool* h_results_bool = new bool[N];
        pRRTC::check_collision_scene(h_configs.data(), N, scene, model,
                                     collision_margin, h_results_bool);
        for (int i = 0; i < N; i++)
            h_results_raw[i] = h_results_bool[i] ? 1 : 0;
        delete[] h_results_bool;
    }

    py::array_t<bool> result(N);
    auto buf = result.mutable_unchecked<1>();
    for (int i = 0; i < N; i++)
        buf(i) = (h_results_raw[i] != 0);

    return result;
}

// ---------------------------------------------------------------------------
// Python-facing MITStarResult (anytime planner result with cost history)
// ---------------------------------------------------------------------------
struct PyMITStarResult {
    bool   solved;
    py::array_t<float> path;     // shape (N, dim), start->goal order
    int    total_nodes;
    int    total_batches;
    float  cost;
    py::list cost_history;
    py::list time_history_ms;
    double wall_time_ms;
    int    total_forward_edges_evaluated;
    int    total_reverse_edges_evaluated;
    int    reverse_restarts;
    float  eis_cost;
    int    total_iterations;
    // FIX_B3 diagnostics
    int    sparse_midpoints_checked;
    int    sparse_self_coll_hits;
    int    sparse_scene_coll_hits;
    int    full_midpoints_checked;
    int    full_self_coll_hits;
    int    full_scene_coll_hits;
};

static PyMITStarResult convert_mitstar_result(const MITStarResult& res, int dim) {
    const int n = static_cast<int>(res.path.size());

    py::array_t<float> path({n, dim});
    auto buf = path.mutable_unchecked<2>();

    // Path is already in start->goal order from MIT*
    for (int i = 0; i < n; ++i) {
        const auto& cfg = res.path[i];
        for (int j = 0; j < dim; ++j)
            buf(i, j) = cfg[j];
    }

    py::list cost_hist;
    for (float c : res.cost_history) cost_hist.append(c);
    py::list time_hist;
    for (float t : res.time_history_ms) time_hist.append(t);

    return PyMITStarResult{
        res.solved,
        std::move(path),
        res.total_nodes,
        res.total_batches,
        res.cost,
        std::move(cost_hist),
        std::move(time_hist),
        static_cast<double>(res.wall_ns) / 1e6,
        res.total_forward_edges_evaluated,
        res.total_reverse_edges_evaluated,
        res.reverse_restarts,
        res.eis_cost,
        res.total_iterations,
        // FIX_B3 diagnostics
        res.sparse_midpoints_checked,
        res.sparse_self_coll_hits,
        res.sparse_scene_coll_hits,
        res.full_midpoints_checked,
        res.full_self_coll_hits,
        res.full_scene_coll_hits
    };
}

// ---------------------------------------------------------------------------
// mitstar_solve_scene — MIT* with scene collision data
// ---------------------------------------------------------------------------
static PyMITStarResult mitstar_solve_scene(
    py::array_t<float> start_np,
    py::array_t<float> goals_np,
    SceneCollisionDataBuilder& scene_builder,
    MITStar_settings& settings,
    RobotModelBuilder& robot,
    py::object bufs_obj)
{
    auto& model = robot.build_on_device();
    int dim = model.n_dof;

    auto s = start_np.unchecked<1>();
    if (s.shape(0) != dim)
        throw std::runtime_error("start size mismatch with robot n_dof");
    std::vector<float> start(dim);
    for (int i = 0; i < dim; ++i) start[i] = s(i);

    auto g = goals_np.unchecked<2>();
    if (g.shape(1) != dim)
        throw std::runtime_error("goals columns mismatch with robot n_dof");
    std::vector<std::vector<float>> goals(g.shape(0), std::vector<float>(dim));
    for (int i = 0; i < g.shape(0); ++i)
        for (int j = 0; j < dim; ++j)
            goals[i][j] = g(i, j);

    MITStarBuffers* bufs_ptr = nullptr;
    if (!bufs_obj.is_none())
        bufs_ptr = bufs_obj.cast<MITStarBuffers*>();

    auto scene = scene_builder.build();
    auto res = MITStar::solve_runtime_scene(start, goals, scene, settings, model, bufs_ptr);
    return convert_mitstar_result(res, dim);
}

// ---------------------------------------------------------------------------
// Python-facing MHAStarResult (SMHA* CPU search + GPU batched lazy edges)
// ---------------------------------------------------------------------------
struct PyMHAStarResult {
    bool   solved;
    py::array_t<float> path;   // shape (N, dim), start->goal order
    float  cost;
    int    expansions;
    int    edges_evaluated;
    double wall_time_ms;
};

static PyMHAStarResult convert_mhastar_result(const MHAStarResult& res, int dim) {
    const int n = static_cast<int>(res.path.size());
    py::array_t<float> path({n, dim});
    auto buf = path.mutable_unchecked<2>();
    for (int i = 0; i < n; ++i) {
        const auto& cfg = res.path[i];
        for (int j = 0; j < dim; ++j)
            buf(i, j) = cfg[j];
    }
    return PyMHAStarResult{
        res.solved,
        std::move(path),
        res.cost,
        res.expansions,
        res.edges_evaluated,
        static_cast<double>(res.wall_ns) / 1e6
    };
}

// ---------------------------------------------------------------------------
// mhastar_solve_scene — SMHA* with scene collision data
// ---------------------------------------------------------------------------
static PyMHAStarResult mhastar_solve_scene(
    py::array_t<float> start_np,
    py::array_t<float> goals_np,
    SceneCollisionDataBuilder& scene_builder,
    MHAStar_settings& settings,
    RobotModelBuilder& robot)
{
    auto& model = robot.build_on_device();
    int dim = model.n_dof;

    auto s = start_np.unchecked<1>();
    if (s.shape(0) != dim)
        throw std::runtime_error("start size mismatch with robot n_dof");
    std::vector<float> start(dim);
    for (int i = 0; i < dim; ++i) start[i] = s(i);

    auto g = goals_np.unchecked<2>();
    if (g.shape(1) != dim)
        throw std::runtime_error("goals columns mismatch with robot n_dof");
    std::vector<std::vector<float>> goals(g.shape(0), std::vector<float>(dim));
    for (int i = 0; i < g.shape(0); ++i)
        for (int j = 0; j < dim; ++j)
            goals[i][j] = g(i, j);

    auto scene = scene_builder.build();
    auto res = MHAStar::solve_runtime_scene(start, goals, scene, settings, model);
    return convert_mhastar_result(res, dim);
}

// ---------------------------------------------------------------------------
// Python-facing WPASEResult (wPA*SE CPU parallel search + GPU batched edges)
// ---------------------------------------------------------------------------
struct PyWPASEResult {
    bool   solved;
    py::array_t<float> path;   // shape (N, dim), start->goal order
    float  cost;
    int    expansions;
    int    edges_evaluated;
    double wall_time_ms;
};

static PyWPASEResult convert_wpase_result(const WPASEResult& res, int dim) {
    const int n = static_cast<int>(res.path.size());
    py::array_t<float> path({n, dim});
    auto buf = path.mutable_unchecked<2>();
    for (int i = 0; i < n; ++i) {
        const auto& cfg = res.path[i];
        for (int j = 0; j < dim; ++j)
            buf(i, j) = cfg[j];
    }
    return PyWPASEResult{
        res.solved,
        std::move(path),
        res.cost,
        res.expansions,
        res.edges_evaluated,
        static_cast<double>(res.wall_ns) / 1e6
    };
}

// ---------------------------------------------------------------------------
// wpase_solve_scene — wPA*SE with scene collision data
// ---------------------------------------------------------------------------
static PyWPASEResult wpase_solve_scene(
    py::array_t<float> start_np,
    py::array_t<float> goals_np,
    SceneCollisionDataBuilder& scene_builder,
    WPASE_settings& settings,
    RobotModelBuilder& robot)
{
    auto& model = robot.build_on_device();
    int dim = model.n_dof;

    auto s = start_np.unchecked<1>();
    if (s.shape(0) != dim)
        throw std::runtime_error("start size mismatch with robot n_dof");
    std::vector<float> start(dim);
    for (int i = 0; i < dim; ++i) start[i] = s(i);

    auto g = goals_np.unchecked<2>();
    if (g.shape(1) != dim)
        throw std::runtime_error("goals columns mismatch with robot n_dof");
    std::vector<std::vector<float>> goals(g.shape(0), std::vector<float>(dim));
    for (int i = 0; i < g.shape(0); ++i)
        for (int j = 0; j < dim; ++j)
            goals[i][j] = g(i, j);

    auto scene = scene_builder.build();
    auto res = WPASE::solve_runtime_scene(start, goals, scene, settings, model);
    return convert_wpase_result(res, dim);
}

// ---------------------------------------------------------------------------
// Python-facing STOMPResult
// ---------------------------------------------------------------------------
struct PySTOMPResult {
    bool   solved;
    py::array_t<float> path;   // shape (T, dim), start->goal order
    int    iterations_run;
    float  final_state_cost;
    float  final_control_cost;
    float  final_total_cost;
    double wall_time_ms;
};

static PySTOMPResult convert_stomp_result(const STOMPResult& res, int dim) {
    const int n = static_cast<int>(res.path.size());

    py::array_t<float> path({n, dim});
    auto buf = path.mutable_unchecked<2>();
    for (int i = 0; i < n; ++i) {
        const auto& cfg = res.path[i];
        for (int j = 0; j < dim; ++j)
            buf(i, j) = cfg[j];
    }

    return PySTOMPResult{
        res.solved,
        std::move(path),
        res.iterations_run,
        res.final_state_cost,
        res.final_control_cost,
        res.final_total_cost,
        static_cast<double>(res.wall_ns) / 1e6
    };
}

// ---------------------------------------------------------------------------
// stomp_solve_scene — STOMP with scene collision data (single goal)
// ---------------------------------------------------------------------------
static PySTOMPResult stomp_solve_scene(
    py::array_t<float> start_np,
    py::array_t<float> goal_np,
    SceneCollisionDataBuilder& scene_builder,
    STOMP_settings& settings,
    RobotModelBuilder& robot,
    py::object bufs_obj)
{
    auto& model = robot.build_on_device();
    int dim = model.n_dof;

    auto s = start_np.unchecked<1>();
    if (s.shape(0) != dim)
        throw std::runtime_error("start size mismatch with robot n_dof");
    std::vector<float> start(dim);
    for (int i = 0; i < dim; ++i) start[i] = s(i);

    auto g = goal_np.unchecked<1>();
    if (g.shape(0) != dim)
        throw std::runtime_error("goal size mismatch with robot n_dof");
    std::vector<std::vector<float>> goals(1, std::vector<float>(dim));
    for (int j = 0; j < dim; ++j) goals[0][j] = g(j);

    STOMPBuffers* bufs_ptr = nullptr;
    if (!bufs_obj.is_none())
        bufs_ptr = bufs_obj.cast<STOMPBuffers*>();

    auto scene = scene_builder.build();
    auto res = STOMP::solve_runtime_scene(start, goals, scene, settings, model, bufs_ptr);
    return convert_stomp_result(res, dim);
}

// ---------------------------------------------------------------------------
// Python-facing CHOMPResult (mirrors PySTOMPResult so the two optimization
// representatives report an identical cost signal to the benchmark).
// ---------------------------------------------------------------------------
struct PyCHOMPResult {
    bool   solved;
    py::array_t<float> path;   // shape (T, dim), start->goal order
    int    iterations_run;
    float  final_state_cost;
    float  final_control_cost;
    float  final_total_cost;
    double wall_time_ms;
};

static PyCHOMPResult convert_chomp_result(const CHOMPResult& res, int dim) {
    const int n = static_cast<int>(res.path.size());

    py::array_t<float> path({n, dim});
    auto buf = path.mutable_unchecked<2>();
    for (int i = 0; i < n; ++i) {
        const auto& cfg = res.path[i];
        for (int j = 0; j < dim; ++j)
            buf(i, j) = cfg[j];
    }

    return PyCHOMPResult{
        res.solved,
        std::move(path),
        res.iterations_run,
        res.final_state_cost,
        res.final_control_cost,
        res.final_total_cost,
        static_cast<double>(res.wall_ns) / 1e6
    };
}

// ---------------------------------------------------------------------------
// chomp_solve_scene — CHOMP with scene collision data (single goal)
// ---------------------------------------------------------------------------
static PyCHOMPResult chomp_solve_scene(
    py::array_t<float> start_np,
    py::array_t<float> goal_np,
    SceneCollisionDataBuilder& scene_builder,
    CHOMP_settings& settings,
    RobotModelBuilder& robot,
    py::object bufs_obj)
{
    auto& model = robot.build_on_device();
    int dim = model.n_dof;

    auto s = start_np.unchecked<1>();
    if (s.shape(0) != dim)
        throw std::runtime_error("start size mismatch with robot n_dof");
    std::vector<float> start(dim);
    for (int i = 0; i < dim; ++i) start[i] = s(i);

    auto g = goal_np.unchecked<1>();
    if (g.shape(0) != dim)
        throw std::runtime_error("goal size mismatch with robot n_dof");
    std::vector<std::vector<float>> goals(1, std::vector<float>(dim));
    for (int j = 0; j < dim; ++j) goals[0][j] = g(j);

    CHOMPBuffers* bufs_ptr = nullptr;
    if (!bufs_obj.is_none())
        bufs_ptr = bufs_obj.cast<CHOMPBuffers*>();

    auto scene = scene_builder.build();
    auto res = CHOMP::solve_runtime_scene(start, goals, scene, settings, model, bufs_ptr);
    return convert_chomp_result(res, dim);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(prrtc, m) {
    m.doc() = "GPU-parallel motion planning substrate (pRRTC / MIT* / MHA* / STOMP / CHOMP)";

    // --- PlannerResult ---
    py::class_<PyPlannerResult>(m, "PlannerResult")
        .def_readonly("solved",       &PyPlannerResult::solved)
        .def_readonly("path",         &PyPlannerResult::path)
        .def_readonly("iters",        &PyPlannerResult::iters)
        .def_readonly("cost",         &PyPlannerResult::cost)
        .def_readonly("wall_time_ms", &PyPlannerResult::wall_time_ms);

    // --- Settings (unconstrained pRRTC) ---
    py::class_<pRRTC_settings>(m, "Settings")
        .def(py::init<>())
        .def_readwrite("max_samples",     &pRRTC_settings::max_samples)
        .def_readwrite("max_iters",       &pRRTC_settings::max_iters)
        .def_readwrite("num_new_configs", &pRRTC_settings::num_new_configs)
        .def_readwrite("granularity",     &pRRTC_settings::granularity)
        .def_readwrite("range",           &pRRTC_settings::range)
        .def_readwrite("balance",         &pRRTC_settings::balance)
        .def_readwrite("tree_ratio",      &pRRTC_settings::tree_ratio)
        .def_readwrite("dynamic_domain",  &pRRTC_settings::dynamic_domain)
        .def_readwrite("dd_alpha",        &pRRTC_settings::dd_alpha)
        .def_readwrite("dd_radius",       &pRRTC_settings::dd_radius)
        .def_readwrite("dd_min_radius",   &pRRTC_settings::dd_min_radius)
        .def_readwrite("enable_mesh_collision", &pRRTC_settings::enable_mesh_collision)
        .def_readwrite("time_limit_ms",          &pRRTC_settings::time_limit_ms)
        .def_readwrite("gpu_clock_rate_khz",     &pRRTC_settings::gpu_clock_rate_khz)
        .def_readwrite("shortcut_path",          &pRRTC_settings::shortcut_path)
        .def_readwrite("valid_segment_length",   &pRRTC_settings::valid_segment_length)
        .def_readwrite("collision_margin",       &pRRTC_settings::collision_margin)
        .def_readwrite("shortcut_collision_margin", &pRRTC_settings::shortcut_collision_margin);

    // --- EnvironmentBuilder ---
    py::class_<EnvironmentBuilder>(m, "EnvironmentBuilder")
        .def(py::init<>())
        .def("add_sphere", &EnvironmentBuilder::add_sphere,
             py::arg("cx"), py::arg("cy"), py::arg("cz"), py::arg("r"))
        .def("add_cuboid", &EnvironmentBuilder::add_cuboid,
             py::arg("cx"), py::arg("cy"), py::arg("cz"),
             py::arg("rho"), py::arg("theta"), py::arg("phi"),
             py::arg("hx"), py::arg("hy"), py::arg("hz"))
        .def("add_cuboid_quat", &EnvironmentBuilder::add_cuboid_quat,
             py::arg("cx"), py::arg("cy"), py::arg("cz"),
             py::arg("qw"), py::arg("qx"), py::arg("qy"), py::arg("qz"),
             py::arg("sx"), py::arg("sy"), py::arg("sz"),
             "Add cuboid with quaternion orientation and full-extent dimensions")
        .def("add_cylinder", &EnvironmentBuilder::add_cylinder,
             py::arg("cx"), py::arg("cy"), py::arg("cz"),
             py::arg("rho"), py::arg("theta"), py::arg("phi"),
             py::arg("radius"), py::arg("length"));

    // --- RobotModelBuilder ---
    py::class_<RobotModelBuilder>(m, "RobotModelBuilder")
        .def(py::init<>())
        .def_readwrite("n_dof",       &RobotModelBuilder::n_dof)
        .def_readwrite("n_joints",    &RobotModelBuilder::n_joints)
        .def_readwrite("fixed_transforms",        &RobotModelBuilder::fixed_transforms)
        .def_readwrite("joint_types",             &RobotModelBuilder::joint_types)
        .def_readwrite("spheres_flat",            &RobotModelBuilder::spheres_flat)
        .def_readwrite("sphere_to_joint",         &RobotModelBuilder::sphere_to_joint)
        .def_readwrite("self_cc_ranges_flat",     &RobotModelBuilder::self_cc_ranges_flat)
        .def_readwrite("approx_spheres_flat",     &RobotModelBuilder::approx_spheres_flat)
        .def_readwrite("approx_sphere_to_joint",  &RobotModelBuilder::approx_sphere_to_joint)
        .def_readwrite("approx_self_cc_ranges_flat", &RobotModelBuilder::approx_self_cc_ranges_flat)
        .def_readwrite("approx_fixed_transforms", &RobotModelBuilder::approx_fixed_transforms)
        .def_readwrite("approx_joint_types",      &RobotModelBuilder::approx_joint_types)
        .def_readwrite("joint_lower",             &RobotModelBuilder::joint_lower)
        .def_readwrite("joint_upper",             &RobotModelBuilder::joint_upper)
        .def_readwrite("joint_parents",           &RobotModelBuilder::joint_parents)
        .def_readwrite("joint_id_to_dof",         &RobotModelBuilder::joint_id_to_dof)
        .def_readwrite("t_memory_idx",            &RobotModelBuilder::t_memory_idx)
        .def_readwrite("dfs_order",               &RobotModelBuilder::dfs_order)
        .def_readwrite("mesh_bvh_nodes",          &RobotModelBuilder::mesh_bvh_nodes)
        .def_readwrite("mesh_tri_vertices",       &RobotModelBuilder::mesh_tri_vertices)
        .def_readwrite("mesh_link_bvh_root",      &RobotModelBuilder::mesh_link_bvh_root)
        .def_readwrite("mesh_link_tri_offset",    &RobotModelBuilder::mesh_link_tri_offset)
        .def_readwrite("mesh_link_tri_count",     &RobotModelBuilder::mesh_link_tri_count)
        .def("build",   [](RobotModelBuilder& self) { self.build_on_device(); },
             "Allocate GPU memory and upload robot model data")
        .def("update_joint_limits", &RobotModelBuilder::update_joint_limits,
             "Re-upload joint_lower/joint_upper/joint_range to GPU without full rebuild")
        .def("destroy", &RobotModelBuilder::destroy);

    // --- SceneCollisionDataBuilder ---
    py::class_<SceneCollisionDataBuilder>(m, "SceneCollisionDataBuilder")
        .def(py::init<>())
        .def_readwrite("obb_dims_ptr",       &SceneCollisionDataBuilder::obb_dims_ptr)
        .def_readwrite("obb_pose_ptr",       &SceneCollisionDataBuilder::obb_pose_ptr)
        .def_readwrite("obb_enable_ptr",     &SceneCollisionDataBuilder::obb_enable_ptr)
        .def_readwrite("n_obbs",             &SceneCollisionDataBuilder::n_obbs)
        .def_readwrite("voxel_params_ptr",   &SceneCollisionDataBuilder::voxel_params_ptr)
        .def_readwrite("voxel_pose_ptr",     &SceneCollisionDataBuilder::voxel_pose_ptr)
        .def_readwrite("voxel_enable_ptr",   &SceneCollisionDataBuilder::voxel_enable_ptr)
        .def_readwrite("voxel_features_ptr", &SceneCollisionDataBuilder::voxel_features_ptr)
        .def_readwrite("n_voxel_layers",          &SceneCollisionDataBuilder::n_voxel_layers)
        .def_readwrite("max_voxels_per_layer",    &SceneCollisionDataBuilder::max_voxels_per_layer)
        .def_readwrite("sphere_acm_mask_ptr",     &SceneCollisionDataBuilder::sphere_acm_mask_ptr)
        .def_readwrite("sphere_obb_acm_mask_ptr", &SceneCollisionDataBuilder::sphere_obb_acm_mask_ptr)
        .def_readwrite("sphere_obb_acm_mask_approx_ptr", &SceneCollisionDataBuilder::sphere_obb_acm_mask_approx_ptr);

    // --- Top-level solve functions ---
    // Backward compatible: hardcoded Panda
    m.def("solve_panda", &solve_panda,
          py::arg("start"), py::arg("goals"),
          py::arg("env"), py::arg("settings"),
          "Unconstrained pRRTC for Panda 7-DOF. Returns PlannerResult with path in start->goal order.");

    // Runtime: any robot via RobotModelBuilder
    m.def("solve", &solve_runtime,
          py::arg("start"), py::arg("goals"),
          py::arg("env"), py::arg("settings"),
          py::arg("robot"),
          "Unconstrained pRRTC for any robot. Requires a RobotModelBuilder.");

    // Scene-based: zero-copy from GPU tensors
    m.def("solve_scene", &solve_runtime_scene,
          py::arg("start"), py::arg("goals"),
          py::arg("scene"), py::arg("settings"),
          py::arg("robot"), py::arg("bufs") = py::none(),
          "Unconstrained pRRTC using scene collision data (zero-copy from GPU tensors).");

    m.def("check_collision_mesh", &check_collision_mesh_py,
          py::arg("configs"), py::arg("scene"), py::arg("robot"),
          "Batch mesh collision check. configs: [N, n_dof]. Returns [N] bool array.");

    m.def("check_collision_scene", &check_collision_scene_py,
          py::arg("configs"), py::arg("scene"), py::arg("robot"),
          py::arg("collision_margin") = 0.0f,
          "Batch scene collision check using planning-consistent FK (__sinf/__cosf). "
          "No approx early-exit, no mesh BVH. Full-sphere check against OBB + ESDF.");

    // --- MITStarSettings ---
    py::class_<MITStar_settings, pRRTC_settings>(m, "MITStarSettings")
        .def(py::init<>())
        .def_readwrite("batch_size",             &MITStar_settings::batch_size)
        .def_readwrite("time_limit_ms",          &MITStar_settings::time_limit_ms)
        .def_readwrite("early_exit_ms",          &MITStar_settings::early_exit_ms)
        .def_readwrite("eta_knn",                &MITStar_settings::eta_knn)
        .def_readwrite("gamma_rgg",              &MITStar_settings::gamma_rgg)
        .def_readwrite("max_neighbors",          &MITStar_settings::max_neighbors)
        .def_readwrite("m_reverse_eval",         &MITStar_settings::m_reverse_eval)
        .def_readwrite("m_forward_eval",         &MITStar_settings::m_forward_eval)
        .def_readwrite("initial_sparse_factor",  &MITStar_settings::initial_sparse_factor)
        .def_readwrite("initial_suboptimality",  &MITStar_settings::initial_suboptimality)
        .def_readwrite("use_eis",                &MITStar_settings::use_eis)
        .def_readwrite("valid_segment_length",   &MITStar_settings::valid_segment_length)
        .def_readwrite("max_nodes",              &MITStar_settings::max_nodes)
        .def_readwrite("max_edges_per_node",     &MITStar_settings::max_edges_per_node)
        .def_readwrite("debug_dump",             &MITStar_settings::debug_dump)
        .def_readwrite("clearance_weight",       &MITStar_settings::clearance_weight)
        .def_readwrite("clearance_epsilon",      &MITStar_settings::clearance_epsilon)
        .def_readwrite("max_clearance_penalty",  &MITStar_settings::max_clearance_penalty)
        .def_readwrite("collision_margin",        &MITStar_settings::collision_margin)
        .def_readwrite("adaptive_batch",         &MITStar_settings::adaptive_batch)
        .def_readwrite("min_batch_size",         &MITStar_settings::min_batch_size)
        .def_readwrite("shortcut_path",          &MITStar_settings::shortcut_path);

    // --- MITStarResult ---
    py::class_<PyMITStarResult>(m, "MITStarResult")
        .def(py::init<>())
        .def_readonly("solved",          &PyMITStarResult::solved)
        .def_readonly("path",            &PyMITStarResult::path)
        .def_readonly("total_nodes",     &PyMITStarResult::total_nodes)
        .def_readonly("total_batches",   &PyMITStarResult::total_batches)
        .def_readonly("cost",            &PyMITStarResult::cost)
        .def_readonly("cost_history",    &PyMITStarResult::cost_history)
        .def_readonly("time_history_ms", &PyMITStarResult::time_history_ms)
        .def_readonly("wall_time_ms",    &PyMITStarResult::wall_time_ms)
        .def_readonly("total_forward_edges_evaluated", &PyMITStarResult::total_forward_edges_evaluated)
        .def_readonly("total_reverse_edges_evaluated", &PyMITStarResult::total_reverse_edges_evaluated)
        .def_readonly("reverse_restarts", &PyMITStarResult::reverse_restarts)
        .def_readonly("eis_cost",        &PyMITStarResult::eis_cost)
        .def_readonly("total_iterations", &PyMITStarResult::total_iterations)
        // FIX_B3 diagnostics
        .def_readonly("sparse_midpoints_checked", &PyMITStarResult::sparse_midpoints_checked)
        .def_readonly("sparse_self_coll_hits",    &PyMITStarResult::sparse_self_coll_hits)
        .def_readonly("sparse_scene_coll_hits",   &PyMITStarResult::sparse_scene_coll_hits)
        .def_readonly("full_midpoints_checked",   &PyMITStarResult::full_midpoints_checked)
        .def_readonly("full_self_coll_hits",      &PyMITStarResult::full_self_coll_hits)
        .def_readonly("full_scene_coll_hits",     &PyMITStarResult::full_scene_coll_hits);

    // --- MIT* solve ---
    m.def("mitstar_solve_scene", &mitstar_solve_scene,
          py::arg("start"), py::arg("goals"),
          py::arg("scene"), py::arg("settings"),
          py::arg("robot"), py::arg("bufs") = py::none(),
          "MIT* asymptotically optimal planner using scene collision data (zero-copy from GPU tensors).");

    // --- MHAStarSettings ---
    py::class_<MHAStar_settings, pRRTC_settings>(m, "MHAStarSettings")
        .def(py::init<>())
        .def_readwrite("w1",             &MHAStar_settings::w1)
        .def_readwrite("w2",             &MHAStar_settings::w2)
        .def_readwrite("num_inad",       &MHAStar_settings::num_inad)
        .def_readwrite("queue_sel",      &MHAStar_settings::queue_sel)
        .def_readwrite("batch_edges",    &MHAStar_settings::batch_edges)
        .def_readwrite("lazy",           &MHAStar_settings::lazy)
        .def_readwrite("lattice_step",   &MHAStar_settings::lattice_step)
        .def_readwrite("grid_cell",      &MHAStar_settings::grid_cell)
        .def_readwrite("goal_radius",    &MHAStar_settings::goal_radius)
        .def_readwrite("goal_connect_steps", &MHAStar_settings::goal_connect_steps)
        .def_readwrite("edge_resolution", &MHAStar_settings::edge_resolution)
        .def_readwrite("check_self",     &MHAStar_settings::check_self)
        .def_readwrite("granularity",    &MHAStar_settings::granularity)
        .def_readwrite("collision_margin", &MHAStar_settings::collision_margin)
        .def_readwrite("time_limit_ms",  &MHAStar_settings::time_limit_ms);

    // --- MHAStarResult ---
    py::class_<PyMHAStarResult>(m, "MHAStarResult")
        .def(py::init<>())
        .def_readonly("solved",          &PyMHAStarResult::solved)
        .def_readonly("path",            &PyMHAStarResult::path)
        .def_readonly("cost",            &PyMHAStarResult::cost)
        .def_readonly("expansions",      &PyMHAStarResult::expansions)
        .def_readonly("edges_evaluated", &PyMHAStarResult::edges_evaluated)
        .def_readonly("wall_time_ms",    &PyMHAStarResult::wall_time_ms);

    // --- MHA* solve ---
    m.def("mhastar_solve_scene", &mhastar_solve_scene,
          py::arg("start"), py::arg("goals"),
          py::arg("scene"), py::arg("settings"),
          py::arg("robot"),
          "SMHA* CPU search with GPU-batched lazy edge evaluation (multi-goal, scene collision data).");

    // --- WPASESettings ---
    py::class_<WPASE_settings, pRRTC_settings>(m, "WPASESettings")
        .def(py::init<>())
        .def_readwrite("heuristic_w",    &WPASE_settings::heuristic_w)
        .def_readwrite("num_parallel",   &WPASE_settings::num_parallel)
        .def_readwrite("batch_edges",    &WPASE_settings::batch_edges)
        .def_readwrite("lattice_step",   &WPASE_settings::lattice_step)
        .def_readwrite("grid_cell",      &WPASE_settings::grid_cell)
        .def_readwrite("goal_radius",    &WPASE_settings::goal_radius)
        .def_readwrite("goal_connect_steps", &WPASE_settings::goal_connect_steps)
        .def_readwrite("edge_resolution", &WPASE_settings::edge_resolution)
        .def_readwrite("check_self",     &WPASE_settings::check_self)
        .def_readwrite("granularity",    &WPASE_settings::granularity)
        .def_readwrite("collision_margin", &WPASE_settings::collision_margin)
        .def_readwrite("time_limit_ms",  &WPASE_settings::time_limit_ms);

    // --- WPASEResult ---
    py::class_<PyWPASEResult>(m, "WPASEResult")
        .def_readonly("solved",          &PyWPASEResult::solved)
        .def_readonly("path",            &PyWPASEResult::path)
        .def_readonly("cost",            &PyWPASEResult::cost)
        .def_readonly("expansions",      &PyWPASEResult::expansions)
        .def_readonly("edges_evaluated", &PyWPASEResult::edges_evaluated)
        .def_readonly("wall_time_ms",    &PyWPASEResult::wall_time_ms);

    // --- wPA*SE solve ---
    m.def("wpase_solve_scene", &wpase_solve_scene,
          py::arg("start"), py::arg("goals"),
          py::arg("scene"), py::arg("settings"),
          py::arg("robot"),
          "wPA*SE CPU parallel-expansion weighted-A* search with GPU-batched edge evaluation (multi-goal, scene collision data).");

    // --- SolverBuffers (pre-allocated GPU buffers for pRRTC) ---
    py::class_<SolverBuffers>(m, "SolverBuffers")
        .def_readonly("owns_memory", &SolverBuffers::owns_memory)
        .def_readonly("max_samples", &SolverBuffers::max_samples)
        .def_readonly("n_dof",       &SolverBuffers::n_dof)
        .def_static("create", &SolverBuffers::create,
            py::arg("max_samples"), py::arg("n_dof"),
            py::arg("num_new_configs"), py::arg("granularity"),
            py::arg("n_joints"), py::arg("enable_mesh"),
            "Allocate GPU buffers for pRRTC solve calls.")
        .def("destroy", &SolverBuffers::destroy,
            "Free all GPU buffers. Safe to call multiple times.");

    // --- MITStarBuffers (pre-allocated GPU buffers for MIT*) ---
    py::class_<MITStarBuffers>(m, "MITStarBuffers")
        .def_readonly("owns_memory", &MITStarBuffers::owns_memory)
        .def_readonly("n_dof",       &MITStarBuffers::n_dof)
        .def_readonly("max_nodes",   &MITStarBuffers::max_nodes)
        .def_static("create", &MITStarBuffers::create,
            py::arg("n_dof"), py::arg("batch_size"),
            py::arg("max_nodes"), py::arg("max_edges_per_node"),
            py::arg("max_neighbors"), py::arg("m_reverse_eval"),
            py::arg("m_forward_eval"), py::arg("enable_clearance") = false,
            "Allocate GPU buffers for MIT* solve calls.")
        .def("destroy", &MITStarBuffers::destroy,
            "Free all GPU buffers. Safe to call multiple times.");

    // --- STOMPSettings ---
    py::class_<STOMP_settings, pRRTC_settings>(m, "STOMPSettings")
        .def(py::init<>())
        .def_readwrite("num_batch",                     &STOMP_settings::num_batch)
        .def_readwrite("num_timesteps",                  &STOMP_settings::num_timesteps)
        .def_readwrite("num_iterations",                 &STOMP_settings::num_iterations)
        .def_readwrite("num_iterations_after_valid",     &STOMP_settings::num_iterations_after_valid)
        .def_readwrite("num_rollouts_new",                &STOMP_settings::num_rollouts_new)
        .def_readwrite("num_rollouts_old",                &STOMP_settings::num_rollouts_old)
        .def_readwrite("delta_t",                         &STOMP_settings::delta_t)
        .def_readwrite("control_cost_weight",             &STOMP_settings::control_cost_weight)
        .def_readwrite("collision_cost_weight",           &STOMP_settings::collision_cost_weight)
        .def_readwrite("exponentiated_cost_sensitivity",  &STOMP_settings::exponentiated_cost_sensitivity)
        .def_readwrite("world_collision_margin",          &STOMP_settings::world_collision_margin)
        .def_readwrite("self_collision_margin",           &STOMP_settings::self_collision_margin)
        .def_readwrite("convergence_eps",                 &STOMP_settings::convergence_eps)
        .def_readwrite("collision_free_tol",              &STOMP_settings::collision_free_tol)
        .def_readwrite("collision_check_step",            &STOMP_settings::collision_check_step)
        .def_readwrite("noise_scale",                     &STOMP_settings::noise_scale);

    // --- STOMPResult ---
    py::class_<PySTOMPResult>(m, "STOMPResult")
        .def_readonly("solved",              &PySTOMPResult::solved)
        .def_readonly("path",                &PySTOMPResult::path)
        .def_readonly("iterations_run",       &PySTOMPResult::iterations_run)
        .def_readonly("final_state_cost",     &PySTOMPResult::final_state_cost)
        .def_readonly("final_control_cost",   &PySTOMPResult::final_control_cost)
        .def_readonly("final_total_cost",     &PySTOMPResult::final_total_cost)
        .def_readonly("wall_time_ms",         &PySTOMPResult::wall_time_ms);

    m.def("stomp_solve_scene", &stomp_solve_scene,
        py::arg("start"), py::arg("goal"), py::arg("scene"),
        py::arg("settings"), py::arg("robot"), py::arg("bufs") = py::none(),
        "Run GPU-native STOMP against zero-copy scene collision data (single goal).");

    // --- STOMPBuffers (pre-allocated GPU buffers for STOMP) ---
    py::class_<STOMPBuffers>(m, "STOMPBuffers")
        .def_readonly("owns_memory",       &STOMPBuffers::owns_memory)
        .def_readonly("n_dof",             &STOMPBuffers::n_dof)
        .def_readonly("num_timesteps",     &STOMPBuffers::num_timesteps)
        .def_readonly("num_rollouts_new",  &STOMPBuffers::num_rollouts_new)
        .def_readonly("num_rollouts_old",  &STOMPBuffers::num_rollouts_old)
        .def_readonly("num_rollouts_all",  &STOMPBuffers::num_rollouts_all)
        .def_readonly("num_batch",         &STOMPBuffers::num_batch)
        .def_static("create", &STOMPBuffers::create,
            py::arg("n_dof"), py::arg("num_timesteps"),
            py::arg("num_rollouts_new"), py::arg("num_rollouts_old"),
            py::arg("num_batch") = 1,
            "Allocate GPU buffers for STOMP solve calls.")
        .def("destroy", &STOMPBuffers::destroy,
            "Free all GPU buffers. Safe to call multiple times.");

    // --- CHOMPSettings ---
    py::class_<CHOMP_settings, pRRTC_settings>(m, "CHOMPSettings")
        .def(py::init<>())
        .def_readwrite("num_timesteps",                  &CHOMP_settings::num_timesteps)
        .def_readwrite("num_iterations",                 &CHOMP_settings::num_iterations)
        .def_readwrite("num_iterations_after_valid",     &CHOMP_settings::num_iterations_after_valid)
        .def_readwrite("step_size",                       &CHOMP_settings::step_size)
        .def_readwrite("smoothness_weight",               &CHOMP_settings::smoothness_weight)
        .def_readwrite("collision_cost_weight",           &CHOMP_settings::collision_cost_weight)
        .def_readwrite("fd_epsilon",                      &CHOMP_settings::fd_epsilon)
        .def_readwrite("world_collision_margin",          &CHOMP_settings::world_collision_margin)
        .def_readwrite("self_collision_margin",           &CHOMP_settings::self_collision_margin)
        .def_readwrite("delta_t",                         &CHOMP_settings::delta_t)
        .def_readwrite("convergence_eps",                 &CHOMP_settings::convergence_eps)
        .def_readwrite("collision_free_tol",              &CHOMP_settings::collision_free_tol)
        .def_readwrite("collision_check_step",            &CHOMP_settings::collision_check_step);

    // --- CHOMPResult ---
    py::class_<PyCHOMPResult>(m, "CHOMPResult")
        .def_readonly("solved",              &PyCHOMPResult::solved)
        .def_readonly("path",                &PyCHOMPResult::path)
        .def_readonly("iterations_run",       &PyCHOMPResult::iterations_run)
        .def_readonly("final_state_cost",     &PyCHOMPResult::final_state_cost)
        .def_readonly("final_control_cost",   &PyCHOMPResult::final_control_cost)
        .def_readonly("final_total_cost",     &PyCHOMPResult::final_total_cost)
        .def_readonly("wall_time_ms",         &PyCHOMPResult::wall_time_ms);

    m.def("chomp_solve_scene", &chomp_solve_scene,
        py::arg("start"), py::arg("goal"), py::arg("scene"),
        py::arg("settings"), py::arg("robot"), py::arg("bufs") = py::none(),
        "Run GPU-native CHOMP against zero-copy scene collision data (single goal).");

    // --- CHOMPBuffers (pre-allocated GPU buffers for CHOMP) ---
    py::class_<CHOMPBuffers>(m, "CHOMPBuffers")
        .def_readonly("owns_memory",       &CHOMPBuffers::owns_memory)
        .def_readonly("n_dof",             &CHOMPBuffers::n_dof)
        .def_readonly("num_timesteps",     &CHOMPBuffers::num_timesteps)
        .def_static("create", &CHOMPBuffers::create,
            py::arg("n_dof"), py::arg("num_timesteps"),
            "Allocate GPU buffers for CHOMP solve calls.")
        .def("destroy", &CHOMPBuffers::destroy,
            "Free all GPU buffers. Safe to call multiple times.");
}
