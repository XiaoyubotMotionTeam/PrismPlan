#pragma once

#include <vector>
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <cmath>

#include "Robots.hh"
#include "robot_model.cuh"
#include "src/collision/environment.hh"
#include "src/collision/scene_collision_data.hh"
#include "pRRTC_settings.hh"
#include "MITStar_settings.hh"
#include "MHAStar_settings.hh"
#include "WPASE_settings.hh"
#include "solver_buffers.hh"
#include "mitstar_buffers.hh"
#include "STOMP_settings.hh"
#include "stomp_buffers.hh"
#include "CHOMP_settings.hh"
#include "chomp_buffers.hh"

template <typename Robot>
struct PlannerResult {
    bool solved = false;
    std::vector<typename Robot::Configuration> path; 
    int start_tree_size = 0;
    int goal_tree_size = 0;
    int path_length = 0;
    int iters = 0;
    float cost = 0.0;
    std::size_t wall_ns = 0; // wall time of the solve function
    std::size_t kernel_ns = 0; // just kernel runtime
    std::size_t copy_ns = 0; // time to copy start/goals to gpu and copy path and path size back
};

template <typename Robot>
inline float l2dist(typename Robot::Configuration &a, typename Robot::Configuration &b)
{
    float res = 0;
    float diff;
    for (int i = 0; i < a.size(); i++) {
        diff = a[i] - b[i];
        res += diff * diff;
    }
    return sqrt(res);
}

template <typename Robot>
inline void print_cfg_ptr(float *config) {
    for (int i = 0; i < Robot::dimension; i++) {
        std::cout << config[i] << " ";
    }
    std::cout << "\n";
}

template <typename Robot>
inline void print_cfg(typename Robot::Configuration &config) {
    for (int i = 0; i < Robot::dimension; i++) {
        std::cout << config[i] << " ";
    }
    std::cout << "\n";
}

template <typename Robot>
inline void print_cfg_to_ss(typename Robot::Configuration &config, std::stringstream &out) {
    for (int i = 0; i < Robot::dimension; i++) {
        out << config[i] << " ";
    }
    out << "\\n";
}



inline std::size_t get_elapsed_nanoseconds(const std::chrono::time_point<std::chrono::steady_clock> &start)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}

/* This file handles the declarations of each solve function so that they may be called from .cpp files. Implementation is in .cu files.*/
namespace pRRTC {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment, pRRTC_settings &settings);
}

// Runtime (data-driven) planner result — dimension not known at compile time
struct RuntimePlannerResult {
    bool solved = false;
    std::vector<std::vector<float>> path;
    int start_tree_size = 0;
    int goal_tree_size = 0;
    int path_length = 0;
    int iters = 0;
    float cost = 0.0f;
    std::size_t wall_ns = 0;
    std::size_t kernel_ns = 0;
    std::size_t copy_ns = 0;
};

#include "src/collision/mesh_collision_data.hh"

// Runtime solve declarations
namespace pRRTC {
    RuntimePlannerResult solve_runtime(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        ppln::collision::Environment<float>& h_environment,
        pRRTC_settings& settings,
        ppln::RobotModel& model
    );
}

// Scene-based runtime solve declarations (zero-copy from GPU tensors)
namespace pRRTC {
    RuntimePlannerResult solve_runtime_scene(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        ppln::collision::SceneCollisionData& scene,
        pRRTC_settings& settings,
        ppln::RobotModel& model,
        SolverBuffers* bufs = nullptr
    );
}

// Standalone mesh collision check (batch of configs)
namespace pRRTC {
    void check_collision_mesh(
        const float* h_configs,       // [N * n_dof] host array
        int N,                        // number of configurations
        ppln::collision::SceneCollisionData& scene,
        ppln::RobotModel& model,
        bool* h_results               // [N] output: true = collision
    );
}

// Scene collision check using planning-consistent FK (__sinf/__cosf).
// No approx early-exit, no mesh BVH. Full-sphere check against OBB + ESDF.
namespace pRRTC {
    void check_collision_scene(
        const float* h_configs,       // [N * n_dof] host array
        int N,                        // number of configurations
        ppln::collision::SceneCollisionData& scene,
        ppln::RobotModel& model,
        float collision_margin,       // extra margin added to sphere radii
        bool* h_results               // [N] output: true = collision
    );
}

// MIT* result — extends RuntimePlannerResult with anytime cost history
struct MITStarResult {
    bool solved = false;
    std::vector<std::vector<float>> path;      // best path found (start→goal order)
    int total_nodes = 0;
    int total_batches = 0;
    float cost = std::numeric_limits<float>::infinity();
    std::vector<float> cost_history;            // cost after each batch that improves
    std::vector<float> time_history_ms;         // wall time at each improvement
    std::size_t wall_ns = 0;
    int total_forward_edges_evaluated = 0;
    int total_reverse_edges_evaluated = 0;
    int reverse_restarts = 0;
    float eis_cost = std::numeric_limits<float>::infinity();
    int total_iterations = 0;

    // FIX_B3 diagnostics: edge CC self-collision / scene-collision counters
    int sparse_midpoints_checked = 0;
    int sparse_self_coll_hits = 0;
    int sparse_scene_coll_hits = 0;
    int full_midpoints_checked = 0;
    int full_self_coll_hits = 0;
    int full_scene_coll_hits = 0;
};

// MIT* solve declarations
namespace MITStar {
    MITStarResult solve_runtime_scene(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        ppln::collision::SceneCollisionData& scene,
        MITStar_settings& settings,
        ppln::RobotModel& model,
        MITStarBuffers* bufs = nullptr
    );
}

// MHA* (Shared Multi-Heuristic A*) result — CPU search + GPU batched lazy edges
struct MHAStarResult {
    bool solved = false;
    std::vector<std::vector<float>> path;   // start->goal order
    float cost = std::numeric_limits<float>::infinity();
    int expansions = 0;
    int edges_evaluated = 0;                // exact GPU edge count
    std::size_t wall_ns = 0;
};

// MHA* solve declarations
namespace MHAStar {
    MHAStarResult solve_runtime_scene(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        ppln::collision::SceneCollisionData& scene,
        MHAStar_settings& settings,
        ppln::RobotModel& model
    );
}

// wPA*SE (weighted Parallel A* for Slow Expansions) result — CPU parallel-
// expansion weighted-A* search + GPU batched edge evaluation
struct WPASEResult {
    bool solved = false;
    std::vector<std::vector<float>> path;   // start->goal order
    float cost = std::numeric_limits<float>::infinity();
    int expansions = 0;
    int edges_evaluated = 0;                // exact GPU edge count
    std::size_t wall_ns = 0;
};

// wPA*SE solve declarations
namespace WPASE {
    WPASEResult solve_runtime_scene(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        ppln::collision::SceneCollisionData& scene,
        WPASE_settings& settings,
        ppln::RobotModel& model
    );
}

// STOMP result — single-goal trajectory optimizer, no anytime tree bookkeeping
struct STOMPResult {
    bool solved = false;
    std::vector<std::vector<float>> path;   // optimized geometric waypoints (T x D), start->goal order
    int iterations_run = 0;
    float final_state_cost = 0.0f;
    float final_control_cost = 0.0f;
    float final_total_cost = 0.0f;
    std::size_t wall_ns = 0;
};

// STOMP solve declarations
namespace STOMP {
    STOMPResult solve_runtime_scene(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        ppln::collision::SceneCollisionData& scene,
        STOMP_settings& settings,
        ppln::RobotModel& model,
        STOMPBuffers* bufs = nullptr
    );
}

// CHOMP result — gradient trajectory optimizer, single-goal; mirrors STOMPResult
// so the two optimization representatives report a comparable cost signal.
struct CHOMPResult {
    bool solved = false;
    std::vector<std::vector<float>> path;   // optimized geometric waypoints (T x D), start->goal order
    int iterations_run = 0;
    float final_state_cost = 0.0f;
    float final_control_cost = 0.0f;
    float final_total_cost = 0.0f;
    std::size_t wall_ns = 0;
};

// CHOMP solve declarations
namespace CHOMP {
    CHOMPResult solve_runtime_scene(
        std::vector<float>& start,
        std::vector<std::vector<float>>& goals,
        ppln::collision::SceneCollisionData& scene,
        CHOMP_settings& settings,
        ppln::RobotModel& model,
        CHOMPBuffers* bufs = nullptr
    );
}
