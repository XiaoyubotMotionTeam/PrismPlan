// ============================================================================
// WPASE::solve_runtime_scene — data-driven entry point for the CPU wPA*SE
// (weighted Parallel A* for Slow Expansions) search with GPU-batched edge
// evaluation.
//
// This is the ONLY translation unit that includes wpase_search.hh (which in
// turn pulls in edge_eval.cuh with the __global__ evaluate_edges_batch kernel),
// so the kernel is defined exactly once — mirroring the per-solver static-lib
// layout of MITStar.cu / stomp.cu / mhastar_solve.cu.
//
// Search design (host-computable, NO FK — RobotModel FK/limit data lives on the
// device): single OPEN queue keyed f = g + w*h with an admissible joint-space
// L2 heuristic to the nearest goal; each round expands a mutually-independent
// set of states (PA*SE rule, pairwise L2 heuristic) whose successor edges are
// all confirmed on the GPU in one batch. Successors are a joint lattice
// (+/- lattice_step per joint, clipped to limits copied to host once here).
// ============================================================================

#include "src/planning/Planners.hh"
#include "src/planning/wpase_search.hh"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace WPASE {

WPASEResult solve_runtime_scene(
    std::vector<float>& start,
    std::vector<std::vector<float>>& goals,
    ppln::collision::SceneCollisionData& scene,
    WPASE_settings& settings,
    ppln::RobotModel& model)
{
    using namespace ppln::search;
    const int dim = model.n_dof;

    // NOTE: fully-qualified to disambiguate from ppln::search::WPASEResult,
    // which is visible via the `using namespace ppln::search` above. This is the
    // global (Planners.hh / pybind) result.
    ::WPASEResult out;
    if (dim <= 0 || dim > ppln::MAX_DIM || goals.empty())
        return out;

    // --- Joint limits: copy device -> host once for the lattice clip ---
    std::vector<float> lo(dim), hi(dim);
    cudaMemcpy(lo.data(), model.joint_lower, dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hi.data(), model.joint_upper, dim * sizeof(float), cudaMemcpyDeviceToHost);
    std::array<std::array<float, 2>, ppln::MAX_DIM> limits{};
    for (int i = 0; i < dim; ++i) { limits[i][0] = lo[i]; limits[i][1] = hi[i]; }

    // Per-DoF normalisation from those same limits (dof_scale.hh), so
    // lattice_step / grid_cell / goal_radius mean the same fraction of travel on
    // a 0.386 m prismatic torso as on a 6.3 rad revolute joint.
    const ppln::DofScale scale = ppln::make_dof_scale(lo.data(), hi.data(), dim);

    // --- start / goals into fixed-size WConfig ---
    WConfig sc{};
    for (int i = 0; i < dim; ++i) sc[i] = start[i];
    std::vector<WConfig> gs;
    gs.reserve(goals.size());
    for (const auto& g : goals) {
        WConfig c{};
        for (int i = 0; i < dim; ++i) c[i] = g[i];
        gs.push_back(c);
    }

    // --- GPU edge evaluator: cap must hold a full round's successor edges ---
    // Each expansion now emits 2*dim lattice successors plus, near the goal, one
    // goal-connection edge per goal.
    const int round_edges = settings.num_parallel * (2 * dim + (int)gs.size());
    const int cap = std::max(settings.batch_edges, round_edges);
    EdgeEvalGpu gpu(dim, cap, scene, model);

    // --- planner settings ---
    WPASESettings ws;
    ws.heuristic_w  = settings.heuristic_w;
    ws.num_parallel = std::max(1, settings.num_parallel);
    ws.dim          = dim;
    ws.time_limit_ms = settings.time_limit_ms;
    ws.batch_edges  = cap;
    ws.edge = EdgeEvalParams{ settings.granularity, settings.collision_margin,
                              settings.check_self, /*compute_clearance*/ false,
                              /*enable_mesh*/ false, settings.edge_resolution };

    // --- pluggable problem definition ---
    // Lattice (and the dedup grid that must line up with it) anchored at the
    // first goal, so the goal is itself a lattice vertex — see mhastar_solve.cu.
    const WConfig& anchor = gs[0];

    auto succ    = with_wpase_goal_connect(
                       make_wpase_lattice_succ(dim, settings.lattice_step, scale, anchor, &limits),
                       gs, scale, dim,
                       settings.goal_connect_steps * settings.lattice_step);
    auto key_fn  = make_wpase_grid_key(dim, settings.grid_cell, scale, anchor);
    auto heur_fn = make_wpase_l2_heur(gs, dim);
    auto binh_fn = make_wpase_binary_l2(dim);
    // Tight tolerance, not grown with the lattice step — see mhastar_solve.cu.
    auto goal_fn = make_wpase_goal_radius(gs, dim, settings.goal_radius, scale);

    WPASEPlanner planner(ws, &gpu, succ, heur_fn, binh_fn, goal_fn, key_fn);
    auto r = planner.solve(sc);

    out.solved          = r.solved;
    out.cost            = r.cost;
    out.expansions      = r.expansions;
    out.edges_evaluated = r.edges_evaluated;
    out.wall_ns         = (std::size_t)(r.elapsed_ms * 1e6f);
    out.path.reserve(r.path.size());
    for (const auto& c : r.path) {
        std::vector<float> v(dim);
        for (int i = 0; i < dim; ++i) v[i] = c[i];
        out.path.push_back(std::move(v));
    }
    return out;
}

} // namespace WPASE
