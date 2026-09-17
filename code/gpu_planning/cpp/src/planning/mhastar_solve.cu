// ============================================================================
// MHAStar::solve_runtime_scene — data-driven entry point for the CPU SMHA*
// search with GPU-batched lazy edge evaluation.
//
// This is the ONLY translation unit that includes mhastar_search.hh (which in
// turn pulls in edge_eval.cuh with the __global__ evaluate_edges_batch kernel),
// so the kernel is defined exactly once — mirroring the per-solver static-lib
// layout of MITStar.cu / stomp.cu.
//
// Heuristic design (host-computable, NO FK — RobotModel FK/limit data lives on
// the device):
//   * anchor queue 0  : admissible joint-space L2 to the nearest goal.
//   * inadmissible 1..N: the same L2 with per-DoF weights in [1, infl], so each
//     MAY overestimate the cost-to-go. Odd qi inflates proximal joints, even qi
//     distal ones, giving two genuinely distinct functions. All queues share g.
//     Only two are distinct, so num_inad > 2 buys duplicates.
// Successors are a joint lattice (+/- lattice_step per joint, clipped to limits
// copied to host once here). Edge validity+cost is confirmed on the GPU.
// ============================================================================

#include "src/planning/Planners.hh"
#include "src/planning/mhastar_search.hh"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace MHAStar {

MHAStarResult solve_runtime_scene(
    std::vector<float>& start,
    std::vector<std::vector<float>>& goals,
    ppln::collision::SceneCollisionData& scene,
    MHAStar_settings& settings,
    ppln::RobotModel& model)
{
    using namespace ppln::search;
    const int dim = model.n_dof;

    // NOTE: fully-qualified to disambiguate from ppln::search::MHAStarResult,
    // which becomes visible in the global scope via the `using namespace
    // ppln::search` above. This is the global (Planners.hh / pybind) result.
    ::MHAStarResult out;
    if (dim <= 0 || dim > ppln::MAX_DIM || goals.empty())
        return out;

    // --- Joint limits: copy device -> host once for the lattice clip ---
    std::vector<float> lo(dim), hi(dim);
    cudaMemcpy(lo.data(), model.joint_lower, dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hi.data(), model.joint_upper, dim * sizeof(float), cudaMemcpyDeviceToHost);
    std::array<std::array<float, 2>, ppln::MAX_DIM> limits{};
    for (int i = 0; i < dim; ++i) { limits[i][0] = lo[i]; limits[i][1] = hi[i]; }

    // Per-DoF normalisation from those same limits, so lattice_step, grid_cell
    // and goal_radius mean the same fraction of travel on a 0.386 m prismatic
    // torso as on a 6.3 rad revolute joint (dof_scale.hh).
    const ppln::DofScale scale = ppln::make_dof_scale(lo.data(), hi.data(), dim);

    // --- start / goals into fixed-size Config ---
    Config sc{};
    for (int i = 0; i < dim; ++i) sc[i] = start[i];
    std::vector<Config> gs;
    gs.reserve(goals.size());
    for (const auto& g : goals) {
        Config c{};
        for (int i = 0; i < dim; ++i) c[i] = g[i];
        gs.push_back(c);
    }

    // --- GPU lazy-edge evaluator (persistent device/pinned buffers) ---
    EdgeEvalGpu gpu(dim, settings.batch_edges, scene, model);

    // --- planner settings ---
    MHAStarSettings ms;
    ms.w1           = settings.w1;
    ms.w2           = settings.w2;
    ms.num_inad     = settings.num_inad;
    ms.dim          = dim;
    ms.time_limit_ms = settings.time_limit_ms;
    ms.batch_edges  = settings.batch_edges;
    ms.lazy         = settings.lazy;
    ms.queue_sel    = (settings.queue_sel == 1) ? QueueSel::BestKey : QueueSel::RoundRobin;
    ms.edge = EdgeEvalParams{ settings.granularity, settings.collision_margin,
                              settings.check_self, /*compute_clearance*/ false,
                              /*enable_mesh*/ false, settings.edge_resolution };

    // --- pluggable problem definition ---
    // The lattice (and the dedup grid that must line up with it) is anchored at
    // the first goal, so the goal is itself a lattice vertex and exactly
    // reachable. Start-anchored, the nearest vertex to the goal is in collision
    // on 47 of 70 real MBM Fetch problems even though the goal is free.
    const Config& anchor = gs[0];

    auto succ = with_goal_connect(
        make_lattice_succ(dim, settings.lattice_step, scale, anchor, &limits),
        gs, scale, dim, settings.goal_connect_steps * settings.lattice_step);

    // Tight goal tolerance, NOT grown with the lattice step: the lattice is
    // anchored at gs[0] and goal-connect emits every goal as a successor, so a
    // goal is an exactly reachable vertex. A step-proportional tolerance would
    // instead declare success up to sqrt(dim)*step/2 away, with no evidence that
    // the remaining motion is collision-free.
    const float radius = settings.goal_radius;
    auto goal_fn = [gs, dim, radius, scale](const Config& q) {
        for (const auto& g : gs) {
            float d = 0.f;
            for (int i = 0; i < dim; ++i) {
                float e = (q[i] - g[i]) * scale.inv[i];
                d += e * e;
            }
            if (std::sqrt(d) <= radius) return true;
        }
        return false;
    };

    auto key_fn = make_grid_key(dim, settings.grid_cell, scale, anchor);

    // heur(qi, q):
    //   qi == 0    : the anchor -- admissible joint-space L2 to the nearest goal.
    //   qi >= 1    : genuinely INADMISSIBLE -- the same L2 with per-DoF weights
    //                >= 1, so each MAY OVERESTIMATE the true cost-to-go, which is
    //                the whole point of an inadmissible heuristic: it is allowed
    //                to be more aggressive than the anchor in order to escape
    //                local minima faster, with the w2 anchor bound retaining the
    //                suboptimality guarantee.
    //
    // Queues alternate which end of the body they inflate, so the "small fixed
    // set of inadmissible heuristics" is a set of DISTINCT functions:
    //   odd  qi -> proximal joints inflated: fix gross repositioning first.
    //   even qi -> distal joints inflated:   fix fine alignment first.
    // On a branching whole-body robot this splits along the branches for free --
    // Baxter's dofs 0-6 are the left arm and 7-13 the right, so queue 1 drives
    // one arm and queue 2 the other.
    //
    // This replaces an earlier version whose qi>=1 weights were (dim-i)/dim, i.e.
    // <= 1. Those made h_i <= h_0, so every "inadmissible" queue held a heuristic
    // strictly DOMINATED by the anchor, and since SBPL's GetBestHeuristicID
    // starts at index 1 and treats the anchor as a fallback, MHA* spent most of
    // its budget on a LESS informed search than plain weighted A*. Measured at a
    // 1 s budget, real MBM, eager, step 0.50: turning those queues off entirely
    // (num_inad=0) BEAT keeping them on Fetch (0.34 vs 0.20) and Baxter
    // (0.03 vs 0.00); they only helped on Panda (0.91 vs 0.86), where the budget
    // is not the binding constraint. They were also all identical to each other,
    // so num_inad=2 bought two copies of one function.
    const float infl = 2.0f;   // max per-DoF inflation on an inadmissible queue
    auto heur_fn = [gs, dim, infl](int qi, const Config& q) -> float {
        float best = std::numeric_limits<float>::infinity();
        const float span = (dim > 1) ? (float)(dim - 1) : 1.0f;
        for (const auto& g : gs) {
            float d = 0.f;
            if (qi == 0) {
                for (int i = 0; i < dim; ++i) { float e = q[i] - g[i]; d += e * e; }
            } else {
                const bool proximal = (qi % 2) == 1;
                for (int i = 0; i < dim; ++i) {
                    const float t = proximal ? (float)(dim - 1 - i) / span
                                             : (float)i / span;
                    const float w = 1.0f + (infl - 1.0f) * t;   // in [1, infl]
                    float e = (q[i] - g[i]) * w;
                    d += e * e;
                }
            }
            best = std::fmin(best, std::sqrt(d));
        }
        return best;
    };

    MHAStarPlanner planner(ms, &gpu, succ, heur_fn, goal_fn, key_fn);
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

} // namespace MHAStar
