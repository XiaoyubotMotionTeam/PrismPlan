#pragma once

// ============================================================================
// wPA*SE (weighted Parallel A* for Slow Expansions) on CPU, with GPU-batched
// edge evaluation.
//
// Algorithm reference: weighted PA*SE (Mukherjee et al.; MIT-licensed reference
// impl shohinm/parallel_search, PasePlanner). Running PA*SE with a heuristic
// weight w > 1 yields wPA*SE with bounded suboptimality = w.
//
// Architecture (substrate-consistent with MHA*): the best-first search runs on
// the CPU (single OPEN queue keyed f = g + w*h, CLOSED set, bookkeeping) while
// the expensive "slow expansions" (edge validity + cost) are batched onto the
// GPU via EdgeEvalGpu / evaluate_edges_batch.
//
// PA*SE parallelism mapping: reference PA*SE expands, in parallel across CPU
// threads, a set of states that are mutually INDEPENDENT (no being-expanded
// state can lower another's g within the weight bound). Here that same
// independent set is expanded together and ALL their successor edges are
// evaluated in a SINGLE GPU batch — i.e. the parallel slow-expansions are
// realised by GPU batch width rather than CPU threads (collision eval is
// already a batch-parallel kernel).
//
// Independence rule (per PasePlanner): candidate s is dependent on an already
// selected state s' iff  g(s) > g(s') + w * h_binary(s' -> s).  Only mutually
// independent states are expanded in the same round.
//
// This is a SKELETON: queue mechanics + independent-set + batch loop are wired;
// heuristics, successor generation, goal test and the pairwise heuristic are
// pluggable via callbacks. NOT compiled or run here (needs the CUDA build).
// ============================================================================

#include "src/planning/edge_batch_bridge.hh"
#include "src/planning/edge_eval.cuh"      // EdgeEvalParams
#include "src/planning/robot_model.cuh"    // ppln::MAX_DIM
#include "src/planning/dof_scale.hh"       // ppln::DofScale

#include <vector>
#include <array>
#include <queue>
#include <unordered_map>
#include <functional>
#include <limits>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <utility>

namespace ppln::search {

// Config alias mirrors mhastar_search.hh. wpase_solve.cu is the ONLY TU that
// includes this header, so there is no ODR clash with the MHA* copy.
using WConfig = std::array<float, ppln::MAX_DIM>;

struct WPASESettings {
    float heuristic_w = 5.0f;     // f(s) = g(s) + w * h(s); suboptimality bound = w
    int   num_parallel = 4;       // max mutually-independent states expanded per round
    int   dim = 6;
    float time_limit_ms = 1000.0f;
    int   batch_edges = 256;      // GPU edge-eval cap (>= num_parallel*2*dim recommended)
    EdgeEvalParams edge = { /*granularity*/ 16, /*margin*/ 0.0f,
                            /*check_self*/ true, /*compute_clearance*/ false,
                            /*enable_mesh*/ false, /*resolution*/ 0.025f };
};

struct WPASEResult {
    bool solved = false;
    std::vector<WConfig> path;    // start .. goal
    float cost = std::numeric_limits<float>::infinity();
    int expansions = 0;
    int edges_evaluated = 0;
    float elapsed_ms = 0.0f;
};

class WPASEPlanner {
public:
    // Pluggable problem definition.
    //   succ_fn(id, q)     -> successor configs (lattice / primitives)
    //   heur_fn(q)         -> admissible unary heuristic h(q)
    //   binh_fn(a, b)      -> pairwise heuristic h(a -> b) for the independence test
    //   goal_fn(q)         -> true if q is a goal
    //   key_of(q)          -> discretization key for the node table (lattice cell)
    using SuccFn = std::function<std::vector<WConfig>(int, const WConfig&)>;
    using HeurFn = std::function<float(const WConfig&)>;
    using BinHeurFn = std::function<float(const WConfig&, const WConfig&)>;
    using GoalFn = std::function<bool(const WConfig&)>;
    using KeyFn  = std::function<uint64_t(const WConfig&)>;

    WPASEPlanner(const WPASESettings& s, EdgeEvalGpu* gpu,
                 SuccFn succ, HeurFn heur, BinHeurFn binh, GoalFn goal, KeyFn key)
        : s_(s), gpu_(gpu), succ_(std::move(succ)), heur_(std::move(heur)),
          binh_(std::move(binh)), goal_(std::move(goal)), key_(std::move(key)) {}

    WPASEResult solve(const WConfig& start);

private:
    struct Node {
        WConfig q;
        float  g = std::numeric_limits<float>::infinity();  // best known cost-to-come
        float  v = std::numeric_limits<float>::infinity();  // settled value at expansion
        int    parent = -1;
        bool   closed = false;      // expanded (settled) — never re-expanded
        bool   in_batch = false;    // selected in the current round (dedup guard)
        float  h = 0.f;             // cached admissible heuristic h(q)
    };

    struct OpenEntry {
        double key;
        int    node;
        bool operator>(const OpenEntry& o) const { return key > o.key; }
    };
    using MinHeap = std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<OpenEntry>>;

    double key_for(int id) const {
        const Node& n = nodes_[id];
        return (double)n.g + (double)s_.heuristic_w * (double)n.h;
    }

    int get_or_make(const WConfig& q) {
        uint64_t k = key_(q);
        auto it = table_.find(k);
        if (it != table_.end()) return it->second;
        int id = (int)nodes_.size();
        Node n; n.q = q; n.h = heur_(q);
        nodes_.push_back(std::move(n));
        table_.emplace(k, id);
        return id;
    }

    // Pull up to num_parallel mutually-independent, non-closed states off the
    // OPEN front (in f-order). Deferred (dependent / stale) pops are pushed back.
    // Returns the selected node ids; sets goal_hit/goal_id if a popped state is a goal.
    std::vector<int> select_independent_set(bool& goal_hit);

    // Expand the whole independent set: settle+close each, generate all successor
    // edges, evaluate them on the GPU in one batch, relax children, reinsert.
    void expand_batch(const std::vector<int>& batch);

    WPASESettings s_;
    EdgeEvalGpu*  gpu_;
    SuccFn succ_; HeurFn heur_; BinHeurFn binh_; GoalFn goal_; KeyFn key_;

    std::vector<Node> nodes_;
    std::unordered_map<uint64_t,int> table_;
    MinHeap open_;

    int goal_id_ = -1;
    int gpu_edges_ = 0;   // exact count of edges evaluated on the GPU
};

// ---------------------------------------------------------------------------
inline std::vector<int> WPASEPlanner::select_independent_set(bool& goal_hit) {
    goal_hit = false;
    std::vector<int> batch;
    std::vector<int> deferred;   // dependent-on-batch pops, re-inserted after

    while ((int)batch.size() < s_.num_parallel && !open_.empty()) {
        int id = open_.top().node;
        open_.pop();
        Node& n = nodes_[id];
        if (n.closed || n.in_batch) continue;         // stale / already selected

        if (goal_(n.q)) { goal_id_ = id; goal_hit = true; break; }

        // Independence: dependent iff some already-selected s' could still lower
        // this state's g within the weight bound (PA*SE rule).
        bool independent = true;
        for (int bid : batch) {
            const Node& b = nodes_[bid];
            if ((double)n.g > (double)b.g + (double)s_.heuristic_w * (double)binh_(b.q, n.q)) {
                independent = false;
                break;
            }
        }
        if (independent) {
            n.in_batch = true;
            batch.push_back(id);
        } else {
            deferred.push_back(id);   // defer to a later round
        }
    }

    // Return deferred states to OPEN (keys unchanged).
    for (int id : deferred)
        open_.push({ key_for(id), id });

    return batch;
}

// ---------------------------------------------------------------------------
inline void WPASEPlanner::expand_batch(const std::vector<int>& batch) {
    // Settle + close every state in the independent set, then gather successors.
    // NB: get_or_make() may reallocate nodes_, so gather parent snapshots first
    // and refer to nodes strictly by id afterwards.
    struct Edge { int parent; WConfig cq; };
    std::vector<Edge> edges;

    for (int id : batch) {
        Node& n = nodes_[id];
        n.v = n.g;                    // settle
        n.closed = true;
        n.in_batch = false;
        WConfig sq = n.q;             // snapshot before any push_back
        for (const WConfig& cq : succ_(id, sq))
            edges.push_back({ id, cq });
    }

    int m = (int)edges.size();
    if (m == 0) return;

    // Flatten and evaluate on the GPU (chunked to the evaluator cap).
    const int dim = s_.dim;
    std::vector<float> from_flat, to_flat;
    std::vector<uint8_t> valid; std::vector<float> cost, clr;

    int e = 0;
    while (e < m) {
        int chunk = std::min(m - e, gpu_->max_edges());
        from_flat.clear(); to_flat.clear();
        from_flat.reserve((size_t)chunk * dim);
        to_flat.reserve((size_t)chunk * dim);
        for (int k = 0; k < chunk; ++k) {
            const WConfig& pq = nodes_[edges[e + k].parent].q;
            const WConfig& cq = edges[e + k].cq;
            for (int i = 0; i < dim; ++i) from_flat.push_back(pq[i]);
            for (int i = 0; i < dim; ++i) to_flat.push_back(cq[i]);
        }
        gpu_->evaluate(from_flat.data(), to_flat.data(), chunk, s_.edge, valid, cost, clr);
        gpu_edges_ += chunk;

        for (int k = 0; k < chunk; ++k) {
            if (!valid[k]) continue;          // edge in collision -> drop
            int pid = edges[e + k].parent;
            int cid = get_or_make(edges[e + k].cq);
            Node& c = nodes_[cid];
            if (c.closed) continue;           // never re-open a settled state
            float ng = nodes_[pid].v + cost[k];   // true (collision-aware) cost
            if (ng < c.g) {
                c.g = ng;
                c.parent = pid;
                open_.push({ key_for(cid), cid });
            }
        }
        e += chunk;
    }
}

// ---------------------------------------------------------------------------
inline WPASEResult WPASEPlanner::solve(const WConfig& start) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    WPASEResult res;

    int sid = get_or_make(start);
    nodes_[sid].g = 0.f; nodes_[sid].v = 0.f;
    open_.push({ key_for(sid), sid });

    while (true) {
        auto ms = std::chrono::duration<float,std::milli>(clk::now() - t0).count();
        if (ms > s_.time_limit_ms) break;
        if (open_.empty()) break;

        bool goal_hit = false;
        std::vector<int> batch = select_independent_set(goal_hit);
        if (goal_hit) { res.solved = true; break; }
        if (batch.empty()) {
            // Everything popped was stale/closed and nothing was deferrable, or
            // OPEN drained. If OPEN still has deferred work, loop; else stop.
            if (open_.empty()) break;
            continue;
        }

        expand_batch(batch);
        res.expansions += (int)batch.size();
    }

    if (res.solved && goal_id_ >= 0) {
        for (int cur = goal_id_; cur >= 0; cur = nodes_[cur].parent)
            res.path.push_back(nodes_[cur].q);
        std::reverse(res.path.begin(), res.path.end());
        res.cost = nodes_[goal_id_].g;
    }
    res.elapsed_ms = std::chrono::duration<float,std::milli>(clk::now() - t0).count();
    res.edges_evaluated = gpu_edges_;
    return res;
}

// ============================================================================
// Default problem-definition factories (joint lattice). Self-contained copies
// (distinct names / this TU only) so no dependency on mhastar_search.hh.
// ============================================================================

// Snap `q` onto the lattice anchored at `anchor` (per-joint spacing
// step*scale[i]). Clamping is applied to the grid INDEX, not the value, so the
// result is always exactly on the lattice AND inside `limits`.
inline void wpase_snap_to_lattice(WConfig& q, const WConfig& anchor, int dim, float step,
                                  const ppln::DofScale& scale,
                                  const std::array<std::array<float,2>, ppln::MAX_DIM>* limits) {
    for (int k = 0; k < dim; ++k) {
        const float h = step * scale.s[k];
        float idx = std::round((q[k] - anchor[k]) / h);
        if (limits) {
            const float imin = std::ceil (((*limits)[k][0] - anchor[k]) / h);
            const float imax = std::floor(((*limits)[k][1] - anchor[k]) / h);
            if (imin <= imax) {
                if (idx < imin) idx = imin;
                if (idx > imax) idx = imax;
            }
        }
        q[k] = anchor[k] + idx * h;
    }
}

// Joint-lattice successors, PER-DoF NORMALISED: joint i moves by
// step * scale[i], i.e. the same fraction of every joint's travel (see
// dof_scale.hh).
//
// The lattice is ANCHORED AT `anchor` (the goal) so the goal is itself a lattice
// vertex and exactly reachable. Start-anchored, the nearest vertex to the goal
// is in collision on 47 of 70 real MBM Fetch problems even though the goal is
// free, which makes those goals unreachable by construction. Successors are
// snapped back onto the anchored grid -- a no-op except for the start's own
// successors, the start being the only off-grid node.
inline WPASEPlanner::SuccFn
make_wpase_lattice_succ(int dim, float step, const ppln::DofScale& scale,
                        const WConfig& anchor,
                        const std::array<std::array<float,2>, ppln::MAX_DIM>* limits = nullptr) {
    return [dim, step, scale, anchor, limits](int /*id*/, const WConfig& q) {
        std::vector<WConfig> out;
        out.reserve(2 * dim);
        for (int i = 0; i < dim; ++i) {
            const float h = step * scale.s[i];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                WConfig n = q;
                n[i] = q[i] + (float)sgn * h;
                wpase_snap_to_lattice(n, anchor, dim, step, scale, limits);
                out.push_back(n);
            }
        }
        return out;
    };
}

// Goal-connection successor: emit each goal directly once the node is within
// `connect` (normalised L2), so the final straight line is offered to the GPU
// edge check instead of having to be walked out one lattice cell at a time.
inline WPASEPlanner::SuccFn
with_wpase_goal_connect(WPASEPlanner::SuccFn base, std::vector<WConfig> goals,
                        const ppln::DofScale& scale, int dim, float connect) {
    if (!(connect > 0.f)) return base;
    return [base, goals, scale, dim, connect](int id, const WConfig& q) {
        std::vector<WConfig> out = base(id, q);
        for (const WConfig& g : goals) {
            float d = 0.f;
            for (int i = 0; i < dim; ++i) {
                float e = (g[i] - q[i]) * scale.inv[i];
                d += e * e;
            }
            if (std::sqrt(d) <= connect) out.push_back(g);
        }
        return out;
    };
}

// Grid-cell key: quantise each joint to round((q-anchor)/(cell*scale[i])) and
// FNV-1a hash-combine. Quantised relative to the SAME anchor the lattice uses,
// or dedup bucket boundaries stop lining up with lattice vertices.
inline WPASEPlanner::KeyFn
make_wpase_grid_key(int dim, float cell, const ppln::DofScale& scale, const WConfig& anchor) {
    return [dim, cell, scale, anchor](const WConfig& q) -> uint64_t {
        uint64_t h = 1469598103934665603ull;   // FNV offset basis
        for (int i = 0; i < dim; ++i) {
            int64_t c = (int64_t)std::llround((q[i] - anchor[i]) / (cell * scale.s[i]));
            uint64_t u = (uint64_t)c;
            for (int b = 0; b < 8; ++b) {
                h ^= (u & 0xff);
                h *= 1099511628211ull;          // FNV prime
                u >>= 8;
            }
        }
        return h;
    };
}

// Admissible joint-space L2 heuristic to the nearest of `goals` (unary).
inline WPASEPlanner::HeurFn
make_wpase_l2_heur(const std::vector<WConfig>& goals, int dim) {
    return [goals, dim](const WConfig& q) {
        float best = std::numeric_limits<float>::infinity();
        for (const auto& g : goals) {
            float d = 0.f;
            for (int i = 0; i < dim; ++i) { float e = q[i] - g[i]; d += e * e; }
            best = std::fmin(best, std::sqrt(d));
        }
        return best;
    };
}

// Pairwise joint-space L2 heuristic h(a -> b) used by the independence test.
inline WPASEPlanner::BinHeurFn
make_wpase_binary_l2(int dim) {
    return [dim](const WConfig& a, const WConfig& b) {
        float d = 0.f;
        for (int i = 0; i < dim; ++i) { float e = a[i] - b[i]; d += e * e; }
        return std::sqrt(d);
    };
}

// Goal test: within `radius` of any of `goals` in NORMALISED joint space, so the
// tolerance is the same fraction of travel on every joint instead of adding
// metres to radians on a robot with a prismatic axis.
inline WPASEPlanner::GoalFn
make_wpase_goal_radius(const std::vector<WConfig>& goals, int dim, float radius,
                       const ppln::DofScale& scale) {
    return [goals, dim, radius, scale](const WConfig& q) {
        for (const auto& g : goals) {
            float d = 0.f;
            for (int i = 0; i < dim; ++i) {
                float e = (q[i] - g[i]) * scale.inv[i];
                d += e * e;
            }
            if (std::sqrt(d) <= radius) return true;
        }
        return false;
    };
}

} // namespace ppln::search
