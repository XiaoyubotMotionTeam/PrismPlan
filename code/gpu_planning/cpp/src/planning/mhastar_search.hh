#pragma once

// ============================================================================
// SMHA* (Shared Multi-Heuristic A*) on CPU, with GPU-batched lazy edge eval.
//
// Architecture (ePA*SE / GePA*SE style): the best-first search runs on the CPU
// (queues, CLOSED, bookkeeping) while the expensive edge validity+cost checks
// are deferred (lazy) and flushed to the GPU in batches via EdgeEvalGpu.
//
// Mirrors SBPL MHAPlanner semantics:
//   * key(s,i) = g(s) + w1 * h_i(s)   (inflation_eps = w1)
//   * anchor bound test: use inadmissible queue i only while
//       key(OPEN[i].min) <= w2 * key(OPEN[0].min)   (anchor_eps = w2)
//   * SHARED g across all queues; on expansion, close in all queues.
//   * lazy successors: insertLazyList(...) with is_true_cost=false; a state's
//     incoming edge is truly evaluated (on GPU) only when it is popped. This
//     pop-time evaluation is exactly the GPU batching point.
//
// This is a SKELETON: queue mechanics + lazy-batch loop are wired; heuristics,
// successor generation, and goal test are pluggable via callbacks. NOT compiled
// or run yet (needs the pRRTC CUDA build in ur_container).
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
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <utility>

namespace ppln::search {

using Config = std::array<float, ppln::MAX_DIM>;

// GetBestHeuristicID strategy. RoundRobin = SBPL default; BestKey = lightweight
// Meta-A* surrogate (pick the within-bound inadmissible queue with smallest key,
// i.e. the one most likely to make immediate progress). DTS (Thompson sampling)
// is intentionally left out of this POC.
enum class QueueSel { RoundRobin, BestKey };

struct MHAStarSettings {
    float w1 = 5.0f;              // inflation_eps (heuristic inflation)
    float w2 = 2.0f;              // anchor_eps (bounded-suboptimality on anchor)
    int   num_inad = 2;           // # inadmissible heuristics (queues 1..num_inad)
    int   dim = 6;
    float time_limit_ms = 500.0f;
    int   batch_edges = 256;      // lazy edges flushed per GPU call (<= EdgeEvalGpu cap)
    // Edge-evaluation policy. true = SBPL insertLazyList (verify a state's
    // incoming edge only when it is popped); false = verify ALL successors of an
    // expanded state in one GPU call before inserting any of them. Laziness only
    // pays when most edges are valid: it trades a correct priority ordering for
    // skipped evaluations, and on cluttered MBM scenes ~50 % of lattice edges
    // collide, so half the frontier's keys are optimistic fiction.
    bool  lazy = true;
    QueueSel queue_sel = QueueSel::RoundRobin;
    EdgeEvalParams edge = { /*granularity*/ 16, /*margin*/ 0.0f,
                            /*check_self*/ true, /*compute_clearance*/ false,
                            /*enable_mesh*/ false, /*resolution*/ 0.025f };
};

struct MHAStarResult {
    bool solved = false;
    std::vector<Config> path;     // start .. goal
    float cost = std::numeric_limits<float>::infinity();
    int expansions = 0;
    int edges_evaluated = 0;
    float elapsed_ms = 0.0f;
};

class MHAStarPlanner {
public:
    // Pluggable problem definition.
    //   succ_fn(id, q)     -> list of successor configs (lattice / primitives)
    //   heur_fn(i, q)      -> heuristic i in [0, num_inad]; 0 must be admissible
    //   goal_fn(q)         -> true if q is a goal
    //   key_of(id)         -> discretization key for the node table (lattice cell)
    using SuccFn = std::function<std::vector<Config>(int, const Config&)>;
    using HeurFn = std::function<float(int, const Config&)>;
    using GoalFn = std::function<bool(const Config&)>;
    using KeyFn  = std::function<uint64_t(const Config&)>;

    MHAStarPlanner(const MHAStarSettings& s, EdgeEvalGpu* gpu,
                   SuccFn succ, HeurFn heur, GoalFn goal, KeyFn key)
        : s_(s), gpu_(gpu), succ_(std::move(succ)), heur_(std::move(heur)),
          goal_(std::move(goal)), key_(std::move(key)),
          nq_(s.num_inad + 1) {}

    MHAStarResult solve(const Config& start);

private:
    // ---- node table (SHARED g/v across all queues) ----
    struct Node {
        Config q;
        float  g = std::numeric_limits<float>::infinity();  // best known cost-to-come
        float  v = std::numeric_limits<float>::infinity();  // expanded (settled) value
        int    parent = -1;
        bool   is_true_cost = false;   // incoming edge (parent->this) GPU-verified?
        bool   in_pending = false;     // already queued for GPU verification this round?
        std::vector<int8_t> closed;    // per-queue CLOSED (iteration_closed analogue)
        std::vector<float>  h;         // cached per-queue heuristic h_i(q) (state-fixed)

        // Lazy-A* candidate incoming edges: every (parent, lazy_cost) offered so
        // far, minus the ones the GPU has proven to be in collision. Needed
        // because a lazy pop can invalidate the current parent, and without an
        // alternative to fall back on the state would keep the optimistic g of
        // the rejected edge and become permanently unreachable.
        std::vector<std::pair<int,float>> cands;
        std::vector<int> rejected;     // parents with a GPU-invalidated edge
    };

    struct OpenEntry {
        double key;
        int    node;
        bool operator>(const OpenEntry& o) const { return key > o.key; }
    };
    using MinHeap = std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<OpenEntry>>;

    double key_for(int id, int qi) const {
        const Node& n = nodes_[id];
        return (double)n.g + (double)s_.w1 * (double)n.h[qi];   // cached heuristic
    }

    int get_or_make(const Config& q) {
        uint64_t k = key_(q);
        auto it = table_.find(k);
        if (it != table_.end()) return it->second;
        int id = (int)nodes_.size();
        Node n; n.q = q; n.closed.assign(nq_, 0);
        n.h.resize(nq_);
        for (int qi = 0; qi < nq_; ++qi) n.h[qi] = heur_(qi, q);  // compute once
        nodes_.push_back(std::move(n));
        table_.emplace(k, id);
        return id;
    }

    // insertLazyList analogue: relax parent->child with a *lazy* (unverified)
    // edge cost, push into all queues, mark is_true_cost=false.
    void insert_lazy(int parent, const Config& cq, float lazy_cost);

    // Flush accumulated lazy-pop edges to the GPU, backfill true cost, reinsert.
    void flush_lazy_batch();

    int  best_queue();   // GetBestHeuristicID (round-robin) + anchor bound gate
    void expand(int id); // v(s)=g(s); close in all queues; generate lazy successors

    MHAStarSettings s_;
    EdgeEvalGpu*    gpu_;
    SuccFn succ_; HeurFn heur_; GoalFn goal_; KeyFn key_;
    int nq_;

    std::vector<Node> nodes_;
    std::unordered_map<uint64_t,int> table_;
    std::vector<MinHeap> open_;                 // open_[0]=anchor, 1..num_inad inadmissible

    // pending lazy edges popped this round, awaiting GPU verification
    struct PendingEdge { int parent; int child; };
    std::vector<PendingEdge> pending_;
    std::vector<float> pending_from_, pending_to_;   // flattened cfg buffers

    std::vector<int> queue_expands_;  // per-queue expansion count (SBPL queue_expands)
    int goal_id_ = -1;
    int gpu_edges_ = 0;    // exact count of edges evaluated on the GPU
    int invalid_edges_ = 0;// of those, how many came back in collision
    double gpu_ms_ = 0.0;  // wall time spent inside EdgeEvalGpu::evaluate
    int gpu_calls_ = 0;    // number of evaluate() calls (== expansions when eager)
    float min_h_ = std::numeric_limits<float>::infinity();  // closest expanded node to goal
};

// ---------------------------------------------------------------------------
inline void MHAStarPlanner::insert_lazy(int parent, const Config& cq, float lazy_cost) {
    int cid = get_or_make(cq);
    Node& c = nodes_[cid];

    // SBPL insertLazyList's first line: if the state is already settled at a cost
    // this edge cannot beat, the offer is dead on arrival -- don't record it as a
    // candidate and don't push a heap entry that will only be discarded at pop.
    if (c.v <= nodes_[parent].v + lazy_cost) return;

    // Never re-offer an edge the GPU already rejected.
    for (int r : c.rejected) if (r == parent) return;

    bool known = false;
    for (auto& kv : c.cands)
        if (kv.first == parent) { kv.second = std::fmin(kv.second, lazy_cost); known = true; break; }
    if (!known) c.cands.emplace_back(parent, lazy_cost);

    float ng = nodes_[parent].g + lazy_cost;
    if (ng < c.g) {
        c.g = ng;
        c.parent = parent;
        c.is_true_cost = false;      // lazy: parent->child edge not yet GPU-verified
        for (int qi = 0; qi < nq_; ++qi) {
            // push into every queue with that queue's key (SMHA shared-g)
            if (qi == 0 || c.closed[qi] == 0)   // anchor always; inad only if not closed
                open_[qi].push({ key_for(cid, qi), cid });
        }
    }
}

// ---------------------------------------------------------------------------
inline int MHAStarPlanner::best_queue() {
    // Anchor min key.
    while (!open_[0].empty() && nodes_[open_[0].top().node].closed[0])
        open_[0].pop();
    double anchor_min = open_[0].empty()
        ? std::numeric_limits<double>::infinity()
        : open_[0].top().key;
    const double bound = (double)s_.w2 * anchor_min;

    if (s_.queue_sel == QueueSel::BestKey) {
        // Meta-A* surrogate: among inadmissible queues within the w2 bound,
        // pick the smallest key.
        int    best_qi = 0;
        double best_key = anchor_min;
        for (int qi = 1; qi <= s_.num_inad; ++qi) {
            while (!open_[qi].empty() && nodes_[open_[qi].top().node].closed[qi])
                open_[qi].pop();
            if (open_[qi].empty()) continue;
            double k = open_[qi].top().key;
            if (k <= bound && k < best_key) { best_key = k; best_qi = qi; }
        }
        return best_qi;
    }

    // Round-robin over inadmissible queues, honoring the w2 anchor bound.
    // SBPL's GetBestHeuristicID picks the eligible queue with the FEWEST
    // expansions so far (queue_expands[]) rather than cycling a counter, so a
    // queue whose min key repeatedly falls outside the bound does not lose its
    // turn permanently. starting_ind = 1: the anchor is only used as a fallback.
    int    best_qi = -1;
    int    best_expands = std::numeric_limits<int>::max();
    for (int qi = 1; qi <= s_.num_inad; ++qi) {
        while (!open_[qi].empty() && nodes_[open_[qi].top().node].closed[qi])
            open_[qi].pop();
        if (open_[qi].empty()) continue;
        if (open_[qi].top().key > bound) continue;      // outside the anchor bound
        if (queue_expands_[qi] < best_expands) { best_expands = queue_expands_[qi]; best_qi = qi; }
    }
    if (best_qi >= 0) return best_qi;
    return 0;                          // fall back to anchor
}

// ---------------------------------------------------------------------------
inline void MHAStarPlanner::flush_lazy_batch() {
    int m = (int)pending_.size();
    if (m == 0) return;

    std::vector<uint8_t> valid; std::vector<float> cost, clr;
    {
        auto t0 = std::chrono::steady_clock::now();
        gpu_->evaluate(pending_from_.data(), pending_to_.data(), m, s_.edge, valid, cost, clr);
        gpu_ms_ += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
        ++gpu_calls_;
    }
    gpu_edges_ += m;

    for (int e = 0; e < m; ++e) {
        int pid = pending_[e].parent, cid = pending_[e].child;
        Node& c = nodes_[cid];
        c.in_pending = false;                          // released from the batch

        if (!valid[e]) {
            // Edge in collision. Retiring it is not enough: the state's current
            // g/parent came from THIS edge, so leaving them in place keeps an
            // optimistic g that no real parent can ever beat, and the state is
            // silently lost (it is in no queue -- the lazy pop took it out
            // without closing it). Roll back to the next-best candidate parent
            // and re-open, or drop the state to infinity if none is left.
            ++invalid_edges_;
            c.rejected.push_back(pid);
            for (size_t i = 0; i < c.cands.size(); ++i)
                if (c.cands[i].first == pid) { c.cands.erase(c.cands.begin() + i); break; }

            if (c.is_true_cost || c.parent != pid) continue;   // current parent unaffected

            c.g = std::numeric_limits<float>::infinity();
            c.parent = -1;
            for (const auto& kv : c.cands) {
                float ng = nodes_[kv.first].v + kv.second;      // settled v(parent)
                if (ng < c.g) { c.g = ng; c.parent = kv.first; }
            }
            if (c.parent >= 0)
                for (int qi = 0; qi < nq_; ++qi)
                    if (qi == 0 || c.closed[qi] == 0)
                        open_[qi].push({ key_for(cid, qi), cid });
            continue;
        }

        float true_g = nodes_[pid].v + cost[e];        // use settled v(parent)
        if (true_g < c.g || !c.is_true_cost) {
            c.g = true_g;
            c.parent = pid;
            c.is_true_cost = true;                     // now GPU-verified
            for (int qi = 0; qi < nq_; ++qi)
                if (qi == 0 || c.closed[qi] == 0)
                    open_[qi].push({ key_for(cid, qi), cid });
        }
    }
    pending_.clear(); pending_from_.clear(); pending_to_.clear();
}

// ---------------------------------------------------------------------------
inline void MHAStarPlanner::expand(int id) {
    // Snapshot state into locals BEFORE the successor loop: insert_lazy() ->
    // get_or_make() may push_back into nodes_ and reallocate, which would
    // dangle any Node& into the vector held across the loop.
    Config sq = nodes_[id].q;
    float  sg = nodes_[id].g;
    nodes_[id].v = sg;                                   // settle
    for (int qi = 0; qi < nq_; ++qi) nodes_[id].closed[qi] = 1;  // close all (SMHA)
    if (nodes_[id].h[0] < min_h_) min_h_ = nodes_[id].h[0];

    const std::vector<Config> succs = succ_(id, sq);

    if (!s_.lazy) {
        // Eager: confirm every successor edge in ONE GPU call, then insert only
        // the survivors with their true cost. Costs 2*dim edges per expansion
        // instead of ~2, but every key on OPEN is then backed by a verified edge.
        const int m = (int)succs.size();
        std::vector<float> from, to;
        from.reserve((size_t)m * s_.dim); to.reserve((size_t)m * s_.dim);
        for (const Config& cq : succs) {
            for (int i = 0; i < s_.dim; ++i) from.push_back(sq[i]);
            for (int i = 0; i < s_.dim; ++i) to.push_back(cq[i]);
        }
        std::vector<uint8_t> valid; std::vector<float> cost, clr;
        {
            auto t0 = std::chrono::steady_clock::now();
            gpu_->evaluate(from.data(), to.data(), m, s_.edge, valid, cost, clr);
            gpu_ms_ += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
            ++gpu_calls_;
        }
        gpu_edges_ += m;
        for (int e = 0; e < m; ++e) {
            if (!valid[e]) { ++invalid_edges_; continue; }
            int cid = get_or_make(succs[e]);
            Node& c = nodes_[cid];
            float ng = sg + cost[e];
            if (ng < c.g || !c.is_true_cost) {
                c.g = ng;
                c.parent = id;
                c.is_true_cost = true;
                for (int qi = 0; qi < nq_; ++qi)
                    if (qi == 0 || c.closed[qi] == 0)
                        open_[qi].push({ key_for(cid, qi), cid });
            }
        }
        return;
    }

    for (const Config& cq : succs) {
        // lazy edge cost = straight-line joint distance (admissible lower bound);
        // the true (collision-aware) cost is confirmed on GPU at pop time.
        float d = 0.f;
        for (int i = 0; i < s_.dim; ++i) { float dq = cq[i] - sq[i]; d += dq * dq; }
        insert_lazy(id, cq, std::sqrt(d));
    }
}

// ---------------------------------------------------------------------------
inline MHAStarResult MHAStarPlanner::solve(const Config& start) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    MHAStarResult res;

    open_.assign(nq_, MinHeap());
    queue_expands_.assign(nq_, 0);
    int sid = get_or_make(start);
    nodes_[sid].g = 0.f; nodes_[sid].v = 0.f; nodes_[sid].is_true_cost = true;
    for (int qi = 0; qi < nq_; ++qi) open_[qi].push({ key_for(sid, qi), sid });

    // Loop while there is *any* work left: OPEN entries OR lazy edges still
    // awaiting GPU verification. Draining the anchor is NOT a termination
    // condition on its own -- a pending flush can reinsert nodes (incl. goal).
    while (true) {
        auto ms = std::chrono::duration<float,std::milli>(clk::now() - t0).count();
        if (ms > s_.time_limit_ms) break;

        if (open_[0].empty()) {
            if (!pending_.empty()) { flush_lazy_batch(); continue; }
            break;                                  // nothing left to verify
        }

        int qi = best_queue();
        if (open_[qi].empty()) {
            if (!pending_.empty()) { flush_lazy_batch(); continue; }
            break;
        }
        int id = open_[qi].top().node; open_[qi].pop();
        Node& n = nodes_[id];
        if (n.closed[qi]) continue;                 // stale heap entry
        // Orphaned by an invalid-edge rollback: every candidate parent of this
        // node has been proven in collision, so it has no cost-to-come and no
        // parent chain. Stale entries from before the rollback are still in the
        // queues; expanding (or worse, goal-testing) one would claim a path that
        // does not exist.
        if (!std::isfinite(n.g)) continue;

        // Lazy gate: if this node's incoming edge is unverified, defer to GPU
        // batch instead of expanding now. Guard against double-enqueue: the
        // same node may sit in several queues, and would otherwise be pushed
        // to pending_ once per queue before the flush flips is_true_cost.
        if (!n.is_true_cost && n.parent >= 0) {
            if (!n.in_pending) {
                n.in_pending = true;
                pending_.push_back({ n.parent, id });
                const Config& pq = nodes_[n.parent].q;
                for (int i = 0; i < s_.dim; ++i) pending_from_.push_back(pq[i]);
                for (int i = 0; i < s_.dim; ++i) pending_to_.push_back(n.q[i]);
                if ((int)pending_.size() >= s_.batch_edges) flush_lazy_batch();
            }
            continue;
        }

        if (goal_(n.q)) { goal_id_ = id; res.solved = true; break; }
        expand(id);
        ++res.expansions;
        ++queue_expands_[qi];

        // Opportunistic flush so the GPU is not starved between expansions.
        if ((int)pending_.size() >= s_.batch_edges) flush_lazy_batch();
    }
    // Drain any remaining lazy edges (e.g. goal reached via lazy pop).
    flush_lazy_batch();

    if (res.solved) {
        for (int cur = goal_id_; cur >= 0; cur = nodes_[cur].parent)
            res.path.push_back(nodes_[cur].q);
        std::reverse(res.path.begin(), res.path.end());
        res.cost = nodes_[goal_id_].g;
    }
    res.elapsed_ms = std::chrono::duration<float,std::milli>(clk::now() - t0).count();
    res.edges_evaluated = gpu_edges_;   // exact GPU edge count
    if (const char* v = std::getenv("MHASTAR_DEBUG"))
        if (v[0] == '1')
            std::fprintf(stderr, "[mha] solved=%d exp=%d nodes=%zu edges=%d invalid=%d (%.0f%%) h0=%.3f minh=%.3f gpu=%.0fms/%d calls (%.0fus/call, %.0f%% of wall)\n",
                         (int)res.solved, res.expansions, nodes_.size(),
                         gpu_edges_, invalid_edges_,
                         gpu_edges_ ? 100.0 * invalid_edges_ / gpu_edges_ : 0.0,
                         nodes_[0].h[0], min_h_,
                         gpu_ms_, gpu_calls_,
                         gpu_calls_ ? 1000.0 * gpu_ms_ / gpu_calls_ : 0.0,
                         res.elapsed_ms > 0 ? 100.0 * gpu_ms_ / res.elapsed_ms : 0.0);
    return res;
}

// ============================================================================
// Default problem-definition factories (joint lattice). These make the planner
// directly runnable; swap in domain-specific versions as needed.
// ============================================================================

// Snap `q` onto the lattice anchored at `anchor` (per-joint spacing
// step*scale[i]). Clamping is applied to the grid INDEX, not the value, so the
// result is always exactly on the lattice AND inside `limits`.
inline void snap_to_lattice(Config& q, const Config& anchor, int dim, float step,
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
// The lattice is ANCHORED AT `anchor` (the goal), not at wherever the start
// happens to sit, so the goal is itself a lattice vertex and is exactly
// reachable. With a start-anchored lattice the search can only stop at the
// nearest vertex to the goal, and on real MBM Fetch problems that vertex is in
// collision in 47 of 70 cases even though the goal is free -- the goal's free
// pocket is narrower than half a lattice cell, making those goals unreachable
// by construction. Anchored at the goal, every one of those 70 problems has at
// least one collision-free neighbour of the goal to arrive from.
//
// Every successor is snapped back onto the anchored grid. That is a no-op for
// nodes already on it; the only node that is not is the start, so the snap
// happens exactly once, on the start's own successors.
inline MHAStarPlanner::SuccFn
make_lattice_succ(int dim, float step, const ppln::DofScale& scale,
                  const Config& anchor,
                  const std::array<std::array<float,2>, ppln::MAX_DIM>* limits = nullptr) {
    return [dim, step, scale, anchor, limits](int /*id*/, const Config& q) {
        std::vector<Config> out;
        out.reserve(2 * dim);
        for (int i = 0; i < dim; ++i) {
            const float h = step * scale.s[i];
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                Config n = q;
                n[i] = q[i] + (float)sgn * h;
                snap_to_lattice(n, anchor, dim, step, scale, limits);
                out.push_back(n);
            }
        }
        return out;
    };
}

// Goal-connection successor. Pure lattice walking can only reach a goal by
// landing inside the goal ball, so the goal itself is never a successor and the
// final approach must be spelled out one cell at a time. Emitting each goal as a
// candidate successor once a node is within `connect` (normalised L2) hands the
// final straight line to the GPU edge check, which either confirms it or
// discards it -- the same goal-connection step SBPL-style lattice planners use.
inline MHAStarPlanner::SuccFn
with_goal_connect(MHAStarPlanner::SuccFn base, std::vector<Config> goals,
                  const ppln::DofScale& scale, int dim, float connect) {
    if (!(connect > 0.f)) return base;
    return [base, goals, scale, dim, connect](int id, const Config& q) {
        std::vector<Config> out = base(id, q);
        for (const Config& g : goals) {
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
// FNV-1a hash-combine the integer coords into a uint64. Quantised relative to
// the SAME anchor the lattice uses, or dedup bucket boundaries stop lining up
// with lattice vertices.
inline MHAStarPlanner::KeyFn
make_grid_key(int dim, float cell, const ppln::DofScale& scale, const Config& anchor) {
    return [dim, cell, scale, anchor](const Config& q) -> uint64_t {
        uint64_t h = 1469598103934665603ull;   // FNV offset basis
        for (int i = 0; i < dim; ++i) {
            int64_t c = (int64_t)std::llround((q[i] - anchor[i]) / (cell * scale.s[i]));
            uint64_t u = (uint64_t)c;
            for (int b = 0; b < 8; ++b) {       // hash the 8 bytes of the coord
                h ^= (u & 0xff);
                h *= 1099511628211ull;          // FNV prime
                u >>= 8;
            }
        }
        return h;
    };
}

// Admissible joint-space L2 heuristic to a single goal (queue 0). Inadmissible
// heuristics (queues 1..num_inad) are domain-specific and supplied by the caller.
// Deliberately NOT normalised: g comes from the GPU edge kernel as a raw
// configuration-space L2 length, so h must use the same metric to stay
// admissible. Normalisation applies to the discretization, not to the cost.
inline MHAStarPlanner::HeurFn
make_l2_heur(const Config& goal, int dim) {
    return [goal, dim](int /*qi*/, const Config& q) {
        float d = 0.f;
        for (int i = 0; i < dim; ++i) { float dq = q[i] - goal[i]; d += dq * dq; }
        return std::sqrt(d);
    };
}

// Goal test: within `radius` of `goal` in NORMALISED joint space, so the
// tolerance is the same fraction of travel on every joint instead of adding
// metres to radians on a robot with a prismatic axis.
inline MHAStarPlanner::GoalFn
make_goal_radius(const Config& goal, int dim, float radius,
                 const ppln::DofScale& scale) {
    return [goal, dim, radius, scale](const Config& q) {
        float d = 0.f;
        for (int i = 0; i < dim; ++i) {
            float dq = (q[i] - goal[i]) * scale.inv[i];
            d += dq * dq;
        }
        return std::sqrt(d) <= radius;
    };
}

} // namespace ppln::search
