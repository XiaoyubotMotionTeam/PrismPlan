#pragma once

#include "pRRTC_settings.hh"

// Settings for the CPU SMHA* search with GPU-batched lazy edge evaluation
// (see mhastar_search.hh / edge_eval.cuh). Extends pRRTC_settings so it shares
// the common granularity / collision_margin / time_limit_ms fields; the MHA*
// specific knobs (heuristic inflation, anchor bound, lattice discretization)
// are added below.
struct MHAStar_settings : pRRTC_settings {
    // --- Multi-heuristic weights (SBPL MHAPlanner semantics) ---
    float w1 = 5.0f;          // inflation_eps: key(s,i) = g(s) + w1 * h_i(s)
    float w2 = 2.0f;          // anchor_eps: use inad queue i while key(i) <= w2 * key(anchor)
    int   num_inad = 2;       // # inadmissible heuristics (queues 1..num_inad)

    // --- Queue selection strategy: 0 = RoundRobin (SBPL default), 1 = BestKey ---
    int   queue_sel = 0;

    // --- GPU lazy-edge batching ---
    int   batch_edges = 256;  // lazy edges flushed per GPU call (== EdgeEvalGpu cap)

    // Edge-evaluation policy: true = SBPL insertLazyList (a state's incoming edge
    // is confirmed only when the state is popped); false = confirm all 2*dim
    // successor edges of an expanded state in one GPU call before inserting any.
    //
    // Default is FALSE (eager). Two independent reasons:
    //  1. It is what the paper specifies -- "flushes ... the successors of one
    //     expansion (MHA*)" -- whereas pop-time single-edge verification is the
    //     SBPL variant, kept available for reference comparison.
    //  2. Measured better everywhere. Laziness only pays when most edges are
    //     valid; the lattice's true invalid rate on real MBM scenes is ~21 %, and
    //     lazy probing preferentially verifies the most attractive-looking edges
    //     (exactly the ones cutting corners through obstacles), so its observed
    //     rate inflates to ~60 % and most keys on OPEN are optimistic fiction.
    //     Real MBM, 1 s, goal_radius 0.15 fixed, lazy -> eager:
    //       Panda @0.50  0.43 -> 0.91 (cost 8.42 -> 7.77)
    //       Fetch @0.50  0.00 -> 0.21
    bool  lazy = false;

    // --- Joint lattice discretization ---
    // All three are in NORMALISED units: joint i actually gets value*scale[i],
    // where scale[i] is joint i's travel relative to the robot's mean travel
    // (dof_scale.hh). On a homogeneous arm every scale[i] is ~1 and these behave
    // exactly as the old scalar radians.
    float lattice_step = 0.50f;   // +/- step per joint => 2*dim neighbours
    float grid_cell    = 0.25f;   // node-table quantization cell for dedup
    float goal_radius  = 0.15f;   // L2 goal tolerance, used as-is. NOT grown
                                  // with lattice_step: the lattice is anchored
                                  // at the goal and goal-connect emits it as a
                                  // successor, so the goal is an exactly
                                  // reachable vertex. A step-proportional
                                  // tolerance would declare success up to
                                  // sqrt(dim)*step/2 away with no evidence the
                                  // remaining motion is collision-free.

    // Goal-connection radius in multiples of lattice_step; 0 disables. Without
    // it the goal is never a successor and is reachable only by landing inside
    // the goal ball one cell at a time.
    float goal_connect_steps = 8.0f;

    // --- Edge evaluation ---
    bool  check_self = true;      // run self-collision check on each edge

    // Configuration-space spacing between collision checks along an edge. When
    // > 0 it REPLACES granularity: each edge gets ceil(len/edge_resolution)
    // points, so a goal-connection edge an order of magnitude longer than a
    // lattice edge is checked proportionately instead of at the same fixed
    // count (which would under-check it and report a tunnelling path as valid).
    float edge_resolution = 0.025f;

    // time_limit_ms / granularity / collision_margin inherited from pRRTC_settings.
    // Reasonable MHA* defaults (base defaults are tuned for pRRTC):
    MHAStar_settings() {
        time_limit_ms = 1000.0f;
        // 32 (not 16) because a goal-connection edge is up to
        // goal_connect_steps longer than a lattice edge and must still be
        // sampled at roughly pRRTC's 0.025 rad resolution -- an under-checked
        // long edge would report a tunnelling path as a success.
        granularity   = 32;
        collision_margin = 0.0f;
    }
};
