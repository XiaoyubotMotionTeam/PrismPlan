#pragma once

#include "pRRTC_settings.hh"

// Settings for wPA*SE (weighted Parallel A* for Slow Expansions): a CPU
// bounded-suboptimal parallel-expansion weighted-A* search whose "slow
// expansions" (edge validity + cost) are offloaded to the shared GPU substrate
// in batches (see wpase_search.hh / edge_batch_bridge.hh / edge_eval.cuh).
// Extends pRRTC_settings so it shares granularity / collision_margin /
// time_limit_ms; the wPA*SE-specific knobs are added below.
//
// Algorithm reference: weighted PA*SE (Mukherjee et al.), i.e. PasePlanner run
// with heuristic_weight > 1; suboptimality bound = heuristic_w.
struct WPASE_settings : pRRTC_settings {
    // --- weighted-A* / bounded suboptimality ---
    float heuristic_w = 5.0f;     // f(s) = g(s) + heuristic_w * h(s); bound = w

    // --- parallel slow expansions (PA*SE) ---
    // Max mutually-INDEPENDENT states expanded per round; their successor edges
    // are flushed to the GPU in a single evaluate() call. Analogue of PA*SE's
    // num_threads (here the parallelism is realised by GPU batch width).
    int   num_parallel = 4;

    // --- GPU lazy-edge batching ---
    int   batch_edges = 256;      // edges flushed per GPU call (<= EdgeEvalGpu cap)

    // --- joint lattice discretization ---
    // All three are in NORMALISED units: joint i gets value*scale[i], where
    // scale[i] is joint i's travel relative to the robot's mean travel
    // (dof_scale.hh). On a homogeneous arm every scale[i] is ~1 and these behave
    // exactly as the old scalar radians.
    // Step measured on real MBM (see config/planners/wpase.yaml): 0.50 is the
    // success-rate peak on Fetch and Baxter, and improves path cost on Panda.
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

    // --- edge evaluation ---
    bool  check_self = true;      // run self-collision check on each edge

    // Configuration-space spacing between collision checks along an edge. When
    // > 0 it REPLACES granularity: each edge gets ceil(len/edge_resolution)
    // points, so a goal-connection edge an order of magnitude longer than a
    // lattice edge is checked proportionately instead of at the same fixed
    // count (which would under-check it and report a tunnelling path as valid).
    float edge_resolution = 0.025f;

    // time_limit_ms / granularity / collision_margin inherited from pRRTC_settings.
    // Reasonable wPA*SE defaults (base defaults are tuned for pRRTC):
    WPASE_settings() {
        time_limit_ms = 1000.0f;
        // 32 (not 16) because a goal-connection edge is up to
        // goal_connect_steps longer than a lattice edge and must still be
        // sampled at roughly pRRTC's 0.025 rad resolution -- an under-checked
        // long edge would report a tunnelling path as a success.
        granularity   = 32;
        collision_margin = 0.0f;
    }
};
