#pragma once

#include "pRRTC_settings.hh"

struct MITStar_settings : pRRTC_settings {
    // --- Batch / iteration control ---
    int batch_size = 64;            // samples per batch (also the MAX / allocation cap)
    float time_limit_ms = 500.0f;    // wall-clock time limit

    // --- Adaptive batch size (MIT* "A": AdaptiveBatchSize, DecayMethod::LOG) ---
    // When enabled, the per-batch sample count shrinks from batch_size (max)
    // toward min_batch_size as the informed-set hyperellipsoid area shrinks with
    // an improving solution, mirroring ompl::geometric::mitstar::AdaptiveBatchSize
    // (LOG/sigmoid decay). Before the first solution the full batch_size is used.
    // Buffers are always allocated for the max (batch_size), so the adaptive value
    // must stay in [min_batch_size, batch_size]. Disable => fixed batch_size every
    // batch (bit-identical to the pre-adaptive behaviour).
    bool adaptive_batch = true;
    int min_batch_size = 32;         // lower bound (minSamples) for adaptive decay

    // --- Early exit after solution ---
    // After finding a solution, if cost doesn't improve for this many ms,
    // exit early instead of burning full time_limit_ms.  0 = disabled.
    float early_exit_ms = 0.0f;

    // --- Connectivity ---
    float eta_knn = 1.001f;          // OMPL k-NN factor: k = ceil(eta * e * (1+1/d) * log(n))
    float gamma_rgg = 2.0f;         // RGG radius factor for radius-NN: r = gamma * (V/zeta_d)^(1/d) * (log(n)/n)^(1/d)
    int max_neighbors = 32;          // max neighbors per NN query


    // --- Reverse search ---
    int m_reverse_eval = 128;              // edges evaluated per reverse search iteration
    // FIX_B2: was 2000 → only 1 midpoint per edge in reverse CC.
    // factor=2 gives sparse_resolution = 0.05*2 = 0.1 rad → ~7 checks
    // for a typical 0.7 rad edge.  Much better heuristic accuracy.
    float initial_sparse_factor = 2.0f;

    // --- Forward search ---
    int m_forward_eval = 512;              // edges evaluated per forward search iteration

    // --- Suboptimality ---
    float initial_suboptimality = 1e10f;   // ~INF before first solution

    // --- EIS (Estimated Initial Solution) ---
    bool use_eis = true;

    // --- Valid segment length (CC resolution) ---
    float valid_segment_length = 0.025f;   // CC resolution (rad); halved from 0.05 to prevent false-free edges near obstacles

    // --- Collision margin (meters) ---
    // Inflates sphere radii during BOTH node CC (batch_cc_kernel) and
    // edge CC (sparse_cc_kernel, full_cc_kernel).  Keeps planned paths
    // further from obstacles so post-plan interpolation doesn't clip.
    float collision_margin = 0.0f;

    // --- ESDF clearance cost ---
    // Penalises edges near obstacles:
    //   penalty = min(clearance_weight / max(min_clearance, clearance_epsilon), max_clearance_penalty)
    //   weight  = L2_dist * (1 + penalty)
    // Set clearance_weight = 0 to disable (pure L2, default).
    float clearance_weight = 0.0f;
    float clearance_epsilon = 0.005f;       // 5mm floor
    float max_clearance_penalty = 5.0f;     // cap: weight <= L2 * 6.0

    // --- Path shortcutting ---
    // After path extraction, try to skip intermediate waypoints by checking
    // if direct connections are collision-free.  Runs a single GPU kernel
    // on all O(N²) skip-edge candidates, then greedy forward scan on host.
    bool shortcut_path = true;

    // --- Graph capacity ---
    int max_nodes = 50000;
    int max_edges_per_node = 32;

    // --- Debug / diagnostics ---
    bool debug_dump = false;     // write final state binary to /tmp/mitstar_dump.bin
};
