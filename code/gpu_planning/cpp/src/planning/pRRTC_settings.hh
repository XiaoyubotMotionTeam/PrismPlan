#pragma once

struct pRRTC_settings {
    int max_samples = 1000000;
    int max_iters = 1000000;
    float time_limit_ms = 4000.0f;   // GPU kernel wall-clock limit (0 = no limit)
    int gpu_clock_rate_khz = 0;      // filled from cudaDeviceProp.clockRate at init
    int num_new_configs = 600;
    int granularity = 16;
    float range = 0.5;

    int balance = 1; // 0 = no balance, 1 = balance strategy 1, 2 = balance strategy 2
    float tree_ratio = 0.5; // 0.5 for balance=1, 1.0 for balance=2

    bool dynamic_domain = true;
    float dd_alpha = 0.0001;
    float dd_radius = 4.0;
    float dd_min_radius = 1.0;

    bool enable_mesh_collision = false;  // GPU BVH mesh precision collision

    // Path shortcutting (post-processing)
    bool shortcut_path = false;
    float valid_segment_length = 0.025f;
    float collision_margin = 0.0f;

    // Extra safety margin for shortcut CC (meters).
    // Shortcut should be MORE conservative than post-plan validation
    // (which uses collision_margin=0) so that interpolated points after
    // time-parameterization never clip obstacles.
    // Effective shortcut margin = collision_margin + shortcut_collision_margin.
    float shortcut_collision_margin = 0.0005f;  // 0.5 mm
};