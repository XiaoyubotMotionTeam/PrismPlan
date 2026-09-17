#pragma once

// Shared Halton quasi-random state used by pRRTC and MITStar.
// Extracted from individual .cu files to allow SolverBuffers to reference
// the type in a header (solver_buffers.hh).

#include "robot_model.cuh"  // ppln::MAX_DIM

struct HaltonState_runtime {
    float b[ppln::MAX_DIM];
    float n[ppln::MAX_DIM];
    float d[ppln::MAX_DIM];
};
