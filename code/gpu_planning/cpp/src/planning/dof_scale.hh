#pragma once

// ============================================================================
// Per-DoF normalisation derived from the robot's own joint limits.
//
// A whole-body composite concatenates heterogeneous joints into one tree: on
// Fetch the prismatic torso travels 0.386 m while every revolute joint spans
// 2.7-6.3 rad. Any SCALAR step, tolerance, noise stddev or gradient step is
// therefore simultaneously far too coarse for the prismatic axis and far too
// fine for the revolute ones, and an L2 norm over the raw vector adds metres to
// radians. That is why parameters tuned on 7-DoF Panda do not transfer.
//
// scale[i] is joint i's travel relative to the robot's mean travel, so a
// normalised quantity s means s * scale[i] on joint i: the same FRACTION of
// every joint's range, whatever its unit. On a homogeneous arm all scale[i]
// are ~1 and the normalised value equals the old scalar one, so Panda-tuned
// defaults keep their meaning.
// ============================================================================

#include "src/planning/robot_model.cuh"   // ppln::MAX_DIM

#include <array>
#include <cmath>

namespace ppln {

struct DofScale {
    std::array<float, MAX_DIM> s{};     // scale[i] = range[i] / mean_range
    std::array<float, MAX_DIM> inv{};   // 1 / s[i]
    int dim = 0;
};

// `lo` / `hi` are host copies of RobotModel::joint_lower / joint_upper.
inline DofScale make_dof_scale(const float* lo, const float* hi, int dim) {
    DofScale d;
    d.dim = dim;
    float sum = 0.f;
    for (int i = 0; i < dim; ++i) sum += std::fabs(hi[i] - lo[i]);
    // A degenerate model (all limits equal) must not produce inf/NaN scales.
    const float mean = (dim > 0 && sum > 0.f) ? sum / (float)dim : 1.0f;
    for (int i = 0; i < dim; ++i) {
        float r = std::fabs(hi[i] - lo[i]) / mean;
        if (!(r > 1e-6f)) r = 1.0f;
        d.s[i]   = r;
        d.inv[i] = 1.0f / r;
    }
    for (int i = dim; i < MAX_DIM; ++i) { d.s[i] = 1.0f; d.inv[i] = 1.0f; }
    return d;
}

}  // namespace ppln
