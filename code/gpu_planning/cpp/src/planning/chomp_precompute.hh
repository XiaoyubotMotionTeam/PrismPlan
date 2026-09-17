#pragma once

#include <vector>

// Host-only Eigen precompute for CHOMP, kept out of chomp.cu so nvcc never has
// to parse <Eigen/Dense> (its product_type_selector template metaprogramming
// trips a parser bug in device-code compilation on some nvcc versions) — same
// rationale as stomp_precompute.
namespace CHOMP {

// Raw control-cost quadratic form R (row-major, T*T): the ros-industrial 5-point
// acceleration matrix (FINITE_DIFF_RULE_LENGTH=7, coeffs {-1/12,16/12,-30/12,
// 16/12,-1/12} on a +/-6-padded grid, central T*T block of dt*A^T*A). This is
// the SAME metric STOMP uses (compute_control_cost_R); the smoothness gradient
// is R theta and the per-timestep smoothness cost is theta .* (R theta).
std::vector<float> compute_control_cost_R(int num_timesteps, float delta_t);

// Covariant preconditioner A^{-1} = (R + 1e-8 I)^{-1} (row-major, T*T). CHOMP's
// covariant update is theta <- theta - step_size * A^{-1} grad; using the
// smoothness metric R as A pulls the trajectory toward the (zero-acceleration)
// straight-line seed between the pinned endpoints while the obstacle gradient
// pushes it clear.
std::vector<float> compute_smoothness_Ainv(int num_timesteps, float delta_t);

}  // namespace CHOMP
