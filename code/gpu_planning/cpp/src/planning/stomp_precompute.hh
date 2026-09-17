#pragma once

#include <vector>

// Host-only Eigen precompute, kept out of stomp.cu so nvcc never has to
// parse <Eigen/Dense> (its product_type_selector template metaprogramming
// trips a parser bug in device-code compilation on some nvcc versions).
namespace STOMP {

// Cholesky factor L (row-major, num_timesteps*num_timesteps) such that
// L L^T = R_inv. R is the ros-industrial 5-point acceleration control-cost
// matrix (FINITE_DIFF_RULE_LENGTH=7, coeffs {-1/12,16/12,-30/12,16/12,-1/12}
// on a ±6-padded grid, central T*T block of dt*A^T*A). R_inv = (R+1e-8 I)^-1
// is GLOBALLY rescaled so its max entry == 1/num_timesteps, resymmetrized,
// then factored. noise_scale linearly scales the returned L (i.e. the noise
// stddev); 1.0 = baseline.
std::vector<float> compute_cholesky_L(int num_timesteps, float delta_t, float noise_scale = 1.0f);

// STOMP update-smoothing projection matrix M (row-major, T*T).
// Kalakrishnan 2011 §III: the raw per-timestep weighted noise average
// delta_raw must be projected through M = R^-1 with EACH COLUMN scaled so its
// maximum entry == 1/T. Uses the SAME 5-point R as compute_cholesky_L /
// compute_control_cost_R. Distinct from compute_cholesky_L, which GLOBALLY
// rescales R_inv for the sampling covariance — M uses PER-COLUMN
// normalisation and is NOT factored.
std::vector<float> compute_projection_M(int num_timesteps, float delta_t);

// Raw control-cost quadratic form R (row-major, T*T): the un-inverted,
// un-normalised 5-point acceleration matrix underlying both compute_cholesky_L
// and compute_projection_M. Uploaded to the device so the per-rollout
// smoothness cost is evaluated as theta^T R theta under the identical metric.
std::vector<float> compute_control_cost_R(int num_timesteps, float delta_t);

}  // namespace STOMP
