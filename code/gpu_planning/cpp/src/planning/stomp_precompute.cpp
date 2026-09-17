#include "stomp_precompute.hh"

#include <Eigen/Dense>
#include <algorithm>

namespace STOMP {

namespace {

// ros-industrial/stomp finite-difference rule: FINITE_DIFF_RULE_LENGTH = 7,
// half-width = 3. The acceleration row is the 5-tap 4th-order central stencil
// {-1/12, 16/12, -30/12, 16/12, -1/12} (padded to 7 with leading/trailing 0).
// Building it on a matrix padded by (RULE_LENGTH-1)=6 on each side and then
// extracting the central T*T block gives the same boundary treatment as
// generateSmoothingMatrix(): near-boundary rows couple to the zero pads, so R
// is the exact control-cost quadratic form used by both the sampling
// covariance and the M projection.
constexpr int kFiniteDiffRuleLength = 7;
constexpr int kPad = kFiniteDiffRuleLength - 1;  // 6

// R = dt * A_padded^T * A_padded, central (T,T) block. A_padded is the 5-point
// acceleration operator on the padded index grid. Shared by every consumer so
// L, M and the per-rollout control cost all use one identical metric.
Eigen::MatrixXf build_control_cost_R(int Tn, float delta_t) {
    const int Np = Tn + 2 * kPad;
    const float inv_dt2 = 1.0f / (delta_t * delta_t);
    // coeffs indexed by (j + 3), j in [-3, 3]
    const float coeff[kFiniteDiffRuleLength] = {
        0.0f, -1.0f / 12.0f, 16.0f / 12.0f, -30.0f / 12.0f, 16.0f / 12.0f, -1.0f / 12.0f, 0.0f};

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(Np, Np);
    for (int i = 0; i < Np; i++) {
        for (int j = -kFiniteDiffRuleLength / 2; j <= kFiniteDiffRuleLength / 2; j++) {
            const int idx = i + j;
            if (idx < 0 || idx >= Np) continue;
            A(i, idx) = inv_dt2 * coeff[j + kFiniteDiffRuleLength / 2];
        }
    }

    Eigen::MatrixXf R_padded = delta_t * (A.transpose() * A);
    return R_padded.block(kPad, kPad, Tn, Tn);
}

}  // namespace

std::vector<float> compute_cholesky_L(int num_timesteps, float delta_t, float noise_scale) {
    const int Tn = num_timesteps;

    Eigen::MatrixXf Rm = build_control_cost_R(Tn, delta_t);
    Rm += 1.0e-8f * Eigen::MatrixXf::Identity(Tn, Tn);
    Eigen::MatrixXf Rinv = Rm.inverse();
    const float max_entry = std::max(Rinv.maxCoeff(), 1.0e-12f);
    Rinv *= ((1.0f / (float)Tn) / max_entry);
    Rinv = 0.5f * (Rinv + Rinv.transpose());

    Eigen::LLT<Eigen::MatrixXf> llt(Rinv);
    Eigen::MatrixXf L = llt.matrixL();

    // Global exploration-amplitude knob: noise = L @ z, so scaling L scales the
    // sampled noise stddev linearly (cov scales by noise_scale^2). 1.0 = baseline.
    const float s = (noise_scale > 0.0f) ? noise_scale : 1.0f;

    std::vector<float> h_L((size_t)Tn * Tn);
    for (int i = 0; i < Tn; i++)
        for (int j = 0; j < Tn; j++)
            h_L[(size_t)i * Tn + j] = s * L(i, j);
    return h_L;
}

std::vector<float> compute_projection_M(int num_timesteps, float delta_t) {
    const int Tn = num_timesteps;

    // Same 5-point control-cost R as compute_cholesky_L / compute_control_cost_R,
    // so M and the sampling covariance share one metric.
    Eigen::MatrixXf Rm = build_control_cost_R(Tn, delta_t);
    Rm += 1.0e-8f * Eigen::MatrixXf::Identity(Tn, Tn);
    Eigen::MatrixXf Rinv = Rm.inverse();

    // Per-column normalisation: scale column j so max_i Rinv(i,j) == 1/Tn
    // (ros-industrial generateSmoothingMatrix column-max scaling).
    const float inv_T = 1.0f / (float)Tn;
    Eigen::MatrixXf M(Tn, Tn);
    for (int j = 0; j < Tn; j++) {
        float col_max = Rinv.col(j).maxCoeff();
        float scale = inv_T / std::max(col_max, 1.0e-12f);
        M.col(j) = Rinv.col(j) * scale;
    }

    std::vector<float> h_M((size_t)Tn * Tn);
    for (int i = 0; i < Tn; i++)
        for (int j = 0; j < Tn; j++)
            h_M[(size_t)i * Tn + j] = M(i, j);
    return h_M;
}

std::vector<float> compute_control_cost_R(int num_timesteps, float delta_t) {
    const int Tn = num_timesteps;

    // The raw (un-inverted, un-normalised) control-cost quadratic form R.
    // Per-rollout smoothness cost = theta^T R theta, evaluated on device as
    // control_costs[d,t] = theta[t] * (R theta)[t] so the row-sum recovers the
    // quadratic form — identical to ros-industrial computeParametersControlCosts.
    Eigen::MatrixXf R = build_control_cost_R(Tn, delta_t);

    std::vector<float> h_R((size_t)Tn * Tn);
    for (int i = 0; i < Tn; i++)
        for (int j = 0; j < Tn; j++)
            h_R[(size_t)i * Tn + j] = R(i, j);
    return h_R;
}

}  // namespace STOMP
