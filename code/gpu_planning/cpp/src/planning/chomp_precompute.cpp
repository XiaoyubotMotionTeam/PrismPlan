#include "chomp_precompute.hh"

#include <Eigen/Dense>

namespace CHOMP {

namespace {

// ros-industrial/stomp finite-difference rule: FINITE_DIFF_RULE_LENGTH = 7,
// half-width = 3. The acceleration row is the 5-tap 4th-order central stencil
// {-1/12, 16/12, -30/12, 16/12, -1/12} (padded to 7 with leading/trailing 0).
// Building it on a matrix padded by (RULE_LENGTH-1)=6 on each side and then
// extracting the central T*T block gives the same boundary treatment as STOMP's
// build_control_cost_R — CHOMP's smoothness metric is therefore bit-identical to
// STOMP's control-cost metric, so the two optimizers penalise smoothness the
// same way on the shared substrate.
constexpr int kFiniteDiffRuleLength = 7;
constexpr int kPad = kFiniteDiffRuleLength - 1;  // 6

Eigen::MatrixXf build_control_cost_R(int Tn, float delta_t) {
    const int Np = Tn + 2 * kPad;
    const float inv_dt2 = 1.0f / (delta_t * delta_t);
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

std::vector<float> compute_control_cost_R(int num_timesteps, float delta_t) {
    const int Tn = num_timesteps;
    Eigen::MatrixXf R = build_control_cost_R(Tn, delta_t);

    std::vector<float> h_R((size_t)Tn * Tn);
    for (int i = 0; i < Tn; i++)
        for (int j = 0; j < Tn; j++)
            h_R[(size_t)i * Tn + j] = R(i, j);
    return h_R;
}

std::vector<float> compute_smoothness_Ainv(int num_timesteps, float delta_t) {
    const int Tn = num_timesteps;
    Eigen::MatrixXf Rm = build_control_cost_R(Tn, delta_t);
    Rm += 1.0e-8f * Eigen::MatrixXf::Identity(Tn, Tn);
    Eigen::MatrixXf Ainv = Rm.inverse();

    std::vector<float> h_Ainv((size_t)Tn * Tn);
    for (int i = 0; i < Tn; i++)
        for (int j = 0; j < Tn; j++)
            h_Ainv[(size_t)i * Tn + j] = Ainv(i, j);
    return h_Ainv;
}

}  // namespace CHOMP
