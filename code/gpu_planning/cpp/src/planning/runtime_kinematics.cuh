#pragma once

// Runtime (data-driven) FK and collision functions.
// Extracted from panda.cuh — these are robot-agnostic, reading from
// ppln::RobotModel global memory instead of __constant__ memory.

#include "src/planning/utils.cuh"
#include "src/planning/robot_model.cuh"

namespace ppln::collision {

// ============================================================================
// Forward kinematics — full sphere model
// ============================================================================
// 4-thread cooperative: each thread handles one column of the 4x4 transform.
__device__ void fk_runtime(
    const ppln::RobotModel& model,
    const float* q,
    volatile float* sphere_pos,
    float *T,
    const int tid
) {
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    const int n_slots = model.n_t_slots;
    int T_offset = batch_ind * n_slots * 16;
    float T_step_col[4];
    float *T_base = T + T_offset;

    // Seed every scratch slot with the identity.
    for (int s = 0; s < n_slots; ++s) {
        float *T_col_s = T_base + s * 16 + col_ind * 4;
        for (int r = 0; r < 4; r++) T_col_s[r] = 0;
        T_col_s[col_ind] = 1;
    }
    __syncwarp();

    // Walk the kinematic tree in DFS order: every parent is visited before its
    // children, so joint i composes onto the slot its parent left behind. For a
    // serial chain this is the old loop verbatim (parent slot == own slot).
    for (int j = 0; j < model.n_joints; ++j) {
        const int i = model.dfs_order[j];
        const int slot = model.t_memory_idx[i];
        const int parent_slot = model.t_memory_idx[model.joint_parents[i]];

        if (j > 0) {
            int ft_addr_start = i * 16;
            int joint_type = model.joint_types[i];
            const float qi = q[model.joint_id_to_dof[i]];
            if (joint_type <= Z_PRISM) {
                prism_fn(&model.fixed_transforms[ft_addr_start], qi, col_ind, T_step_col, joint_type);
            } else if (joint_type == X_ROT) {
                xrot_fn(&model.fixed_transforms[ft_addr_start], qi, col_ind, T_step_col);
            } else if (joint_type == Y_ROT) {
                yrot_fn(&model.fixed_transforms[ft_addr_start], qi, col_ind, T_step_col);
            } else if (joint_type == Z_ROT) {
                zrot_fn(&model.fixed_transforms[ft_addr_start], qi, col_ind, T_step_col);
            }
            // Buffer the whole column before publishing it: the four cooperating
            // threads each read every column of the parent slot, so writing in
            // place would race when parent slot == own slot.
            float T_col_tmp[4];
            for (int r = 0; r < 4; r++) {
                T_col_tmp[r] = dot4_col(&T_base[parent_slot * 16 + r], T_step_col);
            }
            __syncwarp();
            for (int r = 0; r < 4; r++) {
                T_base[slot * 16 + col_ind * 4 + r] = T_col_tmp[r];
            }
        }
        __syncwarp();

        const int s_beg = model.sphere_joint_offsets[i];
        const int s_end = model.sphere_joint_offsets[i + 1];
        for (int k = s_beg; k < s_end; ++k) {
            const int si = model.sphere_order[k];
            if (col_ind < 3) {
                sphere_pos[si * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] =
                    T_base[slot*16 + col_ind]       * model.spheres[si].x +
                    T_base[slot*16 + col_ind + M]   * model.spheres[si].y +
                    T_base[slot*16 + col_ind + M*2] * model.spheres[si].z +
                    T_base[slot*16 + col_ind + M*3];
            }
        }
        __syncwarp();
    }
}

// ============================================================================
// Forward kinematics — per-joint world frames, single thread
// ============================================================================
// Same tree walk as fk_runtime, but emits every joint's world transform instead
// of sphere positions: the mesh/BVH path transforms triangles, not spheres, so
// it needs the frames themselves. Output is column-major per joint,
// joint_transforms[i*16 + c*4 + r] = T_i[r][c].
__device__ void fk_joint_transforms_runtime(
    const ppln::RobotModel& model,
    const float* q,
    float* joint_transforms          // [n_joints * 16]
) {
    for (int j = 0; j < model.n_joints; ++j) {
        const int i = model.dfs_order[j];
        float* out = &joint_transforms[i * 16];

        if (j == 0) {                       // root frame is the identity
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 4; r++)
                    out[c * 4 + r] = (r == c) ? 1.0f : 0.0f;
            continue;
        }

        // fixed_transforms is ROW-MAJOR: ft[r*4 + c] = FT[r][c].
        const float* ft = &model.fixed_transforms[i * 16];
        const int joint_type = model.joint_types[i];
        const float qi = q[model.joint_id_to_dof[i]];
        const float sn = __sinf(qi), cs = __cosf(qi);

        float step[4][4];                   // step = FT * joint_motion(qi)
        if (joint_type == Z_ROT) {
            for (int r = 0; r < 4; r++) {
                int rb = r * 4;
                step[r][0] =  ft[rb+0] * cs + ft[rb+1] * sn;
                step[r][1] = -ft[rb+0] * sn + ft[rb+1] * cs;
                step[r][2] = ft[rb+2];
                step[r][3] = ft[rb+3];
            }
        } else if (joint_type == X_ROT) {
            for (int r = 0; r < 4; r++) {
                int rb = r * 4;
                step[r][0] = ft[rb+0];
                step[r][1] =  ft[rb+1] * cs + ft[rb+2] * sn;
                step[r][2] = -ft[rb+1] * sn + ft[rb+2] * cs;
                step[r][3] = ft[rb+3];
            }
        } else if (joint_type == Y_ROT) {
            for (int r = 0; r < 4; r++) {
                int rb = r * 4;
                step[r][0] = ft[rb+0] * cs - ft[rb+2] * sn;
                step[r][1] = ft[rb+1];
                step[r][2] = ft[rb+0] * sn + ft[rb+2] * cs;
                step[r][3] = ft[rb+3];
            }
        } else if (joint_type >= X_PRISM && joint_type <= Z_PRISM) {
            for (int r = 0; r < 4; r++) {
                int rb = r * 4;
                step[r][0] = ft[rb+0];
                step[r][1] = ft[rb+1];
                step[r][2] = ft[rb+2];
                step[r][3] = ft[rb + joint_type] * qi + ft[rb+3];
            }
        } else {                            // FIXED or unknown
            for (int r = 0; r < 4; r++) {
                int rb = r * 4;
                step[r][0] = ft[rb+0];
                step[r][1] = ft[rb+1];
                step[r][2] = ft[rb+2];
                step[r][3] = ft[rb+3];
            }
        }

        const float* par = &joint_transforms[model.joint_parents[i] * 16];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float acc = 0.0f;
                for (int k = 0; k < 4; k++) acc += par[k * 4 + r] * step[k][c];
                out[c * 4 + r] = acc;
            }
        }
    }
}

// ============================================================================
// Forward kinematics — approximate sphere model
// ============================================================================
__device__ void fk_approx_runtime(
    const ppln::RobotModel& model,
    const float* q,
    volatile float* sphere_pos_approx,
    float *T,
    const int tid
) {
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    const int n_slots = model.n_t_slots;
    int T_offset = batch_ind * n_slots * 16;
    float T_step_col[4];
    float *T_base = T + T_offset;

    for (int s = 0; s < n_slots; ++s) {
        float *T_col_s = T_base + s * 16 + col_ind * 4;
        for (int r = 0; r < 4; r++) T_col_s[r] = 0;
        T_col_s[col_ind] = 1;
    }
    __syncwarp();

    // Same tree walk as fk_runtime, over the coarse gate model. The approx
    // sphere set is one bounding sphere per joint, so it shares the topology.
    for (int j = 0; j < model.n_joints; ++j) {
        const int i = model.dfs_order[j];
        const int slot = model.t_memory_idx[i];
        const int parent_slot = model.t_memory_idx[model.joint_parents[i]];

        if (j > 0) {
            int ft_addr_start = i * 16;
            int joint_type = model.approx_joint_types[i];
            const float qi = q[model.joint_id_to_dof[i]];
            if (joint_type <= Z_PRISM) {
                prism_fn(&model.approx_fixed_transforms[ft_addr_start], qi, col_ind, T_step_col, joint_type);
            } else if (joint_type == X_ROT) {
                xrot_fn(&model.approx_fixed_transforms[ft_addr_start], qi, col_ind, T_step_col);
            } else if (joint_type == Y_ROT) {
                yrot_fn(&model.approx_fixed_transforms[ft_addr_start], qi, col_ind, T_step_col);
            } else if (joint_type == Z_ROT) {
                zrot_fn(&model.approx_fixed_transforms[ft_addr_start], qi, col_ind, T_step_col);
            }
            float T_col_tmp[4];
            for (int r = 0; r < 4; r++) {
                T_col_tmp[r] = dot4_col(&T_base[parent_slot * 16 + r], T_step_col);
            }
            __syncwarp();
            for (int r = 0; r < 4; r++) {
                T_base[slot * 16 + col_ind * 4 + r] = T_col_tmp[r];
            }
        }
        __syncwarp();

        const int s_beg = model.approx_sphere_joint_offsets[i];
        const int s_end = model.approx_sphere_joint_offsets[i + 1];
        for (int k = s_beg; k < s_end; ++k) {
            const int si = model.approx_sphere_order[k];
            if (col_ind < 3) {
                sphere_pos_approx[si * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] =
                    T_base[slot*16 + col_ind]       * model.approx_spheres[si].x +
                    T_base[slot*16 + col_ind + M]   * model.approx_spheres[si].y +
                    T_base[slot*16 + col_ind + M*2] * model.approx_spheres[si].z +
                    T_base[slot*16 + col_ind + M*3];
            }
        }
        __syncwarp();
    }
}

// ============================================================================
// Self-collision check — full model
// ============================================================================
__device__ bool self_collision_check_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos,
    volatile int* joint_in_collision,
    const int tid
) {
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision = false;

    for (int i = thread_ind; i < model.n_self_cc_ranges; i += 4) {
        if (ppln::device_utils::warp_any_active_mask(has_collision)) return false;
        int sphere_1_ind = model.self_cc_ranges[i * 3 + 0];
        float sphere_1[3] = {
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        int start = model.self_cc_ranges[i * 3 + 1];
        int end   = model.self_cc_ranges[i * 3 + 2];
        for (int j = start; j <= end; j++) {
            float sphere_2[3] = {
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            if (ppln::device_utils::sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], model.spheres[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], model.spheres[j].w
            )) {
                has_collision = true;
            }
        }
    }
    return !has_collision;
}

// ============================================================================
// Self-collision min-hinge penalty — full model
// ============================================================================
// Mirrors self_collision_check_runtime's self_cc_ranges traversal, but
// instead of a binary early-exit, computes the minimum pairwise clearance
// dist-(w_i+w_j) across ALL pairs (no early exit) and returns the hinge
// relu(self_margin - min_clearance). self_collision_check_runtime cannot be
// reused since it stops at the first collision, whereas STOMP's smooth
// self-collision cost needs the actual minimum clearance.
__device__ float self_min_hinge_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos,
    const int tid,
    float self_margin
) {
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    float my_min = 1000.0f;

    for (int i = thread_ind; i < model.n_self_cc_ranges; i += 4) {
        int sphere_1_ind = model.self_cc_ranges[i * 3 + 0];
        float sphere_1[3] = {
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        int start = model.self_cc_ranges[i * 3 + 1];
        int end   = model.self_cc_ranges[i * 3 + 2];
        for (int j = start; j <= end; j++) {
            float dx = sphere_1[0] - sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 0];
            float dy = sphere_1[1] - sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 1];
            float dz = sphere_1[2] - sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 2];
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);
            float clr = dist - (model.spheres[sphere_1_ind].w + model.spheres[j].w);
            my_min = fminf(my_min, clr);
        }
    }

    for (int offset = 2; offset >= 1; offset >>= 1) {
        float other = __shfl_xor_sync(0xf, my_min, offset);
        my_min = fminf(my_min, other);
    }

    return fmaxf(0.0f, self_margin - my_min);
}

// ============================================================================
// Environment collision check — full model (legacy Environment path)
// ============================================================================
__device__ bool env_collision_check_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos,
    volatile int* joint_in_collision,
    Environment<float> *env,
    const int tid
) {
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision = false;
    int ns = model.n_spheres;
    int rem = ns % 4;

    for (int i = ns - 1 - thread_ind; i >= rem; i -= 4) {
        if (joint_in_collision[20*batch_ind + model.sphere_to_joint[i]] > 0 &&
            ppln::device_utils::sphere_environment_in_collision(
                env,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                model.spheres[i].w
            )
        ) {
            has_collision = true;
        }
        if (ppln::device_utils::warp_any_full_mask(has_collision)) return false;
    }

    if (thread_ind < rem) {
        int i = thread_ind;
        if (joint_in_collision[20*batch_ind + model.sphere_to_joint[i]] > 0 &&
            ppln::device_utils::sphere_environment_in_collision(
                env,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                model.spheres[i].w
            )
        ) {
            has_collision = true;
        }
    }
    return !has_collision;
}

// ============================================================================
// Self-collision check — approximate model
// ============================================================================
__device__ bool self_collision_check_approx_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos_approx,
    volatile int* joint_in_collision,
    const int tid
) {
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;

    for (int i = thread_ind; i < model.n_approx_self_cc_ranges; i += 4) {
        int sphere_1_ind = model.approx_self_cc_ranges[i * 3 + 0];
        float sphere_1[3] = {
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        int start = model.approx_self_cc_ranges[i * 3 + 1];
        int end   = model.approx_self_cc_ranges[i * 3 + 2];
        for (int j = start; j <= end; j++) {
            float sphere_2[3] = {
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            if (ppln::device_utils::sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], model.approx_spheres[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], model.approx_spheres[j].w
            )) {
                atomicAdd((int*)&joint_in_collision[20*batch_ind + model.approx_sphere_to_joint[sphere_1_ind]], 1);
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Environment collision check — approximate model (legacy Environment path)
// ============================================================================
__device__ bool env_collision_check_approx_runtime(
    const ppln::RobotModel& model,
    volatile float* sphere_pos_approx,
    volatile int* joint_in_collision,
    Environment<float> *env,
    const int tid
) {
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;
    int na = model.n_approx_spheres;
    int chunk = na / 4;

    for (int i = chunk * thread_ind; i < chunk * (thread_ind + 1) && i < na; i++) {
        if (ppln::device_utils::sphere_environment_in_collision(
            env,
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            model.approx_spheres[i].w
        )) {
            atomicAdd((int*)&joint_in_collision[20*batch_ind + model.approx_sphere_to_joint[i]], 1);
            out = false;
        }
    }

    // Handle remaining spheres
    int handled = chunk * 4;
    if (handled + thread_ind < na) {
        int i = handled + thread_ind;
        if (ppln::device_utils::sphere_environment_in_collision(
            env,
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            model.approx_spheres[i].w
        )) {
            atomicAdd((int*)&joint_in_collision[20*batch_ind + model.approx_sphere_to_joint[i]], 1);
            out = false;
        }
    }
    return out;
}

// ============================================================================
// FK for mesh collision — outputs per-joint 4x4 transforms to global memory
// ============================================================================
// Does NOT compute sphere positions. Used when mesh collision is enabled.
// Each of the 4 cooperating threads writes one column of each joint's 4x4.
__device__ void fk_mesh_transforms_runtime(
    const ppln::RobotModel& model,
    const float* q,
    float *T,                  // [BATCH_SIZE * 16] shared memory scratch
    float *out_transforms,     // global memory, offset by batch_ind already applied by caller
    const int tid
) {
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    int T_offset = batch_ind * 16;
    float T_step_col[4];
    float *T_base = T + T_offset;
    float *T_col = T_base + col_ind * 4;

    // Each batch_ind writes to its own output slice
    float *my_out = out_transforms + batch_ind * model.n_joints * 16;

    // Initialize to identity
    for (int r = 0; r < 4; r++) T_col[r] = 0;
    T_col[col_ind] = 1;

    for (int i = 0; i < model.n_joints; ++i) {
        if (i > 0) {
            int ft_addr_start = i * 16;
            int joint_type = model.joint_types[i];
            if (joint_type <= X_PRISM) {
                prism_fn(&model.fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col, joint_type);
            } else if (joint_type == X_ROT) {
                xrot_fn(&model.fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            } else if (joint_type == Y_ROT) {
                yrot_fn(&model.fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            } else if (joint_type == Z_ROT) {
                zrot_fn(&model.fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }
            for (int r = 0; r < 4; r++) {
                T_col[r] = dot4_col(&T_base[r], T_step_col);
            }
        }
        // Write this joint's column to global memory
        int out_offset = i * 16 + col_ind * 4;
        for (int r = 0; r < 4; r++) {
            my_out[out_offset + r] = T_col[r];
        }
    }
}

// ============================================================================
// Utility: scale config from [0,1] to joint limits
// ============================================================================
__device__ __forceinline__ void scale_cfg_runtime(
    const ppln::RobotModel& model,
    float *q
) {
    for (int i = 0; i < model.n_dof; i++) {
        q[i] = q[i] * model.joint_range[i] + model.joint_lower[i];
    }
}

// ============================================================================
// Utility: clamp config to joint limits
// ============================================================================
__device__ __forceinline__ void clamp_to_limits_runtime(
    const ppln::RobotModel& model,
    float *q
) {
    for (int i = 0; i < model.n_dof; i++) {
        q[i] = fmaxf(model.joint_lower[i], fminf(model.joint_upper[i], q[i]));
    }
}

} // namespace ppln::collision
