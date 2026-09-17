#pragma once

// ============================================================================
// The shared feasibility anchor.
//
// Every paradigm -- sampling (pRRTC, MIT*), search (MHA*/wPA*SE edge eval), and
// optimisation (STOMP/CHOMP validity) -- gets its boolean collision verdict from
// this one routine, so a configuration that is feasible for one planner is
// feasible for all of them by construction rather than by three implementations
// happening to agree. That is not decoration: before this was unified, pRRTC and
// the optimizers each carried their own composition of the same primitives, and
// "three implementations that agree" is a weaker property than "one
// implementation" -- weak in exactly the place the cross-paradigm comparison
// rests on.
//
// Two paths deliberately do NOT come through here, and no benchmarked number
// touches either: the mesh-BVH branch (this routine has no mesh counterpart, and
// the MotionBenchMaker evaluation is sphere-vs-OBB throughout), and the legacy
// Environment<float> entry point, which predates SceneCollisionData and would
// need an overload. If you add a mesh path or revive that entry point, route it
// here rather than growing a fourth composition.
//
// Two phases: an approximate FK + CC pass gates an exact confirmation. The
// approximate spheres enclose the true link geometry, so an approximate hit
// only means "maybe" and must be confirmed at full resolution before the
// configuration is rejected; an approximate miss is conclusive.
//
// Thread layout: 4 lanes cooperate on one configuration slot, so a block of
// 4*n_slots threads checks n_slots configurations at once (n_slots = 1 for a
// single node, BATCH_SIZE for a batch of edge waypoints). All slots are OR-ed
// into one block-wide verdict, which is what every caller wants: the edge
// kernels ask "is this whole edge free", and the node kernel runs one slot.
// ============================================================================

#include "src/planning/robot_model.cuh"
#include "src/planning/runtime_kinematics.cuh"
#include "src/collision/scene_collision.cuh"

namespace ppln::collision {

// Block-wide scratch. One instance per block, in shared memory.
struct TwoPhaseFlags {
    int collision;    // any slot in collision
    int scene_hit;    // the collision came from the scene (vs. self) check
    int need_exact;   // approximate pass flagged something to confirm
};

struct TwoPhaseResult {
    bool collision;
    bool scene_hit;
    bool did_full_fk;   // sphere_pos holds this slot's exact FK output
};

// Collective over the whole block: every thread must call this, and the
// returned verdict is uniform. `q` is the calling thread's slot configuration,
// `joint_in_collision` needs 20 ints per slot, `flags` is one shared instance.
__device__ inline TwoPhaseResult two_phase_cc(
    const ppln::RobotModel& model,
    const SceneCollisionData& scene,
    const float* q,
    volatile float* sphere_pos,
    volatile float* approx_sphere_pos,
    float* T,
    volatile int* joint_in_collision,
    TwoPhaseFlags* flags,
    const int tid,
    float collision_margin,
    bool check_self)
{
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    const int jic_lo = batch_ind * 20 + 5 * thread_ind;

    if (tid == 0) {
        flags->collision = 0;
        flags->scene_hit = 0;
        flags->need_exact = 0;
    }
    for (int r = jic_lo; r < jic_lo + 5; ++r) joint_in_collision[r] = 0;
    __syncthreads();

    fk_approx_runtime(model, q, approx_sphere_pos, T, tid);
    __syncthreads();

    bool did_full_fk = false;

    // Environment first. The exact scene check is GATED on the per-joint flags
    // the approximate pass writes, so nothing may touch joint_in_collision
    // between the two.
    if (!scene_collision_check_approx_runtime(
            model, approx_sphere_pos, joint_in_collision, scene, tid,
            collision_margin))
        atomicOr(&flags->need_exact, 1);
    __syncthreads();

    if (flags->need_exact) {
        fk_runtime(model, q, sphere_pos, T, tid);
        did_full_fk = true;
        __syncthreads();
        if (!scene_collision_check_runtime(
                model, sphere_pos, joint_in_collision, scene, tid,
                collision_margin)) {
            atomicOr(&flags->collision, 1);
            atomicOr(&flags->scene_hit, 1);
        }
        __syncthreads();
    }

    if (check_self && !flags->collision) {
        if (tid == 0) flags->need_exact = 0;
        for (int r = jic_lo; r < jic_lo + 5; ++r) joint_in_collision[r] = 0;
        __syncthreads();

        if (!self_collision_check_approx_runtime(
                model, approx_sphere_pos, joint_in_collision, tid))
            atomicOr(&flags->need_exact, 1);
        __syncthreads();

        if (flags->need_exact) {
            if (!did_full_fk) {
                fk_runtime(model, q, sphere_pos, T, tid);
                did_full_fk = true;
                __syncthreads();
            }
            if (!self_collision_check_runtime(
                    model, sphere_pos, joint_in_collision, tid))
                atomicOr(&flags->collision, 1);
            __syncthreads();
        }
    }
    __syncthreads();

    TwoPhaseResult out;
    out.collision = flags->collision != 0;
    out.scene_hit = flags->scene_hit != 0;
    out.did_full_fk = did_full_fk;
    return out;
}

}  // namespace ppln::collision
