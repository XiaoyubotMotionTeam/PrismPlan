# PrismPlan native backend

This directory contains the CUDA/C++ implementation shared by PrismPlan's six
motion planners and the `pybind11` module exposed to Python as `prrtc`.

## Build

From `code/gpu_planning`:

```bash
cmake -S cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cpp/build --parallel
```

Requirements are CMake 3.16–3.22, a C++17 compiler, the CUDA toolkit, Eigen3,
and pybind11. Set `CMAKE_CUDA_ARCHITECTURES` for the target GPU. The build fails
with a clear dependency error when Eigen3, CUDA, or pybind11 is unavailable.

## Layout

- `src/collision/`: shared sphere/OBB collision primitives.
- `src/planning/`: sampling, search, optimization, buffers, and runtime FK.
- `src/python/bindings.cpp`: the common Python extension interface.
- `src/robots/`: retained robot collision and kinematics tables.

Robot YAML assets under `../config/robots/` are generated from the retained
native tables by `../tools/gen_robot_asset.py`.

## Provenance

PrismPlan builds on the Apache-2.0-licensed pRRTC implementation from CoMMALab.
Selected collision data structures are adapted from VAMP; the affected headers
carry direct source references. See `../LICENSE` and `../NOTICE`.
