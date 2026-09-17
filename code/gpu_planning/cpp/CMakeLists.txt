cmake_minimum_required(VERSION 3.16...3.22)

project(
    gpu_planning
    VERSION 0.1
    LANGUAGES CXX CUDA
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)


# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo")  # enable line info in ncu ui


### pybind11 Python module
find_package(pybind11 REQUIRED)
if(pybind11_FOUND)
    # Build each .cu as a separate static library to avoid nvlink
    # multiple-definition errors from __device__ __constant__ variables
    # in shared headers (panda.cuh, utils.cuh).
    # CUDA_RESOLVE_DEVICE_SYMBOLS ON ensures nvlink runs per-library
    # so device symbols are resolved before the host link step.
    add_library(prrtc_cuda STATIC src/planning/pRRTC.cu src/planning/solver_buffers.cu)
    set_target_properties(prrtc_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(prrtc_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(prrtc_cuda PRIVATE Eigen3::Eigen)
    target_compile_options(prrtc_cuda PRIVATE -Wno-narrowing)

    add_library(mitstar_cuda STATIC src/planning/MITStar.cu src/planning/mitstar_buffers.cu)
    set_target_properties(mitstar_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(mitstar_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(mitstar_cuda PRIVATE Eigen3::Eigen)
    target_compile_options(mitstar_cuda PRIVATE -Wno-narrowing)

    add_library(stomp_cuda STATIC src/planning/stomp.cu src/planning/stomp_buffers.cu src/planning/stomp_precompute.cpp)
    set_target_properties(stomp_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(stomp_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(stomp_cuda PRIVATE Eigen3::Eigen)
    target_compile_options(stomp_cuda PRIVATE -Wno-narrowing)

    add_library(chomp_cuda STATIC src/planning/chomp.cu src/planning/chomp_buffers.cu src/planning/chomp_precompute.cpp)
    set_target_properties(chomp_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(chomp_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(chomp_cuda PRIVATE Eigen3::Eigen)
    target_compile_options(chomp_cuda PRIVATE -Wno-narrowing)

    add_library(mhastar_cuda STATIC src/planning/mhastar_solve.cu)
    set_target_properties(mhastar_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(mhastar_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(mhastar_cuda PRIVATE Eigen3::Eigen)
    target_compile_options(mhastar_cuda PRIVATE -Wno-narrowing)

    add_library(wpase_cuda STATIC src/planning/wpase_solve.cu)
    set_target_properties(wpase_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(wpase_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(wpase_cuda PRIVATE Eigen3::Eigen)
    target_compile_options(wpase_cuda PRIVATE -Wno-narrowing)

    pybind11_add_module(prrtc_python src/python/bindings.cpp)

    set_target_properties(prrtc_python PROPERTIES
        OUTPUT_NAME "prrtc"
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION OFF
    )

    target_include_directories(prrtc_python PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(prrtc_python PRIVATE
        prrtc_cuda
        mitstar_cuda
        stomp_cuda
        chomp_cuda
        mhastar_cuda
        wpase_cuda
        Eigen3::Eigen
        CUDA::cudart_static
    )

    # __device__ template specializations in panda.cuh emit identical host stubs
    # from multiple .cu files — safe to allow multiple definitions at link time.
    target_link_options(prrtc_python PRIVATE "LINKER:--allow-multiple-definition")

    target_compile_options(prrtc_python PRIVATE
        -Wno-narrowing
    )
endif()

### ament_cmake integration (for colcon/Bazel builds)
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND AND pybind11_FOUND)
    find_package(ament_cmake_python REQUIRED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(PYTHON_INSTALL_DIR "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/dist-packages")

    install(TARGETS prrtc_python
        LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}"
    )

    # Register PYTHONPATH hook so that setup.bash adds dist-packages to PYTHONPATH
    _ament_cmake_python_register_environment_hook()

    ament_package()
endif()
