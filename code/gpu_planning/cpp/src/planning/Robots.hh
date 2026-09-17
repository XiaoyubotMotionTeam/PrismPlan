#pragma once

#include <array>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace ppln::robots {
    struct Panda
    {
        static constexpr auto name = "panda";
        static constexpr auto dimension = 7;
        using Configuration = std::array<float, dimension>;

        // necessary to generate the scale_cfg function at compile time
        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                5.9342f,
                3.6652f,
                5.9342f,
                3.2289f,
                5.9342f,
                3.9095999999999997f,
                5.9342f
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -2.9671f,
                -1.8326f,
                -2.9671f,
                -3.1416f,
                -2.9671f,
                -0.0873f,
                -2.9671f
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };
        
        // template metaprogramming to generate the scale_cfg function
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    struct Sphere
    {
        static constexpr auto name = "sphere";
        static constexpr auto dimension = 3;
        static constexpr float radius = 1.0;
        using Configuration = std::array<float, dimension>;

        // necessary to generate the scale_cfg function at compile time
        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                10.0f,
                10.0f,
                10.0f
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -5.0f,
                -5.0f,
                -5.0f
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };

        // template metaprogramming to generate the scale_cfg function
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };
}