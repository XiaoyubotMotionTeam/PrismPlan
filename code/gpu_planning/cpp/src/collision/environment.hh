#pragma once

#include <vector>
#include <optional>
#include "shapes.hh"

/* Adapted from https://github.com/KavrakiLab/vamp/blob/main/src/impl/vamp/collision/environment.hh */

namespace ppln::collision
{
    template <typename DataT>
    struct Environment
    {
        Sphere<DataT> *spheres;
        unsigned int num_spheres;

        Capsule<DataT> *capsules;
        unsigned int num_capsules;

        Capsule<DataT> *z_aligned_capsules;
        unsigned int num_z_aligned_capsules;

        Cylinder<DataT> *cylinders;
        unsigned int num_cylinders;

        Cuboid<DataT> *cuboids;
        unsigned int num_cuboids;

        Cuboid<DataT> *z_aligned_cuboids;
        unsigned int num_z_aligned_cuboids;

        // HeightField<DataT> *heightfields;
        // unsigned int num_heightfields;

        Environment() = default;

        // Prevent copy (raw owning pointers — copy would double-free)
        Environment(const Environment&) = delete;
        Environment& operator=(const Environment&) = delete;

        // Allow move
        Environment(Environment&& o) noexcept
            : spheres(o.spheres), num_spheres(o.num_spheres),
              capsules(o.capsules), num_capsules(o.num_capsules),
              z_aligned_capsules(o.z_aligned_capsules), num_z_aligned_capsules(o.num_z_aligned_capsules),
              cylinders(o.cylinders), num_cylinders(o.num_cylinders),
              cuboids(o.cuboids), num_cuboids(o.num_cuboids),
              z_aligned_cuboids(o.z_aligned_cuboids), num_z_aligned_cuboids(o.num_z_aligned_cuboids)
        {
            o.spheres = nullptr; o.num_spheres = 0;
            o.capsules = nullptr; o.num_capsules = 0;
            o.z_aligned_capsules = nullptr; o.num_z_aligned_capsules = 0;
            o.cylinders = nullptr; o.num_cylinders = 0;
            o.cuboids = nullptr; o.num_cuboids = 0;
            o.z_aligned_cuboids = nullptr; o.num_z_aligned_cuboids = 0;
        }
        Environment& operator=(Environment&& o) noexcept {
            if (this != &o) {
                delete[] spheres; delete[] capsules; delete[] cuboids;
                delete[] z_aligned_capsules; delete[] cylinders; delete[] z_aligned_cuboids;
                spheres = o.spheres; num_spheres = o.num_spheres;
                capsules = o.capsules; num_capsules = o.num_capsules;
                z_aligned_capsules = o.z_aligned_capsules; num_z_aligned_capsules = o.num_z_aligned_capsules;
                cylinders = o.cylinders; num_cylinders = o.num_cylinders;
                cuboids = o.cuboids; num_cuboids = o.num_cuboids;
                z_aligned_cuboids = o.z_aligned_cuboids; num_z_aligned_cuboids = o.num_z_aligned_cuboids;
                o.spheres = nullptr; o.num_spheres = 0;
                o.capsules = nullptr; o.num_capsules = 0;
                o.z_aligned_capsules = nullptr; o.num_z_aligned_capsules = 0;
                o.cylinders = nullptr; o.num_cylinders = 0;
                o.cuboids = nullptr; o.num_cuboids = 0;
                o.z_aligned_cuboids = nullptr; o.num_z_aligned_cuboids = 0;
            }
            return *this;
        }

        ~Environment() {
            delete[] spheres;
            delete[] capsules;
            delete[] cuboids;
            delete[] z_aligned_capsules;
            delete[] cylinders;
            delete[] z_aligned_cuboids;
        }
    };
}  // namespace ppln::collision