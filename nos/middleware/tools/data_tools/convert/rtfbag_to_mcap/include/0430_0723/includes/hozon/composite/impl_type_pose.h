/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_POSE_H
#define HOZON_COMPOSITE_IMPL_TYPE_POSE_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_point3d_double.h"
#include "hozon/composite/impl_type_quaternion.h"
#include "hozon/composite/impl_type_vector3.h"
#include "impl_type_float.h"

namespace hozon {
namespace composite {
struct Pose {
    ::hozon::composite::Point3D_double position;
    ::hozon::composite::Quaternion quaternion;
    ::hozon::composite::Vector3 eulerAngle;
    ::hozon::composite::Vector3 rotationVRF;
    ::Float heading;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(position);
        fun(quaternion);
        fun(eulerAngle);
        fun(rotationVRF);
        fun(heading);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(position);
        fun(quaternion);
        fun(eulerAngle);
        fun(rotationVRF);
        fun(heading);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("position", position);
        fun("quaternion", quaternion);
        fun("eulerAngle", eulerAngle);
        fun("rotationVRF", rotationVRF);
        fun("heading", heading);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("position", position);
        fun("quaternion", quaternion);
        fun("eulerAngle", eulerAngle);
        fun("rotationVRF", rotationVRF);
        fun("heading", heading);
    }

    bool operator==(const ::hozon::composite::Pose& t) const
    {
        return (position == t.position) && (quaternion == t.quaternion) && (eulerAngle == t.eulerAngle) && (rotationVRF == t.rotationVRF) && (fabs(static_cast<double>(heading - t.heading)) < DBL_EPSILON);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_POSE_H
