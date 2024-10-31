/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_POSE_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_POSE_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_point3d_soc_mcu.h"
#include "hozon/soc_mcu/impl_type_quaternion_soc_mcu.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc_mcu {
struct Pose_soc_mcu {
    ::hozon::soc_mcu::Point3D_soc_mcu position;
    ::hozon::soc_mcu::Quaternion_soc_mcu quaternion;
    ::hozon::soc_mcu::Point3D_soc_mcu eulerAngle;
    ::hozon::soc_mcu::Point3D_soc_mcu rotationVRF;
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

    bool operator==(const ::hozon::soc_mcu::Pose_soc_mcu& t) const
    {
        return (position == t.position) && (quaternion == t.quaternion) && (eulerAngle == t.eulerAngle) && (rotationVRF == t.rotationVRF) && (fabs(static_cast<double>(heading - t.heading)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_POSE_SOC_MCU_H
