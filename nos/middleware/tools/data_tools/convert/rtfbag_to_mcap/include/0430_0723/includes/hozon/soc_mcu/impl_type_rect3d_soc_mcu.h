/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_RECT3D_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_RECT3D_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_point3f_soc_mcu.h"

namespace hozon {
namespace soc_mcu {
struct Rect3D_soc_mcu {
    ::hozon::soc_mcu::Point3f_soc_mcu Center;
    ::hozon::soc_mcu::Point3f_soc_mcu CenterStdDev;
    ::hozon::soc_mcu::Point3f_soc_mcu SizeLWH;
    ::hozon::soc_mcu::Point3f_soc_mcu SizeStdDev;
    float Orientation;
    float OrientationStdDev;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Center);
        fun(CenterStdDev);
        fun(SizeLWH);
        fun(SizeStdDev);
        fun(Orientation);
        fun(OrientationStdDev);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Center);
        fun(CenterStdDev);
        fun(SizeLWH);
        fun(SizeStdDev);
        fun(Orientation);
        fun(OrientationStdDev);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Center", Center);
        fun("CenterStdDev", CenterStdDev);
        fun("SizeLWH", SizeLWH);
        fun("SizeStdDev", SizeStdDev);
        fun("Orientation", Orientation);
        fun("OrientationStdDev", OrientationStdDev);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Center", Center);
        fun("CenterStdDev", CenterStdDev);
        fun("SizeLWH", SizeLWH);
        fun("SizeStdDev", SizeStdDev);
        fun("Orientation", Orientation);
        fun("OrientationStdDev", OrientationStdDev);
    }

    bool operator==(const ::hozon::soc_mcu::Rect3D_soc_mcu& t) const
    {
        return (Center == t.Center) && (CenterStdDev == t.CenterStdDev) && (SizeLWH == t.SizeLWH) && (SizeStdDev == t.SizeStdDev) && (fabs(static_cast<double>(Orientation - t.Orientation)) < DBL_EPSILON) && (fabs(static_cast<double>(OrientationStdDev - t.OrientationStdDev)) < DBL_EPSILON);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_RECT3D_SOC_MCU_H
