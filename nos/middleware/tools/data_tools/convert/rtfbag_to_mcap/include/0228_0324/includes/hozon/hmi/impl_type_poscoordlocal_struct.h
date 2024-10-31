/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_POSCOORDLOCAL_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_POSCOORDLOCAL_STRUCT_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi {
struct PosCoordLocal_Struct {
    float PosCoordLocal_X;
    float PosCoordLocal_Y;
    float PosCoordLocal_Z;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(PosCoordLocal_X);
        fun(PosCoordLocal_Y);
        fun(PosCoordLocal_Z);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(PosCoordLocal_X);
        fun(PosCoordLocal_Y);
        fun(PosCoordLocal_Z);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("PosCoordLocal_X", PosCoordLocal_X);
        fun("PosCoordLocal_Y", PosCoordLocal_Y);
        fun("PosCoordLocal_Z", PosCoordLocal_Z);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("PosCoordLocal_X", PosCoordLocal_X);
        fun("PosCoordLocal_Y", PosCoordLocal_Y);
        fun("PosCoordLocal_Z", PosCoordLocal_Z);
    }

    bool operator==(const ::hozon::hmi::PosCoordLocal_Struct& t) const
    {
        return (fabs(static_cast<double>(PosCoordLocal_X - t.PosCoordLocal_X)) < DBL_EPSILON) && (fabs(static_cast<double>(PosCoordLocal_Y - t.PosCoordLocal_Y)) < DBL_EPSILON) && (fabs(static_cast<double>(PosCoordLocal_Z - t.PosCoordLocal_Z)) < DBL_EPSILON);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_POSCOORDLOCAL_STRUCT_H
