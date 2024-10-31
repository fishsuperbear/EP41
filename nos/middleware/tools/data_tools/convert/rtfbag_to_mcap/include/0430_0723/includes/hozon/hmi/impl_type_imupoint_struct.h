/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_IMUPOINT_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_IMUPOINT_STRUCT_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi {
struct IMUPoint_Struct {
    float IMUPoint_X;
    float IMUPoint_Y;
    float IMUPoint_Z;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(IMUPoint_X);
        fun(IMUPoint_Y);
        fun(IMUPoint_Z);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(IMUPoint_X);
        fun(IMUPoint_Y);
        fun(IMUPoint_Z);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("IMUPoint_X", IMUPoint_X);
        fun("IMUPoint_Y", IMUPoint_Y);
        fun("IMUPoint_Z", IMUPoint_Z);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("IMUPoint_X", IMUPoint_X);
        fun("IMUPoint_Y", IMUPoint_Y);
        fun("IMUPoint_Z", IMUPoint_Z);
    }

    bool operator==(const ::hozon::hmi::IMUPoint_Struct& t) const
    {
        return (fabs(static_cast<double>(IMUPoint_X - t.IMUPoint_X)) < DBL_EPSILON) && (fabs(static_cast<double>(IMUPoint_Y - t.IMUPoint_Y)) < DBL_EPSILON) && (fabs(static_cast<double>(IMUPoint_Z - t.IMUPoint_Z)) < DBL_EPSILON);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_IMUPOINT_STRUCT_H
