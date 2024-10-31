/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_OBSIZE_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_OBSIZE_STRUCT_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi {
struct ObSize_Struct {
    float ObSize_Length;
    float ObSize_Width;
    float ObSize_Height;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ObSize_Length);
        fun(ObSize_Width);
        fun(ObSize_Height);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ObSize_Length);
        fun(ObSize_Width);
        fun(ObSize_Height);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ObSize_Length", ObSize_Length);
        fun("ObSize_Width", ObSize_Width);
        fun("ObSize_Height", ObSize_Height);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ObSize_Length", ObSize_Length);
        fun("ObSize_Width", ObSize_Width);
        fun("ObSize_Height", ObSize_Height);
    }

    bool operator==(const ::hozon::hmi::ObSize_Struct& t) const
    {
        return (fabs(static_cast<double>(ObSize_Length - t.ObSize_Length)) < DBL_EPSILON) && (fabs(static_cast<double>(ObSize_Width - t.ObSize_Width)) < DBL_EPSILON) && (fabs(static_cast<double>(ObSize_Height - t.ObSize_Height)) < DBL_EPSILON);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_OBSIZE_STRUCT_H
