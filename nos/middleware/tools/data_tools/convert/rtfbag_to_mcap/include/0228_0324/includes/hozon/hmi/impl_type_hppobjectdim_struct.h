/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HPPOBJECTDIM_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_HPPOBJECTDIM_STRUCT_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi {
struct HPPObjectDim_Struct {
    float fXm;
    float fYm;
    float angle;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(fXm);
        fun(fYm);
        fun(angle);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(fXm);
        fun(fYm);
        fun(angle);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("fXm", fXm);
        fun("fYm", fYm);
        fun("angle", angle);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("fXm", fXm);
        fun("fYm", fYm);
        fun("angle", angle);
    }

    bool operator==(const ::hozon::hmi::HPPObjectDim_Struct& t) const
    {
        return (fabs(static_cast<double>(fXm - t.fXm)) < DBL_EPSILON) && (fabs(static_cast<double>(fYm - t.fYm)) < DBL_EPSILON) && (fabs(static_cast<double>(angle - t.angle)) < DBL_EPSILON);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HPPOBJECTDIM_STRUCT_H
