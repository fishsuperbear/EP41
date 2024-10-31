/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HPPDIM5F_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_HPPDIM5F_STRUCT_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi {
struct HPPDim5F_Struct {
    float fXm;
    float fYm;
    float fZm;
    float fYawr;

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
        fun(fZm);
        fun(fYawr);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(fXm);
        fun(fYm);
        fun(fZm);
        fun(fYawr);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("fXm", fXm);
        fun("fYm", fYm);
        fun("fZm", fZm);
        fun("fYawr", fYawr);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("fXm", fXm);
        fun("fYm", fYm);
        fun("fZm", fZm);
        fun("fYawr", fYawr);
    }

    bool operator==(const ::hozon::hmi::HPPDim5F_Struct& t) const
    {
        return (fabs(static_cast<double>(fXm - t.fXm)) < DBL_EPSILON) && (fabs(static_cast<double>(fYm - t.fYm)) < DBL_EPSILON) && (fabs(static_cast<double>(fZm - t.fZm)) < DBL_EPSILON) && (fabs(static_cast<double>(fYawr - t.fYawr)) < DBL_EPSILON);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HPPDIM5F_STRUCT_H
