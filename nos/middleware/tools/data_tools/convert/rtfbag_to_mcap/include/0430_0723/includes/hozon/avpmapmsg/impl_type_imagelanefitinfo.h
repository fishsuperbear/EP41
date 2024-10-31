/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_AVPMAPMSG_IMPL_TYPE_IMAGELANEFITINFO_H
#define HOZON_AVPMAPMSG_IMPL_TYPE_IMAGELANEFITINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "adsfi/impl_type_lanefitparam.h"

namespace hozon {
namespace avpmapmsg {
struct imageLaneFitInfo {
    ::Float xStartVRF;
    ::Float xEndVRF;
    ::adsfi::LaneFitParam coefficients;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(xStartVRF);
        fun(xEndVRF);
        fun(coefficients);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(xStartVRF);
        fun(xEndVRF);
        fun(coefficients);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("xStartVRF", xStartVRF);
        fun("xEndVRF", xEndVRF);
        fun("coefficients", coefficients);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("xStartVRF", xStartVRF);
        fun("xEndVRF", xEndVRF);
        fun("coefficients", coefficients);
    }

    bool operator==(const ::hozon::avpmapmsg::imageLaneFitInfo& t) const
    {
        return (fabs(static_cast<double>(xStartVRF - t.xStartVRF)) < DBL_EPSILON) && (fabs(static_cast<double>(xEndVRF - t.xEndVRF)) < DBL_EPSILON) && (coefficients == t.coefficients);
    }
};
} // namespace avpmapmsg
} // namespace hozon


#endif // HOZON_AVPMAPMSG_IMPL_TYPE_IMAGELANEFITINFO_H
