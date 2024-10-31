/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LANELINE_IMPL_TYPE_LANELINEFIT_H
#define HOZON_LANELINE_IMPL_TYPE_LANELINEFIT_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "hozon/laneline/impl_type_laneparam.h"

namespace hozon {
namespace laneline {
struct LaneLineFit {
    ::Float xStartVRF;
    ::Float xEndVRF;
    ::hozon::laneline::LaneParam coefficients;

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

    bool operator==(const ::hozon::laneline::LaneLineFit& t) const
    {
        return (fabs(static_cast<double>(xStartVRF - t.xStartVRF)) < DBL_EPSILON) && (fabs(static_cast<double>(xEndVRF - t.xEndVRF)) < DBL_EPSILON) && (coefficients == t.coefficients);
    }
};
} // namespace laneline
} // namespace hozon


#endif // HOZON_LANELINE_IMPL_TYPE_LANELINEFIT_H
