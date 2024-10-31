/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_RADAR_IMPL_TYPE_GUARDNAILINFO_H
#define ARA_RADAR_IMPL_TYPE_GUARDNAILINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

namespace ara {
namespace radar {
struct GuardnailInfo {
    ::Float confidence;
    ::Float dxStart;
    ::Float dxEnd;
    ::Float guardnailC0;
    ::Float guardnailC1;
    ::Float guardnailC2;
    ::Float guardnailC3;
    ::Float reserved1;
    ::Float reserved2;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(confidence);
        fun(dxStart);
        fun(dxEnd);
        fun(guardnailC0);
        fun(guardnailC1);
        fun(guardnailC2);
        fun(guardnailC3);
        fun(reserved1);
        fun(reserved2);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(confidence);
        fun(dxStart);
        fun(dxEnd);
        fun(guardnailC0);
        fun(guardnailC1);
        fun(guardnailC2);
        fun(guardnailC3);
        fun(reserved1);
        fun(reserved2);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("confidence", confidence);
        fun("dxStart", dxStart);
        fun("dxEnd", dxEnd);
        fun("guardnailC0", guardnailC0);
        fun("guardnailC1", guardnailC1);
        fun("guardnailC2", guardnailC2);
        fun("guardnailC3", guardnailC3);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("confidence", confidence);
        fun("dxStart", dxStart);
        fun("dxEnd", dxEnd);
        fun("guardnailC0", guardnailC0);
        fun("guardnailC1", guardnailC1);
        fun("guardnailC2", guardnailC2);
        fun("guardnailC3", guardnailC3);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
    }

    bool operator==(const ::ara::radar::GuardnailInfo& t) const
    {
        return (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (fabs(static_cast<double>(dxStart - t.dxStart)) < DBL_EPSILON) && (fabs(static_cast<double>(dxEnd - t.dxEnd)) < DBL_EPSILON) && (fabs(static_cast<double>(guardnailC0 - t.guardnailC0)) < DBL_EPSILON) && (fabs(static_cast<double>(guardnailC1 - t.guardnailC1)) < DBL_EPSILON) && (fabs(static_cast<double>(guardnailC2 - t.guardnailC2)) < DBL_EPSILON) && (fabs(static_cast<double>(guardnailC3 - t.guardnailC3)) < DBL_EPSILON) && (fabs(static_cast<double>(reserved1 - t.reserved1)) < DBL_EPSILON) && (fabs(static_cast<double>(reserved2 - t.reserved2)) < DBL_EPSILON);
    }
};
} // namespace radar
} // namespace ara


#endif // ARA_RADAR_IMPL_TYPE_GUARDNAILINFO_H
