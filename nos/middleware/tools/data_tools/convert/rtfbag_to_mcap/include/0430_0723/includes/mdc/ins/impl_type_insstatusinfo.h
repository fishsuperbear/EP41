/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_INS_IMPL_TYPE_INSSTATUSINFO_H
#define MDC_INS_IMPL_TYPE_INSSTATUSINFO_H
#include <cfloat>
#include <cmath>
#include "ara/gnss/impl_type_header.h"
#include "impl_type_uint32.h"

namespace mdc {
namespace ins {
struct InsStatusInfo {
    ::ara::gnss::Header header;
    ::UInt32 posInitStatus;
    ::UInt32 velInitStatus;
    ::UInt32 headingInitStatus;
    ::UInt32 attitudeInitStatus;
    ::UInt32 insInitStatus;
    ::UInt32 whlspdfusionStatus;
    ::UInt32 faultStatus;
    ::UInt32 reversed1;
    ::UInt32 reversed2;
    ::UInt32 reversed3;
    ::UInt32 reversed4;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(posInitStatus);
        fun(velInitStatus);
        fun(headingInitStatus);
        fun(attitudeInitStatus);
        fun(insInitStatus);
        fun(whlspdfusionStatus);
        fun(faultStatus);
        fun(reversed1);
        fun(reversed2);
        fun(reversed3);
        fun(reversed4);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(posInitStatus);
        fun(velInitStatus);
        fun(headingInitStatus);
        fun(attitudeInitStatus);
        fun(insInitStatus);
        fun(whlspdfusionStatus);
        fun(faultStatus);
        fun(reversed1);
        fun(reversed2);
        fun(reversed3);
        fun(reversed4);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("posInitStatus", posInitStatus);
        fun("velInitStatus", velInitStatus);
        fun("headingInitStatus", headingInitStatus);
        fun("attitudeInitStatus", attitudeInitStatus);
        fun("insInitStatus", insInitStatus);
        fun("whlspdfusionStatus", whlspdfusionStatus);
        fun("faultStatus", faultStatus);
        fun("reversed1", reversed1);
        fun("reversed2", reversed2);
        fun("reversed3", reversed3);
        fun("reversed4", reversed4);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("posInitStatus", posInitStatus);
        fun("velInitStatus", velInitStatus);
        fun("headingInitStatus", headingInitStatus);
        fun("attitudeInitStatus", attitudeInitStatus);
        fun("insInitStatus", insInitStatus);
        fun("whlspdfusionStatus", whlspdfusionStatus);
        fun("faultStatus", faultStatus);
        fun("reversed1", reversed1);
        fun("reversed2", reversed2);
        fun("reversed3", reversed3);
        fun("reversed4", reversed4);
    }

    bool operator==(const ::mdc::ins::InsStatusInfo& t) const
    {
        return (header == t.header) && (posInitStatus == t.posInitStatus) && (velInitStatus == t.velInitStatus) && (headingInitStatus == t.headingInitStatus) && (attitudeInitStatus == t.attitudeInitStatus) && (insInitStatus == t.insInitStatus) && (whlspdfusionStatus == t.whlspdfusionStatus) && (faultStatus == t.faultStatus) && (reversed1 == t.reversed1) && (reversed2 == t.reversed2) && (reversed3 == t.reversed3) && (reversed4 == t.reversed4);
    }
};
} // namespace ins
} // namespace mdc


#endif // MDC_INS_IMPL_TYPE_INSSTATUSINFO_H
