/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_GSV_IMPL_TYPE_GSVINFO_H
#define MDC_GSV_IMPL_TYPE_GSVINFO_H
#include <cfloat>
#include <cmath>
#include "ara/gnss/impl_type_header.h"
#include "impl_type_int16.h"
#include "mdc/gsv/impl_type_satelliteinfovec.h"

namespace mdc {
namespace gsv {
struct GsvInfo {
    ::ara::gnss::Header header;
    ::Int16 num;
    ::mdc::gsv::SatelliteInfoVec satelliteInfoVec;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(num);
        fun(satelliteInfoVec);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(num);
        fun(satelliteInfoVec);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("num", num);
        fun("satelliteInfoVec", satelliteInfoVec);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("num", num);
        fun("satelliteInfoVec", satelliteInfoVec);
    }

    bool operator==(const ::mdc::gsv::GsvInfo& t) const
    {
        return (header == t.header) && (num == t.num) && (satelliteInfoVec == t.satelliteInfoVec);
    }
};
} // namespace gsv
} // namespace mdc


#endif // MDC_GSV_IMPL_TYPE_GSVINFO_H
