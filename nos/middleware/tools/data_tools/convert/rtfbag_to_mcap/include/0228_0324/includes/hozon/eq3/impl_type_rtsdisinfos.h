/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_RTSDISINFOS_H
#define HOZON_EQ3_IMPL_TYPE_RTSDISINFOS_H
#include <cfloat>
#include <cmath>
#include "hozon/eq3/impl_type_rtsdisdataarray.h"

namespace hozon {
namespace eq3 {
struct RTSDisInfos {
    ::hozon::eq3::RtsDisDataArray RTSDisDataArray;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(RTSDisDataArray);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(RTSDisDataArray);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("RTSDisDataArray", RTSDisDataArray);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("RTSDisDataArray", RTSDisDataArray);
    }

    bool operator==(const ::hozon::eq3::RTSDisInfos& t) const
    {
        return (RTSDisDataArray == t.RTSDisDataArray);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_RTSDISINFOS_H
