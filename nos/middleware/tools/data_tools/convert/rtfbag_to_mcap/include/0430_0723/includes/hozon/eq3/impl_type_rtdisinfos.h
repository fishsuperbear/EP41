/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_IMPL_TYPE_RTDISINFOS_H
#define HOZON_EQ3_IMPL_TYPE_RTDISINFOS_H
#include <cfloat>
#include <cmath>
#include "hozon/eq3/impl_type_rtdisdataarray.h"

namespace hozon {
namespace eq3 {
struct RTDisInfos {
    ::hozon::eq3::RtDisDataArray RtDisDataArray;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(RtDisDataArray);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(RtDisDataArray);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("RtDisDataArray", RtDisDataArray);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("RtDisDataArray", RtDisDataArray);
    }

    bool operator==(const ::hozon::eq3::RTDisInfos& t) const
    {
        return (RtDisDataArray == t.RtDisDataArray);
    }
};
} // namespace eq3
} // namespace hozon


#endif // HOZON_EQ3_IMPL_TYPE_RTDISINFOS_H
