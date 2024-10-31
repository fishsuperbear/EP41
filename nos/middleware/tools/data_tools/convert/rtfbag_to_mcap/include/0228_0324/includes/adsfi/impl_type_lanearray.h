/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_LANEARRAY_H
#define ADSFI_IMPL_TYPE_LANEARRAY_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "adsfi/impl_type_lanevector.h"

namespace adsfi {
struct LaneArray {
    ::ara::common::CommonHeader header;
    ::adsfi::LaneVector lane;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(lane);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(lane);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("lane", lane);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("lane", lane);
    }

    bool operator==(const ::adsfi::LaneArray& t) const
    {
        return (header == t.header) && (lane == t.lane);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_LANEARRAY_H
