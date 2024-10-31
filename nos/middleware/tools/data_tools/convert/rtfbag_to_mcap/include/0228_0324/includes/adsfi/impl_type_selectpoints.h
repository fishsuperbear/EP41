/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_SELECTPOINTS_H
#define ADSFI_IMPL_TYPE_SELECTPOINTS_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_pointarray.h"

namespace adsfi {
struct SelectPoints {
    ::ara::common::CommonHeader header;
    ::PointArray points;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(points);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(points);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("points", points);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("points", points);
    }

    bool operator==(const ::adsfi::SelectPoints& t) const
    {
        return (header == t.header) && (points == t.points);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_SELECTPOINTS_H
