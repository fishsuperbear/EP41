/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SELECTPOINT_IMPL_TYPE_SELECTPOINT_H
#define ARA_SELECTPOINT_IMPL_TYPE_SELECTPOINT_H
#include <cfloat>
#include <cmath>
#include "ara/selectpoint/impl_type_header.h"
#include "ara/selectpoint/impl_type_point.h"

namespace ara {
namespace selectpoint {
struct SelectPoint {
    ::ara::selectpoint::Header header;
    ::ara::selectpoint::Point point;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(point);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(point);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("point", point);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("point", point);
    }

    bool operator==(const ::ara::selectpoint::SelectPoint& t) const
    {
        return (header == t.header) && (point == t.point);
    }
};
} // namespace selectpoint
} // namespace ara


#endif // ARA_SELECTPOINT_IMPL_TYPE_SELECTPOINT_H
