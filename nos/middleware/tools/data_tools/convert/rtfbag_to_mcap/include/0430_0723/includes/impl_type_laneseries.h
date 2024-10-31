/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_LANESERIES_H
#define IMPL_TYPE_LANESERIES_H
#include <cfloat>
#include <cmath>
#include "impl_type_lanepiecevector.h"
#include "impl_type_boolean.h"
#include "impl_type_uint32.h"

struct LaneSeries {
    ::LanePieceVector laneSeries;
    ::Boolean isDrivable;
    ::UInt32 directionType;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(laneSeries);
        fun(isDrivable);
        fun(directionType);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(laneSeries);
        fun(isDrivable);
        fun(directionType);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("laneSeries", laneSeries);
        fun("isDrivable", isDrivable);
        fun("directionType", directionType);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("laneSeries", laneSeries);
        fun("isDrivable", isDrivable);
        fun("directionType", directionType);
    }

    bool operator==(const ::LaneSeries& t) const
    {
        return (laneSeries == t.laneSeries) && (isDrivable == t.isDrivable) && (directionType == t.directionType);
    }
};


#endif // IMPL_TYPE_LANESERIES_H
