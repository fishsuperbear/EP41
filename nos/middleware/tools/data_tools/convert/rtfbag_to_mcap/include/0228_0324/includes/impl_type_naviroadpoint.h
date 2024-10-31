/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_NAVIROADPOINT_H
#define IMPL_TYPE_NAVIROADPOINT_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_double.h"
#include "impl_type_point.h"

struct NaviRoadPoint {
    ::String id;
    ::Double s;
    ::Point pose;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(s);
        fun(pose);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(s);
        fun(pose);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("s", s);
        fun("pose", pose);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("s", s);
        fun("pose", pose);
    }

    bool operator==(const ::NaviRoadPoint& t) const
    {
        return (id == t.id) && (fabs(static_cast<double>(s - t.s)) < DBL_EPSILON) && (pose == t.pose);
    }
};


#endif // IMPL_TYPE_NAVIROADPOINT_H
