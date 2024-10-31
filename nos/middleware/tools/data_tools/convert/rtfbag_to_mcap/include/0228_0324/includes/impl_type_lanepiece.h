/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_LANEPIECE_H
#define IMPL_TYPE_LANEPIECE_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_double.h"

struct LanePiece {
    ::String id;
    ::Double startS;
    ::Double endS;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(startS);
        fun(endS);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(startS);
        fun(endS);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("startS", startS);
        fun("endS", endS);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("startS", startS);
        fun("endS", endS);
    }

    bool operator==(const ::LanePiece& t) const
    {
        return (id == t.id) && (fabs(static_cast<double>(startS - t.startS)) < DBL_EPSILON) && (fabs(static_cast<double>(endS - t.endS)) < DBL_EPSILON);
    }
};


#endif // IMPL_TYPE_LANEPIECE_H
