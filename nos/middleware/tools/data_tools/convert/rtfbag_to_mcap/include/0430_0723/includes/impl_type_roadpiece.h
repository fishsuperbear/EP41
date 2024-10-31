/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_ROADPIECE_H
#define IMPL_TYPE_ROADPIECE_H
#include <cfloat>
#include <cmath>
#include "impl_type_laneseriesvector.h"
#include "impl_type_string.h"
#include "impl_type_uint32.h"

struct RoadPiece {
    ::LaneSeriesVector roadPiece;
    ::String id;
    ::UInt32 roadType;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(roadPiece);
        fun(id);
        fun(roadType);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(roadPiece);
        fun(id);
        fun(roadType);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("roadPiece", roadPiece);
        fun("id", id);
        fun("roadType", roadType);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("roadPiece", roadPiece);
        fun("id", id);
        fun("roadType", roadType);
    }

    bool operator==(const ::RoadPiece& t) const
    {
        return (roadPiece == t.roadPiece) && (id == t.id) && (roadType == t.roadType);
    }
};


#endif // IMPL_TYPE_ROADPIECE_H
