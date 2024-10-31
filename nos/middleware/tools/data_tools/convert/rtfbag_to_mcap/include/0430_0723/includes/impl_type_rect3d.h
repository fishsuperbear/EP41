/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_RECT3D_H
#define IMPL_TYPE_RECT3D_H
#include <cfloat>
#include <cmath>
#include "impl_type_point.h"
#include "impl_type_pointarray.h"

struct Rect3d {
    ::Point center;
    ::Point centerCovariance;
    ::Point size;
    ::Point sizeCovariance;
    ::Point orientation;
    ::Point orientationCovariance;
    ::PointArray corners;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(center);
        fun(centerCovariance);
        fun(size);
        fun(sizeCovariance);
        fun(orientation);
        fun(orientationCovariance);
        fun(corners);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(center);
        fun(centerCovariance);
        fun(size);
        fun(sizeCovariance);
        fun(orientation);
        fun(orientationCovariance);
        fun(corners);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("center", center);
        fun("centerCovariance", centerCovariance);
        fun("size", size);
        fun("sizeCovariance", sizeCovariance);
        fun("orientation", orientation);
        fun("orientationCovariance", orientationCovariance);
        fun("corners", corners);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("center", center);
        fun("centerCovariance", centerCovariance);
        fun("size", size);
        fun("sizeCovariance", sizeCovariance);
        fun("orientation", orientation);
        fun("orientationCovariance", orientationCovariance);
        fun("corners", corners);
    }

    bool operator==(const ::Rect3d& t) const
    {
        return (center == t.center) && (centerCovariance == t.centerCovariance) && (size == t.size) && (sizeCovariance == t.sizeCovariance) && (orientation == t.orientation) && (orientationCovariance == t.orientationCovariance) && (corners == t.corners);
    }
};


#endif // IMPL_TYPE_RECT3D_H
