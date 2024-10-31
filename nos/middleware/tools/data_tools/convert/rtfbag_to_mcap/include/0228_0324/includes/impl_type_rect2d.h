/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_RECT2D_H
#define IMPL_TYPE_RECT2D_H
#include <cfloat>
#include <cmath>
#include "impl_type_pointxydouble.h"

struct Rect2d {
    ::PointxyDouble center;
    ::PointxyDouble centerCovariance;
    ::PointxyDouble size;
    ::PointxyDouble sizeCovariance;
    ::PointxyDouble orientation;
    ::PointxyDouble orientationCovariance;
    ::PointxyDouble corners;
    ::PointxyDouble leftTop;

    static bool IsPlane()
    {
        return true;
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
        fun(leftTop);
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
        fun(leftTop);
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
        fun("leftTop", leftTop);
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
        fun("leftTop", leftTop);
    }

    bool operator==(const ::Rect2d& t) const
    {
        return (center == t.center) && (centerCovariance == t.centerCovariance) && (size == t.size) && (sizeCovariance == t.sizeCovariance) && (orientation == t.orientation) && (orientationCovariance == t.orientationCovariance) && (corners == t.corners) && (leftTop == t.leftTop);
    }
};


#endif // IMPL_TYPE_RECT2D_H
