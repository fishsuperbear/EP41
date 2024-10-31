/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_OBJECT_IMPL_TYPE_RECT3D_H
#define HOZON_OBJECT_IMPL_TYPE_RECT3D_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_point.h"
#include "impl_type_double.h"
#include "hozon/object/impl_type_cornervector.h"

namespace hozon {
namespace object {
struct Rect3D {
    ::hozon::composite::Point Center;
    ::hozon::composite::Point CenterStdDev;
    ::hozon::composite::Point SizeLWH;
    ::hozon::composite::Point SizeStdDev;
    ::Double Orientation;
    ::Double OrientationStdDev;
    ::hozon::object::cornerVector Cornes;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Center);
        fun(CenterStdDev);
        fun(SizeLWH);
        fun(SizeStdDev);
        fun(Orientation);
        fun(OrientationStdDev);
        fun(Cornes);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Center);
        fun(CenterStdDev);
        fun(SizeLWH);
        fun(SizeStdDev);
        fun(Orientation);
        fun(OrientationStdDev);
        fun(Cornes);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Center", Center);
        fun("CenterStdDev", CenterStdDev);
        fun("SizeLWH", SizeLWH);
        fun("SizeStdDev", SizeStdDev);
        fun("Orientation", Orientation);
        fun("OrientationStdDev", OrientationStdDev);
        fun("Cornes", Cornes);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Center", Center);
        fun("CenterStdDev", CenterStdDev);
        fun("SizeLWH", SizeLWH);
        fun("SizeStdDev", SizeStdDev);
        fun("Orientation", Orientation);
        fun("OrientationStdDev", OrientationStdDev);
        fun("Cornes", Cornes);
    }

    bool operator==(const ::hozon::object::Rect3D& t) const
    {
        return (Center == t.Center) && (CenterStdDev == t.CenterStdDev) && (SizeLWH == t.SizeLWH) && (SizeStdDev == t.SizeStdDev) && (fabs(static_cast<double>(Orientation - t.Orientation)) < DBL_EPSILON) && (fabs(static_cast<double>(OrientationStdDev - t.OrientationStdDev)) < DBL_EPSILON) && (Cornes == t.Cornes);
    }
};
} // namespace object
} // namespace hozon


#endif // HOZON_OBJECT_IMPL_TYPE_RECT3D_H
