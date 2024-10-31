/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_LOCATIONINFO_H
#define ADSFI_IMPL_TYPE_LOCATIONINFO_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_uint16.h"
#include "impl_type_posewithcovariance.h"
#include "impl_type_twistwithcovariance.h"
#include "impl_type_accelwithcovariance.h"

namespace adsfi {
struct LocationInfo {
    ::ara::common::CommonHeader header;
    ::UInt16 locationState;
    ::PoseWithCovariance pose;
    ::TwistWithCovariance velocity;
    ::AccelWithCovariance acceleration;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(locationState);
        fun(pose);
        fun(velocity);
        fun(acceleration);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locationState);
        fun(pose);
        fun(velocity);
        fun(acceleration);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locationState", locationState);
        fun("pose", pose);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locationState", locationState);
        fun("pose", pose);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
    }

    bool operator==(const ::adsfi::LocationInfo& t) const
    {
        return (header == t.header) && (locationState == t.locationState) && (pose == t.pose) && (velocity == t.velocity) && (acceleration == t.acceleration);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_LOCATIONINFO_H
