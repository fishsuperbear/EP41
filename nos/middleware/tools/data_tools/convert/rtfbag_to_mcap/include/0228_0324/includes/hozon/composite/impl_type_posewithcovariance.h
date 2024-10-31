/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_POSEWITHCOVARIANCE_H
#define HOZON_COMPOSITE_IMPL_TYPE_POSEWITHCOVARIANCE_H
#include <cfloat>
#include <cmath>
#include "hozon/composite/impl_type_pose.h"
#include "impl_type_uint16.h"
#include "hozon/composite/impl_type_floatarray36.h"

namespace hozon {
namespace composite {
struct PoseWithCovariance {
    ::hozon::composite::Pose poseWGS;
    ::hozon::composite::Pose poseLOCAL;
    ::hozon::composite::Pose poseGCJ02;
    ::hozon::composite::Pose poseUTM01;
    ::hozon::composite::Pose poseUTM02;
    ::UInt16 utmZoneID01;
    ::UInt16 utmZoneID02;
    ::hozon::composite::FloatArray36 covariance;
    ::hozon::composite::Pose poseDR;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(poseWGS);
        fun(poseLOCAL);
        fun(poseGCJ02);
        fun(poseUTM01);
        fun(poseUTM02);
        fun(utmZoneID01);
        fun(utmZoneID02);
        fun(covariance);
        fun(poseDR);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(poseWGS);
        fun(poseLOCAL);
        fun(poseGCJ02);
        fun(poseUTM01);
        fun(poseUTM02);
        fun(utmZoneID01);
        fun(utmZoneID02);
        fun(covariance);
        fun(poseDR);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("poseWGS", poseWGS);
        fun("poseLOCAL", poseLOCAL);
        fun("poseGCJ02", poseGCJ02);
        fun("poseUTM01", poseUTM01);
        fun("poseUTM02", poseUTM02);
        fun("utmZoneID01", utmZoneID01);
        fun("utmZoneID02", utmZoneID02);
        fun("covariance", covariance);
        fun("poseDR", poseDR);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("poseWGS", poseWGS);
        fun("poseLOCAL", poseLOCAL);
        fun("poseGCJ02", poseGCJ02);
        fun("poseUTM01", poseUTM01);
        fun("poseUTM02", poseUTM02);
        fun("utmZoneID01", utmZoneID01);
        fun("utmZoneID02", utmZoneID02);
        fun("covariance", covariance);
        fun("poseDR", poseDR);
    }

    bool operator==(const ::hozon::composite::PoseWithCovariance& t) const
    {
        return (poseWGS == t.poseWGS) && (poseLOCAL == t.poseLOCAL) && (poseGCJ02 == t.poseGCJ02) && (poseUTM01 == t.poseUTM01) && (poseUTM02 == t.poseUTM02) && (utmZoneID01 == t.utmZoneID01) && (utmZoneID02 == t.utmZoneID02) && (covariance == t.covariance) && (poseDR == t.poseDR);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_POSEWITHCOVARIANCE_H
