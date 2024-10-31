/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_POSEWITHCOVARIANCE_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_POSEWITHCOVARIANCE_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_pose_soc_mcu.h"
#include "impl_type_uint16.h"

namespace hozon {
namespace soc_mcu {
struct PoseWithCovariance_soc_mcu {
    ::hozon::soc_mcu::Pose_soc_mcu poseWGS;
    ::hozon::soc_mcu::Pose_soc_mcu poseLOCAL;
    ::hozon::soc_mcu::Pose_soc_mcu poseGCJ02;
    ::hozon::soc_mcu::Pose_soc_mcu poseUTM01;
    ::hozon::soc_mcu::Pose_soc_mcu poseUTM02;
    ::UInt16 utmZoneID01;
    ::UInt16 utmZoneID02;

    static bool IsPlane()
    {
        return true;
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
    }

    bool operator==(const ::hozon::soc_mcu::PoseWithCovariance_soc_mcu& t) const
    {
        return (poseWGS == t.poseWGS) && (poseLOCAL == t.poseLOCAL) && (poseGCJ02 == t.poseGCJ02) && (poseUTM01 == t.poseUTM01) && (poseUTM02 == t.poseUTM02) && (utmZoneID01 == t.utmZoneID01) && (utmZoneID02 == t.utmZoneID02);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_POSEWITHCOVARIANCE_SOC_MCU_H
