/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_IMUINSINFOFRAME_H
#define HOZON_SENSORS_IMPL_TYPE_IMUINSINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "impl_type_double.h"
#include "hozon/sensors/impl_type_imuinfoframe.h"
#include "hozon/sensors/impl_type_insinfoframe.h"
#include "hozon/sensors/impl_type_offsetinfoframe.h"

namespace hozon {
namespace sensors {
struct ImuInsInfoFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 gpsWeek;
    ::Double gpsSec;
    ::hozon::sensors::ImuInfoFrame imu_info;
    ::hozon::sensors::InsInfoFrame ins_info;
    ::hozon::sensors::offsetInfoFrame offset_info;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(imu_info);
        fun(ins_info);
        fun(offset_info);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(imu_info);
        fun(ins_info);
        fun(offset_info);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("imu_info", imu_info);
        fun("ins_info", ins_info);
        fun("offset_info", offset_info);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("imu_info", imu_info);
        fun("ins_info", ins_info);
        fun("offset_info", offset_info);
    }

    bool operator==(const ::hozon::sensors::ImuInsInfoFrame& t) const
    {
        return (header == t.header) && (gpsWeek == t.gpsWeek) && (fabs(static_cast<double>(gpsSec - t.gpsSec)) < DBL_EPSILON) && (imu_info == t.imu_info) && (ins_info == t.ins_info) && (offset_info == t.offset_info);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_IMUINSINFOFRAME_H
