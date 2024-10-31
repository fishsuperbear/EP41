/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_POINTCLOUDFRAME_H
#define HOZON_SENSORS_IMPL_TYPE_POINTCLOUDFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "hozon/sensors/impl_type_pointfields.h"
#include "hozon/sensors/impl_type_lidarsn.h"
#include "hozon/sensors/impl_type_lidareolcalibstatus.h"

namespace hozon {
namespace sensors {
struct PointCloudFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 isBigEndian;
    ::UInt32 height;
    ::UInt32 width;
    ::UInt32 pointStep;
    ::UInt32 rowStep;
    ::UInt32 isDense;
    ::hozon::sensors::PointFields data;
    ::hozon::sensors::LidarSN lidarSN;
    ::hozon::sensors::LidarEolCalibStatus eolCalibStatus;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(isBigEndian);
        fun(height);
        fun(width);
        fun(pointStep);
        fun(rowStep);
        fun(isDense);
        fun(data);
        fun(lidarSN);
        fun(eolCalibStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(isBigEndian);
        fun(height);
        fun(width);
        fun(pointStep);
        fun(rowStep);
        fun(isDense);
        fun(data);
        fun(lidarSN);
        fun(eolCalibStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("isBigEndian", isBigEndian);
        fun("height", height);
        fun("width", width);
        fun("pointStep", pointStep);
        fun("rowStep", rowStep);
        fun("isDense", isDense);
        fun("data", data);
        fun("lidarSN", lidarSN);
        fun("eolCalibStatus", eolCalibStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("isBigEndian", isBigEndian);
        fun("height", height);
        fun("width", width);
        fun("pointStep", pointStep);
        fun("rowStep", rowStep);
        fun("isDense", isDense);
        fun("data", data);
        fun("lidarSN", lidarSN);
        fun("eolCalibStatus", eolCalibStatus);
    }

    bool operator==(const ::hozon::sensors::PointCloudFrame& t) const
    {
        return (header == t.header) && (isBigEndian == t.isBigEndian) && (height == t.height) && (width == t.width) && (pointStep == t.pointStep) && (rowStep == t.rowStep) && (isDense == t.isDense) && (data == t.data) && (lidarSN == t.lidarSN) && (eolCalibStatus == t.eolCalibStatus);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_POINTCLOUDFRAME_H
