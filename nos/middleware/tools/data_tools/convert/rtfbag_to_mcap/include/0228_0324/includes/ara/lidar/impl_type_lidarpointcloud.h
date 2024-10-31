/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_LIDAR_IMPL_TYPE_LIDARPOINTCLOUD_H
#define ARA_LIDAR_IMPL_TYPE_LIDARPOINTCLOUD_H
#include <cfloat>
#include <cmath>
#include "ara/lidar/impl_type_header.h"
#include "impl_type_uint32.h"
#include "ara/lidar/impl_type_lidarpointfield.h"

namespace ara {
namespace lidar {
struct LidarPointCloud {
    ::ara::lidar::Header header;
    ::UInt32 isBigEndian;
    ::UInt32 height;
    ::UInt32 width;
    ::UInt32 pointStep;
    ::UInt32 rowStep;
    ::UInt32 isDense;
    ::ara::lidar::LidarPointField data;

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
    }

    bool operator==(const ::ara::lidar::LidarPointCloud& t) const
    {
        return (header == t.header) && (isBigEndian == t.isBigEndian) && (height == t.height) && (width == t.width) && (pointStep == t.pointStep) && (rowStep == t.rowStep) && (isDense == t.isDense) && (data == t.data);
    }
};
} // namespace lidar
} // namespace ara


#endif // ARA_LIDAR_IMPL_TYPE_LIDARPOINTCLOUD_H
