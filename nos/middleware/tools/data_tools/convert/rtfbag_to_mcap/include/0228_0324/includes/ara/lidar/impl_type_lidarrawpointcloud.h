/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_LIDAR_IMPL_TYPE_LIDARRAWPOINTCLOUD_H
#define ARA_LIDAR_IMPL_TYPE_LIDARRAWPOINTCLOUD_H
#include <cfloat>
#include <cmath>
#include "ara/lidar/impl_type_header.h"
#include "ara/lidar/impl_type_vectoruint8.h"

namespace ara {
namespace lidar {
struct LidarRawPointCloud {
    ::ara::lidar::Header header;
    ::ara::lidar::VectorUint8 data;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(data);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(data);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("data", data);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("data", data);
    }

    bool operator==(const ::ara::lidar::LidarRawPointCloud& t) const
    {
        return (header == t.header) && (data == t.data);
    }
};
} // namespace lidar
} // namespace ara


#endif // ARA_LIDAR_IMPL_TYPE_LIDARRAWPOINTCLOUD_H
