/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：OccupancyGrid
 */

#ifndef VIZ_MAP_H
#define VIZ_MAP_H

#include <cstdint>
#include <sys/time.h>
#include "ara/core/vector.h"
#include "viz_header.h"
#include "viz_pose.h"
#include "viz_times.h"

namespace mdc {
namespace visual {
struct MapMetaData {
    Times mapLoadTime;
    float resolution;
    uint32_t width;
    uint32_t height;
    Pose origin;
    MapMetaData() : mapLoadTime(Times::now()), resolution(0.0F), width(0U), height(0U), origin() {}
    MapMetaData(const Times& vTimes, const float& vResolution, const uint32_t& vWidth,
    const uint32_t& vHeight, const Pose& vOrigin)
        : mapLoadTime(vTimes), resolution(vResolution), width(vWidth), height(vHeight), origin(vOrigin) {}
};

struct OccupancyGrid {
    Header header;
    MapMetaData info;
    ara::core::Vector<int8_t> data;
    OccupancyGrid() : header(), info(), data() {}
    OccupancyGrid(const Header& vHeader, const MapMetaData& vInfo, const ara::core::Vector<int8_t>& vData)
        : header(vHeader), info(vInfo), data(vData) {}
};
}
}

#endif // VIZ_MAP_H
