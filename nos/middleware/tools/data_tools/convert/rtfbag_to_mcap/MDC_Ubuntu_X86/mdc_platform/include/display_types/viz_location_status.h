/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: LocationStatus
 */

#ifndef VIZ_LOCATION_STATUS_H
#define VIZ_LOCATION_STATUS_H

#include <cstdint>
#include "viz_header.h"

namespace mdc {
namespace visual {
enum LocationType {
    LOCATION_TYPE_UNKNOWN = 0,    // 未知
    LOCATION_TYPE_SINGLE = 1,     // 单点定位
    LOCATION_TYPE_FUSION = 2,     // 融合定位（激光+视觉）
    LOCATION_TYPE_RTK = 3,        // RTK定位
    LOCATION_TYPE_VISION = 4,     // 视觉定位
    LOCATION_TYPE_LIDAR = 5,      // 激光定位
    LOCATION_TYPE_INTEGRATED = 6, // 组合定位（惯导）
};

struct LocationStatus {
    Header header;
    uint8_t searchedSattelites; // 搜星数，0~100
    uint8_t locationType;       // 定位类型
    uint8_t locationState;      // 定位状态,0~255
    LocationStatus() : header(), searchedSattelites(0U), locationType(0U), locationState(0U) {}
    LocationStatus(const Header& vHeader, const uint8_t& vSattelites, const uint8_t& vType, const uint8_t& vState)
        : header(vHeader),
          searchedSattelites(vSattelites),
          locationType(vType),
          locationState(vState) {}
};
} // namespace visual
} // namespace ara

#endif // VIZ_LOCATION_STATUS_H
