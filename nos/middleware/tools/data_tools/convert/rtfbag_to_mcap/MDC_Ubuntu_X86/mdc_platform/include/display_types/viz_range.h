/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Range
 */

#ifndef VIZ_RANGE_H
#define VIZ_RANGE_H

#include <cstdint>
#include "viz_header.h"

namespace mdc {
namespace visual {
enum class RadiationType : uint8_t {
    ULTRASOUND = 0U,
    INFRARED
};

struct Range {
    Header header;
    RadiationType radiationType;
    float fieldOfView;
    float minRange;
    float maxRange;
    float range;
    Range()
        : header(),
          radiationType(RadiationType::ULTRASOUND),
          fieldOfView(0.0F),
          minRange(0.0F),
          maxRange(0.0F),
          range(0.0F) {}
    Range(const Header& vHeader, const RadiationType& vType, const float& vView, const float& vMin,
    const float& vMmax, const float& vRange)
        : header(vHeader),
          radiationType(vType),
          fieldOfView(vView),
          minRange(vMin),
          maxRange(vMmax),
          range(vRange) {}
};
}
}

#endif // VIZ_RANGE_H
