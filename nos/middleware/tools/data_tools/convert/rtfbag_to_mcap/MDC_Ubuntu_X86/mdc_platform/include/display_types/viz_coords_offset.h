/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：CoordsOffset
 */

#ifndef VIZ_COORDS_OFFSET_H
#define VIZ_COORDS_OFFSET_H

#include <cstdint>

namespace mdc {
namespace visual {
struct CoordsOffset {
    double xOffset;
    double yOffset;
    CoordsOffset() : xOffset(0.0), yOffset(0.0) {}
    CoordsOffset(const double& vX, const double& vY) : xOffset(vX), yOffset(vY) {}
};
}
}

#endif // VIZ_COORDS_OFFSET_H
