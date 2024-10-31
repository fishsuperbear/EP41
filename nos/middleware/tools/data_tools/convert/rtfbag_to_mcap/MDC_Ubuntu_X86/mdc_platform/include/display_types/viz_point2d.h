/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Point2D
 */

#ifndef VIZ_POINT2D_H
#define VIZ_POINT2D_H

#include "viz_header.h"

namespace mdc {
namespace visual {
struct Point2D {
    double x;
    double y;
    Point2D() : x(0.0), y(0.0) {}
    Point2D(const double& vX, const double& vY) : x(vX), y(vY) {}
};
}
}

#endif // VIZ_POINT2D_H