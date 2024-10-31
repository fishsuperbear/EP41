/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Rectangle
 */

#ifndef VIZ_RECTANGLE_H
#define VIZ_RECTANGLE_H

#include "viz_point.h"
#include "ara/core/vector.h"

namespace mdc {
namespace visual {
struct Rectangle {
    double x;
    double y;
    double width;
    double height;
    Rectangle() : x(0.0), y(0.0), width(0.0), height(0.0) {}
    Rectangle(const double& vX, const double& vY, const double& vWidth, const double& vHeight)
        : x(vX), y(vY), width(vWidth), height(vHeight) {}
};
}
}

#endif // VIZ_RECTANGLE_H
