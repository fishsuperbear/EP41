/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Point
 */

#ifndef VIZ_POINT_H
#define VIZ_POINT_H

#include "viz_header.h"

namespace mdc {
namespace visual {
struct Point {
    double x;
    double y;
    double z;
    Point() : x(0.0), y(0.0), z(0.0) {}
    Point(const double& vX, const double& vY, const double& vZ) : x(vX), y(vY), z(vZ) {}
};

struct PointStamped {
    Header header;
    Point point;
    PointStamped() : header(), point() {}
    PointStamped(const Header& vHeader, const Point& vPoint) : header(vHeader), point(vPoint) {}
};

using RecvPointStamped = void (*) (const PointStamped &);
}
}

#endif // VIZ_POINT_H
