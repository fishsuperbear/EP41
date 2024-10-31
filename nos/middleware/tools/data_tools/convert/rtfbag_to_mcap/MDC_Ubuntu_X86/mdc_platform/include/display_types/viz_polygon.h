/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Polygon
 */


#ifndef VIZ_POLYGON_H
#define VIZ_POLYGON_H

#include "ara/core/vector.h"
#include "viz_point.h"

namespace mdc {
namespace visual {
struct Polygon {
    ara::core::Vector<Point> points;
    Polygon() : points() {}
    explicit Polygon(const ara::core::Vector<Point>& vPoints) : points(vPoints) {}
};

struct PolygonStamped {
    Header header;
    Polygon polygon;
    PolygonStamped() : header(), polygon() {}
    PolygonStamped(const Header& vHeader, const Polygon& vPolygon) : header(vHeader), polygon(vPolygon) {}
};
}
}

#endif // VIZ_POLYGON_H
