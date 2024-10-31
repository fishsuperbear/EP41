/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：GridCells
 */

#ifndef VIZ_GRID_CELLS_H
#define VIZ_GRID_CELLS_H

#include <cstdint>
#include "ara/core/vector.h"
#include "viz_header.h"
#include "viz_point.h"

namespace mdc {
namespace visual {
struct GridCells {
    Header header;
    float cellWidth;
    float cellHeight;
    ara::core::Vector<Point> cells;
    GridCells() : header(), cellWidth(0.0F), cellHeight(0.0F), cells() {}
    GridCells(const Header& vHeader, const float& vWidth, const float& vHeight,
    const ara::core::Vector<Point>& vCells)
        : header(vHeader), cellWidth(vWidth), cellHeight(vHeight), cells(vCells) {}
};
}
}

#endif // VIZ_GRID_CELLS_H
