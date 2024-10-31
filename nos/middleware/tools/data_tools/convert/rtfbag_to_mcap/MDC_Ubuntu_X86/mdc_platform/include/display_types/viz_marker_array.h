/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：MarkerArray
 */

#ifndef VIZ_MARKER_ARRAY_H
#define VIZ_MARKER_ARRAY_H

#include "ara/core/vector.h"
#include "viz_marker.h"

namespace mdc {
namespace visual {
struct MarkerArray {
    ara::core::Vector<Marker> markers;
    MarkerArray() : markers() {}
    explicit MarkerArray(const ara::core::Vector<Marker>& vMarkers) : markers(vMarkers) {}
};
}
}

#endif // VIZ_MARKER_ARRAY_H
