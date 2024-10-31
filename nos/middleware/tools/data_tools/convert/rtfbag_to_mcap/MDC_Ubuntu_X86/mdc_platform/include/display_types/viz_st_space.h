/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：StSpace
 */

#ifndef VIZ_ST_SPACE_H
#define VIZ_ST_SPACE_H

#include <cstdint>
#include "ara/core/string.h"
#include "ara/core/vector.h"

namespace mdc {
namespace visual {
struct StPoint {
    int32_t type;
    ara::core::String id;
    double  sValue;
    double  tValue;
    StPoint() : type(0), id(), sValue(0.0), tValue(0.0) {}
    StPoint(const int32_t& vType, const ara::core::String& vId, const double& vSValue, const double& vTValue)
        : type(vType), id(vId), sValue(vSValue), tValue(vTValue) {}
};

struct StSpace {
    ara::core::Vector<StPoint> firstStTrack;
    ara::core::Vector<StPoint> optimizedStTrack;
    ara::core::Vector<StPoint> stSpeedLimit;
    ara::core::Vector<StPoint> stObjects;
    StSpace() : firstStTrack(), optimizedStTrack(), stSpeedLimit(), stObjects() {}
    StSpace(const ara::core::Vector<StPoint>& vFirstStTrack, const ara::core::Vector<StPoint>& vOptimizedStTrack,
    const ara::core::Vector<StPoint>& vStSpeedLimit, const ara::core::Vector<StPoint>& vStObjects)
        : firstStTrack(vFirstStTrack),
          optimizedStTrack(vOptimizedStTrack),
          stSpeedLimit(vStSpeedLimit),
          stObjects(vStObjects) {}
};
}
}

#endif // VIZ_ST_SPACE_H
