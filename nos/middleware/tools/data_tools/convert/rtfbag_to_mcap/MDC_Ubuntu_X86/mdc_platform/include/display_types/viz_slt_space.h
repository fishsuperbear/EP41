/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：SltSpace
 */

#ifndef VIZ_SLT_SPACE_H
#define VIZ_SLT_SPACE_H

#include <cstdint>
#include "ara/core/string.h"
#include "ara/core/vector.h"

namespace mdc {
namespace visual {
struct SltPoint {
    int32_t type;
    ara::core::String id;
    double sValue;
    double lValue;
    double tValue;
    SltPoint() : type(0), id(), sValue(0.0), lValue(0.0), tValue(0.0) {}
    SltPoint(const int32_t& vType, const ara::core::String& vId, const double& vSValue,
    const double& vLValue, const double& vTValue)
        : type(vType), id(vId), sValue(vSValue), lValue(vLValue), tValue(vTValue) {}
};

struct SltSpace {
    ara::core::Vector<SltPoint> optimizedSltTrack;
    ara::core::Vector<SltPoint> sltSpeedLimit;
    ara::core::Vector<SltPoint> sltObjects;
    SltSpace() : optimizedSltTrack(), sltSpeedLimit(), sltObjects() {}
    SltSpace(const ara::core::Vector<SltPoint>& vOptimizedSltTrack, const ara::core::Vector<SltPoint>& vSltSpeedLimit,
    const ara::core::Vector<SltPoint>& vSltObjects)
        : optimizedSltTrack(vOptimizedSltTrack), sltSpeedLimit(vSltSpeedLimit), sltObjects(vSltObjects) {}
};
}
}

#endif // VIZ_SLT_SPACE_H
