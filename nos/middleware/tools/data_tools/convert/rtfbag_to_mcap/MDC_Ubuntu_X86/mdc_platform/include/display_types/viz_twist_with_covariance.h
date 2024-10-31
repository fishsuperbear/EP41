/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：TwistWithCovariance
 */

#ifndef VIZ_TWIST_WITH_COVARIANCE_H
#define VIZ_TWIST_WITH_COVARIANCE_H

#include <cstdint>
#include "viz_twist.h"
#include "ara/core/vector.h"

namespace mdc {
namespace visual {
const uint32_t THIRTYSIX = 36U;
struct TwistWithCovariance {
    Twist twistData;
    double covariance[THIRTYSIX];
    TwistWithCovariance() : twistData(), covariance() {}
    TwistWithCovariance(const Twist& vTwistData, const double (&vCovariance)[THIRTYSIX])
        : twistData(vTwistData)
    {
        for (uint32_t i = 0U; i < THIRTYSIX; i++) {
            covariance[i] = vCovariance[i];
        }
    }
};
}
}

#endif // VIZ_TWIST_WITH_COVARIANCE_H
