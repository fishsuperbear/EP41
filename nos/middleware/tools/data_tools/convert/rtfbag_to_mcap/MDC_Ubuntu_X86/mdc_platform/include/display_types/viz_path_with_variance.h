/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：PathWithVariance
 */

#ifndef VIZ_PATH_WITH_VARIANCE_H
#define VIZ_PATH_WITH_VARIANCE_H

#include <cstdint>
#include "ara/core/vector.h"

namespace mdc {
namespace visual {
const uint32_t THREE = 3U;
struct Pose2DWithVariance {
    double x;
    double y;
    double theta;
    double variance[THREE];
    Pose2DWithVariance() : x(0.0), y(0.0), theta(0.0), variance() {}
    Pose2DWithVariance(const double& vX, const double& vY, const double& vTheta,
    const double (&vVariance)[THREE])
        : x(vX), y(vY), theta(vTheta)
    {
        for (uint32_t i = 0U; i < THREE; i++) {
            variance[i] = vVariance[i];
        }
    }
};

struct PathWithVariance {
    ara::core::Vector<Pose2DWithVariance> poses;
    PathWithVariance() : poses() {}
    explicit PathWithVariance(const ara::core::Vector<Pose2DWithVariance>& vPoses) : poses(vPoses) {}
};
}
}

#endif // VIZ_PATH_WITH_VARIANCE_H
