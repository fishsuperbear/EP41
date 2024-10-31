/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Quaternion
 */

#ifndef VIZ_QUATERNION_H
#define VIZ_QUATERNION_H

namespace mdc {
namespace visual {
struct Quaternion {
    double x;
    double y;
    double z;
    double w;
    Quaternion() : x(0.0), y(0.0), z(0.0), w(1.0) {}
    Quaternion(const double& vX, const double& vY, const double& vZ, const double& vW)
        : x(vX), y(vY), z(vZ), w(vW) {}
};
}
}

#endif // VIZ_QUATERNION_H
