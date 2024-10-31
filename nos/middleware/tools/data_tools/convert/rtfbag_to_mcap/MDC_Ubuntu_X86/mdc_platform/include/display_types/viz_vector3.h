/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Vector3
 */


#ifndef VIZ_VECTOR3_H
#define VIZ_VECTOR3_H

namespace mdc {
namespace visual {
struct Vector3 {
    double x;
    double y;
    double z;
    Vector3() : x(0.0), y(0.0), z(0.0) {}
    Vector3(const double& vX, const double& vY, const double& vZ) : x(vX), y(vY), z(vZ) {}
};
}
}

#endif // VIZ_VECTOR3_H
