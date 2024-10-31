/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Twist
 */


#ifndef VIZ_TWIST_H
#define VIZ_TWIST_H

#include "viz_vector3.h"

namespace mdc {
namespace visual {
struct Twist {
    Vector3 linear;
    Vector3 angular;
    Twist() : linear(), angular() {}
    Twist(const Vector3& vLinear, const Vector3& vAngular) : linear(vLinear), angular(vAngular) {}
};
}
}

#endif // VIZ_TWIST_H
