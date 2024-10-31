/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Transform
 */

#ifndef VIZ_TRANSFORM_H
#define VIZ_TRANSFORM_H

#include "viz_vector3.h"
#include "viz_quaternion.h"

namespace mdc {
namespace visual {
struct Transform {
    Vector3 translation;
    Quaternion rotation;
    Transform() : translation(), rotation() {}
    Transform(const Vector3& vTranslation, const Quaternion& vRotation)
        : translation(vTranslation), rotation(vRotation) {}
};
}
}

#endif // VIZ_TRANSFORM_H
