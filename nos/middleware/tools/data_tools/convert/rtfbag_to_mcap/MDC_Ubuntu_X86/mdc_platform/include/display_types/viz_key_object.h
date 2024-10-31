/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：KeyObject
 */

#ifndef VIZ_KEY_OBJECT_H
#define VIZ_KEY_OBJECT_H

#include <cstdint>
#include "viz_header.h"
#include "viz_pose.h"
#include "viz_quaternion.h"

namespace mdc {
namespace visual {
enum KeyObjectClassification {
    KEYOBJECT_CLASSIFICATION_UNKNOWN = 0,    // 未知
    KEYOBJECT_CLASSIFICATION_CAR = 1,        // 小汽车
    KEYOBJECT_CLASSIFICATION_TRUCK = 2,      // 卡车
    KEYOBJECT_CLASSIFICATION_PEDESTRIAN = 3, // 行人
    KEYOBJECT_CLASSIFICATION_PEDESTRIAN_SITTING = 4,
    KEYOBJECT_CLASSIFICATION_CYCLIST = 5,
    KEYOBJECT_CLASSIFICATION_TRAM = 6,
    KEYOBJECT_CLASSIFICATION_MISC = 7,
    KEYOBJECT_CLASSIFICATION_BUS = 8,
};

struct KeyObject {
    Header header;
    int32_t id;                  // 关键障碍物ID
    uint8_t classification;      // 障碍物类别
    double speed;                // 障碍物速度
    Quaternion speedOrientation; // 速度方向
    double lateralDistance;      // 横向距离
    double verticalDistance;     // 纵向距离
    KeyObject()
        : header(),
          id(0),
          classification(0U),
          speed(0.0),
          speedOrientation(),
          lateralDistance(0.0),
          verticalDistance(0.0) {}
    KeyObject(const Header& vHeader, const int32_t& vId, const uint8_t& vClassification,
        const double& vSpeed, const Quaternion& vOrientation, const double& vLateral, const double& vVertical)
        : header(vHeader),
          id(vId),
          classification(vClassification),
          speed(vSpeed),
          speedOrientation(vOrientation),
          lateralDistance(vLateral),
          verticalDistance(vVertical) {}
};
} // namespace visual
} // namespace ara

#endif // VIZ_KEYOBJECT_H
