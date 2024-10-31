/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Pose
 */

#ifndef VIZ_POSE_H
#define VIZ_POSE_H

#include "viz_header.h"
#include "viz_point.h"
#include "viz_quaternion.h"

namespace mdc {
namespace visual {
struct Pose {
    Point position;
    Quaternion orientation;
    Pose() : position(), orientation() {}
    Pose(const Point& vPosition, const Quaternion& vOrientation) : position(vPosition), orientation(vOrientation) {}
};

struct PoseStamped {
    Header header;
    Pose pose;
    PoseStamped() : header(), pose() {}
    PoseStamped(const Header& vHeader, const Pose& vPose) : header(vHeader), pose(vPose) {}
};
}
}

#endif // VIZ_POSE_H
