/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Path
 */

#ifndef VIZ_PATH_H
#define VIZ_PATH_H

#include "ara/core/vector.h"

#include "viz_header.h"
#include "viz_pose.h"

namespace mdc {
namespace visual {
struct Path {
    Header header;
    ara::core::Vector<PoseStamped> poses;
    Path() : header(), poses() {}
    Path(const Header& vHeader, const ara::core::Vector<PoseStamped>& vPoses) : header(vHeader), poses(vPoses) {}
};
}
}

#endif // VIZ_PATH_H
