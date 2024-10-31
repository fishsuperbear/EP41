/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：TF
 */

#ifndef VIZ_TF_H
#define VIZ_TF_H

#include "ara/core/string.h"

#include "viz_header.h"
#include "viz_transform.h"

namespace mdc {
namespace visual {
struct Tf {
    Header header;
    ara::core::String childFrameId;
    Transform transform;
    Tf() : header(), childFrameId(), transform() {}
    Tf(const Header& vHeader, const ara::core::String& vChildFrameId, const Transform& vTransform)
        : header(vHeader), childFrameId(vChildFrameId), transform(vTransform) {}
};
}
}

#endif // VIZ_TF_H
