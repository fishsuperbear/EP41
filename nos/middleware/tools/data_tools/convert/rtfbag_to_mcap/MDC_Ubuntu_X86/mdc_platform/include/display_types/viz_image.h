/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Image
 */

#ifndef VIZ_IMAGE_H
#define VIZ_IMAGE_H

#include "viz_header.h"
#include "ara/core/vector.h"

namespace mdc {
namespace visual {
struct Image {
    Header header;
    ara::core::Vector<uint8_t> frameData;
    Image() : header(), frameData() {}
    Image(const Header& vHeader, const ara::core::Vector<uint8_t>& vFrames) : header(vHeader), frameData(vFrames) {}
};
}
}

#endif // VIZ_IMAGE_H
