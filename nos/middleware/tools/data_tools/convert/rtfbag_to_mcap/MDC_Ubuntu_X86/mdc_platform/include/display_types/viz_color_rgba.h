/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：ColorRGBA
 */

#ifndef VIZ_COLOR_RGBA_H
#define VIZ_COLOR_RGBA_H


namespace mdc {
namespace visual {
struct ColorRGBA {
    float r;
    float g;
    float b;
    float a;
    ColorRGBA() : r(0.0F), g(0.0F), b(0.0F), a(1.0F) {}
    ColorRGBA(const float& vR, const float& vG, const float& vB, const float& vA)
        : r(vR), g(vG), b(vB), a(vA) {}
};
}
}

#endif // VIZ_COLOR_RGBA_H
