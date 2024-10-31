/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：ImageRectangleArray
 */

#ifndef VIZ_IMAGE_RECTANGLE_ARRAY_H
#define VIZ_IMAGE_RECTANGLE_ARRAY_H

#include <cstdint>
#include <string>
#include "viz_header.h"
#include "viz_rectangle.h"

namespace mdc {
namespace visual {
struct ImageRectangle {
    Header header;
    Rectangle rectImage;
    uint8_t classification;          // Must be a value specified in the CLASSIFICATION_* constants above.
    double classificationConfidence; // A value between 0.0 and 1.0. #Optional for fusion output
    double classificationAgeSeconds; // Current classification duration in seconds. #Optional for fusion output
    int32_t classificationAgeCount;  // Current classification duration in number of frames. #Optional for fusion output
    ara::core::String textDisplay; // 实际发送时限制了长度，超过255字节会截断

    ImageRectangle()
        : header(),
          rectImage(),
          classification(0U),
          classificationConfidence(0.0),
          classificationAgeSeconds(0.0),
          classificationAgeCount(0),
          textDisplay()
    {}
    ImageRectangle(const Header& vHeader, const Rectangle& vRectImage, const uint8_t& vClassification,
        const double& vConfidence, const double& vAgeSeconds, const int32_t& vAgeCount,
        const ara::core::String& text)
        : header(vHeader),
          rectImage(vRectImage),
          classification(vClassification),
          classificationConfidence(vConfidence),
          classificationAgeSeconds(vAgeSeconds),
          classificationAgeCount(vAgeCount),
          textDisplay(text)
    {}
};

struct ImageRectangleArray {
    Header header;
    ara::core::Vector<ImageRectangle> imageRectangleList;
    ImageRectangleArray() : header(), imageRectangleList() {}
    ImageRectangleArray(const Header& vHeader, const ara::core::Vector<ImageRectangle>& vRectangles)
        : header(vHeader), imageRectangleList(vRectangles) {}
};
}
}

#endif // VIZ_IMAGERECTANGLEARRAY_H
