/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：VizImageLineArray
 */

#ifndef VIZ_IMAGE_LINE_ARRAY_H
#define VIZ_IMAGE_LINE_ARRAY_H
#include <cstdint>
#include <string>
#include "viz_header.h"
#include "viz_point2d.h"

namespace mdc {
namespace visual {
struct ImageLine {
    Header header;
    Point2D startPoint;
    Point2D endPoint; // Must be a value specified in the CLASSIFICATION_* constants above.
    // 多项式系数: "ay^3+by^2+cy+d = x"
    double a;
    double b;
    double c;
    double d;

    uint8_t classification;
    double classificationConfidence; // A value between 0.0 and 1.0. #Optional for fusion output
    double classificationAgeSeconds; // Current classification duration in seconds. #Optional for fusion output
    int32_t classificationAgeCount;  // Current classification duration in number of frames. #Optional for fusion output
    ara::core::String textDisplay; // 实际发送时限制了长度，超过255字节会截断

    ImageLine()
        : header(),
          startPoint(),
          endPoint(),
          a(0.0),
          b(0.0),
          c(0.0),
          d(0.0),
          classification(0U),
          classificationConfidence(0.0),
          classificationAgeSeconds(0.0),
          classificationAgeCount(0),
          textDisplay()
    {}
    ImageLine(const Header& vHeader, const Point2D& vStart, const Point2D& vEnd, const double& vA,
            const double& vB, const double& vC, const double& vD, const uint8_t& vClassification,
            const double& vConfidence, const double& vAgeSeconds, const int32_t& vAgeCount,
            const ara::core::String& text)
        : header(vHeader),
          startPoint(vStart),
          endPoint(vEnd),
          a(vA),
          b(vB),
          c(vC),
          d(vD),
          classification(vClassification),
          classificationConfidence(vConfidence),
          classificationAgeSeconds(vAgeSeconds),
          classificationAgeCount(vAgeCount),
          textDisplay(text)
    {}
};

struct ImageLineArray {
    Header header;
    ara::core::Vector<ImageLine> imageLineList;
    ImageLineArray() : header(), imageLineList() {}
    ImageLineArray(const Header& vHeader, const ara::core::Vector<ImageLine>& vLines)
        : header(vHeader), imageLineList(vLines) {}
};
}
}
#endif // VIZ_IMAGE_LINE_ARRAY_H
