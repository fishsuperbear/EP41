/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：PlotData
 */

#ifndef VIZ_PLOT_DATA_H
#define VIZ_PLOT_DATA_H

#include <cstdint>
#include "ara/core/vector.h"

namespace mdc {
namespace visual {
struct PlotPoint {
    double xValue;
    double yValue;
    PlotPoint() : xValue(0.0), yValue(0.0) {}
    PlotPoint(const double& vX, const double& vY) : xValue(vX), yValue(vY) {}
};

struct PlotData {
    ara::core::Vector<PlotPoint> firstCurve;
    ara::core::Vector<PlotPoint> secondCurve;
    double meanSquareDifference;
    PlotData() : firstCurve(), secondCurve(), meanSquareDifference(0.0) {}
    PlotData(const ara::core::Vector<PlotPoint>& vFirst, const ara::core::Vector<PlotPoint>& vSecond,
    const double& vDifference)
        : firstCurve(vFirst), secondCurve(vSecond), meanSquareDifference(vDifference) {}
};
}
}

#endif // VIZ_PLOT_POINT_H
