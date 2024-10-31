/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: 可视化内置结构：Radar Detect Array
 */

#ifndef VIZ_RADAR_DETECT_ARRAY_H
#define VIZ_RADAR_DETECT_ARRAY_H
#include <cstdint>
#include <string>
#include "viz_header.h"
#include "viz_radar_state.h"

namespace mdc {
namespace visual {
struct RadarDetect {
    uint8_t id;
    uint8_t idPair;
    uint8_t coordinate;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float rcs;
    float snr;
    float xRms;
    float yRms;
    float zRms;
    float vxRms;
    float vyRms;
    float xQuality;
    float yQuality;
    float zQuality;
    float vxQuality;
    float vyQuality;
    uint8_t existProbability;
    uint8_t falseProbability;
    uint8_t movProperty;
    uint8_t invalidState;
    uint8_t ambiguity;
    RadarDetect()
        : id(0U), idPair(0U), coordinate(0U), x(0.0F), y(0.0F), z(0.0F), vx(0.0F), vy(0.0F), rcs(0.0F),
          snr(0.0F), xRms(0.0F), yRms(0.0F), zRms(0.0F), vxRms(0.0F), vyRms(0.0F), xQuality(0.0F),
          yQuality(0.0F), zQuality(0.0F), vxQuality(0.0F), vyQuality(0.0F), existProbability(0U),
          falseProbability(0U), movProperty(0U), invalidState(0U), ambiguity(0U) {}
    RadarDetect(const uint8_t& vId, const uint8_t& vIdPair, const uint8_t& vCoordinate, const float& vX,
        const float& vY, const float& vZ, const float& vVx, const float& vVy, const float& vRcs, const float& vSnr,
        const float& vXRms, const float& vYRms, const float& vZRms, const float& vVxRms, const float& vVyRms,
        const float& vXQuality, const float& vYQuality, const float& vZQuality, const float& vVxQuality,
        const float& vVyQuality, const uint8_t& vExistProbability, const uint8_t& vFalseProbability,
        const uint8_t& vMovProperty, const uint8_t& vInvalidState, const uint8_t& vSmbiguity)
        : id(vId), idPair(vIdPair), coordinate(vCoordinate), x(vX), y(vY), z(vZ), vx(vVx), vy(vVy), rcs(vRcs),
          snr(vSnr), xRms(vXRms), yRms(vYRms), zRms(vZRms), vxRms(vVxRms), vyRms(vVyRms), xQuality(vXQuality),
          yQuality(vYQuality), zQuality(vZQuality), vxQuality(vVxQuality), vyQuality(vVyQuality),
          existProbability(vExistProbability), falseProbability(vFalseProbability), movProperty(vMovProperty),
          invalidState(vInvalidState), ambiguity(vSmbiguity) {}
};

struct RadarDetectArray {
    Header header;
    uint8_t sensorId;
    RadarState radarState;
    ara::core::Vector<RadarDetect> detectList;
    RadarDetectArray() : header(), sensorId(0U), radarState(), detectList() {}
    RadarDetectArray(const Header& vHeader, const uint8_t& vSensorId, const RadarState& vRadarState,
        const ara::core::Vector<RadarDetect>& vDetectList)
        : header(vHeader), sensorId(vSensorId), radarState(vRadarState), detectList(vDetectList) {}
};
}
}
#endif // VIZ_RADAR_DETECT_ARRAY_H
