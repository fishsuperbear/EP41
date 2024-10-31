/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: 可视化内置结构：Radar Track Array
 */

#ifndef VIZ_RADAR_TRACK_ARRAY_H
#define VIZ_RADAR_TRACK_ARRAY_H
#include <cstdint>
#include <string>
#include "viz_header.h"
#include "viz_radar_state.h"

namespace mdc {
namespace visual {
struct RadarTrack {
    uint8_t id;
    uint8_t idState;
    float lifetime;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float ax;
    float ay;
    float rcs;
    float snr;
    float xRms;
    float yRms;
    float zRms;
    float vxRms;
    float vyRms;
    float axRms;
    float ayRms;
    float orientation;
    float orientRms;
    float yawRate;
    float length;
    float width;
    float height;
    float yawRateRms;
    float lengthRms;
    float widthRms;
    float heightRms;
    float xQuality;
    float yQuality;
    float zQuality;
    float vxQuality;
    float vyQuality;
    float axQuality;
    float ayQuality;
    float orientationQuality;
    float yawRateQuality;
    float lengthQuality;
    float widthQuality;
    float heightQuality;
    float underpassProbability;
    float overpassProbability;
    uint8_t existProbability;
    uint8_t movProperty;
    uint8_t trackState;
    uint8_t trackType;

    RadarTrack()
        : id(0U), idState(0U), lifetime(0.0F), x(0.0F), y(0.0F), z(0.0F), vx(0.0F), vy(0.0F), ax(0.0F), ay(0.0F),
          rcs(0.0F), snr(0.0F), xRms(0.0F), yRms(0.0F), zRms(0.0F), vxRms(0.0F), vyRms(0.0F), axRms(0.0F),
          ayRms(0.0F), orientation(0.0F), orientRms(0.0F), yawRate(0.0F), length(0.0F), width(0.0F), height(0.0F),
          yawRateRms(0.0F), lengthRms(0.0F), widthRms(0.0F), heightRms(0.0F), xQuality(0.0F), yQuality(0.0F),
          zQuality(0.0F), vxQuality(0.0F), vyQuality(0.0F), axQuality(0.0F), ayQuality(0.0F),
          orientationQuality(0.0F), yawRateQuality(0.0F), lengthQuality(0.0F), widthQuality(0.0F),
          heightQuality(0.0F), underpassProbability(0.0F), overpassProbability(0.0F),
          existProbability(0U), movProperty(0U), trackState(0U), trackType(0U) {}

    RadarTrack(const uint8_t& vId, const uint8_t& vIdState, const float& vLifetime, const float& vX, const float& vY,
        const float& vZ, const float& vVx, const float& vVy, const float& vSx, const float& vSy,
        const float& vTcs, const float& vSnr, const float& vXRms, const float& vYRms, const float& vZRms,
        const float& vVxRms, const float& vVyRms, const float& vAxRms, const float& vAyRms,
        const float& vOrientation, const float& vOrientRms, const float& vYawRate, const float& vLength,
        const float& vWidth, const float& vHeight, const float& vYawRateRms, const float& vLengthRms,
        const float& vWidthRms, const float& vHeightRms, const float& vXQuality, const float& vYQuality,
        const float& vZQuality, const float& vVxQuality, const float& vVyQuality, const float& vAxQuality,
        const float& vAyQuality, const float& vOrientationQuality, const float& vyawRateQuality,
        const float& vLengthQuality, const float& vWidthQuality, const float& vHeightQuality,
        const float& vUnderpassProbability, const float& vOverpassProbability, const uint8_t& vExistProbability,
        const uint8_t& vMovProperty, const uint8_t& vTrackState, const uint8_t& vTrackType)
        : id(vId), idState(vIdState), lifetime(vLifetime), x(vX), y(vY), z(vZ), vx(vVx), vy(vVy), ax(vSx), ay(vSy),
          rcs(vTcs), snr(vSnr), xRms(vXRms), yRms(vYRms), zRms(vZRms), vxRms(vVxRms), vyRms(vVyRms), axRms(vAxRms),
          ayRms(vAyRms), orientation(vOrientation), orientRms(vOrientRms), yawRate(vYawRate), length(vLength),
          width(vWidth), height(vHeight), yawRateRms(vYawRateRms), lengthRms(vLengthRms), widthRms(vWidthRms),
          heightRms(vHeightRms), xQuality(vXQuality), yQuality(vYQuality), zQuality(vZQuality), vxQuality(vVxQuality),
          vyQuality(vVyQuality), axQuality(vAxQuality), ayQuality(vAyQuality), orientationQuality(vOrientationQuality),
          yawRateQuality(vyawRateQuality), lengthQuality(vLengthQuality), widthQuality(vWidthQuality),
          heightQuality(vHeightQuality), underpassProbability(vUnderpassProbability),
          overpassProbability(vOverpassProbability), existProbability(vExistProbability),
          movProperty(vMovProperty), trackState(vTrackState), trackType(vTrackType) {}
};

struct RadarTrackArray {
    Header header;
    uint8_t sensorId;
    RadarState radarState;
    ara::core::Vector<RadarTrack> trackList;
    RadarTrackArray() : header(), sensorId(0U), radarState(), trackList() {}
    RadarTrackArray(const Header& vHeader, const uint8_t& vSensorId, const RadarState& vRadarState,
        const ara::core::Vector<RadarTrack>& vTrackList)
        : header(vHeader), sensorId(vSensorId), radarState(vRadarState), trackList(vTrackList) {}
};
}
}
#endif // VIZ_RADAR_TRACK_ARRAY_H
