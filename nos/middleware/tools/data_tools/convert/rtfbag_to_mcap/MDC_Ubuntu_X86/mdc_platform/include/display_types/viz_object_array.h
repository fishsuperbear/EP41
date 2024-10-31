/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：ObjectArray
 */

#ifndef VIZ_OBJECT_ARRAY_H
#define VIZ_OBJECT_ARRAY_H

#include <cstdint>
#include <string>
#include "viz_header.h"
#include "viz_point.h"
#include "viz_point2d.h"
#include "viz_polygon.h"
#include "viz_twist_with_covariance.h"
#include "viz_path_with_variance.h"
#include "viz_rectangle.h"
#include "ara/core/vector.h"

namespace mdc {
namespace visual {
enum ObjectClassification {
    CLASSIFICATION_UNKNOWN = 0,             // 未知
    CLASSIFICATION_CAR = 1,                 // 小汽车
    CLASSIFICATION_TRUCK = 2,               // 卡车
    CLASSIFICATION_PEDESTRIAN = 3,          // 行人
    CLASSIFICATION_PEDESTRIAN_SITTING = 4,
    CLASSIFICATION_CYCLIST = 5,
    CLASSIFICATION_TRAM = 6,
    CLASSIFICATION_MISC = 7,
    CLASSIFICATION_BUS = 8
};

struct Object {
    Header header;
    // Object ID.
    int32_t objectId;
    // Track ID.
    int32_t trackId;
    // The existence probability, 0~1. Optional for fusion output.
    double existenceProbability;
    // The number of measurements the track has received. Optional for fusion output.
    int32_t measurementAgeCount;
    // User defined object classification id.
    uint8_t classification;
    // A value between 0.0 and 1.0.
    double classificationConfidence;
    // Current classification duration in seconds. Optional for fusion output.
    double classificationAgeSeconds;
    // Current classification duration in number of frames. Optional for fusion output.
    int32_t classificationAgeCount;
    // Box center Point, m.
    Point objectBoxCenter;
    // Covariance of bounding box center. Optional for fusion output.
    Point objectBoxCenterCovariance;
    // x/y/z represent bounding box size, m.
    Point objectBoxSize;
    // Covariance of bounding box size. Optional for fusion output.
    Point objectBoxSizeCovariance;
    // Box orientation, rad.
    double objectBoxOrientation;
    // Covariance of object box Orientation. Optional for fusion output.
    double objectBoxOrientationCovariance;
    // Bounding box.
    Polygon boxPolygon;
    // Reference point, m.
    Point referencePoint;
    // Covariance of reference point. Optional for fusion output.
    Point referencePointCovariance;
    // Speed, m/s or rad/s.
    TwistWithCovariance velocity;
    // Rectangle relative to image coordinate from camera DTO output.
    Rectangle boxImage;
    // Polygon relative to image coordinate from camera DTO output.
    ara::core::Vector<Point2D> boxImagePolygon;
    // Multiple possible paths of the object. Optional for prediction output.
    ara::core::Vector<PathWithVariance> intentionPaths;
    // Time step in seconds used in the paths, s. Optional for prediction output.
    double intentionTimeStep;
    ara::core::String textDisplay; // 实际发送时限制了长度，超过255字节会截断

    Object()
        : header(),
          objectId(0),
          trackId(0),
          existenceProbability(0.0),
          measurementAgeCount(0),
          classification(0U),
          classificationConfidence(0.0),
          classificationAgeSeconds(0.0),
          classificationAgeCount(0),
          objectBoxCenter(),
          objectBoxCenterCovariance(),
          objectBoxSize(),
          objectBoxSizeCovariance(),
          objectBoxOrientation(0.0),
          objectBoxOrientationCovariance(0.0),
          boxPolygon(),
          referencePoint(),
          referencePointCovariance(),
          velocity(),
          boxImage(),
          boxImagePolygon(),
          intentionPaths(),
          intentionTimeStep(0.0),
          textDisplay()
    {}

    Object(const Header& vHeader, const int32_t& vObjectId, const int32_t& TrackId, const double& vProbability,
    const int32_t& vMeasurementAgeCount, const uint8_t& vClassification, const double& vConfidence,
    const double& vAgeSeconds, const int32_t& vAgeCount, const Point& vBoxCenter, const Point& vBoxCenterCovariance,
    const Point& vBoxSize, const Point& vBoxSizeCovariance, const double& vObjectBoxOrientation,
    const double& vObjectBoxOrientationCovariance, const Polygon& vPolygon, const Point& vReferencePoint,
    const Point& vPointCovariance, const TwistWithCovariance& vVelocity, const Rectangle& vBoxImage,
    const ara::core::Vector<Point2D>& vBoxImagePolygon, const ara::core::Vector<PathWithVariance>& vIntentionPaths,
    const double& vIntentionTimeStep, const ara::core::String& text)
        : header(vHeader),
          objectId(vObjectId),
          trackId(TrackId),
          existenceProbability(vProbability),
          measurementAgeCount(vMeasurementAgeCount),
          classification(vClassification),
          classificationConfidence(vConfidence),
          classificationAgeSeconds(vAgeSeconds),
          classificationAgeCount(vAgeCount),
          objectBoxCenter(vBoxCenter),
          objectBoxCenterCovariance(vBoxCenterCovariance),
          objectBoxSize(vBoxSize),
          objectBoxSizeCovariance(vBoxSizeCovariance),
          objectBoxOrientation(vObjectBoxOrientation),
          objectBoxOrientationCovariance(vObjectBoxOrientationCovariance),
          boxPolygon(vPolygon),
          referencePoint(vReferencePoint),
          referencePointCovariance(vPointCovariance),
          velocity(vVelocity),
          boxImage(vBoxImage),
          boxImagePolygon(vBoxImagePolygon),
          intentionPaths(vIntentionPaths),
          intentionTimeStep(vIntentionTimeStep),
          textDisplay(text)
    {}
};

struct ObjectArray {
    Header header;
    ara::core::Vector<Object> objectList;
    ObjectArray() : header(), objectList() {}
    ObjectArray(const Header& vHeader, const ara::core::Vector<Object>& vObjectList)
        : header(vHeader), objectList(vObjectList) {}
};
} // namespace visual
} // namespace ara

#endif // VIZ_OBJECT_ARRAY_H
