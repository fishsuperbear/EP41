/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Marker
 */

#ifndef VIZ_MARKER_H
#define VIZ_MARKER_H

#include <cstdint>
#include "ara/core/string.h"
#include "ara/core/vector.h"
#include "viz_header.h"
#include "viz_times.h"
#include "viz_point.h"
#include "viz_pose.h"
#include "viz_vector3.h"
#include "viz_color_rgba.h"

namespace mdc {
namespace visual {
enum class MarkerType : uint8_t {
    ARROW = 0U,
    CUBE,
    SPHERE,
    CYLINDER,
    LINE_STRIP,
    LINE_LIST,
    CUBE_LIST,
    SPHERE_LIST,
    POINTS,
    TEXT_VIEW_FACING,
    MESH_RESOURCE,
    TRIANGLE_LIST
};

enum class MarkerAction : uint8_t {
    ADD = 0U,
    MODIFY = 0U,
    DELETE = 2U,
    DELETEALL = 3U
};

struct Marker {
    Header header;
    ara::core::String ns;
    int32_t id;
    MarkerType type;
    MarkerAction action;
    Pose pose;
    Vector3 scale;
    ColorRGBA color;
    Times lifetime;
    bool frameLocked;
    ara::core::Vector<Point> points;
    ara::core::Vector<ColorRGBA> colors;
    ara::core::String text;
    ara::core::String meshResource;
    bool meshUseEmbeddedMaterials;
    Marker()
        : header(),
          ns(),
          id(0),
          type(MarkerType::TEXT_VIEW_FACING),
          action(MarkerAction::ADD),
          pose(),
          scale(),
          color(),
          lifetime(),
          frameLocked(false),
          points(),
          colors(),
          text(),
          meshResource(),
          meshUseEmbeddedMaterials(false)
    {}
    Marker(const Header& vHeader, const ara::core::String& vNs, const int32_t& vId, const MarkerType& vType,
    const MarkerAction& vAction, const Pose& vPose, const Vector3& vScale, const ColorRGBA& vColor,
    const Times& vLifetime, const bool& vFrameLocked, const ara::core::Vector<Point>& vPoints,
    const ara::core::Vector<ColorRGBA>& vColors, const ara::core::String& vText,
    const ara::core::String& vResource, const bool& vMaterials)
        : header(vHeader),
          ns(vNs),
          id(vId),
          type(vType),
          action(vAction),
          pose(vPose),
          scale(vScale),
          color(vColor),
          lifetime(vLifetime),
          frameLocked(vFrameLocked),
          points(vPoints),
          colors(vColors),
          text(vText),
          meshResource(vResource),
          meshUseEmbeddedMaterials(vMaterials)
    {}
};
}
}

#endif // VIZ_MARKER_H
