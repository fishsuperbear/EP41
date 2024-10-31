/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：RoadInfo
 */

#ifndef VIZ_ROAD_INFO_H
#define VIZ_ROAD_INFO_H

#include <cstdint>
#include "viz_header.h"

namespace mdc {
namespace visual {
enum RoadType {
    LANE_TYPE_UNKNOWN = 0,           // 未知
    LANE_TYPE_NORMAL = 1,            // 常规车道
    LANE_TYPE_EMERGENCY = 2,         // 应急车道
    LANE_TYPE_ESCAPE = 3,            // 避险车道
    LANE_TYPE_BUSLANE = 4,           // 公交车专用车道
    LANE_TYPE_BRT = 5,               // BRT车道
    LANE_TYPE_VARIABLE = 6,          // 可变导向车道
    LANE_TYPE_DISAPPEAR = 7,         // 消失车道
    LANE_TYPE_NEW = 8,               // 新增车道
    LANE_TYPE_ACCESS = 9,            // 入口车道
    LANE_TYPE_EXIT = 10,             // 退出车道
    LANE_TYPE_RAMPLINK = 11,         // 匝道链接车道
    LANE_TYPE_CONSTRUCTING = 12,     // 在建车道
    LANE_TYPE_PARKING_ENTRANCE = 13, // 停车场出入口车道
    LANE_TYPE_PARKING_AREA = 14,     // 停车场内部车道
    LANE_TYPE_TIDAL_LANE = 15,       // 潮汐车道
};

enum FrontIntersectionType {
    INTERSECTION_UNKNOWN = 0,      // 未知
    INTERSECTION_T_JUNCTION = 1,   // T字路口
    INTERSECTION_CROSSROADS = 2,   // 十字路口
    INTERSECTION_Y_JUNCTION = 3,   // Y型路口
    INTERSECTION_MULTI_FOLKS = 4,  // 五岔及以上路口
    INTERSECTION_OBLIQUE = 5,      // 斜交路口
    INTERSECTION_MISALIGNED_T = 6, // 错位T型路口
    INTERSECTION_MISALIGNED_Y = 7, // 错位Y型路口
    INTERSECTION_ROUNDABOUT = 8,   // 环形路口
};

struct RoadInfo {
    Header header;
    uint64_t laneId;                // 车道id
    uint8_t laneType;              // 车道类型
    uint8_t leftLaneType;          // 左相邻车道类型
    uint8_t rightLaneType;         // 右相邻车道类型
    bool canChangeLane;            // 是否可以变道
    bool canDriveByLane;           // 是否可以借道
    double roadCurvature;          // 道路曲率
    double roadGradient;           // 道路坡度
    double distanceToIntersection; // 当前道路离路口的距离
    double roadResidueLength;      // 当前道路的剩余长度
    double distanceToEnd;          // 到终点的距离
    uint8_t frontIntersectionType; // 前方路口类型
    double laneSpeedLimit;         // 当前车道限速，0~300
    RoadInfo()
        : header(),
          laneId(0U),
          laneType(0U),
          leftLaneType(0U),
          rightLaneType(0U),
          canChangeLane(false),
          canDriveByLane(false),
          roadCurvature(0.0),
          roadGradient(0.0),
          distanceToIntersection(0.0),
          roadResidueLength(0.0),
          distanceToEnd(0.0),
          frontIntersectionType(0U),
          laneSpeedLimit(0.0)
    {}
    RoadInfo(const Header& vHeader, const uint64_t& vLaneId, const uint8_t& vLaneType, const uint8_t& vLeftLaneType,
    const uint8_t& vRightLaneType, const bool& cCanChangeLane, const bool& vCanDriveByLane,
    const double& vRoadCurvature,  const double& vRoadGradient, const double& vIntersection,
    const double& vRoadResidueLength, const double& vDistanceToEnd, const uint8_t& vFrontIntersectionType,
    const double& vLaneSpeedLimit)
        : header(vHeader),
          laneId(vLaneId),
          laneType(vLaneType),
          leftLaneType(vLeftLaneType),
          rightLaneType(vRightLaneType),
          canChangeLane(cCanChangeLane),
          canDriveByLane(vCanDriveByLane),
          roadCurvature(vRoadCurvature),
          roadGradient(vRoadGradient),
          distanceToIntersection(vIntersection),
          roadResidueLength(vRoadResidueLength),
          distanceToEnd(vDistanceToEnd),
          frontIntersectionType(vFrontIntersectionType),
          laneSpeedLimit(vLaneSpeedLimit)
    {}
};
} // namespace visual
} // namespace ara

#endif // VIZ_ROADINFO_H