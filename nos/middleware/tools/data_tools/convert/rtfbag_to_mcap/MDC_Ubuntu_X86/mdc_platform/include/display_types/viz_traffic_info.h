/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：TrafficInfo
 */

#ifndef VIZ_TRAFFIC_INFO_H
#define VIZ_TRAFFIC_INFO_H

#include <cstdint>
#include "viz_header.h"

namespace mdc {
namespace visual {
enum TrafficLightState {
    TRAFFIC_LIGHT_STATE_UNKNOWN = 0, // 未知
    TRAFFIC_LIGHT_STATE_GREEN = 1,   // 绿灯
    TRAFFIC_LIGHT_STATE_YELLOW = 2,  // 黄灯
    TRAFFIC_LIGHT_STATE_RED = 3,     // 红灯
};

enum TrafficLightType {
    TRAFFIC_LIGHT_TYPE_UNKNOWN = 0,                  // 未知
    TRAFFIC_LIGHT_TYPE_STRAIGHT,                     // 直行红绿灯
    TRAFFIC_LIGHT_TYPE_TURNLEFT,                     // 左转红绿灯
    TRAFFIC_LIGHT_TYPE_TURNAROUND,                   // 掉头红绿灯
    TRAFFIC_LIGHT_TYPE_TURNLEFT_TURNAROUND,          // 左转掉头红绿灯
    TRAFFIC_LIGHT_TYPE_TURNAROUND_STRAIGHT,          // 掉头直行红绿灯
    TRAFFIC_LIGHT_TYPE_TURNLEFT_STRAIGHT,            // 左转直行红绿灯
    TRAFFIC_LIGHT_TYPE_TURNLEFT_TURNAROUND_STRAIGHT, // 左转掉头直行红绿灯
};

enum TrafficSignType {
    TRAFFIC_SIGN_UNKNOWN = 0,            // 未知
    TRAFFIC_SIGN_CROSSROAD = 1,          // 十字路口
    TRAFFIC_SIGN_T_JUNCTION = 2,         // T形路口
    TRAFFIC_SIGN_Y_JUNCTION = 3,         // Y形路口
    TRAFFIC_SIGN_ROUNDABOUT = 4,         // 环形交叉路口
    TRAFFIC_SIGN_SERIES_BENDS = 5,       // 连续弯道
    TRAFFIC_SIGN_UPHILL = 6,             // 上陡坡
    TRAFFIC_SIGN_DOWNHILL = 7,           // 下陡坡
    TRAFFIC_SIGN_ALERT_PEOPLE = 8,       // 注意行人
    TRAFFIC_SIGN_ALERT_CHILDREN = 9,     // 注意儿童
    TRAFFIC_SIGN_WATCH_LIGHTS = 10,      // 注意信号灯
    TRAFFIC_SIGN_VILLAGE = 11,           // 村庄
    TRAFFIC_SIGN_TUNNEL = 12,            // 隧道
    TRAFFIC_SIGN_ACCIDENT_AREA = 13,     // 事故易发路段
    TRAFFIC_SIGN_SLOW_DOWN = 14,         // 慢行
    TRAFFIC_SIGN_DETOUR_BOTH = 15,       // 左右绕行
    TRAFFIC_SIGN_DETOUR_LEFT = 16,       // 左侧绕行
    TRAFFIC_SIGN_DETOUR_RIGHT = 17,      // 右侧绕行
    TRAFFIC_SIGN_ROAD_CONSTRUCTION = 18, // 施工
    TRAFFIC_SIGN_DANGEROUS = 19,         // 注意危险
    TRAFFIC_SIGN_NO_THOROUGHFARE = 20,   // 禁止通行
    TRAFFIC_SIGN_NO_ENTRY = 21,          // 禁止驶入
    TRAFFIC_SIGN_NO_HONKING = 22,        // 禁止鸣笛
    TRAFFIC_SIGN_NO_TRUCK = 23,          // 禁止XX车辆通行
    TRAFFIC_SIGN_NO_LEFT = 24,           // 禁止左转
    TRAFFIC_SIGN_NO_RIGHT = 25,          // 禁止右转
    TRAFFIC_SIGN_NO_STRAIGHT = 26,       // 禁止直行
    TRAFFIC_SIGN_NO_OVERTAKING = 27,     // 禁止超车
    TRAFFIC_SIGN_NO_U_TURN = 28,         // 禁止掉头
    TRAFFIC_SIGN_NO_LEFT_RIGHT = 29,     // 禁止向左向右转
    TRAFFIC_SIGN_NO_STRAIGHT_LEFT = 30,  // 禁止直行和左转
    TRAFFIC_SIGN_KEEP_LEFT = 31,         // 靠左行驶
    TRAFFIC_SIGN_KEEP_RIGHT = 32,        // 靠右行驶
};

struct TrafficInfo {
    Header header;
    uint8_t trafficLightState;         // 交通信号灯状态
    uint32_t trafficLightReservedTime; // 最近的交通信号灯保留时间
    double trafficLightDistance;       // 距相应停止线的距离
    uint8_t trafficLightLaneId;        // 红绿灯所处车道的id
    uint8_t trafficLightType;          // 红绿灯所控制的车道行车方向
    double speedLimit;                 // 限速牌信息
    uint8_t trafficSign;               // 交通标识牌
    TrafficInfo()
        : header(),
          trafficLightState(0U),
          trafficLightReservedTime(0U),
          trafficLightDistance(0.0),
          trafficLightLaneId(0U),
          trafficLightType(0U),
          speedLimit(0.0),
          trafficSign(0U)
    {}
    TrafficInfo(const Header vHeader, const uint8_t& vTrafficLightState, const uint32_t vTrafficLightReservedTime,
    const double& vTrafficLightDistance, const uint8_t& vTrafficLightLaneId, const uint8_t& vTrafficLightType,
    const double& vSpeedLimit, const uint8_t& vTrafficSign)
        : header(vHeader),
          trafficLightState(vTrafficLightState),
          trafficLightReservedTime(vTrafficLightReservedTime),
          trafficLightDistance(vTrafficLightDistance),
          trafficLightLaneId(vTrafficLightLaneId),
          trafficLightType(vTrafficLightType),
          speedLimit(vSpeedLimit),
          trafficSign(vTrafficSign)
    {}
};
} // namespace visual
} // namespace ara

#endif // VIZ_TRAFFIC_INFO_H
