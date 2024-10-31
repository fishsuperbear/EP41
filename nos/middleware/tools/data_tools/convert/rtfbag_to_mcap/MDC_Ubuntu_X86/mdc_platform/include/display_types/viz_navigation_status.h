/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：NavigationStatus
 */

#ifndef VIZ_NAVIGATION_STATUS_H
#define VIZ_NAVIGATION_STATUS_H

#include <cstdint>
#include "viz_header.h"

namespace mdc {
namespace visual {
enum NavSceneType {
    NAV_SCENE_UNKNOWN = 0,       // 未知
    NAV_SCENE_PARK_ROAD = 1,     // 园区道路
    NAV_SCENE_HIGHWAY = 2,       // 高速道路
    NAV_SCENE_PARKING = 3,       // 泊车场景
    NAV_SCENE_URBAN = 4,         // 城市道路
    NAV_SCENE_OPEN_ROAD = 5,     // 开放道路
    NAV_SCENE_OPEN_CROSSING = 6, // 开放道路路口
    NAV_SCENE_RAMP = 7,          // 匝道
};

enum NavStatus {
    NAV_STATUS_UNKNOWN = 0,       // 未知
    NAV_STATUS_GO_STRAIGHT = 1,   // 直行
    NAV_STATUS_BY_PASS = 2,       // 避障
    NAV_STATUS_STOP = 3,          // 停车
    NAV_STATUS_START = 4,         // 启动
    NAV_STATUS_SUMMON = 5,        // 召唤
    NAV_STATUS_KEEP_LANE = 6,     // 车道保持
    NAV_STATUS_IDLE = 7,          // 怠速
    NAV_STATUS_FOLLOW = 8,        // 跟随
    NAV_STATUS_MERGE_LEFT = 9,    // 向左变道
    NAV_STATUS_MERGE_RIGHT = 10,  // 向右变道
    NAV_STATUS_MERGE_CANCEL = 11, // 变道取消
    NAV_STATUS_TURN_LEFT = 12,    // 左转
    NAV_STATUS_TURN_RIGHT = 13,   // 右转
    NAV_STATUS_U_TURN = 14,       // 调头
    NAV_STATUS_AVOID = 15,        // 避让
};

struct NavigationStatus {
    Header header;
    uint8_t navSceneType;
    uint8_t navStatus;
    NavigationStatus() : header(), navSceneType(0U), navStatus(0U) {}
    NavigationStatus(const Header& vHeader, const uint8_t& vType, const uint8_t& vStatus)
        : header(vHeader), navSceneType(vType), navStatus(vStatus) {}
};
} // namespace visual
} // namespace ara

#endif // VIZ_NAVIGATIONSTATUS_H
