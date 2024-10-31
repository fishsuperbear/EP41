#pragma once
#include <functional>
#include <iostream>
#include <unordered_map>
#include "adf/include/node_base.h"

#define DYNAMIC_OBJECT_SIZE (64)
#define LANE_SEQUENCE_SIZE (8)
#define LANE_SEQUENCE_BEGIN (200)
#define LANE_SEQUENCE_END (400)
#define LANE_SEQUENCE_OFF_SET (300)

namespace hozon {
namespace netaos {
namespace extra {

enum class HmiDynamicObjectType : uint8_t {
    HMI_UNKNOWN = 0,
    HMI_PERSON,
    HMI_BICYCLE,
    HMI_MOTORBIKE,
    HMI_TRICYCLE,
    HMI_CAR,
    HMI_TRUCK,
    HMI_BUS,
    HMI_SPECIAL,
    HMI_SUV,
    HMI_MINIBUS,
    HMI_AMBULANCE,
    HMI_FIRETRUCK,
    HMI_PRODUCTION_VEHICLE,
    HMI_ANIMAL,
    HMI_SMALL_TRUCK,
    HMI_RESERVED_1,
    HMI_RESERVED_2,
    HMI_RESERVED_3,
};

enum class HmiStaticObjectType : uint8_t {
    HMI_STATIC_UNKNOWN = 0,
    HMI_STATIC_CONE = 1,           // 锥筒
    HMI_STATIC_WATER_HORSE,        // 水马
    HMI_STATIC_TRIANGULAR_PLATE,   // 三角板
    HMI_STATIC_BAN,                // 禁停牌
    HMI_STATIC_STOP_BAR,           // 停车杆
    HMI_STATIC_SPEED_HUMP,         // 减速带
    HMI_STATIC_OTHERS,             // 其他
    HMI_STATIC_COLUMN = 8,         // 柱子
    HMI_STATIC_REFLECTIVE_COLUMN,  // 反光立柱
    HMI_STATIC_OPENED_GROUNDLOCK,  // 地锁开
    HMI_STATIC_CLOSED_GROUNDLOCK,  // 地锁关
    HMI_STATIC_WHEELS,             // 轮档
    HMI_STATIC_STOP_LINE,          // 停止线
    HMI_STATIC_ZEBRA,              // 斑马线
    HMI_STATIC_D_ARROW,            // 直行箭头
    HMI_STATIC_L_ARROW,            // 左转箭头
    HMI_STATIC_R_ARROW,            // 右转箭头
    HMI_STATIC_DL_ARROW,           // 直行左转
    HMI_STATIC_DR_ARROW,           // 直行右转
    HMI_STATIC_LR_TURN,            // 左右转
    HMI_STATIC_NO_L_TURN,          // 禁止左转
    HMI_STATIC_NO_R_TURN,          // 禁止右转
    HMI_STATIC_NO_TURN_AROUND,     // 禁止掉头
    HMI_STATIC_TRAFFICLIGHT,       // 红绿灯
};

enum class HmiBrakeLightStatus : uint8_t {
    HMI_UNKNOW,
    HMI_OFF,
    HMI_ON,
};

enum class HmiTurnLightStatus : uint8_t {
    HMI_LIGHT_UNKOWN,
    HMI_LIGHT_LEFT_FLASH,
    HMI_LIGHT_RIGHT_FLASH,
    HMI_LIGHT_OFF,
};

enum class HmiLaneClassType : uint8_t {
    HMI_DOTTED,                   // 单虚线
    HMI_SOLID,                    // 单实线
    HMI_LEFT_DOTTED_RIGHT_SOLID,  // 左侧虚线，右侧实线
    HMI_LEFT_SOLID_RIGHT_DOTTED,  // 左侧实线，右侧虚线
    HMI_DOUBLE_DOTTED,            // 双虚线
    HMI_DOUBLE_SOLID,             // 双实线

    HMI_TRIPLE_LINES,  // 三线
    HMI_ROAD_EDGE,     // 道路边缘
    HMI_BOLD_DASHED,   // 粗虚线
    HMI_FISH_BONE,     // 鱼骨线
};

enum class HmiLaneLineSequence : uint8_t {
    HMI_LEFT_ROAD_EDGE = 0,
    HMI_LEFT = 100,
    HMI_LEFT_LEFT = 99,
    HMI_RIGHT = 101,
    HMI_RIGHT_RIGHT = 102,
    HMI_RIGHT_ROAD_EDGE = 200,
};

enum class HmiLaneColor : uint8_t {
    HMI_WHITE = 0,
    HMI_YELLOW = 1,
    HMI_ORANGE = 2,
    HMI_BLUE = 3,
    HMI_GREEN = 4,
    HMI_GRAY = 5,
    HMI_LEFT_WHITE_RIGHT_YELLOW = 6,
    HMI_LEFT_YELLOW_RIGHT_WHITE = 7,
    HMI_OTHERS = 8,
};
}  // namespace extra
}  // namespace netaos
}  // namespace hozon
