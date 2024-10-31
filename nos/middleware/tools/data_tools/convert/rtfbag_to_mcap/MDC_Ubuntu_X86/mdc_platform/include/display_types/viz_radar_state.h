/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: 可视化内置结构：Radar state
 */

#ifndef VIZ_RADAR_STATE_H
#define VIZ_RADAR_STATE_H
#include <cstdint>

namespace mdc {
namespace visual {
enum MovProperty {
    MOVING_PROPERTY_MOVING = 0, // 移动
    MOVING_PROPERTY_STATIONARY = 1, // 静止
    MOVING_PROPERTY_ONCOMING = 2, // 靠近中
    MOVING_PROPERTY_STATIONARY_CANDIDATE = 3, // 静止候选
    MOVING_PROPERTY_UNKOWN = 4, // 未知
    MOVING_PROPERTY_CROSSING_STATIONARY = 5, // 横穿静止
    MOVING_PROPERTY_CROSSING_MOVING = 6, // 横穿移动
    MOVING_PROPERTY_STOPPED = 7, // 停止
};

enum Ambiguity {
    AMBIG_STATE_INVALID = 0, // 无效
    AMBIG_STATE_AMBIGUOUS = 1, // 模糊
    AMBIG_STATE_STAGGERED_RAMP = 2, // 不明确
    AMBIG_STATE_UNAMBIGUOUS = 3, // 明确
    AMBIG_STATE_STATIONARY_CANDIDATES = 4, // 确定
};

enum InvalidState {
    INVALID_STATE_VALID = 0, // 无效
    INVALID_STATE_INVALID_LOW_RCS = 1, // 由于RCS较低而无效
    INVALID_STATE_INVALID_NEAR_FIELD_ARTEFACT = 2, // 由于近处区域虚假现象而无效
    INVALID_STATE_INVALID_NOT_CONFIRMED_IN_RANGE = 3, // 远处的Cluster无效，因为未在近距离内确认
    INVALID_STATE_VALID_LOW_RCS = 4, // 具有低RCS的有效Cluster
    INVALID_STATE_RESERVED_1 = 5, // 预留
    INVALID_STATE_INVALID_MIRROR = 6, // 由于高镜像概率而导致无效的Cluster
    INVALID_STATE_INVALID_OUTSIDE_FOV = 7, // 因外部传感器视野无效
    INVALID_STATE_VALID_AZIMUTH_CORRECTION = 8, // 由于高度而具有方位角校正的有效Cluster
    INVALID_STATE_VALID_CHILD = 9, // 具有多点高概率的有效Cluster
    INVALID_STATE_VALID_50DEG_ARTEFACT = 10, // 具有50%虚假目标高概率的有效Cluster
    INVALID_STATE_VALID_NO_LOCAL_MAXIMUM = 11, // 没有最大值的有效Cluster
    INVALID_STATE_VALID_ARTEFACT = 12, // 具有高虚假目标概率的有效聚类
    INVALID_STATE_RESERVED_2 = 13, // 预留
    INVALID_STATE_INVALID_HARMONICS = 14, // 由于谐波导致的无效Cluster
    INVALID_STATE_VALID_ABOVE_95M_IN_NEAR_RANGE = 15, // 有效Cluster在近范围内超过95米
    INVALID_STATE_VALID_MULTITARGET = 16, // 具有高多目标概率的有效Cluster
    INVALID_STATE_VALID_SUSPICIOUS_ANGLE = 17, // 具有可疑角度的有效Cluster
};

enum TrackState {
    TRACK_STATE_DELETED = 0, // 已删除，最后一次上报
    TRACK_STATE_INITED = 1, // 新生
    TRACK_STATE_MEASURED = 2, // 已确认
    TRACK_STATE_PREDICTED = 3, // 未确认，跟踪信息不准确
    TRACK_STATE_DELETED_FOR_MERGE = 4, // 已删除，和其他目标合并
    TRACK_STATE_NEW_FOR_MERGE = 5, // 合并后新生目标
};

enum TrackType {
    TRACK_TYPE_POINT = 0, // 点目标
    TRACK_TYPE_CAR = 1, // 汽车
    TRACK_TYPE_TRUCK = 2, // 卡车
    TRACK_TYPE_PEDESTRIAN = 3, // 行人
    TRACK_TYPE_MOTORCYCLE = 4, // 摩托车
    TRACK_TYPE_BICYCLE = 5, // 自行车
    TRACK_TYPE_WIDE = 6, // 宽目标
    TRACK_TYPE_RESERVED = 7, // 保留
    TRACK_TYPE_BRIDGE = 8, // 桥梁
    TRACK_TYPE_NOT_IN_USE = 9, // 未使用
};

struct RadarState {
    uint8_t sensorId;
    uint8_t nvmReadStatus;
    uint8_t nvmWriteStatus;
    uint8_t persistentError;
    uint8_t temporaryError;
    uint8_t interferenceError;
    uint8_t temperatureError;
    uint8_t voltageError;
    uint16_t maxDistance;
    uint8_t sortIndex;
    uint8_t radarPower;
    uint8_t ctrlRelay;
    uint8_t outputType;
    uint8_t sendQuality;
    uint8_t sendExtinfo;
    uint8_t motionRxState;
    uint8_t rcsThreshold;
    uint8_t blockError;
    uint8_t broadcastError;
    uint8_t electricAxisError;
    uint8_t configError;
    uint8_t calibrationError;
    uint8_t connectorDirection;
    uint8_t can0WorkMode;
    uint8_t can1WorkMode;
    uint8_t dualCanMode;
    uint8_t timmingMode;
    uint8_t powerMode;
    uint8_t performanceMode;
    uint8_t radarPosition;
    uint8_t hwError;

    RadarState()
        : sensorId(0U), nvmReadStatus(0U), nvmWriteStatus(0U), persistentError(0U), temporaryError(0U),
          interferenceError(0U), temperatureError(0U), voltageError(0U), maxDistance(0U), sortIndex(0U),
          radarPower(0U), ctrlRelay(0U), outputType(0U), sendQuality(0U), sendExtinfo(0U), motionRxState(0U),
          rcsThreshold(0U), blockError(0U), broadcastError(0U), electricAxisError(0U), configError(0U),
          calibrationError(0U), connectorDirection(0U), can0WorkMode(0U), can1WorkMode(0U), dualCanMode(0U),
          timmingMode(0U), powerMode(0U), performanceMode(0U), radarPosition(0U), hwError(0U) {}
    RadarState(const uint8_t& vSensorId, const uint8_t& vNvmReadStatus, const uint8_t& vNvmWriteStatus,
        const uint8_t& vPersistentError, const uint8_t& vTemporaryError, const uint8_t& vInterferenceError,
        const uint8_t& vTemperatureError, const uint8_t& vVoltageError, const uint16_t& vMaxDistance,
        const uint8_t& vSortIndex, const uint8_t& vRadarPower, const uint8_t& vCtrlRelay, const uint8_t& vOutputType,
        const uint8_t& vSendQuality, const uint8_t& vSendExtinfo, const uint8_t& vMotionRxState,
        const uint8_t& vRcsThreshold, const uint8_t& vBlockError, const uint8_t& vBroadcastError,
        const uint8_t& vElectricAxisError, const uint8_t& vConfigError, const uint8_t& vCalibrationError,
        const uint8_t& vConnectorDirection, const uint8_t& vCan0UWorkMode, const uint8_t& vCan1WorkMode,
        const uint8_t& vDualCanMode, const uint8_t& vTimmingMode, const uint8_t& vPowerMode,
        const uint8_t& vPerformanceMode, const uint8_t& vRadarPosition, const uint8_t& vHwError)
        : sensorId(vSensorId), nvmReadStatus(vNvmReadStatus), nvmWriteStatus(vNvmWriteStatus),
          persistentError(vPersistentError), temporaryError(vTemporaryError), interferenceError(vInterferenceError),
          temperatureError(vTemperatureError), voltageError(vVoltageError), maxDistance(vMaxDistance),
          sortIndex(vSortIndex), radarPower(vRadarPower), ctrlRelay(vCtrlRelay), outputType(vOutputType),
          sendQuality(vSendQuality), sendExtinfo(vSendExtinfo), motionRxState(vMotionRxState),
          rcsThreshold(vRcsThreshold), blockError(vBlockError), broadcastError(vBroadcastError),
          electricAxisError(vElectricAxisError), configError(vConfigError), calibrationError(vCalibrationError),
          connectorDirection(vConnectorDirection), can0WorkMode(vCan0UWorkMode), can1WorkMode(vCan1WorkMode),
          dualCanMode(vDualCanMode), timmingMode(vTimmingMode), powerMode(vPowerMode),
          performanceMode(vPerformanceMode), radarPosition(vRadarPosition), hwError(vHwError) {}
};
}
}
#endif // VIZ_RADAR_STATE_H
