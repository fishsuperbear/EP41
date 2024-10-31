/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: phm def
 */

#ifndef PHM_COMMON_DEF_H
#define PHM_COMMON_DEF_H

#include <vector>
#include <string>
#include <atomic>

namespace hozon {
namespace netaos {
namespace phm {

typedef enum PhmMonitorType {
    PHM_MONITOR_TYPE_UNKNOWN = 0,
    PHM_MONITOR_TYPE_ALIVE,
    PHM_MONITOR_TYPE_DEADLINE,
    PHM_MONITOR_TYPE_LOGIC
} PhmMonitorType_t;

struct FaultClusterItem {
    std::string clusterName;
    uint8_t bitPosition;
};

struct FaultStrategy {
    uint8_t notifyMcu;
    uint8_t notifyApp;
    uint8_t restartproc;
};

struct FaultAction {
    uint8_t record;
    uint8_t analysis;
    uint8_t dtcMapping;
    FaultStrategy strategy;
};

struct PhmConfigInfo {
    std::atomic<bool> LoadFlag{false};
    std::string LogContextName{"PHM"};
    std::uint8_t LogLevel{3};
    std::string DebugSwitch{"off"};
    std::uint32_t SystemCheckTime{60000};
};

typedef struct PhmFaultInfo {
    uint32_t faultId;
    uint8_t faultObj;
    uint32_t faultClusterId;
    uint16_t faultLevel;
    FaultAction faultAction;
    std::string faultProcess;
    std::string faultDescribe;
    std::string faultDomain;
    uint64_t faultOccurTime;
    uint8_t faultStatus;
    uint32_t dtcCode;
} phm_fault_info_t;

typedef struct PhmTask {
    PhmMonitorType_t monitorType;
    std::vector<uint32_t> checkPointId;
    std::vector<uint32_t> parameter;
    uint32_t faultId;
    uint8_t faultObj;
} phm_task_t;

const std::string INNER_PREFIX = "inner";

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_COMMON_DEF_H
