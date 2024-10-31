
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 配置管理数据类型定义
 * Created on: Feb 7, 2023
 *
 */

#ifndef SERVICE_CONFIG_SERVER_INCLUDE_CFG_SERVER_DATA_DEF_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_CFG_SERVER_DATA_DEF_H_
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "struct2x/struct2x.h"  // SERIALIZE

namespace hozon {
namespace netaos {
namespace cfg {
typedef struct MonitorClient {
    std::string MonitorClientName;
    CfgResultCode MonitorUpdateFlag;
    MonitorClient() {
        MonitorClientName.clear();
        MonitorUpdateFlag = CONFIG_OK;
    }
    template <typename T>
    void serialize(T& t) {
        SERIALIZE(t, MonitorClientName, MonitorUpdateFlag);
    }
} MonitorClient_t;
typedef struct ParamValueInfo {
    std::string paramValue;
    uint8_t dataType;
    uint32_t dataSize;
    std::string defaultparamValue;
    std::string lastupdateClientName;
    std::string lastupdateTime;
    std::vector<MonitorClient_t> monitorClientsVec;
    uint8_t perFlag;
    uint8_t storageFlag;
    ParamValueInfo() {
        paramValue.clear();
        dataType = 0;
        dataSize = 0;
        defaultparamValue.clear();
        lastupdateTime = "NONE";
        lastupdateClientName = "NONE";
        monitorClientsVec.clear();
        perFlag = 0;
        storageFlag = 0;
    }

    template <typename T>
    void serialize(T& t) {
        SERIALIZE(t, paramValue, dataType, dataSize, defaultparamValue, lastupdateClientName, lastupdateTime, monitorClientsVec, perFlag, storageFlag);
    }
} ParamValueInfo_t;

// debug struct
enum ConfigMethodEventType : std::uint8_t { CFG_SETPARAM, CFG_GETPARAM, CFG_NOTIFY };
struct ClientListInfo {
    std::vector<std::string> monitor_params;
    std::string start_time;
    std::string end_time;
    uint8_t running_status;  // 0 notconnect， 1 connected， 2 disconnect
    std::map<std::string, std::map<ConfigMethodEventType, int32_t>> methodevent_info;
    ClientListInfo() {
        monitor_params.clear();
        start_time.clear();
        end_time.clear();
        methodevent_info.clear();
        running_status = 0;
    }
};

struct CfgServerData {
    std::map<std::string, ParamValueInfo_t> cfgParamDataMap_;
    std::vector<std::string> connectClientsVec_;
    std::map<std::string, ClientListInfo> cfgClientInfoMap_;
    CfgServerData() {
        cfgParamDataMap_.clear();
        connectClientsVec_.clear();
        cfgClientInfoMap_.clear();
    }
    template <typename T>
    void serialize(T& t) {
        SERIALIZE(t, cfgParamDataMap_);
    }
};

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_CONFIG_SERVER_INCLUDE_CFG_SERVER_DATA_DEF_H_
