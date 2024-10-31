/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: phm server config
 */

#ifndef PHM_SERVER_CONFIG_H
#define PHM_SERVER_CONFIG_H

#include <mutex>
#include <unordered_map>

#include "phm_server/include/common/phm_server_def.h"

namespace hozon {
namespace netaos {
namespace phm_server {

class PHMServerConfig {
public:
    static PHMServerConfig* getInstance();

    void Init();
    void DeInit();

    void LoadPhmConfig();
    const PhmConfigInfo& GetPhmConfigInfo() {return phm_config_info_;}
    bool GetFaultInfoByFault(const uint32_t fault, FaultInfo& faultInfo);
    uint32_t GetDtcByFault(const uint32_t fault);
    uint32_t GetFaultByDtc(const uint32_t dtc);

    // process config
    bool GetProcInfoByName(const std::string procName, ProcInfo& procInfo);
    std::string GetProcInfoByFaultKey(const uint32_t faultId, const uint32_t faultObj);
    uint8_t getProcRetryCountByName(const std::string& procName);

    bool IsOverMaxRecordCount(const uint32_t fault);
    void UpdateFaultRecordCount(const uint32_t fault);
    void UpdateFaultStatus(const uint32_t fault, const Fault_t& faultInfo);
    void QueryCurrentOccuredFault(const uint32_t fault, std::vector<uint32_t>& faultList);
    void QueryAllOccuredFault(std::vector<uint32_t>& faultList);
    void QueryPrintConfigData();
    bool getFaultCluster(const uint32_t faultKey, std::vector<FaultClusterItem>& outCluster);

    // Fault lock interface
    std::string GetLockFaultToHMISwitch();
    std::string GetLockFaultCurrentVersion();
    bool IsBlockedFault(uint32_t fault);
    bool GetLockFaultInfo(uint32_t fault, FaultLockInfo &faultLockInfo);
    void GetLockFaultInfos(std::vector<FaultLockInfo> &faultLockInfos);

    void SetLockFaultInfo(uint32_t fault, FaultLockInfo &faultLockInfo);
    void SetLockFaultInfo(uint32_t fault, uint32_t lockedNumber);
    void LockFaultDataToFile();

private:
    char* GetJsonAll(const char *fname);
    int32_t LoadConfig();
    int32_t ParseFaultJson();
    int32_t ParseProcJson();
    int32_t ParseFaultClusterJson();
    int32_t ParseFaultLockJson();

private:
    PHMServerConfig();
    PHMServerConfig(const PHMServerConfig &);
    PHMServerConfig & operator = (const PHMServerConfig &);

private:
    static std::mutex mtx_;
    static PHMServerConfig* instance_;

    // phm config info
    PhmConfigInfo phm_config_info_;

    // unordered_map<faultKey, FaultInfo>
    std::unordered_map<uint32_t, FaultInfo> fault_info_map_;

    // unordered_map<procName, ProcInfo>
    std::unordered_map<std::string, ProcInfo> proc_info_map_;

    // unordered_map<dtcId, faultKey>
    std::unordered_map<uint32_t, uint32_t> dtc_fault_info_map_;

    // unordered_map<clusterName, levelvec>
    std::unordered_map<std::string, std::vector<uint32_t>> phm_cluster_maxlevel_map_;

    // unordered_map<fault, FaultClusterItem>
    std::unordered_map<uint32_t, std::vector<FaultClusterItem>> phm_fault_cluster_level_map_;

    // unordered_map<fault, FaultLockInfo>
    std::unordered_map<uint32_t, FaultLockInfo> fault_lock_info_map_;
    bool file_change_flag_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_SERVER_CONFIG_H
