/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: system monitor info
 */

#ifndef SYSTEM_MONITOR_INFO
#define SYSTEM_MONITOR_INFO

#include <mutex>
#include "system_monitor_info_def.h"
#include "devm/include/devm_device_info.h"

using namespace hozon::netaos::devm;

class SystemMonitorInfo {
public:
    static SystemMonitorInfo* getInstance();

    void Init();
    void DeInit();

    std::string GetMonitorInfo(const std::string& monitorType);
    std::string GetMonitorInfoFromFile(const std::string& monitorType);

private:
    SystemMonitorInfo();
    SystemMonitorInfo(const SystemMonitorInfo &);
    SystemMonitorInfo & operator = (const SystemMonitorInfo &);

protected:
    std::string GetCpuInfo();
    std::string GetDiskInfo();
    std::string GetMemInfo();
    std::string GetNetWorkInfo();
    std::string GetProcessInfo();
    std::string GetTemperatureInfo();
    std::string GetVoltageInfo();

private:
    char* GetJsonAll(const char *fname);
    void ParseSystemMonitorConfigJson();

    int32_t GetCpuStatus(CpuStatus *pCpu);
    int32_t GetPidCpuStatus(std::vector<PidCpuInfo>& pidStatus);
    int32_t GetDiskStatus(std::vector<DiskInfo>& diskStatus);
    int32_t GetMemStatus(MemInfo& memStatus);
    int32_t GetPidMemStatus(std::vector<PidMemInfo>& pidStatus);
    std::string GetNetWorkStatus(const std::string& nic);
    void GetProcessStatus();
    void GetTempInfo(TemperatureData& tempInfo);
    void GetVoltage(VoltageData& voltage);

    static std::string Popen(const std::string& cmd, const bool eFlag = false);
    static std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr);
    static std::string Replace(std::string str, const std::string& pattern, const std::string& to);
    static std::string GetCurrentTime();

private:
    static std::mutex mtx_;
    static SystemMonitorInfo* instance_;

    std::vector<std::string> network_nic_list_;
    std::unordered_map<std::string, bool> process_run_status_map_;

    std::shared_ptr<DevmClientDeviceInfo> device_info_;
};

#endif  // SYSTEM_MONITOR_INFO