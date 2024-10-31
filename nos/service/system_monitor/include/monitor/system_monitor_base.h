
#ifndef SYSTEM_MONITOR_BASE_H
#define SYSTEM_MONITOR_BASE_H

#include <unordered_map>
#include "system_monitor/include/common/system_monitor_def.h"
#include "system_monitor/include/common/system_monitor_config.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

class SystemMonitorBase {

public:
    SystemMonitorBase(const SystemMonitorSubFunctionInfo& funcInfo, const std::string& fileName = "");
    virtual ~SystemMonitorBase();

    // monitor name, id
    std::string GetMonitorName();
    SystemMonitorSubFunctionId GetMonitorId();

    // monitor switch
    void SetMonitorSwitch(const std::string& monitorSwitch);
    std::string GetMonitorSwitch();

    // monitor cycle
    void SetMonitorCycle(const uint32_t monitorCycle);
    uint32_t GetMonitorCycle();

    // is record file
    void SetRecordFileCycle(const uint32_t recordFileCycle);
    uint32_t GetRecordFileCycle();

    // record file path
    void SetRecordFilePath(const std::string& recordFilePath);
    std::string GetRecordFilePath();

    // is alarm
    void SetIsAlarm(const bool isAlarm);
    bool GetIsAlarm();

    // alarm value
    void SetAlarmValue(const uint8_t alarmValue);
    uint8_t GetAlarmValue();

    // post processing switch
    void SetPostProcessingSwitch(const std::string& postProcessingSwitch);
    std::string GetPostProcessingSwitch();

    void StartRecord();
    void StopRecord();
    void SetRecordStr(const std::string& recordStr);
    void WriteDataToFile(const bool overwrite = false);
    void RefreshFile(const std::string& reason);
    bool CopyFile(const std::string& from, const std::string& to);

    void Notify(const std::string info);
    void Alarm(const std::string info);
    void ReportFault(const uint32_t fault, const uint8_t faultStatus);

    virtual void Start() = 0;
    virtual void Stop() = 0;
    virtual void Control(const SystemMonitorSubFunctionControlType& controlType, const std::string controlValue);

private:
    static std::string GetCurrentTime();

private:
    SystemMonitorBase(const SystemMonitorBase &);
    SystemMonitorBase & operator = (const SystemMonitorBase &);

private:
    SystemMonitorSubFunctionInfo sub_function_info_;
    std::string record_file_name_;
    std::string record_str_;
    bool record_stop_flag_;
    uint32_t record_size_;

    std::unordered_map<uint32_t, uint8_t> fault_status_map_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_BASE_H
