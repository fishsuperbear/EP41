#ifndef SYSTEM_MONITOR_CPU_MONITOR_H
#define SYSTEM_MONITOR_CPU_MONITOR_H

#include <inttypes.h>
#include <vector>
#include "system_monitor/include/monitor/system_monitor_base.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

typedef struct {
    float usr;
    float nice;
    float sys;
    float iowait;
    float irq;
    float soft;
    float steal;
    float guest;
    float gnice;
    float idle;
} CpuInfo;

struct PidCpuInfo {
    uint32_t uid;
    uint32_t pid;
    float usr;
    float system;
    float guest;
    float wait;
    float cpu;
    uint32_t cpuName;
    char command[100];
};

typedef struct {
    CpuInfo cpu_all;
    std::vector<CpuInfo> cup_details;
} CpuStatus;

class SystemMonitorCpuMonitor : public SystemMonitorBase {
public:
    SystemMonitorCpuMonitor(const SystemMonitorSubFunctionInfo& funcInfo);
    virtual ~SystemMonitorCpuMonitor();

    virtual void Start();
    virtual void Stop();

private:
    int32_t GetCpuStatus(CpuStatus *pCpu);
    int32_t GetPidCpuStatus(std::vector<PidCpuInfo>& pidStatus);

private:
    bool stop_flag_;
    uint32_t alarm_count_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_CPU_MONITOR_H
