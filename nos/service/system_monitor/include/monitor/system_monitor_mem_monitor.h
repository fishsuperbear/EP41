#ifndef SYSTEM_MONITOR_MEM_MONITOR_H
#define SYSTEM_MONITOR_MEM_MONITOR_H

#include <inttypes.h>
#include <string>
#include <vector>
#include "system_monitor/include/monitor/system_monitor_base.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

struct MemInfo {
    uint32_t total;
    uint32_t used;
    uint32_t free;
    uint32_t shared;
    uint32_t available;
};

struct PidMemInfo {
    uint32_t uid;
    uint32_t pid;
    float minflt;
    float majflt;
    uint32_t vsz;
    uint32_t rss;
    float mem;
    char command[100];
};

class SystemMonitorMemMonitor : public SystemMonitorBase {
public:
    SystemMonitorMemMonitor(const SystemMonitorSubFunctionInfo& funcInfo);
    virtual ~SystemMonitorMemMonitor();

    virtual void Start();
    virtual void Stop();

private:
    int32_t GetMemStatus(MemInfo& memStatus);
    int32_t GetPidMemStatus(std::vector<PidMemInfo>& pidStatus);

    static std::string Replace(std::string str, const std::string& pattern, const std::string& to);

private:
    bool stop_flag_;
    uint32_t alarm_count_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_MEM_MONITOR_H
