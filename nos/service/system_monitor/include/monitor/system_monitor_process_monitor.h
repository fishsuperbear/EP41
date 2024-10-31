#ifndef SYSTEM_MONITOR_PROCESS_MONITOR_H
#define SYSTEM_MONITOR_PROCESS_MONITOR_H

#include <inttypes.h>
#include <string>
#include <vector>
#include "system_monitor/include/monitor/system_monitor_base.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

class SystemMonitorProcessMonitor : public SystemMonitorBase {
public:
    SystemMonitorProcessMonitor(const SystemMonitorSubFunctionInfo& funcInfo);
    virtual ~SystemMonitorProcessMonitor();

    virtual void Start();
    virtual void Stop();

private:
    bool GetProcessStatus();

private:
    bool stop_flag_;
    // unordered_map<processName, processRunStatus>
    std::unordered_map<std::string, bool> process_run_status_map_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_PROCESS_MONITOR_H
