#ifndef SYSTEM_MONITOR_NETWORK_MONITOR_H
#define SYSTEM_MONITOR_NETWORK_MONITOR_H

#include <inttypes.h>
#include <string>
#include <vector>
#include "system_monitor/include/monitor/system_monitor_base.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

struct NetworkInfo {
    std::string notifyStr;
    uint32_t rxErrors;
    uint32_t txErrors;
};

class SystemMonitorNetworkMonitor : public SystemMonitorBase {
public:
    SystemMonitorNetworkMonitor(const SystemMonitorSubFunctionInfo& funcInfo);
    virtual ~SystemMonitorNetworkMonitor();

    virtual void Start();
    virtual void Stop();

private:
    void GetNetWorkStatus(const bool start);
    void GetNetWorkRxAndTxErrors(const std::string& info, uint32_t& rxErrors, uint32_t& txErrors);

    static std::string Popen(const std::string& cmd, const bool eFlag = false);
    static std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr);

private:
    bool stop_flag_;
    // unordered_map<nic, networkinfo>
    std::unordered_map<std::string, NetworkInfo> network_nic_info_map_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_NETWORK_MONITOR_H
