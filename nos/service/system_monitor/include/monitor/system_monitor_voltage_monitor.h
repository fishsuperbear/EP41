#ifndef SYSTEM_MONITOR_VOLTAGE_MONITOR_H
#define SYSTEM_MONITOR_VOLTAGE_MONITOR_H

#include <inttypes.h>
#include <string>
#include <vector>
#include "system_monitor/include/monitor/system_monitor_base.h"
#include "devm/include/devm_device_info.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

using namespace hozon::netaos::devm;

class SystemMonitorVoltageMonitor : public SystemMonitorBase {
public:
    SystemMonitorVoltageMonitor(const SystemMonitorSubFunctionInfo& funcInfo);
    virtual ~SystemMonitorVoltageMonitor();

    virtual void Start();
    virtual void Stop();

private:
    int32_t GetVoltage(VoltageData& voltage);

private:
    bool stop_flag_;

    std::shared_ptr<DevmClientDeviceInfo> device_info_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_VOLTAGE_MONITOR_H
