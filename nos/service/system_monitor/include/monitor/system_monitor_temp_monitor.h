#ifndef SYSTEM_MONITOR_TEMP_MONITOR_H
#define SYSTEM_MONITOR_TEMP_MONITOR_H

#include <inttypes.h>
#include <vector>
#include "system_monitor/include/monitor/system_monitor_base.h"
#include "devm/include/devm_device_info.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

using namespace hozon::netaos::devm;

class SystemMonitorTempMonitor : public SystemMonitorBase {
public:
    SystemMonitorTempMonitor(const SystemMonitorSubFunctionInfo& funcInfo);
    virtual ~SystemMonitorTempMonitor();

    virtual void Start();
    virtual void Stop();

private:
    int32_t GetTempInfo(TemperatureData& tempInfo);
    float GetTemperature(const std::string& tempPath);

private:
    bool stop_flag_;

    std::shared_ptr<DevmClientDeviceInfo> device_info_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_TEMP_MONITOR_H
