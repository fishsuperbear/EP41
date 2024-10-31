#ifndef SYSTEM_MONITOR_MNAND_HS_MONITOR_H
#define SYSTEM_MONITOR_MNAND_HS_MONITOR_H

#include <inttypes.h>
#include <vector>
#include "system_monitor/include/util/nvmnand.h"
#include "system_monitor/include/monitor/system_monitor_base.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

struct MnandHsInfo {
    std::string type;
    std::string node;
    std::string preEolInfo;
    std::string deviceLifeTimeEstA;
    std::string deviceLifeTimeEstB;
};

class SystemMonitorMnandHsMonitor : public SystemMonitorBase {
public:
    SystemMonitorMnandHsMonitor(const SystemMonitorSubFunctionInfo& funcInfo);
    virtual ~SystemMonitorMnandHsMonitor();

    virtual void Start();
    virtual void Stop();

private:
    void GetUfsStatus(std::vector<MnandHsInfo>& ufsStatus);
    void GetEmmcStatus(std::vector<MnandHsInfo>& emmcStatus);
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_MNAND_HS_MONITOR_H
