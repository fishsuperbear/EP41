
#ifndef PHM_HEALTH_MONITOR_H
#define PHM_HEALTH_MONITOR_H

#include <mutex>
#include <thread>
#include "phm_server/include/common/phm_server_def.h"

namespace hozon {
namespace netaos {
namespace phm_server {

class PhmHealthGdbMonitor;
class HealthMonitor {

public:
    HealthMonitor();
    ~HealthMonitor();

    void StartMonitor();
    void StopMonitor();


private:
    HealthMonitor(const HealthMonitor &);
    HealthMonitor & operator = (const HealthMonitor &);

    void Run();
    void SendProcFault(const std::string& procName, const uint8_t faultStatus);

private:
    bool is_start;
    std::thread work_thread_;
    std::unique_ptr<PhmHealthGdbMonitor> m_upPhmHealthResourcesMonitor;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_HEALTH_MONITOR_H