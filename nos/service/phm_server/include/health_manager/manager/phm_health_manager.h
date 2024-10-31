
#ifndef PHM_HEALTH_MANAGER_H
#define PHM_HEALTH_MANAGER_H

#include <mutex>
#include "phm_server/include/health_manager/monitor/phm_health_monitor.h"

namespace hozon {
namespace netaos {
namespace phm_server {

class HealthManager {

public:
    static HealthManager* getInstance();

    void Init();
    void DeInit();

private:
    HealthManager();
    HealthManager(const HealthManager &);
    HealthManager & operator = (const HealthManager &);

private:
    static std::mutex mtx_;
    static HealthManager* instance_;

    std::unique_ptr<HealthMonitor> health_monitor_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_HEALTH_MANAGER_H