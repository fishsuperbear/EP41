#include "phm_server/include/health_manager/manager/phm_health_manager.h"
#include "phm_server/include/common/phm_server_logger.h"

namespace hozon {
namespace netaos {
namespace phm_server {

HealthManager* HealthManager::instance_ = nullptr;
std::mutex HealthManager::mtx_;

HealthManager*
HealthManager::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new HealthManager();
        }
    }

    return instance_;
}

HealthManager::HealthManager()
{
}

void
HealthManager::Init()
{
    PHMS_INFO << "HealthManager::Init";
    health_monitor_ = std::make_unique<HealthMonitor>();
    health_monitor_->StartMonitor();
}

void
HealthManager::DeInit()
{
    PHMS_INFO << "HealthManager::DeInit";
    health_monitor_->StopMonitor();
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
