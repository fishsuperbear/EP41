#include <thread>
#include "phm_server/include/phm_server.h"
#include "phm_server/include/common/function_statistics.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/fault_manager/manager/phm_fault_manager.h"
#include "phm_server/include/health_manager/manager/phm_health_manager.h"
#include "phm_server/include/common/phm_server_persistency.h"
#include "phm_server/include/fault_lock/phm_fault_lock.h"

namespace hozon {
namespace netaos {
namespace phm_server {

PhmServer::PhmServer()
: stop_flag_(false)
{
}

PhmServer::~PhmServer()
{
}

void
PhmServer::Init()
{
    FunctionStatistics func("PhmServer::Init");
    PHMServerConfig::getInstance()->Init();
    // PHMServerPersistency::getInstance()->Init();
    FaultManager::getInstance()->Init();
    HealthManager::getInstance()->Init();
    // FaultLock::getInstance()->Init();
}

void
PhmServer::DeInit()
{
    FunctionStatistics func("PhmServer::DeInit");
    // FaultLock::getInstance()->DeInit();
    HealthManager::getInstance()->DeInit();
    FaultManager::getInstance()->DeInit();
    // PHMServerPersistency::getInstance()->DeInit();
    PHMServerConfig::getInstance()->DeInit();
}

void
PhmServer::Run()
{
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    DeInit();
}

void
PhmServer::Stop()
{
    stop_flag_ = true;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon