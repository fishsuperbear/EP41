#include <thread>

#include "system_monitor/include/system_monitor.h"
#include "system_monitor/include/common/system_monitor_config.h"
#include "system_monitor/include/handler/system_monitor_handler.h"
#include "system_monitor/include/manager/system_monitor_manager.h"
#include "system_monitor/include/common/system_monitor_logger.h"
#include "system_monitor/include/common/function_statistics.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

SystemMonitor::SystemMonitor()
: stop_flag_(false)
{
}

SystemMonitor::~SystemMonitor()
{
}

void
SystemMonitor::Init()
{
    STMM_INFO << "SystemMonitor::Init";
    FunctionStatistics func("SystemMonitor::Init finish, ");
    SystemMonitorConfig::getInstance()->Init();
    SystemMonitorHandler::getInstance()->Init();
    SystemMonitorManager::getInstance()->Init();
}

void
SystemMonitor::DeInit()
{
    STMM_INFO << "SystemMonitor::DeInit";
    FunctionStatistics func("SystemMonitor::DeInit finish, ");
    SystemMonitorManager::getInstance()->DeInit();
    SystemMonitorHandler::getInstance()->DeInit();
    SystemMonitorConfig::getInstance()->DeInit();
}

void
SystemMonitor::Run()
{
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    DeInit();
}

void
SystemMonitor::Stop()
{
    stop_flag_ = true;
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon