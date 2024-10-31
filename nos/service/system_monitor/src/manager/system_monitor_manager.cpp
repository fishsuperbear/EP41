#include "system_monitor/include/manager/system_monitor_manager.h"
#include "system_monitor/include/common/system_monitor_config.h"
#include "system_monitor/include/common/system_monitor_logger.h"
#include "system_monitor/include/monitor/system_monitor_cpu_monitor.h"
#include "system_monitor/include/monitor/system_monitor_mnand_hs_monitor.h"
#include "system_monitor/include/monitor/system_monitor_temp_monitor.h"
#include "system_monitor/include/monitor/system_monitor_disk_monitor.h"
#include "system_monitor/include/monitor/system_monitor_mem_monitor.h"
#include "system_monitor/include/monitor/system_monitor_filesystem_monitor.h"
#include "system_monitor/include/monitor/system_monitor_voltage_monitor.h"
#include "system_monitor/include/monitor/system_monitor_process_monitor.h"
#include "system_monitor/include/monitor/system_monitor_network_monitor.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

SystemMonitorManager* SystemMonitorManager::instance_ = nullptr;
std::mutex SystemMonitorManager::mtx_;

SystemMonitorManager::SystemMonitorManager()
{
}

SystemMonitorManager*
SystemMonitorManager::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new SystemMonitorManager();
        }
    }

    return instance_;
}

void
SystemMonitorManager::Init()
{
    STMM_INFO << "SystemMonitorManager::Init";
    SystemMonitorConfigInfo configInfo = SystemMonitorConfig::getInstance()->GetSystemMonitorConfigInfo();
    monitor_service_ptr_map_.insert(std::make_pair(SystemMonitorSubFunctionId::kCpuMonitor, new SystemMonitorCpuMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kCpuMonitor])));
    monitor_service_ptr_map_.insert(std::make_pair(SystemMonitorSubFunctionId::kEmmcMonitor, new SystemMonitorMnandHsMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kEmmcMonitor])));
    monitor_service_ptr_map_.insert(std::make_pair(SystemMonitorSubFunctionId::kTemperatureMonitor, new SystemMonitorTempMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kTemperatureMonitor])));
    monitor_service_ptr_map_.insert(std::make_pair(SystemMonitorSubFunctionId::kDiskMonitor, new SystemMonitorDiskMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kDiskMonitor])));
    monitor_service_ptr_map_.insert(std::make_pair(SystemMonitorSubFunctionId::kMemMonitor, new SystemMonitorMemMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kMemMonitor])));
    monitor_service_ptr_map_.insert(std::make_pair(SystemMonitorSubFunctionId::kFileSystemMonitor, new SystemMonitorFileSystemMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kFileSystemMonitor])));
    monitor_service_ptr_map_.insert(std::make_pair(SystemMonitorSubFunctionId::kVoltageMonitor, new SystemMonitorVoltageMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kVoltageMonitor])));
    monitor_service_ptr_map_.insert(std::make_pair(SystemMonitorSubFunctionId::kProcessMonitor, new SystemMonitorProcessMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kProcessMonitor])));
    monitor_service_ptr_map_.insert(std::make_pair(SystemMonitorSubFunctionId::kNetworkMonitor, new SystemMonitorNetworkMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kNetworkMonitor])));

    Start();
}

void
SystemMonitorManager::DeInit()
{
    STMM_INFO << "SystemMonitorManager::DeInit";
    Stop();
    for (auto& item : monitor_service_ptr_map_) {
        if (nullptr != item.second) {
            delete item.second;
            item.second = nullptr;
        }
    }

    monitor_service_ptr_map_.clear();
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
SystemMonitorManager::ControlEvent(const SystemMonitorControlEventInfo& info)
{
    STMM_INFO << "SystemMonitorManager::ControlEvent monitor_id: " << info.id;
    if (SystemMonitorSubFunctionId::kAllMonitor == info.id) {
        for (auto& item : monitor_service_ptr_map_) {
            if (nullptr != item.second) {
                item.second->Control(info.type, info.value);
            }
        }
    }
    else {
        if (nullptr != monitor_service_ptr_map_[info.id]) {
            monitor_service_ptr_map_[info.id]->Control(info.type, info.value);
        }
    }
}

void
SystemMonitorManager::RefreshEvent(const std::string& reason)
{
    STMM_INFO << "SystemMonitorManager::RefreshEvent";
    for (auto& item : monitor_service_ptr_map_) {
        if (nullptr != item.second) {
            item.second->RefreshFile(reason);
        }
    }
}

void
SystemMonitorManager::Start()
{
    STMM_INFO << "SystemMonitorManager::Start";
    const SystemMonitorConfigInfo& configInfo = SystemMonitorConfig::getInstance()->GetSystemMonitorConfigInfo();
    for (auto& item : configInfo.subFunction) {
        if (nullptr != monitor_service_ptr_map_[item.first]) {
            if ("on" == item.second.monitorSwitch) {
                monitor_service_ptr_map_[item.first]->Start();
            }
        }
    }
}

void
SystemMonitorManager::Stop()
{
    STMM_INFO << "SystemMonitorManager::Stop";
    for (auto& item : monitor_service_ptr_map_) {
        if (nullptr != item.second) {
            item.second->Stop();
        }
    }
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon