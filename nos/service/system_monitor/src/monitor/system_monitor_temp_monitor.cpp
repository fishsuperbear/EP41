#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include <thread>
#include <fstream>
#include "system_monitor/include/monitor/system_monitor_temp_monitor.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

const std::string TEMP_MONITOR_RECORD_FILE_NAME = "temp_monitor.log";

SystemMonitorTempMonitor::SystemMonitorTempMonitor(const SystemMonitorSubFunctionInfo& funcInfo)
: SystemMonitorBase(funcInfo, TEMP_MONITOR_RECORD_FILE_NAME)
, stop_flag_(false)
, device_info_(std::make_shared<DevmClientDeviceInfo>())
{
}

SystemMonitorTempMonitor::~SystemMonitorTempMonitor()
{
}

void
SystemMonitorTempMonitor::Start()
{
    STMM_INFO << "SystemMonitorTempMonitor::Start";
    stop_flag_ = false;
    if (GetRecordFileCycle()) {
        StartRecord();
    }

    std::thread stmm_temp([this]() {
        while (!stop_flag_) {
            TemperatureData tempInfo{0};
            int32_t res = GetTempInfo(tempInfo);
            if (0 != res) {
                STMM_WARN << "SystemMonitorTempMonitor get temp failed. failcode: " << res;
                std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
                continue;
            }

            std::ostringstream stream;
            stream << "Current temperature:\n\t temp_soc: " << std::fixed << std::setprecision(2) << tempInfo.temp_soc << "℃"
                                       << "\n\t temp_mcu: " << std::fixed << std::setprecision(2) << tempInfo.temp_mcu << "℃"
                                       << "\n\t temp_ext0: " << std::fixed << std::setprecision(2) << tempInfo.temp_ext0 << "℃"
                                       << "\n\t temp_ext1: " << std::fixed << std::setprecision(2) << tempInfo.temp_ext1 << "℃\n";
            std::string notifyStr = stream.str();
            Notify(notifyStr);
            SetRecordStr(notifyStr);
            std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
        }
    });

    pthread_setname_np(stmm_temp.native_handle(), "stmm_temp");
    stmm_temp.detach();
}

void
SystemMonitorTempMonitor::Stop()
{
    STMM_INFO << "SystemMonitorTempMonitor::Stop";
    stop_flag_ = true;
    StopRecord();
}

int32_t
SystemMonitorTempMonitor::GetTempInfo(TemperatureData& tempInfo)
{
    if (nullptr == device_info_) {
        return -1;
    }

    if (!device_info_->GetTemperature(tempInfo))
    {
        return -2;
    }

    return 0;
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
