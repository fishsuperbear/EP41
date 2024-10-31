#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include <thread>
#include <fstream>
#include "system_monitor/include/monitor/system_monitor_voltage_monitor.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

const std::string VOLTAGE_MONITOR_RECORD_FILE_NAME = "voltage_monitor.log";

SystemMonitorVoltageMonitor::SystemMonitorVoltageMonitor(const SystemMonitorSubFunctionInfo& funcInfo)
: SystemMonitorBase(funcInfo, VOLTAGE_MONITOR_RECORD_FILE_NAME)
, stop_flag_(false)
, device_info_(std::make_shared<DevmClientDeviceInfo>())
{
}

SystemMonitorVoltageMonitor::~SystemMonitorVoltageMonitor()
{
}

void
SystemMonitorVoltageMonitor::Start()
{
    STMM_INFO << "SystemMonitorVoltageMonitor::Start";
    stop_flag_ = false;
    if (GetRecordFileCycle()) {
        StartRecord();
    }

    std::thread stmm_voltage([this]() {
        while (!stop_flag_) {
            VoltageData voltage {0};
            int32_t res = GetVoltage(voltage);
            if (0 != res) {
                STMM_WARN << "SystemMonitorVoltageMonitor get voltage failed. failcode: " << res;
                std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
                continue;
            }

            std::ostringstream stream;
            stream << "Current kl15 voltage: " << ((0 == voltage.kl15) ? "ACCOFF" : "ACCON") << "\n";
            stream << "Current kl30 voltage: " << std::fixed << std::setprecision(2) << voltage.kl30 << "V\n";
            std::string notifyStr = stream.str();
            Notify(notifyStr);
            SetRecordStr(notifyStr);
            std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
        }
    });

    pthread_setname_np(stmm_voltage.native_handle(), "stmm_voltage");
    stmm_voltage.detach();
}

void
SystemMonitorVoltageMonitor::Stop()
{
    STMM_INFO << "SystemMonitorVoltageMonitor::Stop";
    stop_flag_ = true;
    StopRecord();
}

int32_t
SystemMonitorVoltageMonitor::GetVoltage(VoltageData& voltage)
{
    if (nullptr == device_info_) {
        return -1;
    }

    if (!device_info_->GetVoltage(voltage))
    {
        return -2;
    }

    return 0;
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
