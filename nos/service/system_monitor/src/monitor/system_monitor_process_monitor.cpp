#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include <thread>
#include <fstream>
#include "system_monitor/include/monitor/system_monitor_process_monitor.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

const std::string PROCESS_MONITOR_RECORD_FILE_NAME = "process_monitor.log";
const std::unordered_map<std::string, uint32_t> PROCESS_FAULT_MAP = {
    {"execution_manager", 410001}, {"PowerManager", 410002}, {"extwdg", 410003}
};

SystemMonitorProcessMonitor::SystemMonitorProcessMonitor(const SystemMonitorSubFunctionInfo& funcInfo)
: SystemMonitorBase(funcInfo, PROCESS_MONITOR_RECORD_FILE_NAME)
, stop_flag_(false)
{
    std::vector<std::string> processList = SystemMonitorConfig::getInstance()->GetProcessMonitorNameList();
    for (auto& item : processList) {
        process_run_status_map_.insert(std::make_pair(item, false));
    }
}

SystemMonitorProcessMonitor::~SystemMonitorProcessMonitor()
{
    process_run_status_map_.clear();
}

void
SystemMonitorProcessMonitor::Start()
{
    STMM_INFO << "SystemMonitorProcessMonitor::Start";
    stop_flag_ = false;
    if (GetRecordFileCycle()) {
        StartRecord();
    }

    std::thread stmm_process([this]() {
        bool start = true;
        while (!stop_flag_) {
            if (start || GetProcessStatus()) {
                std::string notifyStr = "Process run status:\n\t";
                for (auto& item : process_run_status_map_) {
                    notifyStr += item.first;
                    notifyStr += (item.second ? ": Running" : ": Not Running");
                    notifyStr += "\n\t";
                }

                Notify(notifyStr);
                if (start) {
                    start = false;
                }

                SetRecordStr(notifyStr);
                // report fault
                for (auto& item : PROCESS_FAULT_MAP) {
                    if (process_run_status_map_[item.first]) {
                        ReportFault(item.second, 0);
                    }
                    else {
                        ReportFault(item.second, 1);
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
        }
    });

    pthread_setname_np(stmm_process.native_handle(), "stmm_process");
    stmm_process.detach();
}

void
SystemMonitorProcessMonitor::Stop()
{
    STMM_INFO << "SystemMonitorProcessMonitor::Stop";
    stop_flag_ = true;
    StopRecord();
}

bool
SystemMonitorProcessMonitor::GetProcessStatus()
{
    bool bResult = false;
    FILE *fp;
    for (auto& item : process_run_status_map_) {
        bool runStatus = false;
        std::string cmd = "ps -ef | grep -w " + item.first + " | grep -v grep";
        fp = popen(cmd.c_str(), "r");
        if(EOF != fgetc(fp)) {
            runStatus = true;
        }

        if (runStatus != item.second) {
            item.second = runStatus;
            bResult = true;
        }

        pclose(fp);
    }

    return bResult;
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
