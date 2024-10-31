#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include <thread>
#include <fstream>
#include <regex>
#include "system_monitor/include/monitor/system_monitor_mem_monitor.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

// 调用linux系统命令获取mem信息
#define MEM_CMD "free -m | awk '{print $2,$3,$4,$5,$7}' | sed -n '2,$p'"
#define PID_MEM_CMD "/app/bin/pidstat -r |sort -r -n -k 8 | head -10 | awk '{print $2,$3,$4,$5,$6,$7,$8,$9}'"
const std::string MEM_MONITOR_RECORD_FILE_NAME = "mem_monitor.log";
const uint32_t MEM_MORE_THAN_80_FAULT = 406001;

SystemMonitorMemMonitor::SystemMonitorMemMonitor(const SystemMonitorSubFunctionInfo& funcInfo)
: SystemMonitorBase(funcInfo, MEM_MONITOR_RECORD_FILE_NAME)
, stop_flag_(false)
, alarm_count_(0)
{
}

SystemMonitorMemMonitor::~SystemMonitorMemMonitor()
{
}

void
SystemMonitorMemMonitor::Start()
{
    STMM_INFO << "SystemMonitorMemMonitor::Start";
    stop_flag_ = false;
    if (GetRecordFileCycle()) {
        StartRecord();
    }

    std::thread stmm_mem([this]() {
        while (!stop_flag_) {
            MemInfo memStatus;
            int32_t res = GetMemStatus(memStatus);
            if (!(res > 0)) {
                STMM_WARN << "SystemMonitorMemMonitor get mem info failed.";
                std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
                continue;
            }

            char buffer[200];
            sprintf(buffer, "read mem:\n\t total: %dM\n\t used: %dM\n\t free: %dM\n\t shared: %dM\n\t available: %dM\n",
                             memStatus.total, memStatus.used, memStatus.free, memStatus.shared, memStatus.available);

            std::string notifyStr = CONTROL_OUTPUT_LINE + buffer;
            // get large page memory
            std::ifstream ifs;
            ifs.open("/proc/buddyinfo", std::ios::in | std::ios::binary);
            if (ifs.is_open()) {
                std::stringstream stream;
                stream << ifs.rdbuf();
                std::string buddyinfo = stream.str();
                buddyinfo = Replace(buddyinfo, "Node", "");
                buddyinfo = Replace(buddyinfo, ", zone", "");
                if ("" != buddyinfo) {
                    notifyStr += "Node   Zone  Pcs(0) Pcs(1) Pcs(2) Pcs(3) Pcs(4) Pcs(5) Pcs(6) Pcs(7) Pcs(8) Pcs(9) Pcs(10)\n";
                    notifyStr += buddyinfo;
                }

                ifs.close();
            }

            notifyStr += CONTROL_OUTPUT_LINE;
            Notify(notifyStr);
            std::string alarmStr = "";
            if (((memStatus.total - memStatus.free) * 100 / memStatus.total) > GetAlarmValue()) {
                alarm_count_++;
                if (alarm_count_ >= 3) {
                    std::vector<PidMemInfo> pidMem;
                    int32_t res = GetPidMemStatus(pidMem);
                    if (!(res > 0)) {
                        STMM_WARN << "SystemMonitorMemMonitor get pid mem info failed.";
                        std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
                        continue;
                    }

                    char buffer[5000];
                    uint i = sprintf(buffer, "UID        PID     minflt/s     majflt/s       vsz       rss       mem      Command\n");
                    for (auto& item : pidMem) {
                        i += sprintf(buffer + i, "%-4d %9d %10.2f %11.2f %14d %9d %7.2f%%     %s\n",
                                    item.uid, item.pid, item.minflt, item.majflt, item.vsz, item.rss, item.mem, item.command);
                    }

                    const std::string prefix = "MEM usage rate is too high, top ten process as follows:\n";
                    alarmStr = prefix + CONTROL_OUTPUT_LINE + buffer + CONTROL_OUTPUT_LINE;
                    Alarm(alarmStr);
                    ReportFault(MEM_MORE_THAN_80_FAULT, 1);
                    alarm_count_ = 0;
                }
            }
            else {
                ReportFault(MEM_MORE_THAN_80_FAULT, 0);
            }

            SetRecordStr(notifyStr + alarmStr);
            std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
        }
    });

    pthread_setname_np(stmm_mem.native_handle(), "stmm_mem");
    stmm_mem.detach();
}

void
SystemMonitorMemMonitor::Stop()
{
    STMM_INFO << "SystemMonitorMemMonitor::Stop";
    stop_flag_ = true;
    StopRecord();
}

int32_t
SystemMonitorMemMonitor::GetMemStatus(MemInfo& memStatus)
{
    FILE *fp = popen(MEM_CMD, "r");
    if (fp == NULL) {
        return -1;
    }

    int result = fscanf(fp, "%u %u %u %u %u",
        &memStatus.total, &memStatus.used, &memStatus.free, &memStatus.shared, &memStatus.available);

    pclose(fp);
    return result;
}

int32_t
SystemMonitorMemMonitor::GetPidMemStatus(std::vector<PidMemInfo>& pidMem)
{
    FILE *fp = popen(PID_MEM_CMD, "r");
    if (fp == NULL) {
        return -1;
    }

    PidMemInfo item;
    pidMem.clear();
    while (fscanf(fp, "%u %u %f %f %u %u %f %s",
        &item.uid, &item.pid, &item.minflt, &item.majflt, &item.vsz, &item.rss, &item.mem, item.command) == 8) {
        pidMem.push_back(item);
    }

    pclose(fp);
    return pidMem.size();
}

std::string
SystemMonitorMemMonitor::Replace(std::string str, const std::string& pattern, const std::string& to)
{
    std::regex r(pattern);
    return std::regex_replace(str, r, to);
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
