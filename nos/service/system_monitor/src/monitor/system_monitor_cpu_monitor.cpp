#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include <thread>
#include <fstream>
#include "system_monitor/include/monitor/system_monitor_cpu_monitor.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

// 调用linux系统命令获取cpu信息
#define CPU_CMD "/app/bin/mpstat -P ALL | awk '{print $3,$4,$5,$6,$7,$8,$9,$10,$11,$12}' | sed -n '4,$p'"
#define PID_CPU_CMD "/app/bin/pidstat -u |sort -r -n -k 8 | head -10 | awk '{print $2,$3,$4,$5,$6,$7,$8,$9,$10}'"
#define FLAME_GRAPH "/app/scripts/perf_helper.sh perf -p $pid"
const std::string CPU_MONITOR_RECORD_FILE_NAME = "cpu_monitor.log";
const uint32_t CPU_MORE_THAN_80_FAULT = 405001;

SystemMonitorCpuMonitor::SystemMonitorCpuMonitor(const SystemMonitorSubFunctionInfo& funcInfo)
: SystemMonitorBase(funcInfo, CPU_MONITOR_RECORD_FILE_NAME)
, stop_flag_(false)
, alarm_count_(0)
{
}

SystemMonitorCpuMonitor::~SystemMonitorCpuMonitor()
{
}

void
SystemMonitorCpuMonitor::Start()
{
    STMM_INFO << "SystemMonitorCpuMonitor::Start";
    stop_flag_ = false;
    if (GetRecordFileCycle()) {
        StartRecord();
    }

    std::thread stmm_cpu([this]() {
        while (!stop_flag_) {
            CpuStatus tcs;
            int32_t res = GetCpuStatus(&tcs);
            if (!(res > 0)) {
                STMM_WARN << "SystemMonitorCpuMonitor get cpu info failed.";
                std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
                continue;
            }

            char buffer[5000];
            bool alarmFlag = false;
            uint i = sprintf(buffer, "CPU     usr     nice     sys     iowait     irq     soft     steal     guest     gnice     idle\n");
            i += sprintf(buffer + i, "all %7.2f%% %7.2f%% %6.2f%% %7.2f%% %8.2f%% %7.2f%% %7.2f%% %8.2f%% %8.2f%% %9.2f%%\n",
                        tcs.cpu_all.usr, tcs.cpu_all.nice, tcs.cpu_all.sys, tcs.cpu_all.iowait, tcs.cpu_all.irq,
                        tcs.cpu_all.soft, tcs.cpu_all.steal, tcs.cpu_all.guest, tcs.cpu_all.gnice, tcs.cpu_all.idle);
#if 0
            if ((100 - tcs.cpu_all.idle) > 30) {
                FILE* fp= popen(/opt/usr/storage/lwd/perf.sh, "r");
                if (fp == NULL) {
                    STMM_WARN << "/opt/usr/storage/lwd/perf.sh popen failed.";
                    return -1;
                }
            }
#endif

            for (uint j = 0; j < tcs.cup_details.size(); j++) {
                i += sprintf(buffer + i, "%-3d %7.2f%% %7.2f%% %6.2f%% %7.2f%% %8.2f%% %7.2f%% %7.2f%% %8.2f%% %8.2f%% %9.2f%%\n", j,
                        tcs.cup_details.at(j).usr, tcs.cup_details.at(j).nice, tcs.cup_details.at(j).sys, tcs.cup_details.at(j).iowait, tcs.cup_details.at(j).irq,
                        tcs.cup_details.at(j).soft, tcs.cup_details.at(j).steal, tcs.cup_details.at(j).guest, tcs.cup_details.at(j).gnice, tcs.cup_details.at(j).idle);
                if (!alarmFlag) {
                    if ((100 - tcs.cup_details.at(j).idle) > GetAlarmValue()) {
                        alarmFlag = true;
                    }
                }
            }
#if 0
        std::vector<PidCpuInfo> pidCpu;
        int32_t res = GetPidCpuStatus(pidCpu);
        if (!(res > 0)) {
            STMM_WARN << "SystemMonitorCpuMonitor get pid cpu info failed.";
            std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
            continue;
        }
        char buffer[5000];
        uint i = sprintf(buffer, "UID        PID     usr     sys     guest     wait     cpu     CPU     Command\n");
        for (auto& item : pidCpu) {
            i += sprintf(buffer + i, "%-4d %9d %7.2f%% %6.2f%% %7.2f%% %8.2f%% %6.2f%% %6d     %s\n",
                        item.uid, item.pid, item.usr, item.system, item.guest, item.wait, item.cpu, item.cpuName, item.command);
        }

        if (item.cpu > 30) {
            //std::cout << item.cpu;
            char command[100];
            snprintf(command, sizeof(command), "/app/scripts/perf_helper.sh perf -p %d", item.pid);
            FILE* fp= popen(command, "r")
            if (fp == NULL) {
                STMM_WARN << "perf_helper.sh failed.";
                return -1;
            }
        }

#endif
            std::string notifyStr = CONTROL_OUTPUT_LINE + buffer + CONTROL_OUTPUT_LINE;
            Notify(notifyStr);
            std::string alarmStr = "";
            if (alarmFlag) {
                alarm_count_++;
                if (alarm_count_ >= 3) {
                    std::vector<PidCpuInfo> pidCpu;
                    int32_t res = GetPidCpuStatus(pidCpu);
                    if (!(res > 0)) {
                        STMM_WARN << "SystemMonitorCpuMonitor get pid cpu info failed.";
                        std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
                        continue;
                    }

                    char buffer[5000];
                    uint i = sprintf(buffer, "UID        PID     usr     sys     guest     wait     cpu     CPU     Command\n");
                    for (auto& item : pidCpu) {
                        i += sprintf(buffer + i, "%-4d %9d %7.2f%% %6.2f%% %7.2f%% %8.2f%% %6.2f%% %6d     %s\n",
                                    item.uid, item.pid, item.usr, item.system, item.guest, item.wait, item.cpu, item.cpuName, item.command);
                    }

                    if (item.cpu > 70) {
                        "/app/scripts/perf_helper.sh perf -p item.pid"
                        sprintf(item.cpu);
                    }
                    const std::string prefix = "CPU usage rate is too high, top ten process as follows:\n";
                    alarmStr = prefix + CONTROL_OUTPUT_LINE + buffer + CONTROL_OUTPUT_LINE;
                    Alarm(alarmStr);
                    ReportFault(CPU_MORE_THAN_80_FAULT, 1);
                    alarm_count_ = 0;
                }
            }
            else {
                ReportFault(CPU_MORE_THAN_80_FAULT, 0);
            }

            SetRecordStr(notifyStr + alarmStr);
            std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
        }
    });

    pthread_setname_np(stmm_cpu.native_handle(), "stmm_cpu");
    stmm_cpu.detach();
}

void
SystemMonitorCpuMonitor::Stop()
{
    STMM_INFO << "SystemMonitorCpuMonitor::Stop";
    stop_flag_ = true;
    StopRecord();
}

int32_t
SystemMonitorCpuMonitor::GetCpuStatus(CpuStatus *pCpu)
{
    FILE *fp = popen(CPU_CMD, "r");
    if (fp == NULL) {
        return -1;
    }

    CpuInfo plast = {0.0};
    uint8_t index = 0;
    while (fscanf(fp, "%f %f %f %f %f %f %f %f %f %f",
        &plast.usr, &plast.nice, &plast.sys, &plast.iowait, &plast.irq,
        &plast.soft, &plast.steal, &plast.guest, &plast.gnice, &plast.idle) == 10) {
        index++;
        if (index == 1) {
            memcpy(&pCpu->cpu_all, &plast, sizeof(CpuInfo));
        }
        else {
            pCpu->cup_details.push_back(plast);
        }
    }

    pclose(fp);
    return pCpu->cup_details.size();
}

int32_t
SystemMonitorCpuMonitor::GetPidCpuStatus(std::vector<PidCpuInfo>& pidCpu)
{
    FILE *fp = popen(PID_CPU_CMD, "r");
    if (fp == NULL) {
        return -1;
    }

    PidCpuInfo item;
    pidCpu.clear();
    while (fscanf(fp, "%u %u %f %f %f %f %f %u %s",
        &item.uid, &item.pid, &item.usr, &item.system, &item.guest, &item.wait, &item.cpu, &item.cpuName, item.command) == 9) {
        pidCpu.push_back(item);
    }

    pclose(fp);
    return pidCpu.size();
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
