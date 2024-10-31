#include <stdio.h>
#include <string.h>
#include <thread>
#include "system_monitor/include/monitor/system_monitor_mnand_hs_monitor.h"
#include "system_monitor/include/util/nvmnand.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

#define MAX_DESC_BUF_SIZE               144
const std::string HS_MONITOR_RECORD_FILE_NAME = "mnand_monitor.log";

const std::vector<std::string> DEVICE_LIFE_TIME_USED_INFO = {
    "0%-10%", "10%-20%", "20%-30%", "30%-40%", "40%-50%", "50%-60%",
    "60%-70%", "70%-80%", "80%-90%", "90%-100%"
};

SystemMonitorMnandHsMonitor::SystemMonitorMnandHsMonitor(const SystemMonitorSubFunctionInfo& funcInfo)
: SystemMonitorBase(funcInfo, HS_MONITOR_RECORD_FILE_NAME)
{
}

SystemMonitorMnandHsMonitor::~SystemMonitorMnandHsMonitor()
{
}

void
SystemMonitorMnandHsMonitor::Start()
{
    STMM_INFO << "SystemMonitorMnandHsMonitor::Start";
    char buffer[5000];
    uint i = sprintf(buffer, "Type       Node              PreEolInfo        DeviceLifeTimeEstA        DeviceLifeTimeEstB\n");
    std::vector<MnandHsInfo> ufsStatus;
    GetUfsStatus(ufsStatus);
    for (auto& item : ufsStatus) {
        i += sprintf(buffer + i, "%-4s %15s %18s %25s %25s\n",
            item.type.c_str(), item.node.c_str(), item.preEolInfo.c_str(), item.deviceLifeTimeEstA.c_str(), item.deviceLifeTimeEstB.c_str());
    }

    std::vector<MnandHsInfo> emmcStatus;
    GetEmmcStatus(emmcStatus);
    for (auto& item : emmcStatus) {
        i += sprintf(buffer + i, "%-4s %15s %18s %25s %25s\n",
            item.type.c_str(), item.node.c_str(), item.preEolInfo.c_str(), item.deviceLifeTimeEstA.c_str(), item.deviceLifeTimeEstB.c_str());
    }

    std::string notifyStr = CONTROL_OUTPUT_LINE + buffer + CONTROL_OUTPUT_LINE;
    Notify(notifyStr);
    SetRecordStr(notifyStr);
    WriteDataToFile(true);
}

void
SystemMonitorMnandHsMonitor::Stop()
{
    STMM_INFO << "SystemMonitorMnandHsMonitor::Stop";
}

void
SystemMonitorMnandHsMonitor::GetUfsStatus(std::vector<MnandHsInfo>& ufsStatus)
{
    std::vector<std::string> ufsNodeList = SystemMonitorConfig::getInstance()->GetMnandHsMonitorUfsNodeList();
    for (auto& item : ufsNodeList) {
        std::string cmd = "mnand_hs -d " + item + " -desc_rd 9 | awk '{print $4,$5,$6}' | sed -n '5p'";
        FILE *fp = popen(cmd.c_str(), "r");
        if (fp == NULL) {
            continue;
        }

        uint preEolInfo = 0;
        uint deviceLifeTimeEstA = 0;
        uint deviceLifeTimeEstB = 0;
        if (3 != fscanf(fp, "%2X %2X %2X", &preEolInfo, &deviceLifeTimeEstA, &deviceLifeTimeEstB))
        {
            pclose(fp);
            continue;
        }

        pclose(fp);
        MnandHsInfo info;
        info.type = "ufs";
        info.node = item;
        info.preEolInfo = "Normal(" + UINT8_TO_STRING(preEolInfo) + ")";
        if (1 != preEolInfo) {
            info.preEolInfo = "Abnormal(" + UINT8_TO_STRING(preEolInfo) + ")";
        }

        if (deviceLifeTimeEstA >0 && deviceLifeTimeEstA <= 0x0A) {
            info.deviceLifeTimeEstA = DEVICE_LIFE_TIME_USED_INFO[deviceLifeTimeEstA - 1] + "(" + UINT8_TO_STRING(deviceLifeTimeEstA) + ")";
        }
        else {
            info.deviceLifeTimeEstA = "Invalid data(" + UINT8_TO_STRING(deviceLifeTimeEstA) + ")";
        }

        if (deviceLifeTimeEstB >0 && deviceLifeTimeEstB <= 0x0A) {
            info.deviceLifeTimeEstB = DEVICE_LIFE_TIME_USED_INFO[deviceLifeTimeEstB - 1] + "(" + UINT8_TO_STRING(deviceLifeTimeEstB) + ")";
        }
        else {
            info.deviceLifeTimeEstB = "Invalid data(" + UINT8_TO_STRING(deviceLifeTimeEstB) + ")";
        }

        ufsStatus.emplace_back(info);
    }
}

void
SystemMonitorMnandHsMonitor::GetEmmcStatus(std::vector<MnandHsInfo>& emmcStatus)
{
    std::vector<std::string> emmcNodeList = SystemMonitorConfig::getInstance()->GetMnandHsMonitorEmmcNodeList();
    for (auto& item : emmcNodeList) {
        std::string cmd = "mnand_hs -d " + item + " -ext | awk '{print $12,$13,$14}' | sed -n '20p'";
        FILE *fp = popen(cmd.c_str(), "r");
        if (fp == NULL) {
            continue;
        }

        uint preEolInfo = 0;
        uint deviceLifeTimeEstA = 0;
        uint deviceLifeTimeEstB = 0;
        if (3 != fscanf(fp, "%2X %2X %2X", &preEolInfo, &deviceLifeTimeEstA, &deviceLifeTimeEstB))
        {
            pclose(fp);
            continue;
        }

        pclose(fp);
        MnandHsInfo info;
        info.type = "emmc";
        info.node = item;
        info.preEolInfo = "Normal(" + UINT8_TO_STRING(preEolInfo) + ")";
        if (1 != preEolInfo) {
            info.preEolInfo = "Abnormal(" + UINT8_TO_STRING(preEolInfo) + ")";
        }

        if (deviceLifeTimeEstA >0 && deviceLifeTimeEstA <= 0x0A) {
            info.deviceLifeTimeEstA = DEVICE_LIFE_TIME_USED_INFO[deviceLifeTimeEstA - 1] + "(" + UINT8_TO_STRING(deviceLifeTimeEstA) + ")";
        }
        else {
            info.deviceLifeTimeEstA = "Invalid data(" + UINT8_TO_STRING(deviceLifeTimeEstA) + ")";
        }

        if (deviceLifeTimeEstB >0 && deviceLifeTimeEstB <= 0x0A) {
            info.deviceLifeTimeEstB = DEVICE_LIFE_TIME_USED_INFO[deviceLifeTimeEstB - 1] + "(" + UINT8_TO_STRING(deviceLifeTimeEstB) + ")";
        }
        else {
            info.deviceLifeTimeEstB = "Invalid data(" + UINT8_TO_STRING(deviceLifeTimeEstB) + ")";
        }

        emmcStatus.emplace_back(info);
    }
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
