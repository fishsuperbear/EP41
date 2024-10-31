#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include <thread>
#include <fstream>
#include <regex>
#include "system_monitor/include/monitor/system_monitor_network_monitor.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

const uint32_t NETWORK_ERROR_PACKET_LIMIT_VALUE = 10;
const std::string NETWORK_MONITOR_RECORD_FILE_NAME = "network_monitor.log";
const std::unordered_map<std::string, std::vector<uint32_t>> NETWORK_NIC_FAULT_MAP = {
    {"mgbe3_0", {409001, 409002}}
};

SystemMonitorNetworkMonitor::SystemMonitorNetworkMonitor(const SystemMonitorSubFunctionInfo& funcInfo)
: SystemMonitorBase(funcInfo, NETWORK_MONITOR_RECORD_FILE_NAME)
, stop_flag_(false)
{
    std::vector<std::string> nicList = SystemMonitorConfig::getInstance()->GetNetworkMonitorNicList();
    NetworkInfo info;
    for (auto& item : nicList) {
        info.notifyStr = "";
        info.rxErrors = 0;
        info.txErrors = 0;
        network_nic_info_map_.insert(std::make_pair(item, info));
    }
}

SystemMonitorNetworkMonitor::~SystemMonitorNetworkMonitor()
{
    network_nic_info_map_.clear();
}

void
SystemMonitorNetworkMonitor::Start()
{
    STMM_INFO << "SystemMonitorNetworkMonitor::Start";
    stop_flag_ = false;
    if (GetRecordFileCycle()) {
        StartRecord();
    }

    std::thread stmm_network([this]() {
        bool start = true;
        while (!stop_flag_) {
            GetNetWorkStatus(start);
            std::string notifyStr = CONTROL_OUTPUT_LINE;
            for (auto& item : network_nic_info_map_) {
                notifyStr += item.second.notifyStr;
                uint32_t rxErrors = 0, txErrors = 0;
                GetNetWorkRxAndTxErrors(item.second.notifyStr, rxErrors, txErrors);
                auto itr = NETWORK_NIC_FAULT_MAP.find(item.first);
                if (itr != NETWORK_NIC_FAULT_MAP.end()) {
                    if(std::string::npos != item.second.notifyStr.find("Link detected: yes")) {
                        ReportFault(itr->second[0], 0);
                    }

                    if(std::string::npos != item.second.notifyStr.find("Link detected: no")) {
                        ReportFault(itr->second[0], 1);
                    }

                    if (((rxErrors - item.second.rxErrors) < NETWORK_ERROR_PACKET_LIMIT_VALUE) && ((txErrors - item.second.txErrors) < NETWORK_ERROR_PACKET_LIMIT_VALUE)) {
                        ReportFault(itr->second[1], 0);
                    }
                    else {
                        ReportFault(itr->second[1], 1);
                    }
                }

                item.second.rxErrors = rxErrors;
                item.second.txErrors = txErrors;
            }

            if (start) {
                start = false;
            }

            notifyStr += CONTROL_OUTPUT_LINE;
            Notify(notifyStr);
            SetRecordStr(notifyStr);
            std::this_thread::sleep_for(std::chrono::milliseconds(GetMonitorCycle()));
        }
    });

    pthread_setname_np(stmm_network.native_handle(), "stmm_network");
    stmm_network.detach();
}

void
SystemMonitorNetworkMonitor::Stop()
{
    STMM_INFO << "SystemMonitorNetworkMonitor::Stop";
    stop_flag_ = true;
    StopRecord();
}

void
SystemMonitorNetworkMonitor::GetNetWorkStatus(const bool start)
{
    std::string cmd = "";
    std::string networkStatus = "";
    for (auto& item : network_nic_info_map_) {
        if (start) {
            cmd = "ethtool " + item.first;
            networkStatus += Popen(cmd);
        }

        cmd = "ifconfig " + item.first;
        networkStatus += Popen(cmd);

        cmd = "ethtool -S  " + item.first + " | grep error";
        networkStatus += Popen(cmd, true);

        item.second.notifyStr = networkStatus;
        networkStatus = "";
    }
}

void
SystemMonitorNetworkMonitor::GetNetWorkRxAndTxErrors(const std::string& info, uint32_t& rxErrors, uint32_t& txErrors)
{
    std::vector<std::string> vec = Split(info, "RX errors ");
    if (vec.size() > 1) {
        vec = Split(vec[1], "  dropped ");
        if (vec.size() > 1) {
            rxErrors = std::strtoul(vec[0].c_str(), 0, 0);
            vec = Split(vec[1], "TX errors ");
            if (vec.size() > 1) {
                txErrors = std::strtoul(vec[1].c_str(), 0, 0);
            }
        }
    }
}

std::string
SystemMonitorNetworkMonitor::Popen(const std::string& cmd, const bool eFlag)
{
    const uint BUFFER_LEN = 128;
    char buffer[BUFFER_LEN];
    FILE *fp;
    fp = popen(cmd.c_str(), "r");
    if (fp == NULL) {
        return "";
    }

    std::string dataStr = "";
    while (fgets(buffer, BUFFER_LEN - 1, fp) != NULL)
    {
        if (eFlag) {
            dataStr += "   ";
        }

        dataStr += buffer;
    }

    pclose(fp);
    return dataStr;
}

std::vector<std::string>
SystemMonitorNetworkMonitor::Split(const std::string& inputStr, const std::string& regexStr)
{
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
