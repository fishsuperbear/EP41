#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <cinttypes>
#include <fstream>
#include <iomanip>
#include <regex>
#include "json/json.h"
#include "system_monitor_info.h"

SystemMonitorInfo* SystemMonitorInfo::instance_ = nullptr;
std::mutex SystemMonitorInfo::mtx_;

const int MAX_LOAD_SIZE = 1024;

#ifdef BUILD_FOR_MDC
    const std::string SYSTEM_MONITOR_CONFIG_PATH = "/opt/usr/diag_update/mdc-llvm/conf/system_monitor_config.json";
#elif BUILD_FOR_J5
    const std::string SYSTEM_MONITOR_CONFIG_PATH = "/userdata/diag_update/j5/conf/system_monitor_config.json";
#elif BUILD_FOR_ORIN
    const std::string SYSTEM_MONITOR_CONFIG_PATH = "/app/conf/system_monitor_config.json";
#else
    const std::string SYSTEM_MONITOR_CONFIG_PATH = "/app/conf/system_monitor_config.json";
#endif

SystemMonitorInfo::SystemMonitorInfo()
: device_info_(std::make_shared<DevmClientDeviceInfo>())
{
}

SystemMonitorInfo*
SystemMonitorInfo::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new SystemMonitorInfo();
        }
    }

    return instance_;
}

void
SystemMonitorInfo::Init()
{
    ParseSystemMonitorConfigJson();
}

void
SystemMonitorInfo::DeInit()
{
    network_nic_list_.clear();
    process_run_status_map_.clear();
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

std::string
SystemMonitorInfo::GetMonitorInfo(const std::string& monitorType)
{
    std::string monitorInfo = "";
    if ("cpu" == monitorType) {
        monitorInfo = GetCpuInfo();
    }
    else if ("disk" == monitorType) {
        monitorInfo = GetDiskInfo();
    }
    else if ("mem" == monitorType) {
        monitorInfo = GetMemInfo();
    }
    else if ("network" == monitorType) {
        monitorInfo = GetNetWorkInfo();
    }
    else if ("process" == monitorType) {
        monitorInfo = GetProcessInfo();
    }
    else if ("temp" == monitorType) {
        monitorInfo = GetTemperatureInfo();
    }
    else if ("voltage" == monitorType) {
        monitorInfo = GetVoltageInfo();
    }

    if ("" != monitorInfo) {
        monitorInfo += "\n";
    }

    return monitorInfo;
}

std::string
SystemMonitorInfo::GetMonitorInfoFromFile(const std::string& monitorType)
{
    auto itr = MONITOR_RECORD_FILE_PATH_MAP.find(monitorType);
    if (MONITOR_RECORD_FILE_PATH_MAP.end() == itr) {
        return "";
    }

    std::string filePath = "";
    if (0 == access(itr->second[1].c_str(), F_OK)) {
        filePath = itr->second[1];
    }
    else {
        if (0 == access(itr->second[2].c_str(), F_OK)) {
            filePath = itr->second[2];
        }
    }

    std::string monitorInfo = itr->second[0];
    if ("" == filePath) {
        monitorInfo += "No monitor info record!\n\n";
        return monitorInfo;
    }

    std::ifstream ifs;
    ifs.open(filePath, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        monitorInfo += ("Record file: " + filePath + " open failed!\n\n");
        return monitorInfo;
    }

    std::string str = "";
    while(getline(ifs, str))
    {
        str += "\n";
        if(std::string::npos != str.find("Record time: ")) {
            monitorInfo = itr->second[0] + str;
        }
        else {
            monitorInfo += str;
        }
    }

    return monitorInfo;
}

std::string
SystemMonitorInfo::GetCpuInfo()
{
    CpuStatus tcs;
    int32_t res = GetCpuStatus(&tcs);
    if (!(res > 0)) {
        return "";
    }

    char statusBuffer[5000];
    memset(statusBuffer, 0 ,sizeof(statusBuffer));
    uint i = sprintf(statusBuffer, "CPU     usr     nice     sys     iowait     irq     soft     steal     guest     gnice     idle\n");
    i += sprintf(statusBuffer + i, "all %7.2f%% %7.2f%% %6.2f%% %7.2f%% %8.2f%% %7.2f%% %7.2f%% %8.2f%% %8.2f%% %9.2f%%\n",
                tcs.cpu_all.usr, tcs.cpu_all.nice, tcs.cpu_all.sys, tcs.cpu_all.iowait, tcs.cpu_all.irq,
                tcs.cpu_all.soft, tcs.cpu_all.steal, tcs.cpu_all.guest, tcs.cpu_all.gnice, tcs.cpu_all.idle);
    for (uint j = 0; j < tcs.cup_details.size(); j++) {
        i += sprintf(statusBuffer + i, "%-3d %7.2f%% %7.2f%% %6.2f%% %7.2f%% %8.2f%% %7.2f%% %7.2f%% %8.2f%% %8.2f%% %9.2f%%\n", j,
                tcs.cup_details.at(j).usr, tcs.cup_details.at(j).nice, tcs.cup_details.at(j).sys, tcs.cup_details.at(j).iowait, tcs.cup_details.at(j).irq,
                tcs.cup_details.at(j).soft, tcs.cup_details.at(j).steal, tcs.cup_details.at(j).guest, tcs.cup_details.at(j).gnice, tcs.cup_details.at(j).idle);
    }

    std::string cpuInfo = CONTROL_OUTPUT_LINE + statusBuffer + CONTROL_OUTPUT_LINE;
    std::vector<PidCpuInfo> pidCpu;
    res = GetPidCpuStatus(pidCpu);
    if (!(res > 0)) {
        return CPU_INFO_TITLE + GetCurrentTime() + cpuInfo;
    }

    char pidBuffer[5000];
    memset(pidBuffer, 0 ,sizeof(pidBuffer));
    i = sprintf(pidBuffer, "UID        PID     usr     sys     guest     wait     cpu     CPU     Command\n");
    for (auto& item : pidCpu) {
        i += sprintf(pidBuffer + i, "%-4d %9d %7.2f%% %6.2f%% %7.2f%% %8.2f%% %6.2f%% %6d     %s\n",
                    item.uid, item.pid, item.usr, item.system, item.guest, item.wait, item.cpu, item.cpuName, item.command);
    }

    const std::string prefix = "CPU usage rate top ten process as follows:\n";
    cpuInfo += (prefix + CONTROL_OUTPUT_LINE + pidBuffer + CONTROL_OUTPUT_LINE);
    return CPU_INFO_TITLE + GetCurrentTime() + cpuInfo;
}

std::string
SystemMonitorInfo::GetDiskInfo()
{
    std::vector<DiskInfo> diskStatus;
    int32_t res = GetDiskStatus(diskStatus);
    if (!(res > 0)) {
        return "";
    }

    char buffer[5000];
    memset(buffer, 0 ,sizeof(buffer));
    uint i = sprintf(buffer, "Filesystem          Size    Used    Avail    Use     MountedOn\n");
    for (auto& item : diskStatus) {
        i += sprintf(buffer + i, "%-17s %6s %7s %7s %7s     %s\n",
                        item.fileSyetem, item.size, item.used, item.avail, item.use, item.mountedOn);
    }

    std::string diskInfo = CONTROL_OUTPUT_LINE + buffer + CONTROL_OUTPUT_LINE;
    return DISK_INFO_TITLE + GetCurrentTime() + diskInfo;
}

std::string
SystemMonitorInfo::GetMemInfo()
{
    MemInfo memStatus;
    int32_t res = GetMemStatus(memStatus);
    if (!(res > 0)) {
        return "";
    }

    char memBuffer[200];
    memset(memBuffer, 0 ,sizeof(memBuffer));
    sprintf(memBuffer, "read mem:\n\t total: %dM\n\t used: %dM\n\t free: %dM\n\t shared: %dM\n\t available: %dM\n",
                        memStatus.total, memStatus.used, memStatus.free, memStatus.shared, memStatus.available);

    std::string memInfo = CONTROL_OUTPUT_LINE + memBuffer;
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
            memInfo += "Node   Zone  Pcs(0) Pcs(1) Pcs(2) Pcs(3) Pcs(4) Pcs(5) Pcs(6) Pcs(7) Pcs(8) Pcs(9) Pcs(10)\n";
            memInfo += buddyinfo;
        }

        ifs.close();
    }

    memInfo += CONTROL_OUTPUT_LINE;
    std::vector<PidMemInfo> pidMem;
    res = GetPidMemStatus(pidMem);
    if (!(res > 0)) {
        return MEM_INFO_TITLE + GetCurrentTime() + memInfo;
    }

    char pidBuffer[5000];
    memset(pidBuffer, 0 ,sizeof(pidBuffer));
    uint i = sprintf(pidBuffer, "UID        PID     minflt/s     majflt/s       vsz       rss       mem      Command\n");
    for (auto& item : pidMem) {
        i += sprintf(pidBuffer + i, "%-4d %9d %10.2f %11.2f %14d %9d %7.2f%%     %s\n",
                    item.uid, item.pid, item.minflt, item.majflt, item.vsz, item.rss, item.mem, item.command);
    }

    const std::string prefix = "MEM usage rate top ten process as follows:\n";
    memInfo += (prefix + CONTROL_OUTPUT_LINE + pidBuffer + CONTROL_OUTPUT_LINE);
    return MEM_INFO_TITLE + GetCurrentTime() + memInfo;
}

std::string
SystemMonitorInfo::GetNetWorkInfo()
{
    std::string nic_info = "";
    for (auto& item : network_nic_list_) {
        nic_info += GetNetWorkStatus(item);
    }

    if ("" == nic_info) {
        return "";
    }

    std::string networkInfo = CONTROL_OUTPUT_LINE + nic_info + CONTROL_OUTPUT_LINE;
    return NETWORK_INFO_TITLE + GetCurrentTime() + networkInfo;
}

std::string
SystemMonitorInfo::GetProcessInfo()
{
    GetProcessStatus();
    std::string processInfo = "Process run status:\n\t";
    for (auto& item : process_run_status_map_) {
        processInfo += item.first;
        processInfo += (item.second ? ": Running" : ": Not Running");
        processInfo += "\n\t";
    }

    return PROCESS_INFO_TITLE + GetCurrentTime() + processInfo;
}

std::string
SystemMonitorInfo::GetTemperatureInfo()
{
    TemperatureData tempInfo{0};
    GetTempInfo(tempInfo);
    std::ostringstream stream;
    stream << "Current temperature:\n\t temp_soc: " << std::fixed << std::setprecision(2) << tempInfo.temp_soc << "℃"
                                << "\n\t temp_mcu: " << std::fixed << std::setprecision(2) << tempInfo.temp_mcu << "℃"
                                << "\n\t temp_ext0: " << std::fixed << std::setprecision(2) << tempInfo.temp_ext0 << "℃"
                                << "\n\t temp_ext1: " << std::fixed << std::setprecision(2) << tempInfo.temp_ext1 << "℃\n";
    std::string temperatureInfo = stream.str();
    return TEMP_INFO_TITLE + GetCurrentTime() + temperatureInfo;
}

std::string
SystemMonitorInfo::GetVoltageInfo()
{
    VoltageData voltage {0};
    GetVoltage(voltage);
    std::ostringstream stream;
    stream << "Current kl15 voltage: " << ((0 == voltage.kl15) ? "ACCOFF" : "ACCON") << "\n";
    stream << "Current kl30 voltage: " << std::fixed << std::setprecision(2) << voltage.kl30 << "V\n";
    std::string voltageInfo = stream.str();
    return VOLTAGE_INFO_TITLE + GetCurrentTime() + voltageInfo;
}

char*
SystemMonitorInfo::GetJsonAll(const char *fname)
{
    FILE *fp;
    char *str;
    char txt[MAX_LOAD_SIZE];
    int filesize;
    if ((fp = fopen(fname, "r")) == NULL) {
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    filesize = ftell(fp);

    str = (char*)malloc(filesize + 1);
    memset(str, 0, filesize);

    rewind(fp);
    while ((fgets(txt, MAX_LOAD_SIZE, fp)) != NULL) {
        strcat(str, txt);
    }
    fclose(fp);

    return str;
}

void
SystemMonitorInfo::ParseSystemMonitorConfigJson()
{
    char* jsonstr = GetJsonAll(SYSTEM_MONITOR_CONFIG_PATH.c_str());
    if (nullptr == jsonstr) {
        return;
    }

    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    Json::Value  rootValue;
    JSONCPP_STRING errs;

    bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &rootValue, &errs);

    if (!res || !errs.empty()) {
        if (jsonstr != NULL) {
            free(jsonstr);
        }

        return;
    }

    Json::Value nicListValue = rootValue["NetworkMonitorNicList"];
    for (Json::ArrayIndex i = 0; i < nicListValue.size(); ++i) {
        network_nic_list_.push_back(nicListValue[i].asString());
    }

    Json::Value processNameListValue = rootValue["ProcessMonitorNameList"];
    for (Json::ArrayIndex i = 0; i < processNameListValue.size(); ++i) {
        process_run_status_map_.insert(std::make_pair((processNameListValue[i].asString()), false));
    }

    if (jsonstr != NULL) {
        free(jsonstr);
    }
}

int32_t
SystemMonitorInfo::GetCpuStatus(CpuStatus *pCpu)
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
SystemMonitorInfo::GetPidCpuStatus(std::vector<PidCpuInfo>& pidCpu)
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

int32_t
SystemMonitorInfo::GetDiskStatus(std::vector<DiskInfo>& diskStatus)
{
    FILE *fp = popen(DISK_CMD, "r");
    if (fp == NULL) {
        return -1;
    }

    DiskInfo item;
    diskStatus.clear();
    while (fscanf(fp, "%s %s %s %s %s %s",
        item.fileSyetem, item.size, item.used, item.avail, item.use, item.mountedOn) == 6) {
        diskStatus.push_back(item);
    }

    pclose(fp);
    return diskStatus.size();
}

int32_t
SystemMonitorInfo::GetMemStatus(MemInfo& memStatus)
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
SystemMonitorInfo::GetPidMemStatus(std::vector<PidMemInfo>& pidMem)
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
SystemMonitorInfo::GetNetWorkStatus(const std::string& nic)
{
    std::string cmd = "";
    std::string networkStatus = "";
    cmd = "ethtool " + nic;
    networkStatus += Popen(cmd);

    cmd = "ifconfig " + nic;
    networkStatus += Popen(cmd);

    cmd = "ethtool -S  " + nic + " | grep error";
    networkStatus += Popen(cmd, true);

    return networkStatus;
}

void
SystemMonitorInfo::GetProcessStatus()
{
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
        }

        pclose(fp);
    }
}

void
SystemMonitorInfo::GetTempInfo(TemperatureData& tempInfo)
{
    if (nullptr == device_info_) {
        return;
    }

    device_info_->GetTemperature(tempInfo);
}

void
SystemMonitorInfo::GetVoltage(VoltageData& voltage)
{
    if (nullptr == device_info_) {
        return;
    }

    device_info_->GetVoltage(voltage);
}

std::string
SystemMonitorInfo::Popen(const std::string& cmd, const bool eFlag)
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
SystemMonitorInfo::Split(const std::string& inputStr, const std::string& regexStr)
{
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

std::string
SystemMonitorInfo::Replace(std::string str, const std::string& pattern, const std::string& to)
{
    std::regex r(pattern);
    return std::regex_replace(str, r, to);
}

std::string
SystemMonitorInfo::GetCurrentTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y/%m/%d %H:%M:%S", localtime(&timep));
    std::string recordTime = "Record time: " + std::string(tmp) + "\n";
    return recordTime;
}