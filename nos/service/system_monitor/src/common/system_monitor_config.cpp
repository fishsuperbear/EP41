#include <iostream>
#include <algorithm>

#include "json/json.h"
#include "system_monitor/include/common/system_monitor_config.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

SystemMonitorConfig* SystemMonitorConfig::instance_ = nullptr;
std::mutex SystemMonitorConfig::mtx_;

const int MAX_LOAD_SIZE = 1024;

// #ifdef BUILD_FOR_MDC
//     const std::string SYSTEM_MONITOR_CONFIG_PATH = "/opt/usr/diag_update/mdc-llvm/conf/system_monitor_config.json";
// #elif BUILD_FOR_J5
//     const std::string SYSTEM_MONITOR_CONFIG_PATH = "/userdata/diag_update/j5/conf/system_monitor_config.json";
// #elif BUILD_FOR_ORIN
//     const std::string SYSTEM_MONITOR_CONFIG_PATH = "/app/conf/system_monitor_config.json";
// #else
//     const std::string SYSTEM_MONITOR_CONFIG_PATH = "/app/conf/system_monitor_config.json";
// #endif

const std::string SYSTEM_MONITOR_CONFIG_PATH = "/app/conf/system_monitor_config.json";

SystemMonitorConfig::SystemMonitorConfig()
{
}

SystemMonitorConfig*
SystemMonitorConfig::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new SystemMonitorConfig();
        }
    }

    return instance_;
}

void
SystemMonitorConfig::Init()
{
    STMM_INFO << "SystemMonitorConfig::Init";
    // QueryPrintConfigData();
}

void
SystemMonitorConfig::DeInit()
{
    STMM_INFO << "SystemMonitorConfig::DeInit";
    system_monitor_config_info_.subFunction.clear();
    disk_monitor_path_list_.clear();
    disk_monitor_soc_log_move_list_.clear();
    disk_monitor_mcu_log_move_list_.clear();
    process_monitor_name_list_.clear();
    mnand_hs_monitor_ufs_node_list_.clear();
    mnand_hs_monitor_emmc_node_list_.clear();
    file_protect_path_list_.clear();
    file_monitor_path_list_.clear();
    network_monitor_nic_list_.clear();
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
SystemMonitorConfig::LoadSystemMonitorConfig()
{
    ParseSystemMonitorConfigJson();
}

char*
SystemMonitorConfig::GetJsonAll(const char *fname)
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
SystemMonitorConfig::ParseSystemMonitorConfigJson()
{
    std::cout << "SystemMonitorConfig::LoadSystemMonitorConfig configPath: " << SYSTEM_MONITOR_CONFIG_PATH << std::endl;
    // default system monitor config
    {
        // log default config
        system_monitor_config_info_.LogAppName = "SYSTEM_MONITOR";
        system_monitor_config_info_.LogAppDescription = "system_monitor";
        system_monitor_config_info_.LogContextName = "SYSTEM_MONITOR";
        system_monitor_config_info_.LogLevel = 3;
        system_monitor_config_info_.LogMode = 2;
        system_monitor_config_info_.LogFilePath = "./";
        system_monitor_config_info_.MaxLogFileNum = 10;
        system_monitor_config_info_.MaxSizeOfLogFile = 20;
        system_monitor_config_info_.DebugSwitch = "off";

        system_monitor_config_info_.subFunction.clear();
    }
    // default system monitor config

    char* jsonstr = GetJsonAll(SYSTEM_MONITOR_CONFIG_PATH.c_str());

    if (nullptr == jsonstr) {
        std::cout << "SystemMonitorConfig::ParsePhmConfigJson error jsonstr is nullptr." << std::endl;
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

    // load log config
    system_monitor_config_info_.LogAppName = static_cast<std::string>(rootValue["LogAppName"].asString());
    system_monitor_config_info_.LogAppDescription = static_cast<std::string>(rootValue["LogAppDescription"].asString());
    system_monitor_config_info_.LogContextName = static_cast<std::string>(rootValue["LogContextName"].asString());
    system_monitor_config_info_.LogLevel = static_cast<uint8_t>(rootValue["LogLevel"].asUInt());
    system_monitor_config_info_.LogMode = static_cast<uint8_t>(rootValue["LogMode"].asUInt());
    system_monitor_config_info_.LogFilePath = static_cast<std::string>(rootValue["LogFilePath"].asString());
    system_monitor_config_info_.MaxLogFileNum = static_cast<uint32_t>(rootValue["MaxLogFileNum"].asUInt());
    system_monitor_config_info_.MaxSizeOfLogFile = static_cast<uint32_t>(rootValue["MaxSizeOfLogFile"].asUInt());
    system_monitor_config_info_.DebugSwitch = static_cast<std::string>(rootValue["DebugSwitch"].asString());

    // load tcp config
    system_monitor_tcp_config_info_.TcpEthName = static_cast<std::string>(rootValue["TcpEthName"].asString());
    system_monitor_tcp_config_info_.TcpPort = static_cast<uint32_t>(rootValue["TcpPort"].asUInt());
    system_monitor_tcp_config_info_.TcpMaxClients = static_cast<uint8_t>(rootValue["TcpMaxClients"].asUInt());

    SystemMonitorSubFunctionInfo subFunction;
    std::string id = "";
    Json::Value subFunctionValue = rootValue["MonitorSubFunction"];
    for (Json::ArrayIndex i = 0; i < subFunctionValue.size(); ++i) {
        subFunction.shortName = static_cast<std::string>(subFunctionValue[i]["shortName"].asString());
        id = static_cast<std::string>(subFunctionValue[i]["id"].asString());
        subFunction.id = static_cast<SystemMonitorSubFunctionId>(std::strtoul(id.c_str(), 0, 0));
        subFunction.monitorSwitch = static_cast<std::string>(subFunctionValue[i]["monitorSwitch"].asString());
        subFunction.monitorCycle = static_cast<uint32_t>(subFunctionValue[i]["monitorCycle"].asUInt());
        subFunction.recordFileCycle = static_cast<uint32_t>(subFunctionValue[i]["recordFileCycle"].asUInt());
        subFunction.recordFilePath = static_cast<std::string>(subFunctionValue[i]["recordFilePath"].asString());
        subFunction.isAlarm = static_cast<bool>(subFunctionValue[i]["isAlarm"].asBool());
        subFunction.alarmValue = static_cast<uint8_t>(subFunctionValue[i]["alarmValue"].asUInt());
        subFunction.postProcessingSwitch = static_cast<std::string>(subFunctionValue[i]["postProcessingSwitch"].asString());

        system_monitor_config_info_.subFunction.insert(std::make_pair(subFunction.id, subFunction));
    }

    Json::Value diskPathListValue = rootValue["DiskMonitorPathList"];
    SystemMonitorDiskMonitorPartitionInfo partitionInfo;
    SystemMonitorDiskMonitorListInfo listInfo;
    for (Json::ArrayIndex i = 0; i < diskPathListValue.size(); ++i) {
        partitionInfo.deleteFilesByPathList.clear();
        partitionInfo.deleteFilesByWildcardList.clear();
        partitionInfo.partition = diskPathListValue[i]["partition"].asString();
        partitionInfo.reservedSize = diskPathListValue[i]["reservedSize"].asUInt();
        partitionInfo.partitionAlarmValue = diskPathListValue[i]["partitionAlarmValue"].asUInt();
        partitionInfo.isIncludeSpecialPostProcessing = diskPathListValue[i]["isIncludeSpecialPostProcessing"].asBool();
        partitionInfo.isDeleteEmptyDir = diskPathListValue[i]["isDeleteEmptyDir"].asBool();
        for (Json::ArrayIndex j = 0; j < diskPathListValue[i]["deleteFilesByWildcardList"].size(); ++j) {
            listInfo.pathName = diskPathListValue[i]["deleteFilesByWildcardList"][j]["wildcardPath"].asString();
            listInfo.isTraverseSubdir = diskPathListValue[i]["deleteFilesByWildcardList"][j]["isTraverseSubdir"].asBool();
            partitionInfo.deleteFilesByWildcardList.push_back(listInfo);
        }

        for (Json::ArrayIndex k = 0; k < diskPathListValue[i]["deleteFilesByPathList"].size(); ++k) {
            listInfo.pathName = diskPathListValue[i]["deleteFilesByPathList"][k]["filePath"].asString();
            listInfo.isTraverseSubdir = diskPathListValue[i]["deleteFilesByPathList"][k]["isTraverseSubdir"].asBool();
            partitionInfo.deleteFilesByPathList.push_back(listInfo);
        }

        disk_monitor_path_list_.insert(std::make_pair(partitionInfo.partition, partitionInfo));
    }

    Json::Value socLogMoveListValue = rootValue["SocLogMoveList"];
    SystemMonitorDiskMonitorLogMoveListInfo logMoveInfo;
    for (Json::ArrayIndex i = 0; i < socLogMoveListValue.size(); ++i) {
        logMoveInfo.filePrefix = socLogMoveListValue[i]["filePrefix"].asString();
        logMoveInfo.reservedFileNum = socLogMoveListValue[i]["reservedFileNum"].asUInt();
        disk_monitor_soc_log_move_list_.push_back(logMoveInfo);
    }

    Json::Value mcuLogMoveListValue = rootValue["McuLogMoveList"];
    for (Json::ArrayIndex i = 0; i < mcuLogMoveListValue.size(); ++i) {
        logMoveInfo.filePrefix = mcuLogMoveListValue[i]["filePrefix"].asString();
        logMoveInfo.reservedFileNum = mcuLogMoveListValue[i]["reservedFileNum"].asUInt();
        disk_monitor_mcu_log_move_list_.push_back(logMoveInfo);
    }

    Json::Value processNameListValue = rootValue["ProcessMonitorNameList"];
    for (Json::ArrayIndex i = 0; i < processNameListValue.size(); ++i) {
        process_monitor_name_list_.push_back(processNameListValue[i].asString());
    }

    Json::Value ufsNodeListValue = rootValue["MnandHsMonitorUfsNodeList"];
    for (Json::ArrayIndex i = 0; i < ufsNodeListValue.size(); ++i) {
        mnand_hs_monitor_ufs_node_list_.push_back(ufsNodeListValue[i].asString());
    }

    Json::Value emmcNodeListValue = rootValue["MnandHsMonitorEmmcNodeList"];
    for (Json::ArrayIndex i = 0; i < emmcNodeListValue.size(); ++i) {
        mnand_hs_monitor_emmc_node_list_.push_back(emmcNodeListValue[i].asString());
    }

    SystemMonitorFileMonitorInfo fileProtect;
    Json::Value fileProtectListValue = rootValue["FileSystemProtectList"];
    for (Json::ArrayIndex i = 0; i < fileProtectListValue.size(); ++i) {
        fileProtect.pathName = fileProtectListValue[i]["pathName"].asString();
        fileProtect.isRecursive = fileProtectListValue[i]["isRecursive"].asBool();
        file_protect_path_list_.push_back(fileProtect);
    }

    SystemMonitorFileMonitorInfo fileMonitor;
    Json::Value fileMonitorListValue = rootValue["FileSystemMonitorList"];
    for (Json::ArrayIndex i = 0; i < fileMonitorListValue.size(); ++i) {
        fileMonitor.pathName = fileMonitorListValue[i]["pathName"].asString();
        fileMonitor.isRecursive = fileMonitorListValue[i]["isRecursive"].asBool();
        file_monitor_path_list_.push_back(fileMonitor);
    }

    Json::Value nicListValue = rootValue["NetworkMonitorNicList"];
    for (Json::ArrayIndex i = 0; i < nicListValue.size(); ++i) {
        network_monitor_nic_list_.push_back(nicListValue[i].asString());
    }

    if (jsonstr != NULL) {
        free(jsonstr);
    }
}

bool
SystemMonitorConfig::IsDiskMonitorPathList(const std::string& path)
{
    auto itr = disk_monitor_path_list_.find(path);
    if (itr == disk_monitor_path_list_.end()) {
        return false;
    }

    return true;
}

uint8_t
SystemMonitorConfig::GetPartitionAlarmValue(const std::string& path)
{
    auto itr = disk_monitor_path_list_.find(path);
    if (itr == disk_monitor_path_list_.end()) {
        return 100;
    }

    return itr->second.partitionAlarmValue;
}

std::vector<SystemMonitorDiskMonitorLogMoveListInfo>
SystemMonitorConfig::GetDiskMonitorLogMoveList(const SystemMonitorDiskMonitorLogMoveType type)
{
    std::vector<SystemMonitorDiskMonitorLogMoveListInfo> listInfo;
    switch(type)
    {
        case SystemMonitorDiskMonitorLogMoveType::kSoc:
            listInfo.assign(disk_monitor_soc_log_move_list_.begin(), disk_monitor_soc_log_move_list_.end());
            break;
        case SystemMonitorDiskMonitorLogMoveType::kMcu:
            listInfo.assign(disk_monitor_mcu_log_move_list_.begin(), disk_monitor_mcu_log_move_list_.end());
            break;
        default:
            break;
    }

    return listInfo;
}

void
SystemMonitorConfig::QueryPrintConfigData()
{
    /**************data print for test**************/
    STMM_INFO << "SystemMonitorConfig::LoadConfig print system_monitor_config_info_" << " LogAppName: " << system_monitor_config_info_.LogAppName
                                                                              << " LogAppDescription: " << system_monitor_config_info_.LogAppDescription
                                                                              << " ContextName: " << system_monitor_config_info_.LogContextName
                                                                              << " LogLevel: " << static_cast<uint>(system_monitor_config_info_.LogLevel)
                                                                              << " LogMode: " << static_cast<uint>(system_monitor_config_info_.LogMode)
                                                                              << " LogFilePath: " << system_monitor_config_info_.LogFilePath
                                                                              << " MaxLogFileNum: " << system_monitor_config_info_.MaxLogFileNum
                                                                              << " MaxSizeOfLogFile: " << system_monitor_config_info_.MaxSizeOfLogFile
                                                                              << " DebugSwitch: " << system_monitor_config_info_.DebugSwitch;

    for (auto& item : system_monitor_config_info_.subFunction) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print system_monitor_config_info_ subfunction [" << item.first << "]: "
                                                                                     << " shortName: " << item.second.shortName
                                                                                     << " id: " << item.second.id
                                                                                     << " monitorSwitch: " << item.second.monitorSwitch
                                                                                     << " monitorCycle: " << item.second.monitorCycle
                                                                                     << " recordFileCycle: " << item.second.recordFileCycle
                                                                                     << " recordFilePath: " << item.second.recordFilePath
                                                                                     << " isAlarm: " << item.second.isAlarm
                                                                                     << " alarmValue: " << item.second.alarmValue;
    }

    for (auto& item : disk_monitor_path_list_) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print disk_monitor_path_list_ partition: " << item.second.partition
                                                                               << " reservedSize: " << item.second.reservedSize
                                                                               << " partitionAlarmValue: " << item.second.partitionAlarmValue
                                                                               << " isIncludeSpecialPostProcessing: " << item.second.isIncludeSpecialPostProcessing
                                                                               << " isDeleteEmptyDir: " << item.second.isDeleteEmptyDir;
        for (auto& list : item.second.deleteFilesByWildcardList) {
            STMM_INFO << "SystemMonitorConfig::LoadConfig print deleteFilesByWildcardList list.pathName: " << list.pathName
                                                                              << "list.isTraverseSubdir: " << list.isTraverseSubdir;
        }

        for (auto& list : item.second.deleteFilesByPathList) {
            STMM_INFO << "SystemMonitorConfig::LoadConfig print deleteFilesByPathList list.pathName: " << list.pathName
                                                                          << "list.isTraverseSubdir: " << list.isTraverseSubdir;
        }
    }

    for (auto& item : disk_monitor_soc_log_move_list_) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print disk_monitor_soc_log_move_list_ filePrefix: " << item.filePrefix
                                                                                  << " reservedFileNum: " << item.reservedFileNum;
    }

    for (auto& item : disk_monitor_mcu_log_move_list_) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print disk_monitor_mcu_log_move_list_ filePrefix: " << item.filePrefix
                                                                                  << " reservedFileNum: " << item.reservedFileNum;
    }

    for (auto& item : process_monitor_name_list_) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print process_monitor_name_list_ name: " << item;
    }

    for (auto& item : mnand_hs_monitor_ufs_node_list_) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print mnand_hs_monitor_ufs_node_list_ node: " << item;
    }

    for (auto& item : mnand_hs_monitor_emmc_node_list_) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print mnand_hs_monitor_emmc_node_list_ node: " << item;
    }

    for (auto& item : file_protect_path_list_) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print file_protect_path_list_ path: " << item.pathName;
    }

    for (auto& item : file_monitor_path_list_) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print file_monitor_path_list_ path: " << item.pathName;
    }

    for (auto& item : network_monitor_nic_list_) {
        STMM_INFO << "SystemMonitorConfig::LoadConfig print network_monitor_nic_list_ nic: " << item;
    }
    /**************data print for test**************/
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
