#include <iostream>
#include <fstream>

#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/common/data_loader.h"
#include "phm_server/include/common/phm_server_persistency.h"

namespace hozon {
namespace netaos {
namespace phm_server {

std::mutex PHMServerConfig::mtx_;
PHMServerConfig* PHMServerConfig::instance_ = nullptr;

const int MAX_LOAD_SIZE = 1024;

// #ifdef BUILD_FOR_MDC
//     const std::string PHM_CONFIG_PATH = "/opt/usr/diag_update/mdc-llvm/conf/phm_config.json";
//     const std::string PHM_SERVER_FAULT_LIST_PATH = "/opt/usr/diag_update/mdc-llvm/conf/phm_fault_list.json";
//     const std::string PHM_SERVER_PROC_LIST_PATH = "/opt/usr/diag_update/mdc-llvm/conf/phm_proc_monitor.json";
//     const std::string PHM_FAULT_CLUSTER_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_cluster_list.json";
// #elif BUILD_FOR_J5
//     const std::string PHM_CONFIG_PATH = "/userdata/diag_update/j5/conf/phm_config.json";
//     const std::string PHM_SERVER_FAULT_LIST_PATH = "/userdata/diag_update/j5/conf/phm_fault_list.json";
//     const std::string PHM_SERVER_PROC_LIST_PATH = "/userdata/diag_update/j5/conf/phm_proc_monitor.json";
//     const std::string PHM_FAULT_CLUSTER_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_cluster_list.json";
// #elif BUILD_FOR_ORIN
//     const std::string PHM_CONFIG_PATH = "/app/runtime_service/phm_server/conf/phm_config.json";
//     const std::string PHM_SERVER_FAULT_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_list.json";
//     const std::string PHM_SERVER_PROC_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_proc_monitor.json";
//     const std::string PHM_FAULT_CLUSTER_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_cluster_list.json";
// #else
//     const std::string PHM_CONFIG_PATH = "/app/runtime_service/phm_server/conf/phm_config.json";
//     const std::string PHM_SERVER_FAULT_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_list.json";
//     const std::string PHM_SERVER_PROC_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_proc_monitor.json";
//     const std::string PHM_FAULT_CLUSTER_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_cluster_list.json";
// #endif

const std::string PHM_CONFIG_PATH = "/app/runtime_service/phm_server/conf/phm_config.json";
const std::string PHM_SERVER_FAULT_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_list.json";
const std::string PHM_SERVER_PROC_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_proc_monitor.json";
const std::string PHM_FAULT_CLUSTER_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_cluster_list.json";
const std::string PHM_FAULT_LOCK_PATH = "/app/runtime_service/phm_server/conf/phm_fault_lock.json";
const std::string PHM_FAULT_LOCK_OTA_PATH = "/app/runtime_service/phm_server/conf/phm_fault_lock_ota.json";
const std::string FAULT_LOCK_STATUS_PATH = "/opt/usr/col/fm/hz_fault_lock_status.json";

PHMServerConfig::PHMServerConfig()
: file_change_flag_(false)
{
}

PHMServerConfig*
PHMServerConfig::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new PHMServerConfig();
        }
    }

    return instance_;
}

void
PHMServerConfig::Init()
{
    PHMS_INFO << "PHMServerConfig::Init";

    int32_t ret = LoadConfig();
    if (0 != ret) {
        PHMS_ERROR << "PHMServerConfig::Init load config failed! failCode: " << ret;
    }
}

void
PHMServerConfig::DeInit()
{
    PHMS_INFO << "PHMServerConfig::DeInit";
    fault_info_map_.clear();
    dtc_fault_info_map_.clear();
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

int32_t
PHMServerConfig::LoadConfig()
{
    PHMS_INFO << "PHMServerConfig::LoadConfig enter.";
    int ret = ParseFaultJson();
    if (0 != ret) {
        PHMS_ERROR << "PHMServerConfig::LoadConfig ParseFaultJson failed. errorcode: " << ret;
        return -1;
    }
    PHMS_INFO << "PHMServerConfig::LoadConfig fault_info_map_.size: " << fault_info_map_.size()
              << " dtc_fault_info_map_.size: " << dtc_fault_info_map_.size();

    ret = ParseProcJson();
    if (0 != ret) {
        PHMS_ERROR << "PHMServerConfig::LoadConfig ParseProcJson failed. errorcode: " << ret;
        return -1;
    }
    PHMS_INFO << "PHMServerConfig::LoadConfig proc_info_map_.size: " << proc_info_map_.size();

    ret = ParseFaultClusterJson();
    if (0 != ret) {
        PHMS_ERROR << "PHMServerConfig::LoadConfig ParseFaultClusterJson failed. errorcode: " << ret;
        return -1;
    }
    PHMS_INFO << "PHMServerConfig::LoadConfig proc_info_map_.size: " << proc_info_map_.size();

    // ret = ParseFaultLockJson();
    // if (0 != ret) {
    //     PHMS_ERROR << "PHMServerConfig::LoadConfig ParseFaultLockJson failed. errorcode: " << ret;
    //     return -1;
    // }
    // PHMS_INFO << "PHMServerConfig::LoadConfig fault_lock_info_map_.size: " << fault_lock_info_map_.size();

    // QueryPrintConfigData();
    return ret;
}

bool
PHMServerConfig::GetFaultInfoByFault(const uint32_t fault, FaultInfo& faultInfo)
{
    auto itr = fault_info_map_.find(fault);
    if (itr == fault_info_map_.end()) {
        PHMS_WARN << "PHMServerConfig::GetFaultInfoByFault error fault: " << fault;
        return false;
    }

    faultInfo = itr->second;
    return true;
}

bool
PHMServerConfig::GetProcInfoByName(const std::string procName, ProcInfo& procInfo)
{
    auto itr = proc_info_map_.find(procName);
    if (itr == proc_info_map_.end()) {
        PHMS_WARN << "PHMServerConfig::GetProcInfoByName error procName: " << procName.c_str();;
        return false;
    }

    procInfo = itr->second;
    return true;
}

std::string
PHMServerConfig::GetProcInfoByFaultKey(const uint32_t faultId, const uint32_t faultObj)
{
    std::string processName;
    for (auto& i : proc_info_map_) {
        if (faultId == i.second.faultId && faultObj == i.second.faultObj) {
            processName = i.first;
            break;
        }
    }

    return processName;
}

uint8_t
PHMServerConfig::getProcRetryCountByName(const std::string& procName)
{
    auto itr = proc_info_map_.find(procName);
    if (itr == proc_info_map_.end()) {
        PHMS_WARN << "PHMServerConfig::UpdateProcRetryCountByName error procName: " << procName.c_str();;
        return 0;
    }

    return itr->second.retryCount;
}

uint32_t
PHMServerConfig::GetDtcByFault(const uint32_t fault)
{
    auto itr = fault_info_map_.find(fault);
    if (itr == fault_info_map_.end()) {
        PHMS_WARN << "PHMServerConfig::GetDtcByFault error fault: " << fault;
        return 0;
    }

    return static_cast<uint32_t>(itr->second.dtcCode);
}

uint32_t
PHMServerConfig::GetFaultByDtc(const uint32_t dtc)
{
    auto itr = dtc_fault_info_map_.find(dtc);
    if (itr == dtc_fault_info_map_.end()) {
        PHMS_WARN << "PHMServerConfig::GetFaultByDtc error dtc: " << UINT32_TO_STRING(dtc);
        return 0;
    }

    return itr->second;
}

bool
PHMServerConfig::IsOverMaxRecordCount(const uint32_t fault)
{
    auto itr = fault_info_map_.find(fault);
    if (itr == fault_info_map_.end()) {
        PHMS_WARN << "PHMServerConfig::IsOverMaxRecordCount error fault: " << fault;
        return true;
    }

    if (itr->second.faultRecordCount >= phm_config_info_.CollectFaultMaxCnt) {
        PHMS_WARN << "PHMServerConfig::IsOverMaxRecordCount over max record count. recordCount: " << itr->second.faultRecordCount;
        return true;
    }

    return false;
}

void
PHMServerConfig::UpdateFaultRecordCount(const uint32_t fault)
{
    auto itr = fault_info_map_.find(fault);
    if (itr == fault_info_map_.end()) {
        PHMS_WARN << "PHMServerConfig::UpdateFaultRecordCount error fault: " << fault;
        return;
    }

    PHMS_DEBUG << "PHMServerConfig::UpdateFaultRecordCount fault: " << fault << ", count: " << itr->second.faultRecordCount++;
}

void
PHMServerConfig::UpdateFaultStatus(const uint32_t fault, const Fault_t& faultInfo)
{
    PHMS_DEBUG << "PHMServerConfig::UpdateFaultStatus fault: " << fault << " faultStatus: " << faultInfo.faultStatus;
    auto itr = fault_info_map_.find(fault);
    if (itr == fault_info_map_.end()) {
        PHMS_WARN << "PHMServerConfig::UpdateFaultStatus error fault: " << fault;
        return;
    }

    if (faultInfo.faultStatus > 1) {
        PHMS_WARN << "PHMServerConfig::UpdateFaultStatus error faultStatus: " << faultInfo.faultStatus;
        return;
    }

    itr->second.faultDomain = faultInfo.faultDomain;
    itr->second.faultStatus = faultInfo.faultStatus;
    itr->second.faultOccurTime = faultInfo.faultOccurTime;
}

void
PHMServerConfig::QueryCurrentOccuredFault(const uint32_t fault, std::vector<uint32_t>& faultList)
{
    PHMS_DEBUG << "PHMServerConfig::QueryCurrentOccuredFault fault: " << fault;
    faultList.clear();
    auto itr = fault_info_map_.find(fault);
    if ((itr != fault_info_map_.end()) && (1 == itr->second.faultStatus)) {
        faultList.push_back(fault);
    }
}

void
PHMServerConfig::QueryAllOccuredFault(std::vector<uint32_t>& faultList)
{
    PHMS_DEBUG << "PHMServerConfig::QueryAllOccuredFault";
    faultList.clear();
    for (auto& item : fault_info_map_) {
        if (item.second.faultStatus) {
            faultList.push_back(item.first);
        }
    }
}

char*
PHMServerConfig::GetJsonAll(const char *fname)
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
PHMServerConfig::LoadPhmConfig()
{
    std::cout << "PHMServerConfig::LoadPhmConfig config path: " << PHM_CONFIG_PATH << std::endl;
    phm_config_info_.LogAppName = "H002";
    phm_config_info_.LogAppDescription = "phm_server";
    phm_config_info_.LogContextName = "PHM";
    phm_config_info_.LogLevel = 3;
    phm_config_info_.LogMode = 2;
    phm_config_info_.LogFilePath = "./";
    phm_config_info_.MaxLogFileNum = 10;
    phm_config_info_.MaxSizeOfLogFile = 20;
    phm_config_info_.CollectFaultFileSize = 1048576;
    phm_config_info_.CollectFaultMaxCnt = 500;
    phm_config_info_.DebugSwitch = "off";
    phm_config_info_.RestartProcSwitch = "on";
    phm_config_info_.AnalysisSwitch = "on";
    phm_config_info_.SystemCheckTime = 60000;
    phm_config_info_.AnalysisTime = 300000;
    phm_config_info_.LockFaultToHMISwitch = "on";
    phm_config_info_.LockFaultCurrentVersion = "ots";

    char* jsonstr = GetJsonAll(PHM_CONFIG_PATH.c_str());
    if (nullptr == jsonstr) {
        std::cout << "PHMServerConfig::LoadPhmConfig error jsonstr is nullptr." << std::endl;
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

    phm_config_info_.LogAppName = static_cast<std::string>(rootValue["LogAppName"].asString());
    phm_config_info_.LogAppDescription = static_cast<std::string>(rootValue["LogAppDescription"].asString());
    phm_config_info_.LogContextName = static_cast<std::string>(rootValue["LogContextName"].asString());
    phm_config_info_.LogLevel = static_cast<uint8_t>(rootValue["LogLevel"].asUInt());
    phm_config_info_.LogMode = static_cast<uint8_t>(rootValue["LogMode"].asUInt());
    phm_config_info_.LogFilePath = static_cast<std::string>(rootValue["LogFilePath"].asString());
    phm_config_info_.MaxLogFileNum = static_cast<uint32_t>(rootValue["MaxLogFileNum"].asUInt());
    phm_config_info_.MaxSizeOfLogFile = static_cast<uint32_t>(rootValue["MaxSizeOfLogFile"].asUInt());
    phm_config_info_.CollectFaultFileSize = static_cast<uint32_t>(rootValue["CollectFaultFileSize"].asUInt());
    phm_config_info_.CollectFaultMaxCnt = static_cast<uint32_t>(rootValue["CollectFaultMaxCnt"].asUInt());
    phm_config_info_.DebugSwitch = static_cast<std::string>(rootValue["DebugSwitch"].asString());
    phm_config_info_.RestartProcSwitch = static_cast<std::string>(rootValue["RestartProcSwitch"].asString());
    phm_config_info_.AnalysisSwitch = static_cast<std::string>(rootValue["AnalysisSwitch"].asString());
    phm_config_info_.SystemCheckTime = static_cast<uint32_t>(rootValue["SystemCheckTime"].asUInt());
    phm_config_info_.ProcCheckTime = static_cast<uint32_t>(rootValue["ProcCheckTime"].asUInt());
    phm_config_info_.AnalysisTime = static_cast<uint32_t>(rootValue["AnalysisTime"].asUInt());
    phm_config_info_.LockFaultToHMISwitch = static_cast<std::string>(rootValue["LockFaultToHMISwitch"].asString());
    phm_config_info_.LockFaultCurrentVersion = static_cast<std::string>(rootValue["LockFaultCurrentVersion"].asString());

    if (jsonstr != NULL) {
        free(jsonstr);
    }
}

int32_t
PHMServerConfig::ParseFaultJson()
{
    PHMS_INFO << "PHMServerConfig::ParseFaultJson fault list path: " << PHM_SERVER_FAULT_LIST_PATH;
    char* jsonstr = GetJsonAll(PHM_SERVER_FAULT_LIST_PATH.c_str());
    if (nullptr == jsonstr) {
        PHMS_ERROR << "PHMServerConfig::ParseFaultJson jsonstr is nullptr.";
        return -1;
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

        return -2;
    }

    Json::Value & resultValue = rootValue["FaultList"];
    for (uint32_t i = 0; i < resultValue.size(); i++) {
        Json::Value clusterValue = resultValue[i]["FaultCluster"];
        uint32_t clusterId = static_cast<uint32_t>(clusterValue["ClusterID"].asUInt());
        std::string process = clusterValue["Process"].asString();

        Json::Value faultsValue = resultValue[i]["Faults"];
        for (uint32_t j = 0; j < faultsValue.size(); j++) {
            FaultInfo faultInfo;
            faultInfo.faultClusterId = clusterId;
            faultInfo.faultProcess = process;
            uint32_t key = static_cast<uint32_t>(faultsValue[j]["Key"].asUInt());
            faultInfo.faultObj = static_cast<uint8_t>(key % 100);
            faultInfo.faultId = static_cast<uint32_t>((key - faultInfo.faultObj) / 100);
            faultInfo.faultLevel = static_cast<uint16_t>(faultsValue[j]["Level"].asUInt());
            std::string strDtc = faultsValue[j]["DtcCode"].asString();
            faultInfo.dtcCode = static_cast<uint32_t>(std::strtoul(strDtc.c_str(), 0, 0));
            faultInfo.faultDscribe = faultsValue[j]["Dscribe"].asString();
            faultInfo.faultAction.record = static_cast<uint8_t>(faultsValue[j]["Action"]["record"].asUInt());
            faultInfo.faultAction.analysis = static_cast<uint8_t>(faultsValue[j]["Action"]["analysis"].asUInt());
            faultInfo.faultAction.strategy.dtcMapping = static_cast<uint8_t>(faultsValue[j]["Action"]["Strategy"]["dtcMapping"].asUInt());
            faultInfo.faultAction.strategy.notifyMcu = static_cast<uint8_t>(faultsValue[j]["Action"]["Strategy"]["notify_mcu"].asUInt());
            faultInfo.faultAction.strategy.notifyApp = static_cast<uint8_t>(faultsValue[j]["Action"]["Strategy"]["notify_app"].asUInt());
            faultInfo.faultAction.strategy.restartproc = static_cast<uint8_t>(faultsValue[j]["Action"]["Strategy"]["restartproc"].asUInt());

            faultInfo.faultDomain = "";
            faultInfo.faultOccurTime = 0;
            faultInfo.faultStatus = 0;
            faultInfo.faultRecordCount = 0;

            fault_info_map_.insert(std::make_pair(key, faultInfo));
            if (faultInfo.dtcCode) {
                dtc_fault_info_map_.insert(std::make_pair(faultInfo.dtcCode, key));
            }
        }
    }

    if (jsonstr != NULL) {
        free(jsonstr);
    }

    return 0;
}

int32_t
PHMServerConfig::ParseProcJson()
{
    PHMS_INFO << "PHMServerConfig::ParseProcJson proc list path: " << PHM_SERVER_PROC_LIST_PATH;
    char* jsonstr = GetJsonAll(PHM_SERVER_PROC_LIST_PATH.c_str());
    if (nullptr == jsonstr) {
        PHMS_ERROR << "PHMServerConfig::ParseProcJson jsonstr is nullptr.";
        return -1;
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

        return -2;
    }

    Json::Value & resultValue = rootValue["ProcMonitor"];
    ProcInfo procInfo;
    for (uint32_t i = 0; i < resultValue.size(); i++) {
        procInfo.procName = static_cast<std::string>(resultValue[i]["ProcName"].asString());
        procInfo.faultId = static_cast<uint32_t>(resultValue[i]["FaultId"].asUInt());
        procInfo.faultObj = static_cast<uint8_t>(resultValue[i]["FaultObj"].asUInt());
        procInfo.retryCount = static_cast<uint8_t>(resultValue[i]["RetryCount"].asUInt());

        proc_info_map_.insert(std::make_pair(procInfo.procName, procInfo));
    }

    if (jsonstr != NULL) {
        free(jsonstr);
    }

    return 0;
}


int32_t
PHMServerConfig::ParseFaultClusterJson()
{
    PHMS_INFO << "PHMServerConfig::ParseFaultClusterJson config path: " << PHM_FAULT_CLUSTER_LIST_PATH;
    char* jsonstr = GetJsonAll(PHM_FAULT_CLUSTER_LIST_PATH.c_str());
    if (nullptr == jsonstr) {
        PHMS_ERROR << "PHMServerConfig::ParseFaultClusterJson jsonstr is nullptr.";
        return -1;
    }

    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    Json::Value rootValue;
    JSONCPP_STRING errs;
    bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &rootValue, &errs);
    if (!res || !errs.empty()) {
        PHMS_ERROR << "PHMServerConfig::ParseFaultClusterJson jsonreader error.";
        if (jsonstr != NULL) {
            free(jsonstr);
        }
        return -2;
    }

    Json::Value& faultClusterValue = rootValue["Faults"];
    std::vector<FaultClusterItem> faultCluster;
    for (uint32_t i = 0; i < faultClusterValue.size(); i++) {
        uint32_t faultKey = static_cast<uint32_t>(faultClusterValue[i]["FaultKey"].asUInt());
        faultCluster.clear();
        Json::Value & clusterValue = faultClusterValue[i]["Cluster"];
        FaultClusterItem clusterItem;
        for (uint32_t j = 0; j < clusterValue.size(); j++) {
            clusterItem.clusterName = clusterValue[j]["Name"].asString();
            clusterItem.bitPosition = static_cast<uint8_t>(clusterValue[j]["BitPosition"].asUInt());
            faultCluster.emplace_back(clusterItem);
        }

        phm_fault_cluster_level_map_.insert(std::make_pair(faultKey, faultCluster));
    }

    if (jsonstr != NULL) {
        free(jsonstr);
    }

    return 0;
}

int32_t
PHMServerConfig::ParseFaultLockJson()
{
    PHMS_INFO << "PHMServerConfig::ParseFaultLockJson config path: " << FAULT_LOCK_STATUS_PATH;
    std::string lockFaultFile = PHM_FAULT_LOCK_PATH;
    if (phm_config_info_.LockFaultCurrentVersion == "ota") {
        lockFaultFile = PHM_FAULT_LOCK_OTA_PATH;
    }

    if (0 != access(FAULT_LOCK_STATUS_PATH.c_str(), 0)) {
        std::ofstream ofs(FAULT_LOCK_STATUS_PATH);
        ofs.close();
        PHMServerPersistency::getInstance()->CopyFile(lockFaultFile, FAULT_LOCK_STATUS_PATH);
    }

    ReadJsonFileReturnCode ret = ReadJsonFile(FAULT_LOCK_STATUS_PATH, fault_lock_info_map_);
    if (ret != ReadJsonFileReturnCode::OK) {
        PHMS_ERROR << "PHMServerConfig::ParseFaultLockJson load fault lock list file error!";
        return -1;
    }

    return 0;
}

bool
PHMServerConfig::getFaultCluster(const uint32_t faultKey, std::vector<FaultClusterItem>& outCluster)
{
    auto iter = phm_fault_cluster_level_map_.find(faultKey);
    if (iter == phm_fault_cluster_level_map_.end()) {
        return false;
    }

    outCluster = iter->second;
    return true;
}

// Fault lock interface
std::string
PHMServerConfig::GetLockFaultToHMISwitch()
{
    return phm_config_info_.LockFaultToHMISwitch;
}

std::string
PHMServerConfig::GetLockFaultCurrentVersion()
{
    return phm_config_info_.LockFaultCurrentVersion;
}

bool
PHMServerConfig::IsBlockedFault(uint32_t fault)
{
    auto itrFaultLock = fault_lock_info_map_.find(fault);
    if (itrFaultLock == fault_lock_info_map_.end()) {
        return false;
    }

    if (itrFaultLock->second.isBlockedFault) {
        return true;
    }

    return false;
}

bool
PHMServerConfig::GetLockFaultInfo(uint32_t fault, FaultLockInfo &faultLockInfo)
{
    auto itrFaultLock = fault_lock_info_map_.find(fault);
    if (itrFaultLock == fault_lock_info_map_.end()) {
        return false;
    }

    faultLockInfo.faultId = itrFaultLock->second.faultId;
    faultLockInfo.faultObj = itrFaultLock->second.faultObj;
    faultLockInfo.lockCount = itrFaultLock->second.lockCount;
    faultLockInfo.recoverCount = itrFaultLock->second.recoverCount;
    faultLockInfo.faultToHMIData = itrFaultLock->second.faultToHMIData;
    faultLockInfo.lockFaultToHMIData = itrFaultLock->second.lockFaultToHMIData;
    faultLockInfo.isBlockedFault = itrFaultLock->second.isBlockedFault;
    faultLockInfo.faultCount = itrFaultLock->second.faultCount;
    faultLockInfo.isHandled = itrFaultLock->second.isHandled;
    faultLockInfo.lockedNumber = itrFaultLock->second.lockedNumber;
    faultLockInfo.faultRecoverCount = itrFaultLock->second.faultRecoverCount;
    faultLockInfo.isNeedToRecover = itrFaultLock->second.isNeedToRecover;

    return true;
}

void
PHMServerConfig::GetLockFaultInfos(std::vector<FaultLockInfo> &faultLockInfos)
{
    if(fault_lock_info_map_.size() <= 0) {
        return;
    }

    for (auto& item : fault_lock_info_map_) {
        FaultLockInfo faultLockInfo;

        faultLockInfo.faultId = item.second.faultId;
        faultLockInfo.faultObj = item.second.faultObj;
        faultLockInfo.lockCount = item.second.lockCount;
        faultLockInfo.recoverCount = item.second.recoverCount;
        faultLockInfo.faultToHMIData = item.second.faultToHMIData;
        faultLockInfo.lockFaultToHMIData = item.second.lockFaultToHMIData;
        faultLockInfo.isBlockedFault = item.second.isBlockedFault;
        faultLockInfo.faultCount = item.second.faultCount;
        faultLockInfo.isHandled = item.second.isHandled;
        faultLockInfo.lockedNumber = item.second.lockedNumber;
        faultLockInfo.faultRecoverCount = item.second.faultRecoverCount;
        faultLockInfo.isNeedToRecover = item.second.isNeedToRecover;

        faultLockInfos.emplace_back(faultLockInfo);
    }
}

void
PHMServerConfig::SetLockFaultInfo(uint32_t fault, FaultLockInfo &faultLockInfo)
{
    std::lock_guard<std::mutex> lck(mtx_);
    auto itrFaultLock = fault_lock_info_map_.find(fault);
    if (itrFaultLock == fault_lock_info_map_.end()) {
        return;
    }

    fault_lock_info_map_[fault] = faultLockInfo;
    this->file_change_flag_ = true;
}

void
PHMServerConfig::SetLockFaultInfo(uint32_t fault, uint32_t lockedNumber)
{
    std::lock_guard<std::mutex> lck(mtx_);
    auto itrFaultLock = fault_lock_info_map_.find(fault);
    if (itrFaultLock == fault_lock_info_map_.end()) {
        return;
    }

    if (lockedNumber != fault_lock_info_map_[fault].lockedNumber) {
        PHMS_WARN << "PHMServerConfig::SetLockFaultInfo sync lock fault status fault: " << fault << " lockedNumber: " << lockedNumber
                                                    << ", status file lockedNumber: " << fault_lock_info_map_[fault].lockedNumber;
        fault_lock_info_map_[fault].lockedNumber = lockedNumber;
        fault_lock_info_map_[fault].faultCount = 0;
        this->file_change_flag_ = true;
    }
}

void
PHMServerConfig::LockFaultDataToFile()
{
    if (false == this->file_change_flag_) {
        return;
    }

    std::vector<FaultLockInfo> faultLockInfos;
    GetLockFaultInfos(faultLockInfos);
    // update hz_fault_lock
    auto ret = WriteJsonFile(FAULT_LOCK_STATUS_PATH, faultLockInfos);
    // PHMS_DEBUG << "PHMServerConfig::LockFaultDataToFile fault_lock_info_map_ size: " << fault_lock_info_map_.size();
    if (ret != ReadJsonFileReturnCode::OK) {
        PHMS_ERROR << "PHMServerConfig::LockFaultDataToFile update fm fault lock list file error!";
        return;
    }

    this->file_change_flag_ = false;
}

void
PHMServerConfig::QueryPrintConfigData()
{
    /**************data print for test**************/
    PHMS_INFO << "PHMServerConfig::LoadConfig print phm_config_info_"
              << " LogAppName: " << phm_config_info_.LogAppName
              << " LogAppDescription: " << phm_config_info_.LogAppDescription
              << " LogContextName: " << phm_config_info_.LogContextName
              << " LogLevel: " << static_cast<uint>(phm_config_info_.LogLevel)
              << " LogMode: " << static_cast<uint>(phm_config_info_.LogMode)
              << " LogFilePath: " << phm_config_info_.LogFilePath
              << " MaxLogFileNum: " << phm_config_info_.MaxLogFileNum
              << " MaxSizeOfLogFile: " << phm_config_info_.MaxSizeOfLogFile
              << " CollectFaultFileSize: " << phm_config_info_.CollectFaultFileSize
              << " CollectFaultMaxCnt: " << phm_config_info_.CollectFaultMaxCnt
              << " DebugSwitch: " << phm_config_info_.DebugSwitch
              << " AnalysisSwitch: " << phm_config_info_.AnalysisSwitch
              << " RestartProcSwitch: " << phm_config_info_.RestartProcSwitch
              << " SystemCheckTime: " << phm_config_info_.SystemCheckTime
              << " LockFaultToHMISwitch: " << phm_config_info_.LockFaultToHMISwitch
              << " LockFaultCurrentVersion: " << phm_config_info_.LockFaultCurrentVersion;

    int i = 0;
    for (auto& item : fault_info_map_) {
        PHMS_INFO << "PHMServerConfig::LoadConfig print fault_info_map_[" << i << "]"
                  << " faultId: " << item.second.faultId
                  << " faultObj: " << static_cast<uint>(item.second.faultObj)
                  << " faultAction.record: " << static_cast<uint>(item.second.faultAction.record)
                  << " faultAction.analysis: " << static_cast<uint>(item.second.faultAction.analysis)
                  << " faultAction.dtcMapping: " << static_cast<uint>(item.second.faultAction.strategy.dtcMapping)
                  << " faultAction.strategy.notifyApp: " << static_cast<uint>(item.second.faultAction.strategy.notifyApp)
                  << " faultAction.strategy.notifyMcu: " << static_cast<uint>(item.second.faultAction.strategy.notifyMcu)
                  << " faultAction.strategy.restartproc: " << static_cast<uint>(item.second.faultAction.strategy.restartproc)
                  << " faultLevel: " << item.second.faultLevel
                  << " faultDomain: " << item.second.faultDomain
                  << " faultClusterId: " << item.second.faultClusterId
                  << " faultProcess: " << item.second.faultProcess
                  << " faultDscribe: " << item.second.faultDscribe
                  << " dtcCode: " << UINT32_TO_STRING(item.second.dtcCode)
                  << " faultOccurTime: " << UINT32_TO_STRING(item.second.faultOccurTime)
                  << " faultStatus: " << static_cast<uint>(item.second.faultStatus);
        i++;
    }

    i = 0;
    for (auto& item : dtc_fault_info_map_) {
        PHMS_INFO << "PHMServerConfig::LoadConfig print dtc_fault_info_map_[" << i << "]"
                  << " dtc: " << item.first
                  << " faultkey: " << item.second;
        i++;
    }

    i = 0;
    for (auto& item : proc_info_map_) {
        PHMS_INFO << "PHMServerConfig::LoadConfig print proc_info_map_[" << i << "]"
                  << " ProcName: " << item.second.procName.c_str()
                  << " faultId: " << item.second.faultId
                  << " faultObj: " << static_cast<uint>(item.second.faultObj)
                  << " RetryCount: " << static_cast<uint>(item.second.retryCount);
        i++;
    }

    i = 0;
    for (auto& item : fault_lock_info_map_) {
        PHMS_INFO << "PHMServerConfig::LoadConfig print fault_lock_info_map_[" << i << "]"
                  << " faultId: " << item.second.faultId
                  << " faultObj: " << item.second.faultObj
                  << " lockCount: " << item.second.lockCount
                  << " recoverCount: " << item.second.recoverCount
                  << " faultToHMIData: " << item.second.faultToHMIData
                  << " lockFaultToHMIData: " << item.second.lockFaultToHMIData
                  << " isBlockedFault: " << item.second.isBlockedFault
                  << " faultCount: " << item.second.faultCount
                  << " isHandled: " << item.second.isHandled
                  << " lockedNumber: " << item.second.lockedNumber
                  << " faultRecoverCount: " << item.second.faultRecoverCount
                  << " isNeedToRecover: " << item.second.isNeedToRecover;
        i++;
    }
    /**************data print for test**************/
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
/* EOF */
