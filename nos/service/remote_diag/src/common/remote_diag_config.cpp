#include <iostream>
#include <regex>
#include <fstream>

#include "json/json.h"
#include "remote_diag/include/common/remote_diag_logger.h"
#include "remote_diag/include/common/remote_diag_config.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

RemoteDiagConfig* RemoteDiagConfig::instance_ = nullptr;
std::mutex RemoteDiagConfig::mtx_;

const int MAX_LOAD_SIZE = 1024;

#ifdef BUILD_FOR_MDC
    const std::string REMOTE_DIAG_CONFIG_PATH = "/opt/usr/diag_update/mdc-llvm/conf/remote_diag_config.json";
#elif BUILD_FOR_J5
    const std::string REMOTE_DIAG_CONFIG_PATH = "/userdata/diag_update/j5/conf/remote_diag_config.json";
#elif BUILD_FOR_ORIN
    const std::string REMOTE_DIAG_CONFIG_PATH = "/app/conf/remote_diag_config.json";
#else
    const std::string REMOTE_DIAG_CONFIG_PATH = "/app/conf/remote_diag_config.json";
#endif

const std::string DIDS_DATA_FILE_PATH = "/cfg/dids/dids.json";
const std::string DIDS_DATA_BACK_FILE_PATH = "/cfg/dids/dids.json_bak_1";

RemoteDiagConfig::RemoteDiagConfig()
: vin_number_("12345678901234568")
{
}

RemoteDiagConfig*
RemoteDiagConfig::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new RemoteDiagConfig();
        }
    }

    return instance_;
}

void
RemoteDiagConfig::Init()
{
    DGR_INFO << "RemoteDiagConfig::Init";
    // get vin number
    std::string vin = GetVinNumber();
    if ("" != GetVinNumber()) {
        vin_number_ = vin;
        LoadRemoteDiagConfig();
    }

    DGR_INFO << "RemoteDiagConfig::Init vin_number_: " << vin_number_;
    // QueryPrintConfigData();
}

void
RemoteDiagConfig::DeInit()
{
    DGR_INFO << "RemoteDiagConfig::DeInit";
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
RemoteDiagConfig::LoadRemoteDiagConfig()
{
    ParseRemoteDiagConfigJson();
}

char*
RemoteDiagConfig::GetJsonAll(const char *fname)
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
RemoteDiagConfig::ParseRemoteDiagConfigJson()
{
    std::cout << "RemoteDiagConfig::LoadRemoteDiagConfig configPath: " << REMOTE_DIAG_CONFIG_PATH << std::endl;
    // default remote diag config
    {
        // log default config
        remote_diag_config_info_.LogAppName = "REMOTE_DIAG";
        remote_diag_config_info_.LogAppDescription = "remote_diag";
        remote_diag_config_info_.LogContextName = "REMOTE_DIAG";
        remote_diag_config_info_.LogLevel = 3;
        remote_diag_config_info_.LogMode = 2;
        remote_diag_config_info_.LogFilePath = "/opt/usr/log/soc_log";
        remote_diag_config_info_.MaxLogFileNum = 10;
        remote_diag_config_info_.MaxSizeOfLogFile = 20;
        remote_diag_config_info_.DebugSwitch = "off";

        // file transfer default config
        remote_diag_config_info_.FileTransferSize = 1024;
        remote_diag_config_info_.FileCompressDirPath = "./";
        remote_diag_config_info_.FileDownloadDirPath = "./";

        // remote address default config
        remote_diag_config_info_.DiagServerAddress = 0x10C3;
        remote_diag_config_info_.RemoteAddressList = {0xF000, 0xF001};
        remote_diag_config_info_.DoipAddressList = {0x10CA, 0x10CB};
        remote_diag_config_info_.DocanAddressList = {0x10C4, 0x10C7, 0X10C8, 0x10C9, 0x10C5, 0x10CC};

        // rocketmq default config
        remote_diag_config_info_.RocketMQReqGroup = "GID_remotediag_req";
        remote_diag_config_info_.RocketMQResGroup = "GID_remotediag_res";
        remote_diag_config_info_.RocketMQAddress = "10.4.53.97:9876";
        remote_diag_config_info_.RocketMQReqTopic = "remote_diag_req";
        remote_diag_config_info_.RocketMQResTopic = "remote_diag_res";
        remote_diag_config_info_.RocketMQResTag =  "remote_diag";
        remote_diag_config_info_.RocketMQResKeys = "neta";
        remote_diag_config_info_.RocketMQDomain = "diag";
        remote_diag_config_info_.RocketMQAccessKey = "8jCJ3ik16u6yxEZwsekpf0Cd";
        remote_diag_config_info_.RocketMQSecretKey = "ZxeTqqM3j6kJV9dpCh9J2llT";
    }
    // default remote diag config

    char* jsonstr = GetJsonAll(REMOTE_DIAG_CONFIG_PATH.c_str());

    if (nullptr == jsonstr) {
        std::cout << "RemoteDiagConfig::ParsePhmConfigJson error jsonstr is nullptr." << std::endl;
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
    remote_diag_config_info_.LogAppName = static_cast<std::string>(rootValue["LogAppName"].asString());
    remote_diag_config_info_.LogAppDescription = static_cast<std::string>(rootValue["LogAppDescription"].asString());
    remote_diag_config_info_.LogContextName = static_cast<std::string>(rootValue["LogContextName"].asString());
    remote_diag_config_info_.LogLevel = static_cast<uint8_t>(rootValue["LogLevel"].asUInt());
    remote_diag_config_info_.LogMode = static_cast<uint8_t>(rootValue["LogMode"].asUInt());
    remote_diag_config_info_.LogFilePath = static_cast<std::string>(rootValue["LogFilePath"].asString());
    remote_diag_config_info_.MaxLogFileNum = static_cast<uint32_t>(rootValue["MaxLogFileNum"].asUInt());
    remote_diag_config_info_.MaxSizeOfLogFile = static_cast<uint32_t>(rootValue["MaxSizeOfLogFile"].asUInt());
    remote_diag_config_info_.DebugSwitch = static_cast<std::string>(rootValue["DebugSwitch"].asString());

    // load file transfer config
    remote_diag_config_info_.FileTransferSize = static_cast<uint32_t>(rootValue["FileTransferSize"].asUInt());
    remote_diag_config_info_.FileCompressDirPath = static_cast<std::string>(rootValue["FileCompressDirPath"].asString());
    remote_diag_config_info_.FileDownloadDirPath = static_cast<std::string>(rootValue["FileDownloadDirPath"].asString());

    // load remote address config
    remote_diag_config_info_.DiagServerAddress = static_cast<uint16_t>(std::strtoul(rootValue["DiagServerAddress"].asString().c_str(), 0, 0));
    remote_diag_config_info_.RemoteAddressList.clear();
    Json::Value & remoteAddressValue = rootValue["RemoteAddressList"];
    for (uint32_t i = 0; i < remoteAddressValue.size(); i++) {
        remote_diag_config_info_.RemoteAddressList.emplace_back(static_cast<uint16_t>(std::strtoul(remoteAddressValue[i].asString().c_str(), 0, 0)));
    }

    remote_diag_config_info_.DoipAddressList.clear();
    Json::Value & doipAddressValue = rootValue["DoipAddressList"];
    for (uint32_t i = 0; i < doipAddressValue.size(); i++) {
        remote_diag_config_info_.DoipAddressList.emplace_back(static_cast<uint16_t>(std::strtoul(doipAddressValue[i].asString().c_str(), 0, 0)));
    }

    remote_diag_config_info_.DocanAddressList.clear();
    Json::Value & docanAddressValue = rootValue["DocanAddressList"];
    for (uint32_t i = 0; i < docanAddressValue.size(); i++) {
        remote_diag_config_info_.DocanAddressList.emplace_back(static_cast<uint16_t>(std::strtoul(docanAddressValue[i].asString().c_str(), 0, 0)));
    }

    // load rocketmq config
    remote_diag_config_info_.RocketMQReqGroup = static_cast<std::string>(rootValue["RocketMQReqGroup"].asString())  + "_" + vin_number_;
    remote_diag_config_info_.RocketMQResGroup = static_cast<std::string>(rootValue["RocketMQResGroup"].asString())  + "_" + vin_number_;
    remote_diag_config_info_.RocketMQAddress = static_cast<std::string>(rootValue["RocketMQAddress"].asString());
    remote_diag_config_info_.RocketMQReqTopic = static_cast<std::string>(rootValue["RocketMQReqTopic"].asString()) + "_" + vin_number_;
    remote_diag_config_info_.RocketMQResTopic = static_cast<std::string>(rootValue["RocketMQResTopic"].asString()) + "_" + vin_number_;
    remote_diag_config_info_.RocketMQResTag = static_cast<std::string>(rootValue["RocketMQResTag"].asString());
    remote_diag_config_info_.RocketMQResKeys = static_cast<std::string>(rootValue["RocketMQResKeys"].asString());
    remote_diag_config_info_.RocketMQDomain = static_cast<std::string>(rootValue["RocketMQDomain"].asString());
    remote_diag_config_info_.RocketMQAccessKey = static_cast<std::string>(rootValue["RocketMQAccessKey"].asString());
    remote_diag_config_info_.RocketMQSecretKey = static_cast<std::string>(rootValue["RocketMQSecretKey"].asString());

    if (jsonstr != NULL) {
        free(jsonstr);
    }
}

std::string
RemoteDiagConfig::GetVinNumber()
{
    std::string filePath = "";
    if (0 == access(DIDS_DATA_FILE_PATH.c_str(), F_OK)) {
        filePath = DIDS_DATA_FILE_PATH;
    }
    else {
        if (0 == access(DIDS_DATA_BACK_FILE_PATH.c_str(), F_OK)) {
            filePath = DIDS_DATA_BACK_FILE_PATH;
        }
    }

    if ("" == filePath) {
        return "";
    }

    std::ifstream ifs;
    ifs.open(filePath, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        return "";
    }

    std::string vin = "";
    std::string str = "";
    bool bFind = false;
    while (getline(ifs, str))
    {
        if (std::string::npos != str.find("F190")) {
            bFind = true;
            continue;
        }

        if (bFind && (std::string::npos != str.find("string"))) {
            auto vec = Split(str, "\"");
            if (vec.size() > 3) {
                vin = vec[3];
            }

            break;
        }
    }

    ifs.close();
    return vin;
}

std::vector<std::string>
RemoteDiagConfig::Split(const std::string& inputStr, const std::string& regexStr)
{
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

void
RemoteDiagConfig::QueryPrintConfigData()
{
    /**************data print for test**************/
    DGR_INFO << "RemoteDiagConfig::LoadConfig print remote_diag_config_info_" << " LogAppName: " << remote_diag_config_info_.LogAppName
                                                                              << " LogAppDescription: " << remote_diag_config_info_.LogAppDescription
                                                                              << " ContextName: " << remote_diag_config_info_.LogContextName
                                                                              << " LogLevel: " << static_cast<uint>(remote_diag_config_info_.LogLevel)
                                                                              << " LogMode: " << static_cast<uint>(remote_diag_config_info_.LogMode)
                                                                              << " LogFilePath: " << remote_diag_config_info_.LogFilePath
                                                                              << " MaxLogFileNum: " << remote_diag_config_info_.MaxLogFileNum
                                                                              << " MaxSizeOfLogFile: " << remote_diag_config_info_.MaxSizeOfLogFile
                                                                              << " DebugSwitch: " << remote_diag_config_info_.DebugSwitch
                                                                              << " FileTransferSize: " << remote_diag_config_info_.FileTransferSize
                                                                              << " FileCompressDirPath: " << remote_diag_config_info_.FileCompressDirPath
                                                                              << " FileDownloadDirPath: " << remote_diag_config_info_.FileDownloadDirPath
                                                                              << " DiagServerAddress: " << UINT16_TO_STRING(remote_diag_config_info_.DiagServerAddress)
                                                                              << " RemoteAddressList: " << UINT16_VEC_TO_STRING(remote_diag_config_info_.RemoteAddressList)
                                                                              << " DoipAddressList: " << UINT16_VEC_TO_STRING(remote_diag_config_info_.DoipAddressList)
                                                                              << " DocanAddressList: " << UINT16_VEC_TO_STRING(remote_diag_config_info_.DocanAddressList)
                                                                              << " RocketMQReqGroup: " << remote_diag_config_info_.RocketMQReqGroup
                                                                              << " RocketMQResGroup: " << remote_diag_config_info_.RocketMQResGroup
                                                                              << " RocketMQAddress: " << remote_diag_config_info_.RocketMQAddress
                                                                              << " RocketMQReqTopic: " << remote_diag_config_info_.RocketMQReqTopic
                                                                              << " RocketMQResTopic: " << remote_diag_config_info_.RocketMQResTopic
                                                                              << " RocketMQResTag: " << remote_diag_config_info_.RocketMQResTag
                                                                              << " RocketMQResKeys: " << remote_diag_config_info_.RocketMQResKeys
                                                                              << " RocketMQDomain: " << remote_diag_config_info_.RocketMQDomain
                                                                              << " RocketMQAccessKey: " << remote_diag_config_info_.RocketMQAccessKey
                                                                              << " RocketMQSecretKey: " << remote_diag_config_info_.RocketMQSecretKey;
    /**************data print for test**************/
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon