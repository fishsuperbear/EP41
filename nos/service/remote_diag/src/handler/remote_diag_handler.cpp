/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: remote diag handler
*/

#include <thread>
#include <regex>
#include <sys/stat.h>

#include "remote_diag/include/common/remote_diag_logger.h"
#include "remote_diag/include/handler/remote_diag_handler.h"
#include "remote_diag/include/common/remote_diag_config.h"
#include "remote_diag/include/extension/remote_diag_file_transfer.h"
#include "remote_diag/include/extension/remote_diag_dynamic_plugin.h"
#include "remote_diag/include/extension/remote_diag_switch_control.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

RemoteDiagHandler* RemoteDiagHandler::instance_ = nullptr;
std::mutex RemoteDiagHandler::mtx_;

RemoteDiagHandler::RemoteDiagHandler()
: doip_dispatcher_(new RemoteDiagDoipDispatcher())
, rocketmq_dispatcher_(new RemoteDiagRocketMQDispatcher())
, curr_extension_(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_DEFAULT)
, vehicle_speed_(0)
{
}

RemoteDiagHandler*
RemoteDiagHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new RemoteDiagHandler();
        }
    }

    return instance_;
}

void
RemoteDiagHandler::Init()
{
    DGR_INFO << "RemoteDiagHandler::Init";
    const RemoteDiagConfigInfo& configInfo = RemoteDiagConfig::getInstance()->GetRemoteDiagConfigInfo();
    for (auto& item : configInfo.RemoteAddressList) {
        remote_diag_request_status_info_[item] = false;
    }

    if (nullptr != doip_dispatcher_) {
        doip_dispatcher_->Init();
    }

    if (nullptr != rocketmq_dispatcher_) {
        rocketmq_dispatcher_->Init();
    }
}

void
RemoteDiagHandler::DeInit()
{
    DGR_INFO << "RemoteDiagHandler::DeInit";
    if (nullptr != rocketmq_dispatcher_) {
        rocketmq_dispatcher_->DeInit();
        delete rocketmq_dispatcher_;
        rocketmq_dispatcher_ = nullptr;
    }

    if (nullptr != doip_dispatcher_) {
        doip_dispatcher_->DeInit();
        delete doip_dispatcher_;
        doip_dispatcher_ = nullptr;
    }

    remote_diag_request_status_info_.clear();

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }

    DGR_INFO << "RemoteDiagHandler::DeInit finish!";
}

void
RemoteDiagHandler::RecvRemoteMessage(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagHandler::RecvRemoteMessage " << PrintJsonData(message);
    ConnectConfirmReponse(message);
    if (REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kUdsCommand] == message["TYPE"].asString()) {
        UdsCommandReuqest(message);
    }
    else if (REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kFileUpload] == message["TYPE"].asString()) {
        FileUploadReuqest(message);
    }
    else if (REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kFileDownload] == message["TYPE"].asString()) {
        FileDownloadReuqest(message);
    }
    else if (REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kPluginRun] == message["TYPE"].asString()) {
        PluginRunReuqest(message);
    }
    else if (REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kPluginRunResult] == message["TYPE"].asString()) {
        PluginRunResultReuqest(message);
    }
    else if (REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kSwitchControl] == message["TYPE"].asString()) {
        SwitchControlReuqest(message);
    }
    else if (REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kQueryDirInfo] == message["TYPE"].asString()) {
        QueryDirInfoReuqest(message);
    }
    else {
        DGR_WARN << "RemoteDiagHandler::RecvRemoteMessage error dataType: " << message["TYPE"].asString();
    }
}

void
RemoteDiagHandler::ReplyRemoteMessage(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagHandler::ReplyRemoteMessage " << PrintJsonData(message);
    if (nullptr != rocketmq_dispatcher_) {
        rocketmq_dispatcher_->SendMessage(message);
    }
}

void
RemoteDiagHandler::SetRemoteRequestStatus(const uint16_t address, bool status)
{
    std::lock_guard<std::mutex> lck(mtx_);
    auto itr = remote_diag_request_status_info_.find(address);
    if (itr == remote_diag_request_status_info_.end()) {
        return;
    }

    remote_diag_request_status_info_[address] = status;
}

void
RemoteDiagHandler::FileDownloadCompleteCallback(const std::string& filePath)
{
    DGR_INFO << "RemoteDiagHandler::FileDownloadCompleteCallback filePath: " << filePath;
    RemoteDiagDynamicPlugin::getInstance()->CheckPluginResult(filePath);
}

bool
RemoteDiagHandler::IsExecutableRequest(const uint16_t sa, const uint16_t ta, DiagUdsBusType& busType, DiagUdsNrcErrc& nrc)
{
    // Determine if the vehicle is in motion
    if (nullptr != doip_dispatcher_) {
        bool bResult = doip_dispatcher_->GetVehicleSpeed();
        if (!bResult) {
            nrc = DiagUdsNrcErrc::kVehicleInMotion;
            return false;
        }
    }

    if (vehicle_speed_ > 0) {
        nrc = DiagUdsNrcErrc::kVehicleInMotion;
        return false;
    }

    const RemoteDiagConfigInfo& configInfo = RemoteDiagConfig::getInstance()->GetRemoteDiagConfigInfo();
    auto itr = find(configInfo.RemoteAddressList.begin(), configInfo.RemoteAddressList.end(), sa);
    if (itr == configInfo.RemoteAddressList.end()) {
        nrc = DiagUdsNrcErrc::kErrorSa;
        return false;
    }
    else {
        if (remote_diag_request_status_info_[sa]) {
            nrc = DiagUdsNrcErrc::kRequestBusy;
            return false;
        }
    }

    if (configInfo.DiagServerAddress == ta) {
        busType = DiagUdsBusType::kServer;
        return true;
    }

    for (auto& item : configInfo.DoipAddressList) {
        if (item == ta) {
            busType = DiagUdsBusType::kDoip;
            return true;
        }
    }

    for (auto& item : configInfo.DocanAddressList) {
        if (item == ta) {
            busType = DiagUdsBusType::kDocan;
            return true;
        }
    }

    nrc = DiagUdsNrcErrc::kErrorTa;
    return false;
}

void
RemoteDiagHandler::ConnectConfirmReponse(const Json::Value& message)
{
    DGR_DEBUG << "RemoteDiagHandler::ConnectConfirmReponse.";
    if (REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kFileUpload] == message["TYPE"].asString()) {
        uint32_t blockCount = static_cast<uint32_t>(std::strtoul(message["BLOCK_COUNT"].asString().c_str(), 0, 10));
        if (1 != blockCount) {
            return;
        }
    }

    Json::Value respMessage;
    respMessage["SA"] = message["TA"];
    respMessage["TA"] = message["SA"];
    respMessage["TYPE"] = message["TYPE"];
    respMessage["DATA"] = "Connect confirm reponse.";
    ReplyRemoteMessage(respMessage);
}

void
RemoteDiagHandler::UdsCommandReuqest(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagHandler::UdsCommandReuqest.";
    uint16_t sa = static_cast<uint16_t>(std::strtoul(message["SA"].asString().c_str(), 0, 16));
    uint16_t ta = static_cast<uint16_t>(std::strtoul(message["TA"].asString().c_str(), 0, 16));
    std::vector<uint8_t> udsData;
    auto udsDataVec = Split(message["DATA"].asString());
    for (auto& item : udsDataVec) {
        udsData.emplace_back(static_cast<uint8_t>(std::strtoul(item.c_str(), 0, 16)));
    }

    // Negative Response
    DiagUdsBusType busType = DiagUdsBusType::kServer;
    DiagUdsNrcErrc nrc = DiagUdsNrcErrc::kNegativeHead;
    if (!IsExecutableRequest(sa, ta, busType, nrc)) {
        Json::Value respMessage;
        respMessage["SA"] = message["TA"];
        respMessage["TA"] = message["SA"];
        respMessage["TYPE"] = message["TYPE"];
        std::vector<uint8_t> dataVec;
        dataVec.emplace_back(DiagUdsNrcErrc::kNegativeHead);
        dataVec.emplace_back(udsData.at(0));
        dataVec.emplace_back(nrc);
        respMessage["DATA"] = UINT8_VEC_TO_STRING_DATA(dataVec);
        ReplyRemoteMessage(respMessage);
        if (DiagUdsNrcErrc::kRequestBusy == nrc) {
            SetRemoteRequestStatus(sa, false);
        }

        return;
    }

    RemoteDiagReqUdsMessage udsMessage;
    udsMessage.udsSa = sa;
    udsMessage.udsTa = ta;
    udsMessage.busType = busType;
    udsMessage.udsData.assign(udsData.begin(), udsData.end());
    if (nullptr != doip_dispatcher_) {
        doip_dispatcher_->DoipRequestByEquip(udsMessage);
    }
}

void
RemoteDiagHandler::FileUploadReuqest(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagHandler::FileUploadReuqest.";
    uint16_t sa = static_cast<uint16_t>(std::strtoul(message["SA"].asString().c_str(), 0, 16));
    uint16_t ta = static_cast<uint16_t>(std::strtoul(message["TA"].asString().c_str(), 0, 16));
    uint32_t blockSize = static_cast<uint32_t>(std::strtoul(message["BLOCK_SIZE"].asString().c_str(), 0, 10));
    uint32_t blockCount = static_cast<uint32_t>(std::strtoul(message["BLOCK_COUNT"].asString().c_str(), 0, 10));
    // Negative Response
    {
        DiagUdsBusType busType = DiagUdsBusType::kServer;
        DiagUdsNrcErrc nrc = DiagUdsNrcErrc::kNegativeHead;
        Json::Value respMessage;
        respMessage["SA"] = message["TA"];
        respMessage["TA"] = message["SA"];
        respMessage["TYPE"] = message["TYPE"];
        if (!IsExecutableRequest(sa, ta, busType, nrc)) {
            respMessage["DATA"] = NegativeReponseStr(nrc);
            ReplyRemoteMessage(respMessage);
            return;
        }

        if ("" == message["FILE_NAME"].asString()) {
            respMessage["DATA"] = "Error file name!";
            ReplyRemoteMessage(respMessage);
            return;
        }

        if ((blockCount > blockSize) || (0 == blockSize)) {
            respMessage["DATA"] = "Error data block!";
            ReplyRemoteMessage(respMessage);
            return;
        }
    }

    RemoteDiagFileDownloadInfo downloadInfo;
    downloadInfo.sa = message["SA"].asString();
    downloadInfo.ta = message["TA"].asString();
    downloadInfo.downloadDirPath = message["DIR_PATH"].asString();
    downloadInfo.downloadFileName = message["FILE_NAME"].asString();
    downloadInfo.md5 = message["MD5_VALUE"].asString();
    downloadInfo.blockSize = blockSize;
    downloadInfo.blockCount = blockCount;
    downloadInfo.data = message["DATA"].asString();
    RemoteDiagFileTransfer::getInstance()->FileDownload(downloadInfo);
}

void
RemoteDiagHandler::FileDownloadReuqest(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagHandler::FileDownloadReuqest.";
    uint16_t sa = static_cast<uint16_t>(std::strtoul(message["SA"].asString().c_str(), 0, 16));
    uint16_t ta = static_cast<uint16_t>(std::strtoul(message["TA"].asString().c_str(), 0, 16));
    std::string filetype = message["FILE_TYPE"].asString();
    std::string filepath = message["DATA"].asString();
    // Negative Response
    {
        DiagUdsBusType busType = DiagUdsBusType::kServer;
        DiagUdsNrcErrc nrc = DiagUdsNrcErrc::kNegativeHead;
        Json::Value respMessage;
        respMessage["SA"] = message["TA"];
        respMessage["TA"] = message["SA"];
        respMessage["TYPE"] = message["TYPE"];
        respMessage["BLOCK_SIZE"] = "0";
        respMessage["BLOCK_COUNT"] = "0";
        if (!IsExecutableRequest(sa, ta, busType, nrc)) {
            respMessage["DATA"] = NegativeReponseStr(nrc);
            ReplyRemoteMessage(respMessage);
            return;
        }

        if (access(filepath.c_str(), F_OK) != 0) {
            respMessage["DATA"] = "File not exist!";
            ReplyRemoteMessage(respMessage);
            return;
        }

        struct stat statbuf;
        stat(filepath.c_str(), &statbuf);
        auto isDir = S_ISDIR(statbuf.st_mode);
        if ((("DIR" == filetype) && (!isDir)) || (("FILE" == filetype) && (isDir))) {
            respMessage["DATA"] = "Path and type mismatch!";
            ReplyRemoteMessage(respMessage);
            return;
        }
    }

    RemoteDiagFileUploadInfo uploadInfo;
    uploadInfo.sa = message["SA"].asString();
    uploadInfo.ta = message["TA"].asString();
    uploadInfo.uploadFileType = filetype;
    uploadInfo.uploadFilePath = filepath;
    RemoteDiagFileTransfer::getInstance()->FileUpload(uploadInfo);
}

void
RemoteDiagHandler::PluginRunReuqest(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagHandler::PluginRunReuqest.";
    uint16_t sa = static_cast<uint16_t>(std::strtoul(message["SA"].asString().c_str(), 0, 16));
    uint16_t ta = static_cast<uint16_t>(std::strtoul(message["TA"].asString().c_str(), 0, 16));
    // Negative Response
    {
        DiagUdsBusType busType = DiagUdsBusType::kServer;
        DiagUdsNrcErrc nrc = DiagUdsNrcErrc::kNegativeHead;
        Json::Value respMessage;
        respMessage["SA"] = message["TA"];
        respMessage["TA"] = message["SA"];
        respMessage["TYPE"] = message["TYPE"];
        if (!IsExecutableRequest(sa, ta, busType, nrc)) {
            respMessage["DATA"] = NegativeReponseStr(nrc);
            ReplyRemoteMessage(respMessage);
            return;
        }

        if ("" == message["DATA"].asString()) {
            respMessage["DATA"] = "Error plugin package name!";
            ReplyRemoteMessage(respMessage);
            return;
        }
    }

    RemoteDiagPluginRunInfo pluginRunInfo;
    pluginRunInfo.sa = message["SA"].asString();
    pluginRunInfo.ta = message["TA"].asString();
    pluginRunInfo.pluginPackageName = message["DATA"].asString();
    RemoteDiagDynamicPlugin::getInstance()->RunPlugin(pluginRunInfo);
}

void
RemoteDiagHandler::PluginRunResultReuqest(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagHandler::PluginRunResultReuqest.";
    uint16_t sa = static_cast<uint16_t>(std::strtoul(message["SA"].asString().c_str(), 0, 16));
    uint16_t ta = static_cast<uint16_t>(std::strtoul(message["TA"].asString().c_str(), 0, 16));
    // Negative Response
    {
        DiagUdsBusType busType = DiagUdsBusType::kServer;
        DiagUdsNrcErrc nrc = DiagUdsNrcErrc::kNegativeHead;
        Json::Value respMessage;
        respMessage["SA"] = message["TA"];
        respMessage["TA"] = message["SA"];
        respMessage["TYPE"] = message["TYPE"];
        if (REMOTE_DIAG_EXTENSION::REMOTE_DIAG_PLUGIN_RUN == curr_extension_) {
            respMessage["RUN_STATUS"] = REMOTE_DIAG_PLUGIN_RUN_STATUS[RemoteDiagPluginRunStatusType::kExecuting];
            respMessage["DATA"] = "The plugin is currently executing, please get the results later!";
            ReplyRemoteMessage(respMessage);
            return;
        }

        if (!IsExecutableRequest(sa, ta, busType, nrc)) {
            respMessage["RUN_STATUS"] = REMOTE_DIAG_PLUGIN_RUN_STATUS[RemoteDiagPluginRunStatusType::kDefault];
            respMessage["DATA"] = NegativeReponseStr(nrc);
            ReplyRemoteMessage(respMessage);
            return;
        }
    }

    RemoteDiagPluginRunResultInfo resultInfo;
    resultInfo.sa = message["SA"].asString();
    resultInfo.ta = message["TA"].asString();
    resultInfo.pluginName = message["DATA"].asString();
    RemoteDiagDynamicPlugin::getInstance()->GetPluginRunResult(resultInfo);
}

void
RemoteDiagHandler::SwitchControlReuqest(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagHandler::SwitchControlReuqest.";
    uint16_t sa = static_cast<uint16_t>(std::strtoul(message["SA"].asString().c_str(), 0, 16));
    uint16_t ta = static_cast<uint16_t>(std::strtoul(message["TA"].asString().c_str(), 0, 16));
    // Negative Response
    {
        DiagUdsBusType busType = DiagUdsBusType::kServer;
        DiagUdsNrcErrc nrc = DiagUdsNrcErrc::kNegativeHead;
        Json::Value respMessage;
        respMessage["SA"] = message["TA"];
        respMessage["TA"] = message["SA"];
        respMessage["TYPE"] = message["TYPE"];
        if (!IsExecutableRequest(sa, ta, busType, nrc)) {
            respMessage["DATA"] = NegativeReponseStr(nrc);
            ReplyRemoteMessage(respMessage);
            return;
        }
    }

    RemoteDiagSwitchControlInfo switchControlInfo;
    switchControlInfo.sa = message["SA"].asString();
    switchControlInfo.ta = message["TA"].asString();
    switchControlInfo.switchName = message["SWITCH_NAME"].asString();
    switchControlInfo.control = message["DATA"].asString();
    RemoteDiagSwitchControl::getInstance()->SwitchControl(switchControlInfo);
}

void
RemoteDiagHandler::QueryDirInfoReuqest(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagHandler::QueryDirInfoReuqest.";
    uint16_t sa = static_cast<uint16_t>(std::strtoul(message["SA"].asString().c_str(), 0, 16));
    uint16_t ta = static_cast<uint16_t>(std::strtoul(message["TA"].asString().c_str(), 0, 16));
    std::string infotype = message["INFO_TYPE"].asString();
    std::string dirpath = message["DATA"].asString();
    // Negative Response
    {
        DiagUdsBusType busType = DiagUdsBusType::kServer;
        DiagUdsNrcErrc nrc = DiagUdsNrcErrc::kNegativeHead;
        Json::Value respMessage;
        respMessage["SA"] = message["TA"];
        respMessage["TA"] = message["SA"];
        respMessage["TYPE"] = message["TYPE"];
        respMessage["BLOCK_SIZE"] = "0";
        respMessage["BLOCK_COUNT"] = "0";
        if (!IsExecutableRequest(sa, ta, busType, nrc)) {
            respMessage["DATA"] = NegativeReponseStr(nrc);
            ReplyRemoteMessage(respMessage);
            return;
        }

        if (("" == dirpath) || (access(dirpath.c_str(), F_OK) != 0)) {
            respMessage["DATA"] = "Dir path not exist!";
            ReplyRemoteMessage(respMessage);
            return;
        }

        struct stat statbuf;
        stat(dirpath.c_str(), &statbuf);
        auto isDir = S_ISDIR(statbuf.st_mode);
        if (!isDir) {
            respMessage["DATA"] = "Not a dir path!";
            ReplyRemoteMessage(respMessage);
            return;
        }
    }

    RemoteDiagQueryDirInfo queryDirInfo;
    queryDirInfo.sa = message["SA"].asString();
    queryDirInfo.ta = message["TA"].asString();
    queryDirInfo.infoType = message["INFO_TYPE"].asString();
    queryDirInfo.dirFilePath = message["DATA"].asString();
    RemoteDiagFileTransfer::getInstance()->QueryDirInfo(queryDirInfo);
}

std::vector<std::string>
RemoteDiagHandler::Split(const std::string& inputStr, const std::string& regexStr)
{
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

std::string
RemoteDiagHandler::PrintJsonData(const Json::Value& value)
{
    std::string dataStr = "";
    Json::Value::Members mem = value.getMemberNames();
    for (auto& item : mem) {
        if (value[item].type() == Json::stringValue) {
            std::string itemStr = value[item].asString();
            if ("DATA" == item) {
                if (itemStr.size() > 47) {
                    dataStr += item + ": " + itemStr.substr(0, 47) + "..." + " ";
                    continue;
                }
            }

            dataStr += item + ": " + itemStr + " ";
        }
    }

    return dataStr;
}

std::string
RemoteDiagHandler::NegativeReponseStr(const DiagUdsNrcErrc& nrc)
{
    std::string dataStr = "";
    if (DiagUdsNrcErrc::kErrorSa == nrc) {
        dataStr = "Error remote address!";
    }
    else if (DiagUdsNrcErrc::kErrorTa == nrc) {
        dataStr = "Error target address!";
    }
    else if (DiagUdsNrcErrc::kRequestBusy == nrc) {
        dataStr = "Remote address is busy!";
    }
    else if (DiagUdsNrcErrc::kVehicleInMotion == nrc) {
        dataStr = "Remote diagnosis is prohibited while the vehicle is in motion!";
    }

    return dataStr;
}

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon