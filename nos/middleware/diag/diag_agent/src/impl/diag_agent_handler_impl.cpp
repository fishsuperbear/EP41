#include <stdlib.h>
#include <unordered_map>
#include "json/json.h"
#include "diag/diag_agent/include/impl/diag_agent_handler_impl.h"
#include "diag/diag_agent/include/common/diag_agent_logger.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace diag_agent{

const int MAX_LOAD_SIZE = 1024;

int32_t
DiagAgentMethodServer::Process(const std::shared_ptr<uds_data_method> req, std::shared_ptr<uds_data_method> resp)
{
    DGA_INFO << "DiagAgentMethodServer::Process.";
    uint8_t sid = req->sid();
    uint8_t subid = req->subid();
    std::vector<uint8_t> messageData;
    messageData.assign(req->data_vec().begin(), req->data_vec().end());

    if (messageData.size() != req->data_len()) {
        DGA_ERROR << "DiagAgentMethodServer::Process error data. req->data_len: " << req->data_len() << " messageData.size: " << messageData.size();
        return -1;
    }

    DGA_INFO << "DiagAgentMethodServer::Process sid: " << UINT8_TO_STRING(sid) << " subid: " << UINT8_TO_STRING(subid) << " messageData: " << UINT8_VEC_TO_STRING(messageData);
    resp->meta_info(req->meta_info());
    resp->sid(sid);
    resp->subid(subid);
    DiagMessageInfo* messageInfo = new DiagMessageInfo();
    messageInfo->sid = sid;
    messageInfo->subid = subid;
    messageInfo->messageData = messageData;
    bool bResult = deal_diag_message_(messageInfo);
    if (bResult) {
        resp->resp_ack(0);
    }
    else {
        resp->resp_ack(1);
    }

    resp->data_len(messageInfo->messageData.size());
    resp->data_vec().assign(messageInfo->messageData.begin(), messageInfo->messageData.end());
    if (nullptr != messageInfo) {
        delete messageInfo;
        messageInfo = nullptr;
    }

    return 0;
}

DiagAgentMethodServer::~DiagAgentMethodServer()
{}

DiagAgentMethodReceiver::DiagAgentMethodReceiver()
: method_server_(nullptr)
{
}

DiagAgentMethodReceiver::~DiagAgentMethodReceiver()
{
}

void
DiagAgentMethodReceiver::Init(const std::string& processName, std::function<bool(DiagMessageInfo*)> dealDiagMessage)
{
    DGA_INFO << "DiagAgentMethodReceiver::Init.";
    std::shared_ptr<uds_data_methodPubSubType> req_data_type = std::make_shared<uds_data_methodPubSubType>();
    std::shared_ptr<uds_data_methodPubSubType> resp_data_type = std::make_shared<uds_data_methodPubSubType>();
    method_server_ = std::make_shared<DiagAgentMethodServer>(req_data_type, resp_data_type, dealDiagMessage);
    method_server_->Start(0, processName);
    DGA_INFO << "DiagAgentMethodReceiver::Init finish.";
}

void
DiagAgentMethodReceiver::DeInit()
{
    DGA_INFO << "DiagAgentMethodReceiver::DeInit.";
    if (nullptr != method_server_) {
        method_server_->Stop();
        method_server_ = nullptr;
    }
}

DiagAgentHandlerImpl::DiagAgentHandlerImpl()
: method_receiver_(new DiagAgentMethodReceiver())
, process_name_("")
{
}

DiagAgentHandlerImpl::~DiagAgentHandlerImpl()
{
}

DiagAgentInitResultCode
DiagAgentHandlerImpl::Init(const std::string& configPath,
                           std::shared_ptr<DiagAgentDataIdentifier> dataIdentifier,
                           std::shared_ptr<DiagAgentRoutineControl> routineControl)
{
    DGA_INFO << "DiagAgentHandlerImpl::Init configPath: " << configPath;
    if ((nullptr == dataIdentifier) && (nullptr == routineControl)) {
        DGA_WARN << "DiagAgentHandlerImpl::Init dataIdentifier and routineControl all nullptr.";
        return DiagAgentInitResultCode::kAllInstanceNull;
    }

    data_identifier_ = dataIdentifier;
    routine_control_ = routineControl;

    if (!LoadDiagAgentConfig(configPath)) {
        DGA_WARN << "DiagAgentHandlerImpl::Init load config error.";
        return DiagAgentInitResultCode::kLoadConfigError;
    }

    for (auto& item : rid_list_) {
        rid_status_map_.insert(std::make_pair(item.id, DiagRidStatus::kDefault));
    }

    // // print data just for test
    // DGA_INFO << "DiagAgentHandlerImpl::Init process_name_: " << process_name_;
    // for (auto& item : read_did_list_) {
    //     DGA_INFO << "DiagAgentHandlerImpl::Init read did: " << UINT16_TO_STRING(item.id) << " datalen: " << item.dataLen;
    // }

    // for (auto& item : write_did_list_) {
    //     DGA_INFO << "DiagAgentHandlerImpl::Init write did: " << UINT16_TO_STRING(item.id) << " datalen: " << item.dataLen;
    // }

    // for (auto& item : rid_list_) {
    //     DGA_INFO << "DiagAgentHandlerImpl::Init rid: " << UINT16_TO_STRING(item.id) << " isSupportMultiStart: " << item.isSupportMultiStart;
    //     for (auto& sub : item.subFunc) {
    //         DGA_INFO << "DiagAgentHandlerImpl::Init rid subid: " << UINT8_TO_STRING(sub.id) << " reqlen: " << sub.reqLen << " reslen: " << sub.resLen;
    //     }
    // }

    // method receiver init
    if (nullptr != method_receiver_) {
        method_receiver_->Init(process_name_, std::bind(&DiagAgentHandlerImpl::DealWithDiagMessage, this, std::placeholders::_1));
    }

    DGA_INFO << "DiagAgentHandlerImpl::Init finish.";
    return DiagAgentInitResultCode::kSuccess;
}

void
DiagAgentHandlerImpl::DeInit()
{
    DGA_INFO << "DiagAgentHandlerImpl::DeInit.";
    // method receiver deinit
    if (nullptr != method_receiver_) {
        method_receiver_->DeInit();
        delete method_receiver_;
        method_receiver_ = nullptr;
    }

    read_did_list_.clear();
    write_did_list_.clear();
    rid_list_.clear();
    rid_status_map_.clear();
}

bool
DiagAgentHandlerImpl::DealWithDiagMessage(DiagMessageInfo* messageInfo)
{
    DGA_INFO << "DiagAgentHandlerImpl::DealWithDiagMessage sid: " << UINT8_TO_STRING(messageInfo->sid) << " subid: " << UINT8_TO_STRING(messageInfo->subid)
                                                           << " messageData: " << UINT8_VEC_TO_STRING(messageInfo->messageData);
    bool bResult = false;
    // check if the service is supported
    switch (messageInfo->sid)
    {
        case DiagAgentServiceRequestOpc::DIAG_AGENT_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER:
            if ((nullptr == data_identifier_) || (0 == read_did_list_.size())) {
                DGA_WARN << "DiagAgentHandlerImpl::DealWithDiagMessage dataIdentifier is nullptr or error read_did_list_.size: " << read_did_list_.size();
                messageInfo->messageData.clear();
                messageInfo->messageData.push_back(DiagAgentNrcErrc::kServiceNotSupported);
                return false;
            }

            bResult = ReadDataIdentifier(messageInfo->messageData);
            break;
        case DiagAgentServiceRequestOpc::DIAG_AGENT_SERVICE_REQUEST_OPC_WRITE_DATA_IDENTIFIER:
            if ((nullptr == data_identifier_) || (0 == write_did_list_.size())) {
                DGA_WARN << "DiagAgentHandlerImpl::DealWithDiagMessage dataIdentifier is nullptr or error write_did_list_.size: " << write_did_list_.size();
                messageInfo->messageData.clear();
                messageInfo->messageData.push_back(DiagAgentNrcErrc::kServiceNotSupported);
                return false;
            }

            bResult = WriteDataIdentifier(messageInfo->messageData);
            break;
        case DiagAgentServiceRequestOpc::DIAG_AGENT_SERVICE_REQUEST_OPC_ROUTINE_CONTROL:
            if ((nullptr == routine_control_) || (0 == rid_list_.size())) {
                DGA_WARN << "DiagAgentHandlerImpl::DealWithDiagMessage routine_control_ is nullptr or error rid_list_.size: " << rid_list_.size();
                messageInfo->messageData.clear();
                messageInfo->messageData.push_back(DiagAgentNrcErrc::kServiceNotSupported);
                return false;
            }

            bResult = RoutineControl(messageInfo->subid, messageInfo->messageData);
            break;
        default:
            messageInfo->messageData.clear();
            messageInfo->messageData.push_back(DiagAgentNrcErrc::kServiceNotSupported);
            break;
    }

    return bResult;
}

char*
DiagAgentHandlerImpl::GetJsonAll(const char *fname)
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

bool
DiagAgentHandlerImpl::LoadDiagAgentConfig(const std::string& configPath)
{
    char* jsonstr = GetJsonAll(configPath.c_str());
    if (nullptr == jsonstr) {
        DGA_ERROR << "DiagAgentHandlerImpl::Init load config error jsonstr is nullptr.";
        return false;
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

        return false;
    }

    // load process name config
    process_name_ = static_cast<std::string>(rootValue["ProcessName"].asString());
    if ("" == process_name_) {
        DGA_WARN << "DiagAgentHandlerImpl::Init load process name is null.";
        return false;
    }

    // load did config
    std::string did = "";
    bool isSupportWrite = false;
    Json::Value didValue = rootValue["DID"];
    DiagAgentDidDataInfo didInfo;
    for (Json::ArrayIndex i = 0; i < didValue.size(); ++i) {
        did = static_cast<std::string>(didValue[i]["id"].asString());
        didInfo.id = static_cast<uint16_t>(std::strtoul(did.c_str(), 0, 0));
        isSupportWrite = static_cast<bool>(didValue[i]["isSupportWrite"].asBool());
        didInfo.dataLen = static_cast<uint16_t>(didValue[i]["dataSize"].asUInt());

        read_did_list_.push_back(didInfo);
        if (isSupportWrite) {
            write_did_list_.push_back(didInfo);
        }
    }

    // load rid config
    std::string rid = "";
    std::string ridSubFuncId = "";
    Json::Value ridValue = rootValue["RID"];
    DiagAgentRidDataInfo ridInfo;
    DiagAgentRidSubFuncInfo ridSubFuncInfo;
    for (Json::ArrayIndex i = 0; i < ridValue.size(); ++i) {
        rid = static_cast<std::string>(ridValue[i]["id"].asString());
        ridInfo.id = static_cast<uint16_t>(std::strtoul(rid.c_str(), 0, 0));
        ridInfo.isSupportMultiStart = static_cast<bool>(ridValue[i]["isSupportMultiStart"].asBool());
        ridInfo.subFunc.clear();
        Json::Value ridSubFuncValue = ridValue[i]["ridSubFunctions"];
        for (Json::ArrayIndex j = 0; j < ridSubFuncValue.size(); ++j) {
            ridSubFuncId = static_cast<std::string>(ridSubFuncValue[j]["id"].asString());
            ridSubFuncInfo.id = static_cast<uint8_t>(std::strtoul(ridSubFuncId.c_str(), 0, 0));
            ridSubFuncInfo.reqLen = static_cast<uint16_t>(ridSubFuncValue[j]["requestLen"].asUInt());
            ridSubFuncInfo.resLen = static_cast<uint16_t>(ridSubFuncValue[j]["replyLen"].asUInt());
            ridInfo.subFunc.push_back(ridSubFuncInfo);
        }

        rid_list_.push_back(ridInfo);
    }

    if (jsonstr != NULL) {
        free(jsonstr);
    }

    return true;
}

bool
DiagAgentHandlerImpl::ReadDataIdentifier(std::vector<uint8_t>& messageData)
{
    DGA_INFO << "DiagAgentHandlerImpl::ReadDataIdentifier messageData: " << UINT8_VEC_TO_STRING(messageData);
    // check if the minimum length is met
    if (messageData.size() < 2) {
        messageData.clear();
        messageData.push_back(DiagAgentNrcErrc::kIncorrectMessageLengthOrInvalidFormat);
        return false;
    }

    uint16_t did = messageData[0];
    did = (did << 8) | messageData[1];
    messageData.clear();

    // check if the DID is supported
    auto itr = read_did_list_.begin();
    for (; itr != read_did_list_.end(); itr++) {
        if (did == itr->id) {
            break;
        }
    }

    if (read_did_list_.end() == itr) {
        messageData.push_back(DiagAgentNrcErrc::kRequestOutOfRange);
        return false;
    }

    // check whether the service implementation instance is normal
    if (nullptr == data_identifier_) {
        DGA_ERROR << "DiagAgentHandlerImpl::ReadDataIdentifier read did: " << did << " failed, dataIdentifier is nullptr.";
        messageData.push_back(DiagAgentNrcErrc::kConditionsNotCorrect);
        return false;
    }

    bool bResult = data_identifier_->Read(did, messageData);
    // check whether the returned results are correct
    if (!bResult || (static_cast<uint16_t>(messageData.size()) != (itr->dataLen + 2))) {
        DGA_WARN << "DiagAgentHandlerImpl::ReadDataIdentifier read did: " << did << " failed, result: " << bResult << " messageData.size: " << messageData.size();
        messageData.clear();
        messageData.push_back(DiagAgentNrcErrc::kConditionsNotCorrect);
        return false;
    }

    return true;
}

bool
DiagAgentHandlerImpl::WriteDataIdentifier(std::vector<uint8_t>& messageData)
{
    DGA_INFO << "DiagAgentHandlerImpl::WriteDataIdentifier messageData: " << UINT8_VEC_TO_STRING(messageData);
    // check if the minimum length is met
    if (messageData.size() < 3) {
        messageData.clear();
        messageData.push_back(DiagAgentNrcErrc::kIncorrectMessageLengthOrInvalidFormat);
        return false;
    }

    uint16_t did = messageData[0];
    did = (did << 8) | messageData[1];
    std::vector<uint8_t> reqData;
    reqData.assign(messageData.begin() + 2, messageData.end());
    messageData.clear();

    // check if the DID is supported
    auto itr = write_did_list_.begin();
    for (; itr != write_did_list_.end(); itr++) {
        if (did == itr->id) {
            break;
        }
    }

    if (write_did_list_.end() == itr) {
        messageData.push_back(DiagAgentNrcErrc::kRequestOutOfRange);
        return false;
    }

    // check if the request data length is correct
    if (reqData.size() != itr->dataLen) {
        messageData.push_back(DiagAgentNrcErrc::kIncorrectMessageLengthOrInvalidFormat);
        return false;
    }

    // check whether the service implementation instance is normal
    if (nullptr == data_identifier_) {
        DGA_ERROR << "DiagAgentHandlerImpl::WriteDataIdentifier write did: " << did << " failed, dataIdentifier is nullptr.";
        messageData.push_back(DiagAgentNrcErrc::kConditionsNotCorrect);
        return false;
    }

    bool bResult = data_identifier_->Write(did, reqData, messageData);
    // check whether the returned results are correct
    if (!bResult || (messageData.size() != 2)) {
        DGA_WARN << "DiagAgentHandlerImpl::WriteDataIdentifier write did: " << did << " failed, result: " << bResult << " messageData.size: " << messageData.size();
        messageData.clear();
        messageData.push_back(DiagAgentNrcErrc::kConditionsNotCorrect);
        return false;
    }

    return true;
}

bool
DiagAgentHandlerImpl::RoutineControl(const uint8_t subid, std::vector<uint8_t>& messageData)
{
    DGA_INFO << "DiagAgentHandlerImpl::RoutineControl subid: " << UINT8_TO_STRING(subid) << " messageData: " << UINT8_VEC_TO_STRING(messageData);
    // check if the minimum length is met
    if (messageData.size() < 2) {
        messageData.clear();
        messageData.push_back(DiagAgentNrcErrc::kIncorrectMessageLengthOrInvalidFormat);
        return false;
    }

    uint16_t rid = messageData[0];
    rid = (rid << 8) | messageData[1];
    std::vector<uint8_t> reqData;
    reqData.assign(messageData.begin() + 2, messageData.end());
    messageData.clear();

    // check if the RID is supported
    auto itr = rid_list_.begin();
    for (; itr != rid_list_.end(); itr++) {
        if (rid == itr->id) {
            break;
        }
    }

    if (rid_list_.end() == itr) {
        messageData.push_back(DiagAgentNrcErrc::kRequestOutOfRange);
        return false;
    }

    // check if the subfunc is supported
    auto subItr = itr->subFunc.begin();
    for (; subItr != itr->subFunc.end(); subItr++) {
        if (subid == subItr->id) {
            break;
        }
    }

    if (itr->subFunc.end() == subItr) {
        messageData.push_back(DiagAgentNrcErrc::kSubfunctionNotSupported);
        return false;
    }

    // check if the request data length is correct
    if (reqData.size() != subItr->reqLen) {
        messageData.push_back(DiagAgentNrcErrc::kIncorrectMessageLengthOrInvalidFormat);
        return false;
    }

    // check if the sequence of the request is correct
    if (((0x01 == subid) && (DiagRidStatus::kStarted == rid_status_map_[rid]) && (false == itr->isSupportMultiStart)) ||
        ((0x02 == subid) && (DiagRidStatus::kStarted != rid_status_map_[rid])) ||
        ((0x03 == subid) && (DiagRidStatus::kDefault == rid_status_map_[rid]))) {
        messageData.push_back(DiagAgentNrcErrc::kRequestSequenceError);
        return false;
    }

    // check whether the service implementation instance is normal
    if (nullptr == routine_control_) {
        DGA_ERROR << "DiagAgentHandlerImpl::RoutineControl rid: " << rid << " failed, routine_control_ is nullptr.";
        messageData.push_back(DiagAgentNrcErrc::kConditionsNotCorrect);
        return false;
    }

    bool bResult = false;
    switch (subid)
    {
        case 0x01:
            bResult = routine_control_->Start(rid, reqData, messageData);
            if (bResult) {
                rid_status_map_[rid] = DiagRidStatus::kStarted;
            }

            break;
        case 0x02:
            bResult = routine_control_->Stop(rid, reqData, messageData);
            if (bResult) {
                rid_status_map_[rid] = DiagRidStatus::kStopped;
            }

            break;
        case 0x03:
            bResult = routine_control_->Result(rid, reqData, messageData);
            break;
        default:
            messageData.push_back(DiagAgentNrcErrc::kSubfunctionNotSupported);
            break;
    }

    // check whether the returned results are correct
    if (!bResult || (static_cast<uint16_t>(messageData.size()) != subItr->resLen + 2)) {
        DGA_WARN << "DiagAgentHandlerImpl::RoutineControl rid: " << rid << " failed, result: " << bResult << " messageData.size: " << messageData.size();
        messageData.clear();
        messageData.push_back(DiagAgentNrcErrc::kConditionsNotCorrect);
        return false;
    }

    return true;
}

}  // namespace diag_agent
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
