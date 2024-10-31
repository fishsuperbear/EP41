#include <stdlib.h>
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/fault_manager/interactive/phm_fault_diag_handler.h"
#include "phm_server/include/fault_manager/manager/phm_fault_record.h"
#include "phm_server/include/fault_manager/manager/phm_fault_task_handler.h"
#include "phm_server/include/fault_manager/file/phm_file_operate.h"
#include "phm_server/include/common/phm_server_utils.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/fault_manager/manager/phm_fault_dispatcher.h"

namespace hozon {
namespace netaos {
namespace phm_server {

DiagMessageHandler* DiagMessageHandler::instance_ = nullptr;
std::mutex DiagMessageHandler::mtx_;

std::unordered_map<uint16_t, uint> START_REQ_LENGTH = {{0xD000, 6}, {0xD001, 5}, {0xD005, 0}};
std::unordered_map<uint16_t, uint> STOP_REQ_LENGTH = {{0xD000, 0}, {0xD001, 0}};
std::unordered_map<uint16_t, uint> RESULT_REQ_LENGTH = {{0xD002, 4}, {0xD003, 4}, {0xD004, 3}};



int32_t
DiagMessageMethodServer::Process(const std::shared_ptr<uds_data_method> req, std::shared_ptr<uds_data_method> resp)
{
    PHMS_DEBUG << "DiagMessageMethodServer::Process. ";
    uint8_t sid = req->sid();
    uint8_t subid = req->subid();
    std::vector<uint8_t> messageData;
    messageData.assign(req->data_vec().begin(), req->data_vec().end());

    if (messageData.size() != req->data_len()) {
        PHMS_ERROR << "DiagMessageMethodServer::Process error data. req->data_len: " << req->data_len() << " messageData.size: " << messageData.size();
        return -1;
    }

    resp->meta_info(req->meta_info());
    resp->sid(sid);
    resp->subid(subid);
    bool bResult = DiagMessageHandler::getInstance()->DealWithDiagMessage(sid, subid, messageData);
    if (bResult) {
        resp->resp_ack(0);
    }
    else {
        resp->resp_ack(1);
    }

    resp->data_len(messageData.size());
    resp->data_vec().assign(messageData.begin(), messageData.end());
    return 0;
}

DiagMessageMethodServer::~DiagMessageMethodServer()
{}

DiagMessageMethodReceiver::DiagMessageMethodReceiver()
: method_server_(nullptr)
{
}

DiagMessageMethodReceiver::~DiagMessageMethodReceiver()
{
}

void
DiagMessageMethodReceiver::Init()
{
    PHMS_INFO << "DiagMessageMethodReceiver::Init";
    std::shared_ptr<uds_data_methodPubSubType> req_data_type = std::make_shared<uds_data_methodPubSubType>();
    std::shared_ptr<uds_data_methodPubSubType> resp_data_type = std::make_shared<uds_data_methodPubSubType>();
    method_server_ = std::make_shared<DiagMessageMethodServer>(req_data_type, resp_data_type);
    // method_server_->RegisterProcess(std::bind(&DiagMessageMethodServer::Process, method_server_, std::placeholders::_1, std::placeholders::_2));
    method_server_->Start(0, "diag_phm");
}

void
DiagMessageMethodReceiver::DeInit()
{
    PHMS_INFO << "DiagMessageMethodReceiver::DeInit";
    if (nullptr != method_server_) {
        method_server_->Stop();
        method_server_ = nullptr;
    }
}

DiagMessageHandler::DiagMessageHandler()
: method_receiver_(new DiagMessageMethodReceiver())
, time_mgr_(new TimerManager())
, fault_occur_or_recover_timer_fd_(-1)
, fault_occur_or_recover_flag_(false)
, fault_occur_and_recover_timer_fd_(-1)
, fault_occur_and_recover_flag_(false)
{
}

DiagMessageHandler*
DiagMessageHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagMessageHandler();
        }
    }

    return instance_;
}

void
DiagMessageHandler::Init()
{
    PHMS_INFO << "DiagMessageHandler::Init";
    // timer manager init
    if (time_mgr_ != nullptr) {
        time_mgr_->Init();
    }

    // method receiver init
    if (method_receiver_ != nullptr) {
        method_receiver_->Init();
    }
}

void
DiagMessageHandler::DeInit()
{
    PHMS_INFO << "DiagMessageHandler::DeInit";

    // method receiver deinit
    if (method_receiver_ != nullptr) {
        method_receiver_->DeInit();
        delete method_receiver_;
        method_receiver_ = nullptr;
    }

    // timer manager deinit
    if (time_mgr_ != nullptr) {
        time_mgr_->DeInit();
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

bool
DiagMessageHandler::DealWithDiagMessage(const uint8_t sid, const uint8_t subid, std::vector<uint8_t>& messageData)
{
    PHMS_DEBUG << "DiagMessageHandler::DealWithDiagMessage sid: " << UINT8_TO_STRING(sid) << " subid: " << UINT8_TO_STRING(subid)
                                         << " messageData.size: " << messageData.size() << " messageData: " << UINT8_VEC_TO_STRING(messageData);

    if (0x31 != sid) {
        messageData.clear();
        messageData.push_back(0x11);
        return false;
    }

    if (messageData.size() < 2) {
        messageData.clear();
        messageData.push_back(0x13);
        return false;
    }

    uint16_t rid = messageData[0];
    rid = (rid << 8) | messageData[1];
    std::vector<uint8_t> reqData;
    reqData.assign(messageData.begin() + 2, messageData.end());
    messageData.clear();
    bool bResult = false;
    switch (subid)
    {
        case 0x01:
            bResult = Start(rid, reqData, messageData);
            break;
        case 0x02:
            bResult = Stop(rid, reqData, messageData);
            break;
        case 0x03:
            bResult = Result(rid, reqData, messageData);
            break;
        default:
            messageData.push_back(0x12);
            break;
    }

    return bResult;
}

bool
DiagMessageHandler::Start(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::Start | rid: " << UINT16_TO_STRING(rid) << ", reqData size: " << reqData.size()
                                                                               << ", reqData: " << UINT8_VEC_TO_STRING(reqData);
    auto itr = START_REQ_LENGTH.find(rid);
    if (itr == START_REQ_LENGTH.end()) {
        PHMS_ERROR << "DiagMessageHandler::Start rid not support | rid: " << UINT16_TO_STRING(rid);
        respData.push_back(0x31);
        return false;
    }

    if (reqData.size() != itr->second)
    {
        PHMS_ERROR << "DiagMessageHandler::Start error reqdata size | need data size: " << itr->second << ", reqData size: " << reqData.size();
        respData.emplace_back(0x13);
        return false;
    }

    bool bResult = true;
    switch (rid)
    {
        case 0xD000:
            if (fault_occur_or_recover_flag_) {
                PHMS_ERROR << "DiagMessageHandler::Start error sequence.";
                respData.push_back(0x24);
                bResult = false;
            }
            else {
                respData.push_back(static_cast<uint8_t>((rid >> 8) & 0xff));
                respData.push_back(static_cast<uint8_t>(rid & 0xff));
                StartReportFaultOccurOrRecover(reqData, respData);
            }

            break;
        case 0xD001:
            if (fault_occur_and_recover_flag_) {
                PHMS_ERROR << "DiagMessageHandler::Start error sequence.";
                respData.push_back(0x24);
                bResult = false;
            }
            else {
                respData.push_back(static_cast<uint8_t>((rid >> 8) & 0xff));
                respData.push_back(static_cast<uint8_t>(rid & 0xff));
                StartReportFaultOccurAndRecover(reqData, respData);
            }

            break;
        case 0xD005:
            respData.push_back(static_cast<uint8_t>((rid >> 8) & 0xff));
            respData.push_back(static_cast<uint8_t>(rid & 0xff));
            StartRefreshFaultFile(reqData, respData);
            break;
        default:
            break;
    }

    return bResult;
}

void
DiagMessageHandler::StartReportFaultOccurOrRecover(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::StartReportFaultOccurOrRecover.";
    uint8_t cycle = reqData[0];
    occur_or_recover_fault_.faultId = HEX_TO_DEC(reqData[1]) * 10000 + HEX_TO_DEC(reqData[2])  * 100 + HEX_TO_DEC(reqData[3]);
    occur_or_recover_fault_.faultObj = HEX_TO_DEC(reqData[4]);
    occur_or_recover_fault_.faultStatus = reqData[5];

    uint32_t fault = occur_or_recover_fault_.faultId * 100 + occur_or_recover_fault_.faultObj;
    FaultInfo faultInfo;
    if (!PHMServerConfig::getInstance()->GetFaultInfoByFault(fault, faultInfo)) {
        PHMS_ERROR << "PHMServerConfig::StartReportFaultOccurOrRecover not support fault: " << fault;
        respData.emplace_back(0x05);
        return;
    }

    if (occur_or_recover_fault_.faultStatus > 1) {
        PHMS_ERROR << "PHMServerConfig::StartReportFaultOccurOrRecover error input fault status: " << occur_or_recover_fault_.faultStatus;
        respData.emplace_back(0x05);
        return;
    }

    occur_or_recover_fault_.faultDomain = "fault_test";

    if (!cycle) {
        occur_or_recover_fault_.faultOccurTime = PHMUtils::GetCurrentTime();
        FaultDispatcher::getInstance()->ReportFault(occur_or_recover_fault_);
    }
    else {
        if (nullptr == time_mgr_) {
            PHMS_ERROR << "PHMServerConfig::StartReportFaultOccurOrRecover time_mgr_ is nullptr.";
            respData.emplace_back(0x05);
            return;
        }

        uint time = static_cast<uint>(cycle * 1000);
        time_mgr_->StartFdTimer(fault_occur_or_recover_timer_fd_, time,
                                std::bind(&DiagMessageHandler::FaultOccurOrRecover, this, std::placeholders::_1),
                                nullptr, true);
    }

    respData.emplace_back(0x02);
    fault_occur_or_recover_flag_ = true;
}

void
DiagMessageHandler::StartReportFaultOccurAndRecover(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::StartReportFaultOccurAndRecover.";
    uint8_t cycle = reqData[0];
    occur_and_recover_fault_.faultId = HEX_TO_DEC(reqData[1]) * 10000 + HEX_TO_DEC(reqData[2])  * 100 + HEX_TO_DEC(reqData[3]);
    occur_and_recover_fault_.faultObj = HEX_TO_DEC(reqData[4]);
    occur_and_recover_fault_.faultStatus = 0x01;

    uint32_t fault = occur_and_recover_fault_.faultId * 100 + occur_and_recover_fault_.faultObj;
    FaultInfo faultInfo;
    if (!PHMServerConfig::getInstance()->GetFaultInfoByFault(fault, faultInfo)) {
        PHMS_ERROR << "PHMServerConfig::StartReportFaultOccurAndRecover not support fault: " << fault;
        respData.emplace_back(0x05);
        return;
    }

    occur_and_recover_fault_.faultDomain = "fault_test";

    if (!cycle) {
        // fault occur
        occur_and_recover_fault_.faultOccurTime = PHMUtils::GetCurrentTime();
        FaultDispatcher::getInstance()->ReportFault(occur_and_recover_fault_);

        // fault recover
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        occur_and_recover_fault_.faultOccurTime = PHMUtils::GetCurrentTime();
        occur_and_recover_fault_.faultStatus = 0x00;
        FaultDispatcher::getInstance()->ReportFault(occur_and_recover_fault_);
    }
    else {
        if (nullptr == time_mgr_) {
            PHMS_ERROR << "PHMServerConfig::StartReportFaultOccurAndRecover time_mgr_ is nullptr.";
            respData.emplace_back(0x05);
            return;
        }

        uint time = static_cast<uint>(cycle * 1000 / 2);
        time_mgr_->StartFdTimer(fault_occur_and_recover_timer_fd_, time,
                                std::bind(&DiagMessageHandler::FaultOccurAndRecover, this, std::placeholders::_1),
                                nullptr, true);
    }

    respData.emplace_back(0x02);
    fault_occur_and_recover_flag_ = true;
}

void
DiagMessageHandler::StartRefreshFaultFile(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::StartRefreshFaultFile.";
    FileOperate::getInstance()->Sync();
    respData.emplace_back(0x02);
}

bool
DiagMessageHandler::Stop(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::Stop | rid: " << UINT16_TO_STRING(rid) << ", reqData size: " << reqData.size()
                                                                               << ", reqData: " << UINT8_VEC_TO_STRING(reqData);
    auto itr = STOP_REQ_LENGTH.find(rid);
    if (itr == STOP_REQ_LENGTH.end()) {
        PHMS_ERROR << "DiagMessageHandler::Stop rid not support | rid: " << UINT16_TO_STRING(rid);
        respData.push_back(0x31);
        return false;
    }

    if (reqData.size() != itr->second)
    {
        PHMS_ERROR << "DiagMessageHandler::Stop error reqdata size | need data size: " << itr->second << ", reqData size: " << reqData.size();
        respData.emplace_back(0x13);
        return false;
    }

    bool bResult = true;
    switch (rid)
    {
        case 0xD000:
            if (!fault_occur_or_recover_flag_) {
                PHMS_ERROR << "DiagMessageHandler::Stop error sequence.";
                respData.push_back(0x24);
                bResult = false;
            }
            else {
                respData.push_back(static_cast<uint8_t>((rid >> 8) & 0xff));
                respData.push_back(static_cast<uint8_t>(rid & 0xff));
                StopReportFaultOccurOrRecover(reqData, respData);
            }

            break;
        case 0xD001:
            if (!fault_occur_and_recover_flag_) {
                PHMS_ERROR << "DiagMessageHandler::Stop error sequence.";
                respData.push_back(0x24);
                bResult = false;
            }
            else {
                respData.push_back(static_cast<uint8_t>((rid >> 8) & 0xff));
                respData.push_back(static_cast<uint8_t>(rid & 0xff));
                StopReportFaultOccurAndRecover(reqData, respData);
                bResult = true;
            }

            break;
        default:
            break;
    }

    return bResult;
}

void
DiagMessageHandler::StopReportFaultOccurOrRecover(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::StopReportFaultOccurOrRecover.";
    if (-1 != fault_occur_or_recover_timer_fd_) {
        if (nullptr == time_mgr_) {
            PHMS_ERROR << "PHMServerConfig::StopReportFaultOccurOrRecover time_mgr_ is nullptr.";
            respData.emplace_back(0x05);
            return;
        }

        PHMS_INFO << "DiagMessageHandler::StopReportFaultOccurOrRecover fd:" << fault_occur_or_recover_timer_fd_;
        time_mgr_->StopFdTimer(fault_occur_or_recover_timer_fd_);
        fault_occur_or_recover_timer_fd_ = -1;
    }

    respData.emplace_back(0x02);
    fault_occur_or_recover_flag_ = false;
}

void
DiagMessageHandler::StopReportFaultOccurAndRecover(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::StopReportFaultOccurAndRecover.";
    if (-1 != fault_occur_and_recover_timer_fd_) {
        if (nullptr == time_mgr_) {
            PHMS_ERROR << "PHMServerConfig::StopReportFaultOccurAndRecover time_mgr_ is nullptr.";
            respData.emplace_back(0x05);
            return;
        }

        PHMS_INFO << "DiagMessageHandler::StopReportFaultOccurAndRecover fd:" << fault_occur_and_recover_timer_fd_;
        time_mgr_->StopFdTimer(fault_occur_and_recover_timer_fd_);
        fault_occur_and_recover_timer_fd_ = -1;
    }

    respData.emplace_back(0x02);
    fault_occur_and_recover_flag_ = false;
}

bool
DiagMessageHandler::Result(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::Result | rid: " << UINT16_TO_STRING(rid) << ", reqData size: " << reqData.size()
                                                                               << ", reqData: " << UINT8_VEC_TO_STRING(reqData);
    auto itr = RESULT_REQ_LENGTH.find(rid);
    if (itr == RESULT_REQ_LENGTH.end()) {
        PHMS_ERROR << "DiagMessageHandler::Result rid not support | rid: " << UINT16_TO_STRING(rid);
        respData.push_back(0x31);
        return false;
    }

    if (reqData.size() != itr->second)
    {
        PHMS_ERROR << "DiagMessageHandler::Result error reqdata size | need data size: " << itr->second << ", reqData size: " << reqData.size();
        respData.emplace_back(0x13);
        return false;
    }

    bool bResult = true;
    switch (rid)
    {
        case 0xD002:
            respData.push_back(static_cast<uint8_t>((rid >> 8) & 0xff));
            respData.push_back(static_cast<uint8_t>(rid & 0xff));
            ResultQueryCurrentFault(reqData, respData);
            break;
        case 0xD003:
            respData.push_back(static_cast<uint8_t>((rid >> 8) & 0xff));
            respData.push_back(static_cast<uint8_t>(rid & 0xff));
            ResultQueryDtcByFault(reqData, respData);
            break;
        case 0xD004:
            respData.push_back(static_cast<uint8_t>((rid >> 8) & 0xff));
            respData.push_back(static_cast<uint8_t>(rid & 0xff));
            ResultQueryFaultByDtc(reqData, respData);
            break;
        default:
            break;
    }

    return bResult;
}

void
DiagMessageHandler::ResultQueryCurrentFault(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::ResultQueryCurrentFault.";
    uint32_t fault = reqData[0];
    fault = (fault << 8) | reqData[1];
    fault = (fault << 8) | reqData[2];
    fault = (fault << 8) | reqData[3];
    if (0xFFFFFFFF != fault) {
        fault = HEX_TO_DEC(fault);
    }

    std::vector<uint32_t> faultList;
    PHMServerConfig::getInstance()->QueryCurrentOccuredFault(fault, faultList);
    respData.emplace_back(static_cast<uint8_t>(faultList.size()));
    for (auto& item : faultList) {
        fault = DEC_TO_HEX(item);
        respData.push_back(static_cast<uint8_t>((fault >> 24) & 0xff));
        respData.push_back(static_cast<uint8_t>((fault >> 16) & 0xff));
        respData.push_back(static_cast<uint8_t>((fault >> 8) & 0xff));
        respData.push_back(static_cast<uint8_t>(fault & 0xff));
    }
}

void
DiagMessageHandler::ResultQueryDtcByFault(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::ResultQueryDtcByFault.";
    uint32_t fault = reqData[0];
    fault = (fault << 8) | reqData[1];
    fault = (fault << 8) | reqData[2];
    fault = (fault << 8) | reqData[3];
    uint32_t dtc = PHMServerConfig::getInstance()->GetDtcByFault(HEX_TO_DEC(fault));
    respData.push_back(static_cast<uint8_t>((dtc >> 16) & 0xff));
    respData.push_back(static_cast<uint8_t>((dtc >> 8) & 0xff));
    respData.push_back(static_cast<uint8_t>(dtc & 0xff));
}

void
DiagMessageHandler::ResultQueryFaultByDtc(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData)
{
    PHMS_DEBUG << "DiagMessageHandler::ResultQueryFaultByDtc.";
    uint32_t dtc = reqData[0];
    dtc = (dtc << 8) | reqData[1];
    dtc = (dtc << 8) | reqData[2];
    uint32_t fault = DEC_TO_HEX(PHMServerConfig::getInstance()->GetFaultByDtc(dtc));
    respData.push_back(static_cast<uint8_t>((fault >> 24) & 0xff));
    respData.push_back(static_cast<uint8_t>((fault >> 16) & 0xff));
    respData.push_back(static_cast<uint8_t>((fault >> 8) & 0xff));
    respData.push_back(static_cast<uint8_t>(fault & 0xff));
}

void
DiagMessageHandler::FaultOccurOrRecover(void* data)
{
    PHMS_DEBUG << "DiagMessageHandler::FaultOccurOrRecover.";
    occur_or_recover_fault_.faultOccurTime = PHMUtils::GetCurrentTime();
    FaultDispatcher::getInstance()->ReportFault(occur_or_recover_fault_);
}

void
DiagMessageHandler::FaultOccurAndRecover(void* data)
{
    PHMS_DEBUG << "DiagMessageHandler::FaultOccurAndRecover.";
    occur_and_recover_fault_.faultOccurTime = PHMUtils::GetCurrentTime();
    FaultDispatcher::getInstance()->ReportFault(occur_and_recover_fault_);
    occur_and_recover_fault_.faultStatus ^= 0x01;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
