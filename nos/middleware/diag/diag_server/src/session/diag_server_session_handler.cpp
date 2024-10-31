#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/session/diag_server_session_mgr.h"
#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/uds/diag_server_uds_mgr.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/common/diag_server_logger.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerSessionHandler* DiagServerSessionHandler::instance_ = nullptr;
std::mutex DiagServerSessionHandler::mtx_;
std::mutex DiagServerSessionHandler::cursor_mtx_;

DiagServerSessionHandler*
DiagServerSessionHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerSessionHandler();
        }
    }

    return instance_;
}

DiagServerSessionHandler::DiagServerSessionHandler()
: time_mgr_(new TimerManager())
{
}

void
DiagServerSessionHandler::Init()
{
    DG_DEBUG << "DiagServerSessionHandler::Init";
    if (nullptr != time_mgr_) {
        time_mgr_->Init();
    }

    DiagServerSessionMgr::getInstance()->Init();
    support_session_service_.emplace(DIAG_SERVER_SERVICE_REQUEST_OPC_SESSION_CONTROL);
    support_session_service_.emplace(DIAG_SERVER_SERVICE_REQUEST_OPC_SECURITY_ACCESS);
    support_session_service_.emplace(DIAG_SERVER_SERVICE_REQUEST_OPC_TESTER_PRESENT);
}

void
DiagServerSessionHandler::DeInit()
{
    DG_DEBUG << "DiagServerSessionHandler::DeInit";

    support_session_service_.clear();

    DiagServerSessionMgr::getInstance()->DeInit();

    for (auto& item : present_uds_request) {
        time_mgr_->StopFdTimer(item.second.time_fd);
    }

    present_uds_request.clear();

    if (nullptr != time_mgr_) {
        time_mgr_->DeInit();
        time_mgr_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

bool
DiagServerSessionHandler::GeneralBehaviourCheck(const DiagServerUdsMessage& udsMessage)
{
    DiagServerServiceRequestOpc sid = static_cast<DiagServerServiceRequestOpc>(udsMessage.udsData.at(0));
    DG_DEBUG << "DiagServerSessionHandler::GeneralBehaviourCheck Sid " << UINT8_TO_STRING(sid);

    // General server response behaviour
    // SID supported                    NRC 0x11
    if (!DiagServerConfig::getInstance()->QuerySidSupport(sid, udsMessage.taType)) {
        DG_ERROR << "DiagServerSessionHandler::GeneralBehaviourCheck NRC[0x11] [Service Not Supported]";
        ReplyNegativeResponse(sid, udsMessage, kServiceNotSupported);
        return false;
    }

    // Authoritarian                    NRC 0x34 ISO 14229 Page: 393
    // SID supported in active session  NRC 0x7F
    if (!DiagServerConfig::getInstance()->QuerySidSupportInActiveSession(sid, udsMessage.udsSa)) {
        DG_ERROR << "DiagServerSessionHandler::GeneralBehaviourCheck NRC[0x7F] [Service Not Supported In Active Session]";
        ReplyNegativeResponse(sid, udsMessage, kServiceNotSupportedInActiveSession);
        return false;
    }

    // SID security check OK            NRC 0x33
    DiagAccessPermissionDataInfo permission;
    if (DiagServerConfig::getInstance()->QueryAccessPermissionBySid(sid, permission)) {
        bool allowed = false;
        for (auto& item : permission.allowedSecurityLevels) {
            if (0 == item) {
                allowed = true;
                break;
            }

            if (item == DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetSecurityLevel()) {
                allowed = true;
            }
        }

        if (!allowed) {
            DG_ERROR << "DiagServerSessionHandler::GeneralBehaviourCheck NRC[0x33] [Security Access Denied]";
            ReplyNegativeResponse(sid, udsMessage, kSecurityAccessDenied);
            return false;
        }
    }

    // Request message with SubFunction parameter
    if (DiagServerConfig::getInstance()->QuerySidHaveSubFunction(sid)) {
        if (!GeneralBehaviourCheckWithSubFunction(udsMessage)) {
            return false;
        }
    }

    return true;
}

bool
DiagServerSessionHandler::GeneralBehaviourCheckWithSubFunction(const DiagServerUdsMessage& udsMessage)
{
    DiagServerServiceRequestOpc sid = static_cast<DiagServerServiceRequestOpc>(udsMessage.udsData.at(0));
    uint8_t sub_func = udsMessage.udsData.at(1);

    DG_DEBUG << "DiagServerSessionHandler::GeneralBehaviourCheckWithSubFunction Sid " << UINT8_TO_STRING(sid) << " SubFunc " << UINT8_TO_STRING(sub_func);
    // Subfunction supported ever for the SID                   NRC 0x12
    if (!DiagServerConfig::getInstance()->QuerySubFunctionSupportForSid(sid, sub_func, udsMessage.taType)) {
        DG_ERROR << "DiagServerSessionHandler::GeneralBehaviourCheckWithSubFunction NRC[0x12] [Subfunction Not Supported]";
        ReplyNegativeResponse(sid, udsMessage, kSubfunctionNotSupported);
        return false;
    }

    // Authoritarian                                            NRC 0x34 ISO 14229 Page: 393
    // Subfunction supported in active session for the SID      NRC 0x7E
    if (!DiagServerConfig::getInstance()->QuerySubFunctionSupportInActiveSession(sid, sub_func)) {
        DG_ERROR << "DiagServerSessionHandler::GeneralBehaviourCheckWithSubFunction NRC[0x7E] [SubFunction Not Supported In ActiveSession]";
        ReplyNegativeResponse(sid, udsMessage, kSubFunctionNotSupportedInActiveSession);
        return false;
    }

    // Subfunction security check OK                            NRC 0x33
    DiagAccessPermissionDataInfo permission;
    if (DiagServerConfig::getInstance()->QueryAccessPermissionBySidAndSubFunc(sid, sub_func, permission)) {
        bool allowed = false;
        for (auto& item : permission.allowedSecurityLevels) {
            if (0 == item) {
                allowed = true;
                break;
            }

            if (item == DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetSecurityLevel()) {
                allowed = true;
            }
        }

        if (!allowed) {
            DG_ERROR << "DiagServerSessionHandler::GeneralBehaviourCheckWithSubFunction NRC[0x33] [Security Access Denied]";
            ReplyNegativeResponse(sid, udsMessage, kSecurityAccessDenied);
            return false;
        }
    }

    return true;
}

void
DiagServerSessionHandler::RecvUdsMessage(DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSessionHandler::RecvUdsMessage Current Session ["
             << DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetCurrentSession() << "]"
             << " SA: " << udsMessage.udsSa << " TA: " << udsMessage.udsTa
             << " Security Level: " << DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetSecurityLevel();

    if (udsMessage.udsData.empty()) {
        DG_ERROR << "DiagServerSessionHandler::RecvUdsMessage uds data is empty!";
        return;
    }

    // Special handling of session retention
    if ((udsMessage.udsData.size() == 2)
        && (udsMessage.udsData[0] == DIAG_SERVER_SERVICE_REQUEST_OPC_TESTER_PRESENT)
        && (udsMessage.udsData[1] == 0x80)) {
        DiagServerSessionMgr::getInstance()->DealwithSpecialSessionRetention();
        return;
    }

    // Server is busy                   NRC 0x21
    uint8_t sid = udsMessage.udsData.at(0);
    auto itr_client_find = present_uds_request.find(udsMessage.udsSa);
    if (itr_client_find != present_uds_request.end()) {
        DG_ERROR << "DiagServerSessionHandler::RecvUdsMessage NRC[0x21] [Busy Repeat Request]";
        ReplyNegativeResponse(static_cast<DiagServerServiceRequestOpc>(sid), udsMessage, kBusyRepeatRequest);
        return;
    }

    {
        std::lock_guard<std::mutex> lck(cursor_mtx_);
        SessionPending pending;
        pending.sid = sid;
        pending.count = 0;
        pending.maxNumber = DiagServerConfig::getInstance()->GetSidMaxPendingNum(static_cast<DiagServerServiceRequestOpc>(sid));
        pending.source_addr = udsMessage.udsSa;
        pending.is_p2_star_timer = false;
        pending.time_fd = -1;
        uint16_t p2_time = DiagServerConfig::getInstance()->QuerySessionP2Timer(DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetCurrentSession());
        time_mgr_->StartFdTimer(pending.time_fd, p2_time,
                                std::bind(&DiagServerSessionHandler::PendingTimeout,this, std::placeholders::_1),
                                reinterpret_cast<void*>(udsMessage.udsSa), false);
        present_uds_request.insert(std::make_pair(udsMessage.udsSa, pending));
    }

    // Check suppress Postive Rsponse Msg Ind Bit
    if (DiagServerConfig::getInstance()->QuerySidHaveSubFunction(static_cast<DiagServerServiceRequestOpc>(sid))) {
        if (udsMessage.udsData.size() < 2) {
            DG_ERROR << "DiagServerSessionHandler::RecvUdsMessage NRC[0x13] [incorrect message length]";
            ReplyNegativeResponse(static_cast<DiagServerServiceRequestOpc>(sid), udsMessage, kIncorrectMessageLengthOrInvalidFormat);
            return;
        }

        uint8_t sub_func = udsMessage.udsData.at(1);
        if ((sub_func & DIAG_SERVER_SUPPRESS_POS_RES_MSG_INDICATION_BIT) == DIAG_SERVER_SUPPRESS_POS_RES_MSG_INDICATION_BIT) {
            if (DiagServerConfig::getInstance()->QuerySubFunctionSupportSuppressPosMsgindication(static_cast<DiagServerServiceRequestOpc>(sid), udsMessage.udsData.at(1) & 0x7F)) {
                udsMessage.suppressPosRspMsgIndBit = 0x01;
                udsMessage.udsData.at(1) = udsMessage.udsData.at(1) & 0x7F;  // 1000 0001 >> 0000 0001
            }
            else {
                DG_ERROR << "DiagServerSessionHandler::RecvUdsMessage NRC[0xF0] [not support suppress posMsgindication]";
                ReplyNegativeResponse(static_cast<DiagServerServiceRequestOpc>(sid), udsMessage, kNotSupportSuppressPosMsgindication);
                return;
            }
        }
        else {
            udsMessage.suppressPosRspMsgIndBit = 0x00;
        }
    }

    // General behaviour check
    if (!GeneralBehaviourCheck(udsMessage)) {
        // DG_ERROR << "DiagServerSessionHandler::RecvUdsMessage General Behaviour Check Failed!";
        return;
    }

    auto itr_find_service = support_session_service_.find(sid);
    if (itr_find_service != support_session_service_.end()) {
        DiagServerSessionMgr::getInstance()->DealwithSessionLayerService(udsMessage);
    }
    else {
        DiagServerSessionMgr::getInstance()->DealwithApplicationLayerService(udsMessage);
    }
}

void
DiagServerSessionHandler::ReplyUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSessionHandler::ReplyUdsMessage Response Client [" << udsMessage.udsTa << "] " << UINT8_VEC_TO_STRING(udsMessage.udsData);
    auto itr_find = present_uds_request.find(udsMessage.udsTa);
    if (itr_find == present_uds_request.end()) {
        DG_WARN << "DiagServerSessionHandler::ReplyUdsMessage no request! invalid ta:" << udsMessage.udsTa;
        return;
    }

    if (3 == udsMessage.udsData.size()) {
        if ((udsMessage.udsData.at(0) == kNegativeHead) && (udsMessage.udsData.at(2) == kRequestCorrectlyReceivedResponsePending)) {
            DiagServerUdsDataHandler::getInstance()->ReplyUdsMessage(udsMessage);
            return;
        }
    }

    time_mgr_->StopFdTimer(itr_find->second.time_fd);
    present_uds_request.erase(udsMessage.udsTa);

    if ((udsMessage.udsData.at(0) != kNegativeHead) && (udsMessage.suppressPosRspMsgIndBit == 0x01)) {
        DG_DEBUG << "DiagServerSessionHandler::ReplyUdsMessage Not Reply Possitive Response";
        return;
    }

    DiagServerUdsDataHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerSessionHandler::ReplyNegativeResponse(const DiagServerServiceRequestOpc sid, const DiagServerUdsMessage& udsMessage, const DiagServerNrcErrc errorCode)
{
    // DG_DEBUG << "DiagServerSessionHandler::ReplyNegativeResponse";
    DiagServerUdsMessage udsMsg;
    udsMsg.Change(udsMessage);

    udsMsg.udsData.emplace_back(DiagServerNrcErrc::kNegativeHead);
    udsMsg.udsData.emplace_back(sid);
    udsMsg.udsData.emplace_back(errorCode);

    ReplyUdsMessage(udsMsg);
}

void
DiagServerSessionHandler::TransmitUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    // DG_DEBUG << "DiagServerSessionHandler::TransmitUdsMessage sa: " << udsMessage.udsSa << " ta: " << udsMessage.udsTa;
    DiagServerUdsMgr::getInstance()->AnalyzeUdsMessage(udsMessage);
}

void
DiagServerSessionHandler::PendingTimeout(void * data)
{
    std::lock_guard<std::mutex> lck(cursor_mtx_);
    DiagServerSessionMgr::getInstance()->DealwithSpecialSessionRetention(true);
    uint16_t source_addr = reinterpret_cast<long>(data);
    DG_DEBUG << "DiagServerSessionHandler::PendingTimeout Client " << source_addr;

    auto itr_find = present_uds_request.find(source_addr);
    if (itr_find == present_uds_request.end()) {
        DG_WARN << "DiagServerSessionHandler::PendingTimeout Client " << source_addr << " request have been response";
        return;
    }

    DiagServerUdsMessage udsMsg;
    udsMsg.udsSa = source_addr;
    itr_find->second.count++;
    if (itr_find->second.count > itr_find->second.maxNumber) {
        DG_ERROR << "DiagServerSessionHandler::PendingTimeout exceeds the maximum number: " << itr_find->second.maxNumber;
        ReplyNegativeResponse(static_cast<DiagServerServiceRequestOpc>(itr_find->second.sid),
                          udsMsg, kConditionsNotCorrect);
        return;
    }

    ReplyNegativeResponse(static_cast<DiagServerServiceRequestOpc>(itr_find->second.sid),
                          udsMsg, kRequestCorrectlyReceivedResponsePending);

    if (!itr_find->second.is_p2_star_timer) {
        uint16_t p2_star_time = DiagServerConfig::getInstance()->QuerySessionP2StarTimer(DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetCurrentSession());
        itr_find->second.time_fd = -1;
        itr_find->second.is_p2_star_timer = true;
        time_mgr_->StartFdTimer(itr_find->second.time_fd, p2_star_time,
                                std::bind(&DiagServerSessionHandler::PendingTimeout,this, std::placeholders::_1),
                                reinterpret_cast<void*>(source_addr), true);
        DG_DEBUG << "DiagServerSessionHandler::PendingTimeout Start P2*_Server Timer client address " << source_addr << " Timer fd " << itr_find->second.time_fd;
    }
}

void
DiagServerSessionHandler::OnDoipNetlinkStatusChange(const DoipNetlinkStatus doipNetlinkStatus, const uint16_t address)
{
    DG_DEBUG << "DiagServerSessionHandler::OnDoipNetlinkStatusChange Netlink Status " << doipNetlinkStatus
                                                                << " Address " << UINT16_VEC_TO_STRING(address);

    if (DiagServerConfig::getInstance()->GetDiagServerPhysicAddress() == address) {
        DiagServerSessionMgr::getInstance()->DealwithNetlinkStatusChange();
        if (doipNetlinkStatus == DoipNetlinkStatus::kDown) {
            for (auto& item : present_uds_request) {
                time_mgr_->StopFdTimer(item.second.time_fd);
            }

            present_uds_request.clear();
        }
    }
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
