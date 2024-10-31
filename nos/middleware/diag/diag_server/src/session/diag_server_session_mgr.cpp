#include "diag/diag_server/include/event_manager/diag_server_event_handler.h"
#include "diag/diag_server/include/session/diag_server_session_mgr.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/security/diag_server_security_mgr.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerSessionMgr* DiagServerSessionMgr::instance_ = nullptr;
std::mutex DiagServerSessionMgr::mtx_;

DiagServerSessionMgr*
DiagServerSessionMgr::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerSessionMgr();
        }
    }

    return instance_;
}

DiagServerSessionMgr::DiagServerSessionMgr()
: time_fd_(-1)
, time_mgr_(new TimerManager())
{
}

void
DiagServerSessionMgr::Init()
{
    DG_DEBUG << "DiagServerSessionMgr::Init";
    if (nullptr != time_mgr_) {
        time_mgr_->Init();
    }

    diag_session_.SetCurrentSession(kDefaultSession);
    diag_session_.SetSourceAddress(0x00);
    diag_session_.SetSecurityLevel(DiagSecurityLevelId_Non);
}

void
DiagServerSessionMgr::DeInit()
{
    DG_DEBUG << "DiagServerSessionMgr::DeInit";
    if (nullptr != time_mgr_) {
        time_mgr_->DeInit();
        time_mgr_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
DiagServerSessionMgr::DealwithSessionLayerService(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSessionMgr::DealwithSessionLayerService ";

    uint8_t sid = udsMessage.udsData.at(0);
    uint8_t sub_func = udsMessage.udsData.at(1);
    switch (sid)
    {
    case DIAG_SERVER_SERVICE_REQUEST_OPC_SESSION_CONTROL:
    {
        if (udsMessage.udsData.size() != 2) {
            DG_ERROR << "DiagServerSessionMgr::DealwithSessionLayerService NRC[0x13] [Incorrect MessageLength Or Invalid Format]";
            DiagServerSessionHandler::getInstance()->ReplyNegativeResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_SESSION_CONTROL, udsMessage, kIncorrectMessageLengthOrInvalidFormat);
            return;
        }

        if (0x02 == udsMessage.udsData.at(1)) {
            if (!DiagServerTransPortCM::getInstance()->IsCheckUpdateStatusOk()) {
                DG_ERROR << "DiagServerSessionMgr::DealwithSessionLayerService NRC[0x22] [update status condition not met]";
                DiagServerSessionHandler::getInstance()->ReplyNegativeResponse(static_cast<DiagServerServiceRequestOpc>(sid), udsMessage, kConditionsNotCorrect);
                return;
            }
        }

        SessionControlProcess(sub_func, udsMessage);
        break;
    }
    case DIAG_SERVER_SERVICE_REQUEST_OPC_SECURITY_ACCESS:
    {
        SecurityAccessProcess(udsMessage);
        break;
    }
    case DIAG_SERVER_SERVICE_REQUEST_OPC_TESTER_PRESENT:
    {
        if (udsMessage.udsData.size() != 2) {
            DG_ERROR << "DiagServerSessionMgr::DealwithSessionLayerService NRC[0x13] [Incorrect MessageLength Or Invalid Format]";
            DiagServerSessionHandler::getInstance()->ReplyNegativeResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_TESTER_PRESENT, udsMessage, kIncorrectMessageLengthOrInvalidFormat);
            return;
        }

        TestPresentProcess(udsMessage);
        break;
    }
    default:
        break;
    }
}

void
DiagServerSessionMgr::DealwithApplicationLayerService(const DiagServerUdsMessage& udsMessage)
{
    RestartSessionTimer(static_cast<DiagServerSessionCode>(diag_session_.GetCurrentSession()));
    DiagServerSessionHandler::getInstance()->TransmitUdsMessage(udsMessage);
}

void
DiagServerSessionMgr::DealwithSpecialSessionRetention(const bool isPending)
{
    RestartSessionTimer(static_cast<DiagServerSessionCode>(diag_session_.GetCurrentSession()), isPending);
}

void
DiagServerSessionMgr::DealwithNetlinkStatusChange()
{
    ResetUdsService(kDefaultSession);
}

void
DiagServerSessionMgr::SessionControlProcess(const uint8_t income_session, const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Request Client " << UINT16_TO_STRING(udsMessage.udsSa) << " Session " << UINT8_TO_STRING(income_session);
    DiagServerSessionCode current_session = diag_session_.GetCurrentSession();

    if (current_session == kDefaultSession) {
        // Default -> Default
        if (income_session == kDefaultSession) {
            DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Session Switch [Default]->[Default]";
            ReplySessionPositiveResponse(static_cast<DiagServerSessionCode>(income_session), udsMessage);
            return;
        }
        // Default -> Extended
        else if (income_session == kExtendedSession) {
            diag_session_.SetSessionInfo(udsMessage.udsSa, static_cast<DiagServerSessionCode>(income_session), DiagSecurityLevelId_Non);
            diag_session_.Print();
            RestartSessionTimer(static_cast<DiagServerSessionCode>(income_session));
            ResetUdsService(static_cast<DiagServerSessionCode>(income_session));
            DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Session Switch [Default]->[Extended]";
            ReplySessionPositiveResponse(static_cast<DiagServerSessionCode>(income_session), udsMessage);
        }
        // Default -> Programming
        else if (income_session == kProgrammingSession) {
            DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Session Switch [Default]->[Programming]";
            DG_ERROR << "DiagServerSessionMgr::SessionControlProcess NRC[0x22] [Conditions Not Correct]";
            DiagServerSessionHandler::getInstance()->ReplyNegativeResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_SESSION_CONTROL, udsMessage, kConditionsNotCorrect);
        }
    }
    else {
        if (udsMessage.udsSa != diag_session_.GetSourceAddress()) {
            DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Current Session is NonDefault Session, Reject Client Request Addr: " << udsMessage.udsSa;
            DG_ERROR << "DiagServerSessionMgr::SessionControlProcess NRC[0x10] [General Reject]";
            DiagServerSessionHandler::getInstance()->ReplyNegativeResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_SESSION_CONTROL, udsMessage, kGeneralReject);
            return;
        }

        if (current_session == kExtendedSession) {
            // Extended -> Default
            if (income_session == kDefaultSession) {
                StopSessionTimer();
                ResetUdsService(static_cast<DiagServerSessionCode>(income_session));
                DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Session Switch [Extended]->[Default]";

                // revert dtc control setting sw
                DiagServerEventHandler::getInstance()->reportSessionChange(kDefaultSession);
                ReplySessionPositiveResponse(static_cast<DiagServerSessionCode>(income_session), udsMessage);
            }
            // Extended -> Extended
            else if (income_session == kExtendedSession) {
                ResetUdsService(static_cast<DiagServerSessionCode>(income_session));
                RestartSessionTimer(static_cast<DiagServerSessionCode>(income_session));
                DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Session Switch [Extended]->[Extended]";
                ReplySessionPositiveResponse(static_cast<DiagServerSessionCode>(income_session), udsMessage);
            }
            // Extended -> Programming
            else if (income_session == kProgrammingSession) {
                diag_session_.SetCurrentSession(static_cast<DiagServerSessionCode>(income_session));
                ResetUdsService(static_cast<DiagServerSessionCode>(income_session));
                RestartSessionTimer(static_cast<DiagServerSessionCode>(income_session));
                DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Session Switch [Extended]->[Programming]";
                ReplySessionPositiveResponse(static_cast<DiagServerSessionCode>(income_session), udsMessage);
            }
        }
        else if (current_session == kProgrammingSession) {
            // Programming -> Default
            if (income_session == kDefaultSession) {
                diag_session_.SetCurrentSession(static_cast<DiagServerSessionCode>(income_session));
                StopSessionTimer();
                ResetUdsService(static_cast<DiagServerSessionCode>(income_session));
                DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Session Switch [Programming]->[Default]";
                ReplySessionPositiveResponse(static_cast<DiagServerSessionCode>(income_session), udsMessage);
            }
            // Programming -> Extended
            else if (income_session == kExtendedSession) {
                diag_session_.SetCurrentSession(static_cast<DiagServerSessionCode>(income_session));
                ResetUdsService(static_cast<DiagServerSessionCode>(income_session));
                RestartSessionTimer(static_cast<DiagServerSessionCode>(income_session));
                DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Session Switch [Programming]->[Extended]";
                ReplySessionPositiveResponse(static_cast<DiagServerSessionCode>(income_session), udsMessage);
            }
            // Programming -> Programming
            else if (income_session == kProgrammingSession) {
                diag_session_.SetCurrentSession(static_cast<DiagServerSessionCode>(income_session));
                ResetUdsService(static_cast<DiagServerSessionCode>(income_session));
                RestartSessionTimer(static_cast<DiagServerSessionCode>(income_session));
                DG_DEBUG << "DiagServerSessionMgr::SessionControlProcess Session Switch [Programming]->[Programming]";
                ReplySessionPositiveResponse(static_cast<DiagServerSessionCode>(income_session), udsMessage);
            }
        }
    }
}

DiagSessionInfo&
DiagServerSessionMgr::GetDiagSessionInfomation()
{
    return diag_session_;
}

void
DiagServerSessionMgr::SecurityAccessProcess(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSessionMgr::SecurityAccessProcess";
    RestartSessionTimer(diag_session_.GetCurrentSession());
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);
}

void
DiagServerSessionMgr::TestPresentProcess(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSessionMgr::TestPresentProcess";
    if (diag_session_.GetCurrentSession() == kDefaultSession) {
        ReplyTestPresentPositiveResponse(udsMessage);
        return;
    }

    RestartSessionTimer(diag_session_.GetCurrentSession());
    ReplyTestPresentPositiveResponse(udsMessage);
}

void
DiagServerSessionMgr::ResetUdsService(const DiagServerSessionCode session)
{
    DG_DEBUG << "DiagServerSessionMgr::ResetUdsService session " << session;
    NotifySessionStatusChange(session);
}

void
DiagServerSessionMgr::StartSessionTimer(const DiagServerSessionCode session)
{
    DG_DEBUG << "DiagServerSessionMgr::StartSessionTimer session " << session;
    if (time_mgr_ == nullptr) {
        DG_ERROR << "DiagServerSessionMgr::StartSessionTimer time_mgr_ == null";
        return;
    }

    uint16_t s3_time = DiagServerConfig::getInstance()->QuerySessionS3Timer(session);
    time_fd_ = -1;
    time_mgr_->StartFdTimer(time_fd_, s3_time, std::bind(&DiagServerSessionMgr::SessionTimeout, this, std::placeholders::_1), NULL, false);
}

void
DiagServerSessionMgr::StopSessionTimer()
{
    DG_DEBUG << "DiagServerSessionMgr::StopSessionTimer";
    if (time_mgr_ == nullptr) {
        DG_ERROR << "DiagServerSessionMgr::StopSessionTimer time_mgr_ == null";
        return;
    }

    time_mgr_->StopFdTimer(time_fd_);
    time_fd_ = -1;
}

void
DiagServerSessionMgr::RestartSessionTimer(const DiagServerSessionCode session, const bool isPending)
{
    DG_DEBUG << "DiagServerSessionMgr::RestartSessionTimer session : " << session;
    if(DiagServerSessionCode::kDefaultSession == session) {
        DG_INFO << "DiagServerSessionMgr::RestartSessionTimer session is default session.";
        return;
    }

    if (time_mgr_ == nullptr) {
        DG_ERROR << "DiagServerSessionMgr::RestartSessionTimer time_mgr_ == null";
        return;
    }

    uint16_t s3_time = DiagServerConfig::getInstance()->QuerySessionS3Timer(session);
    if (isPending) {
        s3_time += 500;
    }

    time_mgr_->StopFdTimer(time_fd_);
    time_fd_ = -1;
    time_mgr_->StartFdTimer(time_fd_, s3_time, std::bind(&DiagServerSessionMgr::SessionTimeout, this, std::placeholders::_1), NULL, false);
}

void
DiagServerSessionMgr::SessionTimeout(void * data)
{
    DG_DEBUG << "DiagServerSessionMgr::SessionTimeout Session Switch From " << diag_session_.GetCurrentSession() << " To Default";
    ResetUdsService(kDefaultSession);

    if (kExtendedSession == diag_session_.GetCurrentSession()) {
        DiagServerEventHandler::getInstance()->reportSessionChange(kDefaultSession);
    }

    diag_session_.SetSessionInfo(0x00, kDefaultSession, DiagSecurityLevelId_Non);
}

void
DiagServerSessionMgr::RegisterSessionStatusListener(std::function<void(DiagServerSessionCode)> listener)
{
    // DG_DEBUG << "DiagServerSessionMgr::RegisterSessionStatusListener";
    session_listener_list_.emplace_back(listener);
}

void
DiagServerSessionMgr::NotifySessionStatusChange(const DiagServerSessionCode session)
{
    for (auto& item : session_listener_list_) {
        item(session);
    }
}

void
DiagServerSessionMgr::ReplySessionPositiveResponse(const DiagServerSessionCode session, const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSessionMgr::ReplySessionPositiveResponse";
    DiagServerUdsMessage udsMsg;
    udsMsg.Change(udsMessage);

    uint16_t p2 = DiagServerConfig::getInstance()->QuerySessionP2Timer(session);
    uint16_t ps_star = DiagServerConfig::getInstance()->QuerySessionP2StarTimer(session) / 10;

    udsMsg.udsData.emplace_back(DIAG_SERVER_SERVICE_REPLY_OPC_SESSION_CONTROL);
    udsMsg.udsData.emplace_back(session);
    udsMsg.udsData.emplace_back(static_cast<uint8_t>((p2 >> 8) & 0xff));
    udsMsg.udsData.emplace_back(static_cast<uint8_t>(p2 & 0xff));
    udsMsg.udsData.emplace_back(static_cast<uint8_t>((ps_star >> 8) & 0xff));
    udsMsg.udsData.emplace_back(static_cast<uint8_t>(ps_star & 0xff));

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMsg);
}

void
DiagServerSessionMgr::ReplyTestPresentPositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSessionMgr::ReplyTestPresentPositiveResponse";
    DiagServerUdsMessage udsMsg;
    udsMsg.Change(udsMessage);

    udsMsg.udsData.emplace_back(DIAG_SERVER_SERVICE_REPLY_OPC_TESTER_PRESENT);
    udsMsg.udsData.emplace_back(0x00);

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMsg);
}

void
DiagSessionInfo::Print()
{
    DG_DEBUG << "Source Address[" << UINT16_TO_STRING(this->source_addr) << "] Active Session[" << this->current_session << "] Security Level[" << this->security_level << "]";
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
