
#ifndef DIAG_SERVER_SESSION_MGR_H
#define DIAG_SERVER_SESSION_MGR_H

#include <mutex>
#include <vector>
#include <map>
#include "diag/common/include/data_def.h"
#include "diag/common/include/timer_manager.h"
#include "diag/diag_server/include/common/diag_server_def.h"
#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag_server_transport_cm.h"

namespace hozon {
namespace netaos {
namespace diag {

using namespace hozon::netaos::diag::cm_transport;

class DiagSessionInfo {
public:
    uint16_t GetSourceAddress()
    {
        return this->source_addr;
    }
    DiagServerSessionCode GetCurrentSession()
    {
        return this->current_session;
    }
    uint8_t GetSecurityLevel()
    {
        return this->security_level;
    }

    void SetSourceAddress(const uint16_t sourceAddr)
    {
        this->source_addr = sourceAddr;
    }
    void SetCurrentSession(const DiagServerSessionCode session)
    {
        if (session != this->current_session) {
            DiagServerTransPortCM::getInstance()->DiagSessionEventSend(session);
        }

        this->current_session = session;
    }
    void SetSecurityLevel(const uint8_t level)
    {
        this->security_level = level;
    }
    void SetSessionInfo(const uint16_t sourceAddr, const DiagServerSessionCode session, const uint8_t levelId)
    {
        this->source_addr = sourceAddr;
        this->security_level = levelId;
        SetCurrentSession(session);
    }
    void Print();

private:
    uint16_t source_addr;
    DiagServerSessionCode current_session;
    uint8_t security_level;
};

class DiagServerSessionMgr {

public:
    static DiagServerSessionMgr* getInstance();

    void Init();
    void DeInit();

    void RegisterSessionStatusListener(std::function<void(DiagServerSessionCode)> listener);
    DiagSessionInfo& GetDiagSessionInfomation();

    void DealwithSessionLayerService(const DiagServerUdsMessage& udsMessage);
    void DealwithApplicationLayerService(const DiagServerUdsMessage& udsMessage);
    void DealwithSpecialSessionRetention(const bool isPending = false);

    void DealwithNetlinkStatusChange();

private:
    DiagServerSessionMgr();
    DiagServerSessionMgr(const DiagServerSessionMgr &);
    DiagServerSessionMgr & operator = (const DiagServerSessionMgr &);

    void SessionControlProcess(const uint8_t income_session, const DiagServerUdsMessage& udsMessage);
    void SecurityAccessProcess(const DiagServerUdsMessage& udsMessage);
    void TestPresentProcess(const DiagServerUdsMessage& udsMessage);

    void ResetUdsService(const DiagServerSessionCode session);
    void NotifySessionStatusChange(const DiagServerSessionCode session);

    void SessionTimeout(void * data);
    void StartSessionTimer(const DiagServerSessionCode session);
    void RestartSessionTimer(const DiagServerSessionCode session, const bool isPending = false);
    void StopSessionTimer();

    void ReplySessionPositiveResponse(const DiagServerSessionCode session, const DiagServerUdsMessage& udsMessage);
    void ReplyTestPresentPositiveResponse(const DiagServerUdsMessage& udsMessage);

private:
    static std::mutex mtx_;
    static DiagServerSessionMgr* instance_;

    int time_fd_;
    std::unique_ptr<TimerManager> time_mgr_;

    DiagSessionInfo diag_session_;

    std::vector<std::function<void(DiagServerSessionCode)>> session_listener_list_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_SESSION_MGR_H
