#ifndef DIAG_SERVER_SESSION_HANDLER_H
#define DIAG_SERVER_SESSION_HANDLER_H

#include <mutex>
#include <set>
#include <memory>
#include <unordered_map>
#include "diag/common/include/timer_manager.h"
#include "diag/diag_server/include/common/diag_server_def.h"

namespace hozon {
namespace netaos {
namespace diag {

struct SessionPending {
    uint8_t sid;
    uint16_t count;
    uint16_t maxNumber;
    bool is_p2_star_timer;
    uint16_t source_addr;
    int time_fd;
};

class DiagServerSessionHandler {

public:
    static DiagServerSessionHandler* getInstance();

    void Init();
    void DeInit();

    void RecvUdsMessage(DiagServerUdsMessage& udsMessage);
    void ReplyUdsMessage(const DiagServerUdsMessage& udsMessage);

    void TransmitUdsMessage(const DiagServerUdsMessage& udsMessage);

    void ReplyNegativeResponse(const DiagServerServiceRequestOpc sid, const DiagServerUdsMessage& udsMessage, const DiagServerNrcErrc errorCode);
    void OnDoipNetlinkStatusChange(const DoipNetlinkStatus doipNetlinkStatus, const uint16_t address);

private:
    bool GeneralBehaviourCheck(const DiagServerUdsMessage& udsMessage);
    bool GeneralBehaviourCheckWithSubFunction(const DiagServerUdsMessage& udsMessage);
    void PendingTimeout(void * data);
    DiagServerSessionHandler();
    DiagServerSessionHandler(const DiagServerSessionHandler &);
    DiagServerSessionHandler & operator = (const DiagServerSessionHandler &);

private:
    static std::mutex mtx_;
    static std::mutex cursor_mtx_;
    static DiagServerSessionHandler* instance_;

    std::set<uint8_t> support_session_service_;
    std::unique_ptr<TimerManager> time_mgr_;
    std::unordered_map<uint16_t, SessionPending> present_uds_request; // map<client_address, pending>
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_SESSION_HANDLER_H
