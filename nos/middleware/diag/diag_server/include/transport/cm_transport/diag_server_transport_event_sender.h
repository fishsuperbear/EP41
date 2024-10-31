#ifndef DIAG_SERVER_TRANSPORT_EVENT_SENDER_H
#define DIAG_SERVER_TRANSPORT_EVENT_SENDER_H

#include "cm/include/skeleton.h"
#include "diag/diag_server/include/common/diag_server_def.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {

using namespace hozon::netaos::cm;

class DiagServerTransportEventSender {
public:
    DiagServerTransportEventSender();
    ~DiagServerTransportEventSender();

    void Init();
    void DeInit();

    void DiagEventSend(const DiagServerRespUdsMessage& udsMessage);
    void RemoteDiagEventSend(const DiagServerRespUdsMessage& udsMessage);
    void DiagSessionEventSend(const DiagServerSessionCode& session);

private:
    std::shared_ptr<Skeleton> diag_skeleton_;
    std::shared_ptr<Skeleton> remote_diag_skeleton_;
    std::shared_ptr<Skeleton> diag_session_skeleton_;

};

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_TRANSPORT_EVENT_SENDER_H