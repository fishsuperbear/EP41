#ifndef DIAG_SERVER_TRANSPORT_EVENT_RECEIVER_H
#define DIAG_SERVER_TRANSPORT_EVENT_RECEIVER_H

#include "cm/include/proxy.h"
#include "diag/ipc/api/ipc_server.h"
#include "diag/ipc/proto/diag.pb.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {

using namespace hozon::netaos::cm;

class DiagServerTransportEventReceiver : public IPCServer
{
public:
    DiagServerTransportEventReceiver();
    ~DiagServerTransportEventReceiver();

    void Init();
    void DeInit();

    virtual int32_t Process(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp);

private:
    void RemoteDiagEventCallback();

private:
    std::shared_ptr<Proxy> remote_diag_proxy_;

};

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_TRANSPORT_EVENT_RECEIVER_H