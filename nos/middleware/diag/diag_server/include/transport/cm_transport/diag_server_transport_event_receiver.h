#ifndef DIAG_SERVER_TRANSPORT_EVENT_RECEIVER_H
#define DIAG_SERVER_TRANSPORT_EVENT_RECEIVER_H

#include "cm/include/proxy.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {

using namespace hozon::netaos::cm;

class DiagServerTransportEventReceiver {
public:
    DiagServerTransportEventReceiver();
    ~DiagServerTransportEventReceiver();

    void Init();
    void DeInit();

private:
    void DiagEventCallback();
    void RemoteDiagEventCallback();

private:
    std::shared_ptr<Proxy> diag_proxy_;
    std::shared_ptr<Proxy> remote_diag_proxy_;

};

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_TRANSPORT_EVENT_RECEIVER_H