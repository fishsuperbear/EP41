
#ifndef DIAG_SERVER_TRANSPORT_H
#define DIAG_SERVER_TRANSPORT_H

#include <mutex>
#include <iostream>

#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_mgr_impl.h"
#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_handler_doip.h"
#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_handler_docan.h"
#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_handler_dosomeip.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerTransport {

public:
    static DiagServerTransport* getInstance();

    void Init();
    void DeInit();

    void RecvUdsMessage(const DiagServerUdsMessage& udsMessage, const bool someipChannel = false);
    void ReplyUdsMessage(const DiagServerUdsMessage& udsMessage);

    void HandlerStopped(const uint8_t handlerId);
    void NotifyMessageFailure(const DiagServerUdsMessage& udsMessage);
    void TransmitConfirmation(const DiagServerUdsMessage& udsMessage, const bool confirmResult);
    void NotifyDoipNetlinkStatus(const DoipNetlinkStatus doipNetlinkStatus, const uint16_t address);

private:
    DiagServerTransport();
    DiagServerTransport(const DiagServerTransport &);
    DiagServerTransport & operator = (const DiagServerTransport &);
    void SessionTimeout(void * data);

private:
    static DiagServerTransport* instance_;
    static std::mutex mtx_;

    uds_transport::DoIP_UdsTransportProtocolHandler* doip_transport_handler_;
    uds_transport::UdsTransportProtocolMgrImpl* doip_transport_mgr_;
    uds_transport::DoCAN_UdsTransportProtocolHandler* docan_transport_handler_;
    uds_transport::UdsTransportProtocolMgrImpl* docan_transport_mgr_;
    uds_transport::DoSomeIP_UdsTransportProtocolHandler* dosomeip_transport_handler_;
    uds_transport::UdsTransportProtocolMgrImpl* dosomeip_transport_mgr_;

    bool stop_flag_;
    Server_Req_Channel doserverReqChannel_;
    int time_fd_;
    std::unique_ptr<TimerManager> time_mgr_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_TRANSPORT_H
