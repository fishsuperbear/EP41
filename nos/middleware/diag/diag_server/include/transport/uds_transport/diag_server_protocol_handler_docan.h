/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 * Description: diag server protocol handler(DoCAN)
 */

#ifndef DIAG_SERVER_PROTOCOL_HANDLER_DOCAN_H
#define DIAG_SERVER_PROTOCOL_HANDLER_DOCAN_H

#include "diag/diag_server/include/transport/uds_transport/uds_message.h"
#include "diag/diag_server/include/transport/uds_transport/uds_transport_protocol_handler.h"
#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_mgr_impl.h"
#include "diag/diag_server/include/transport/diag_server_transport_service.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {

/**
 * @brief DoCAN_UdsTransportProtocolHandler class
 * Inherited from UdsTransportProtocolHandler class.
 */
class DoCAN_UdsTransportProtocolHandler : public UdsTransportProtocolHandler {

public:

    /**
     * @brief Constructor function of class DoCAN_UdsTransportProtocolHandler.
     * 
     * @param handlerId ID of uds transport protocol handler.
     * @param transportProtocolMgr Mgr of uds transport protocol handler.
     */
    DoCAN_UdsTransportProtocolHandler(const UdsTransportProtocolHandlerID handlerId,
        UdsTransportProtocolMgr& transportProtocolMgr);

    /// @brief Destructor function of class DoCAN_UdsTransportProtocolHandler.
    virtual ~DoCAN_UdsTransportProtocolHandler();

    /// @brief Initializes handler.
    virtual InitializationResult Initialize() const;

    /// @brief Tells the UdsTransportProtocolHandler, that it shall notify the DM core via UdsTransport
    /// ProtocolMgr::ChannelReestablished()) if the given channel has been re-established after next Start().
    virtual bool NotifyReestablishment() const;

    /// @brief Start processing the implemented Uds Transport Protocol.
    virtual bool Start();

    /// @brief Method to indicate that this UdsTransportProtocolHandler should terminate.
    virtual void Stop();

    /**
     * @brief Transmit a Uds message via the underlying Uds Transport Protocol channel.
     * 
     * @param message Const ptr of reply udsmessage.
     * @param transmitChannelId Channel ID.
     * @return void
     */
    virtual void Transmit(UdsMessageConstPtr message, ChannelID transmitChannelId) const;

protected:
    DoCAN_UdsTransportProtocolHandler(const DoCAN_UdsTransportProtocolHandler& other);
    DoCAN_UdsTransportProtocolHandler& operator=(const DoCAN_UdsTransportProtocolHandler& other);

private:
    void DoCanConfirmCallback(docan_confirm* docanConfirm);
    void DoCanIndicationCallback(docan_indication* docanIndication);
};
}  // namespace uds_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_PROTOCOL_HANDLER_DOCAN_H