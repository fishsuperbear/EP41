/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 * Description: diag server protocol handler(DoIP)
 */

#ifndef DIAG_SERVER_PROTOCOL_HANDLER_DOIP_H
#define DIAG_SERVER_PROTOCOL_HANDLER_DOIP_H

#include "diag/diag_server/include/transport/uds_transport/uds_message.h"
#include "diag/diag_server/include/transport/uds_transport/uds_transport_protocol_handler.h"
#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_mgr_impl.h"
#include "diag/diag_server/include/transport/diag_server_transport_service.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {

/**
 * @brief DoIP_UdsTransportProtocolHandler class
 * Inherited from UdsTransportProtocolHandler class.
 */
class DoIP_UdsTransportProtocolHandler : public UdsTransportProtocolHandler {

public:

    /// @brief Type of DoIP confirm result.
    enum DoIPConfirmResult {
        CONFIRM_OK,
        CONFIRM_FAILED,
        CONFIRM_TIMEOUT
    };

    /**
     * @brief Constructor function of class DoIP_UdsTransportProtocolHandler.
     * 
     * @param handlerId ID of uds transport protocol handler.
     * @param transportProtocolMgr Mgr of uds transport protocol handler.
     */
    DoIP_UdsTransportProtocolHandler(const UdsTransportProtocolHandlerID handlerId,
        UdsTransportProtocolMgr& transportProtocolMgr);

    /// @brief Destructor function of class DoIP_UdsTransportProtocolHandler.
    virtual ~DoIP_UdsTransportProtocolHandler();

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

    void DoipResetConfirmResult() {doip_confirm_result_ = DoIPConfirmResult::CONFIRM_TIMEOUT;}

protected:
    DoIP_UdsTransportProtocolHandler(const DoIP_UdsTransportProtocolHandler& other);
    DoIP_UdsTransportProtocolHandler& operator=(const DoIP_UdsTransportProtocolHandler& other);

private:
    void DoipConfirmCallback(doip_confirm_t* doipConfirm);
    void DoipIndicationCallback(doip_indication_t* doipIndication);
    void DoipRouteCallback(doip_route_t* doipRoute);

private:
    // static doip_result_t doip_confirm_result_;
    DoIPConfirmResult doip_confirm_result_;
};
}  // namespace uds_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_PROTOCOL_HANDLER_DOIP_H