/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 * Description: diag server protocol mgr implement
 */

#ifndef DIAG_SERVER_PROTOCOL_MGR_IMPL_H
#define DIAG_SERVER_PROTOCOL_MGR_IMPL_H

#include "diag/diag_server/include/transport/uds_transport/uds_message.h"
#include "diag/diag_server/include/transport/uds_transport/uds_transport_protocol_mgr.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {

/**
 * @brief UdsMessageImpl class
 * Inherited from UdsMessage class.
 */
class UdsMessageImpl : public UdsMessage {
public:

    /// @brief Default Constructor function of class UdsMessageImpl.
    UdsMessageImpl() = default;

    /// @brief Destructor function of class UdsMessageImpl.
    virtual ~UdsMessageImpl(){}

    /**
     * @brief Constructor function of class UdsMessageImpl.
     * 
     * @param udsSa Source address of udsmessage.
     * @param udsTa Target address of udsmessage.
     * @param udsPayload Uds data.
     */
    UdsMessageImpl(const Address udsSa,
        const Address udsTa,
        const ByteVector& udsPayload = {})
    : UdsMessage(udsSa, udsTa, udsPayload)
    {
    }

    /**
     * @brief Constructor function of class UdsMessageImpl.
     * 
     * @param udsSa Source address of udsmessage.
     * @param udsTa Target address of udsmessage.
     * @param requestTa Type of target address(Physical or Functional).
     * @param udsPayload Uds data.
     */
    UdsMessageImpl(const Address udsSa,
        const Address udsTa,
        const TargetAddressType requestTa,
        const ByteVector& udsPayload = {})
    : UdsMessage(udsSa, udsTa, requestTa, udsPayload)
    {
    }
};

/**
 * @brief UdsTransportProtocolMgrImpl class
 * Inherited from UdsTransportProtocolMgr class.
 */
class UdsTransportProtocolMgrImpl : public UdsTransportProtocolMgr {

public:

    /// @brief Constructor function of class UdsTransportProtocolMgrImpl.
    UdsTransportProtocolMgrImpl();

    /// @brief Destructor function of class UdsTransportProtocolMgrImpl.
    virtual ~UdsTransportProtocolMgrImpl();

    /**
     * @brief Notification call from the given transport channel, that it has been reestablished.
     * 
     * @param globalChannelId GlobalChannelIdentifier is type of std::tuple<UdsTransportProtocolHandlerID, ChannelID>
     * @return void
     */
    virtual void ChannelReestablished (GlobalChannelIdentifier globalChannelId) const;

    /**
     * @brief Hands over a valid received Uds message from transport layer to session layer.
     * 
     * @param message Ptr of uds message.
     * @return void
     */
    virtual void HandleMessage(UdsMessagePtr message) const;

    /**
     * @brief Notification from handler, that it has stopped now.
     * 
     * @param handlerId Id of uds transport protocol handler.
     * @return void
     */
    virtual void HandlerStopped(UdsTransportProtocolHandlerID handlerId) const;

    /**
     * @brief Indicates a message start.
     * 
     * @param sourceAddr Source address of uds message.
     * @param targetAddr Target address of uds message.
     * @param type Type of target address(Physical or Functional).
     * @param globalChannelId GlobalChannelIdentifier is type of std::tuple<UdsTransportProtocolHandlerID, ChannelID>
     * @param size Length of uds data.
     * @param priority Priority of the diagnostic session.
     * @param ProtocolKind indicateProtocolKind.
     * @param payloadInfo Uds data.
     * @return std::pair<IndicationResult, UdsMessagePtr>
     */
    virtual std::pair<IndicationResult, UdsMessagePtr> IndicateMessage(
                                                    UdsMessage::Address sourceAddr,
                                                    UdsMessage::Address targetAddr,
                                                    UdsMessage::TargetAddressType type,
                                                    GlobalChannelIdentifier globalChannelId,
                                                    std::size_t size, Priority priority, ProtocolKind protocolKind,
                                                    std::vector<uint8_t>& payloadInfo) const;

    /**
     * @brief Notification about the outcome of a transmit request.
     * 
     * @param message Ptr of uds message.
     * @return void
     */
    virtual void NotifyMessageFailure(UdsMessagePtr message) const;

    /**
     * @brief Notification about the outcome of a transmit request.
     * 
     * @param message Const ptr of uds message.
     * @param result Result of transmit(OK or FAILED).
     * @return void
     */
    virtual void TransmitConfirmation(UdsMessageConstPtr message, TransmissionResult result) const;


private:
    UdsTransportProtocolMgrImpl(const UdsTransportProtocolMgrImpl &);
    UdsTransportProtocolMgrImpl & operator = (const UdsTransportProtocolMgrImpl &);

};
}  // namespace uds_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_TRANSPORT_PROTOCOL_MGR_IMP_H