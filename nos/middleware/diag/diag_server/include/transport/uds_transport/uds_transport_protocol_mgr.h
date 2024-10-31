#ifndef DIAG_UDS_TRANSPORT_PROTOCOL_MGR_H
#define DIAG_UDS_TRANSPORT_PROTOCOL_MGR_H

#include <cstddef>
#include <tuple>
#include <utility>  // std::pair
#include <vector>
#include "protocol_types.h"
#include "uds_message.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {

/// @brief Type of tuple to pack UdsTransportProtocolHandlerID and ChannelID together, to form a global
/// unique (among all used UdsTransportProtocolHandlers within DM) identifier of a UdsTransport
/// Protocol channel.
using GlobalChannelIdentifier = std::tuple<UdsTransportProtocolHandlerID, ChannelID>;

// @uptrace{SWS_DM_00306}
class UdsTransportProtocolMgr {
public:
    // @uptrace{SWS_DM_00384}
    enum IndicationResult {
        INDICATION_OK = 0,
        INDICATION_BUSY = 1,
        INDICATION_OVERFLOW = 2,
        INDICATION_UNKNOWN_TARGET_ADDRESS = 3
    };

    // @uptrace{SWS_DM_00307}
    enum TransmissionResult {
        TRANSMIT_OK = 0,
        TRANSMIT_FAILED = 1,
    };

public:
    UdsTransportProtocolMgr() = default;
    virtual ~UdsTransportProtocolMgr() = default;

    /// @brief Notification call from the given transport channel, that it has been reestablished.
    /// @param GlobalChannelIdentifier global_channel_id,
    /// GlobalChannelIdentifier is type of std::tuple<UdsTransportProtocolHandlerID, ChannelID>
    /// @uptrace{SWS_DM_00313}
    virtual void ChannelReestablished(GlobalChannelIdentifier globalChannelId) const = 0;

    /// @brief Hands over a valid received Uds message from transport layer to session layer.
    /// @param UdsMessagePtr message
    /// @uptrace{SWS_DM_00311}
    virtual void HandleMessage(UdsMessagePtr message) const = 0;

    /// @brief Notification from handler, that it has stopped now.
    /// @param UdsTransportProtocolHandlerID handlerId
    /// @uptrace{SWS_DM_00314}
    virtual void HandlerStopped(UdsTransportProtocolHandlerID handlerId) const = 0;

    /// @brief Indicates a message start.
    /// @param UdsMessage::Address source_addr,UdsMessage::Address target_addr,UdsMessage::TargetAddressType type,
    /// @param GlobalChannelIdentifier global_channel_id,std::size_t size,Priority indicatePriority,
    /// @param ProtocolKind indicateProtocolKind
    /// @uptrace{SWS_DM_00309}
    virtual std::pair<IndicationResult, UdsMessagePtr> IndicateMessage(
                                                    UdsMessage::Address sourceAddr,
                                                    UdsMessage::Address targetAddr,
                                                    UdsMessage::TargetAddressType type,
                                                    GlobalChannelIdentifier globalChannelId,
                                                    std::size_t size, Priority priority, ProtocolKind protocolKind,
                                                    std::vector<uint8_t>& payloadInfo) const = 0;

    /// @brief Indicates, that the message has failure.
    /// @param UdsMessagePtr message
    /// @uptrace{SWS_DM_00310}
    virtual void NotifyMessageFailure(UdsMessagePtr message) const = 0;

    /// @brief Notification about the outcome of a transmit request.
    /// @param UdsMessageConstPtr message, TransmissionResult result
    /// @uptrace{SWS_DM_00312}
    virtual void TransmitConfirmation(UdsMessageConstPtr message, TransmissionResult result) const = 0;
private:
    UdsTransportProtocolMgr& operator=(const UdsTransportProtocolMgr& other) = default;
    UdsTransportProtocolMgr(const UdsTransportProtocolMgr& other) = default;
};

}  // namespace uds_transport
}  // namespace diag 
}  // namespace netaos
}  // namespace hozon

#endif  // DIAG_UDS_TRANSPORT_PROTOCOL_MGR_H
