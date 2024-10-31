/// @file
/// @brief Common Types for the UDS TransportLayer C++ Interfaces

#ifndef DIAG_UDS_TRANSPORT_PROTOCOL_TYPES_H
#define DIAG_UDS_TRANSPORT_PROTOCOL_TYPES_H

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {
// @brief The type of UDS message payloads.
// @uptrace{SWS_DM_00338}
using ByteVector = std::vector<uint8_t>;

// @brief The identifier of a logical (network) channel, over which UDS messages can be sent/received.
// @uptrace{SWS_DM_00337}
using ChannelID = uint32_t;

// @brief The identifier of an Uds Transport Protocol implementation.
// @uptrace{SWS_DM_00336}
using UdsTransportProtocolHandlerID = uint8_t;

// @brief The type of UDS message priority.
// @uptrace{SWS_DM_00451}
using Priority = uint8_t;

// @brief The type of protocol kind
// @uptrace{SWS_DM_00452}
using ProtocolKind = std::string;

}  // namespace uds_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_UDS_TRANSPORT_PROTOCOL_TYPES_H
