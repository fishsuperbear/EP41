#ifndef DIAG_UDS_TRANSPORT_UDS_MESSAGE_H
#define DIAG_UDS_TRANSPORT_UDS_MESSAGE_H

#include <cstdint>  // uint8_t
#include <memory>  // unique_ptr
#include <string>
#include <map>
#include <optional>
#include "protocol_types.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {
    // Data identifier size in uds message
    const size_t DID_DATA_SIZE = 2U;

    // Routine identifier size in uds message
    const size_t RID_SIZE = 2U;

    const size_t FILE_TRANS_HEAD_SIZE = 2U;

namespace MetaInfoKey {
    const std::string PEER_IP = "peerIp";
    const std::string ACTIVATION_TYPE = "activationType";
    const std::string SOURCE_ADDRESS= "sourceAddress";
    const std::string TARGET_ADDRESS= "targetAddress";
    const std::string CONVERSATION_ID= "conversationId";
    const std::string REQUSET_HANDLE= "requestHandle";
    const std::string SESSION= "session";
}

/// @brief Class which represents a UDS message
/// @uptrace **SWS_DM_00291**
class UdsMessage {
public:
    // @brief Type for UDS source and target addresses.
    // @uptrace{SWS_DM_00293}
    using Address = uint16_t;
    // @brief Type for the meta information attached to a UdsMessage.
    // @uptrace{SWS_DM_00294}
    using MetaInfoMap = std::map<std::string, std::string>;
    // @brief Type of target address in UdsMessage.
    // @uptrace{SWS_DM_00296}
    enum class TargetAddressType : std::uint8_t {
        kPhysical = 0U,
        kFunctional = 1U
    };
    using PayloadSizeType = hozon::netaos::diag::uds_transport::ByteVector::size_type;

public:
    /// @uptrace **SWS_DM_09010**
    virtual ~UdsMessage() = default;

    /// @uptrace{SWS_DM_00302}
    virtual void AddMetaInfo(std::shared_ptr<const MetaInfoMap> metaInfo) {
        meta_info_ = *metaInfo;
    };

    MetaInfoMap GetMetaInfo() const {
        return meta_info_;
    }

    std::optional<std::string> GetMetaInfoKeyValue(const std::string& key) const;

    /// @brief Get the UDS message data starting with the SID (A_Data as per ISO)
    /// @uptrace{SWS_DM_00300}
    virtual const uds_transport::ByteVector& GetUdsPayload() const {
        return payload_;
    }

    /// @brief Get the source address of the uds message.
    /// @uptrace{SWS_DM_00297}
    virtual Address GetSa() const noexcept {
        return source_address_;
    }

    /// @brief Get the target address of the uds message.
    /// @uptrace{SWS_DM_00298}
    virtual Address GetTa() const noexcept {
        return target_address_;
    }

    /// @brief Get the target address type of the uds message.
    /// @uptrace{SWS_DM_00299}
    virtual TargetAddressType GetTaType() const noexcept {
        return request_ta_type_;
    }

protected:
    // @brief non public default ctor
    // @uptrace{SWS_DM_09012}
    UdsMessage() = default;
    UdsMessage(const Address udsSa,
        const Address udsTa,
        const hozon::netaos::diag::uds_transport::ByteVector& udsPayload = {})
    : source_address_(udsSa)
    , target_address_(udsTa)
    {
        this->payload_.assign(udsPayload.begin(), udsPayload.end());
    }

    UdsMessage(const Address udsSa,
        const Address udsTa,
        const TargetAddressType requestTa,
        const uds_transport::ByteVector& udsPayload = {})
    : source_address_(udsSa)
    , target_address_(udsTa)
    , request_ta_type_(requestTa)
    {
        this->payload_.assign(udsPayload.begin(), udsPayload.end());
    }

private:
    // @brief Payload of the uds message.
    // @uptrace ***SWS_DM_00291***
    // @uptrace ***SWS_DM_09028***
    hozon::netaos::diag::uds_transport::ByteVector payload_ {};

private:
    Address source_address_ {0U};
    Address target_address_ {0U};
    TargetAddressType request_ta_type_; // request target address type. if this is a response, it means the sa type.
    MetaInfoMap meta_info_;
};

/// @brief unique_ptr for constant UdsMessages as provided by the generic/core DM part towards the
/// UdsTransportLayer-Plugin.
/// @uptrace{SWS_DM_00304}
using UdsMessageConstPtr = std::unique_ptr<const UdsMessage>;

/// @brief unique_ptr for UdsMessagesas provided by the generic/core DM part towards the UdsTransportLayer-Plugin.
/// @uptrace{SWS_DM_00303}
using UdsMessagePtr = std::unique_ptr<UdsMessage>;
}  // namespace uds_transport
}  // namespace diag 
}  // namespace netaos
}  // namespace hozon

#endif  // DIAG_UDS_TRANSPORT_UDS_MESSAGE_H
