/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: RtpsReliableReaderProtocol.hpp
 */

#ifndef DDS_CORE_POLICY_RTPS_RELIABLE_READER_PROTOCOL
#define DDS_CORE_POLICY_RTPS_RELIABLE_READER_PROTOCOL

#include <RT-DDS/dds/core/Duration.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Configures aspects of the RTPS protocol related to a reliable
 * DataReader.
 *
 * It is used to config reliable reader according to RTPS protocol.
 */
class RtpsReliableReaderProtocol {
public:
    /**
     * @brief  The minimum delay to respond to a heartbeat.
     *
     * When a reliable reader receives a heartbeat from a remote writer and
     * finds out that it needs to send back an ACK/NACK message, the reader can
     * choose to delay a while. This sets the value of the minimum delay.
     */
    void MinHeartbeatResponseDelay(dds::core::Duration d) noexcept
    {
        minHeartbeatResponseDelay_ = d;
    }

    /**
     * @brief The maximum delay to respond to a heartbeat.
     *
     * When a reliable reader receives a heartbeat from a remote writer and
     * finds out that it needs to send back an ACK/NACK message, the reader can
     * choose to delay a while. This sets the value of maximum delay.
     */
    void MaxHeartbeatResponseDelay(dds::core::Duration d) noexcept
    {
        maxHeartbeatResponseDelay_ = d;
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    const dds::core::Duration &MinHeartbeatResponseDelay() const noexcept
    {
        return minHeartbeatResponseDelay_;
    }
    /**
     * @brief Getter (see setter with the same name)
     */
    const dds::core::Duration &MaxHeartbeatResponseDelay() const noexcept
    {
        return maxHeartbeatResponseDelay_;
    }

private:
    dds::core::Duration minHeartbeatResponseDelay_{dds::core::Duration::Zero()};
    dds::core::Duration maxHeartbeatResponseDelay_{dds::core::Duration::Zero()};
};
}
}
}

#endif /* DDS_CORE_POLICY_RTPS_RELIABLE_READER_PROTOCOL */

