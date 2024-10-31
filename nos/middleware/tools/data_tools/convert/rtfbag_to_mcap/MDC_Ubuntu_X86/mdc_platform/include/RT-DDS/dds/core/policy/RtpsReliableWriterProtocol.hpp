/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: RtpsReliableWriterProtocol.hpp
 */

#ifndef DDS_CORE_POLICY_RTPS_RELIABLE_WRITER_PROTOCOL_HPP
#define DDS_CORE_POLICY_RTPS_RELIABLE_WRITER_PROTOCOL_HPP

#include <RT-DDS/dds/core/Duration.hpp>

namespace dds {
namespace core {
namespace policy {
class RtpsReliableWriterProtocol {
public:
    /**
     * @brief The period at which to send heartbeats.
     *
     * A reliable writer will send periodic heartbeats at this rate.
     */
    void HeartBeatPeriod(dds::core::Duration d) noexcept
    {
        heartbeatPeriod_ = d;
    };

    /**
     * @brief Getter (see setter with the same name)
     */
    const dds::core::Duration &HeartBeatPeriod() const noexcept
    {
        return heartbeatPeriod_;
    }

private:
    dds::core::Duration heartbeatPeriod_{dds::core::Duration::FromMilliSecs(DEFAULT_HEART_BEAT_DURATION)};
    static const uint32_t DEFAULT_HEART_BEAT_DURATION{100U};
};
}
}
}

#endif /* DDS_CORE_POLICY_RTPS_RELIABLE_WRITER_PROTOCOL_HPP */

