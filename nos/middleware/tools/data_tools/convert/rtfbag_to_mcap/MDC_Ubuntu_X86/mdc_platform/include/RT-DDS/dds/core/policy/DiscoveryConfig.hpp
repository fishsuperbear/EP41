/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DiscoveryConfig.hpp
 */

#ifndef DDS_CORE_POLICY_DISCOVERY_CONFIG_HPP
#define DDS_CORE_POLICY_DISCOVERY_CONFIG_HPP

#include <RT-DDS/dds/core/policy/RtpsReliableWriterProtocol.hpp>
#include <RT-DDS/dds/core/policy/RtpsReliableReaderProtocol.hpp>
#include <RT-DDS/dds/core/policy/RtpsSPDPProtocol.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Configures the discovery mechanism.
 *
 * This QoS policy controls the amount of delay in discovering entities in the
 * system, the amount of discovery traffic in the network, and the unicast SPDP
 * announcement times and intervals.
 */
class DiscoveryConfig {
public:
    /**
     * @brief RTPS protocol-related configuration settings for the RTPS reliable
     * writer associated to a built-in publication writer.
     */
    void PublicationWriter(RtpsReliableWriterProtocol p) noexcept
    {
        publicationWriter_ = p;
    }
    /**
     * @brief RTPS protocol-related configuration settings for the RTPS reliable
     * writer associated to a built-in subscription writer.
     */
    void SubscriptionWriter(RtpsReliableWriterProtocol p) noexcept
    {
        subscriptionWriter_ = p;
    }
    /**
     * @brief RTPS protocol-related configuration settings for the RTPS reliable
     * reader associated to a built-in publication reader.
     */
    void PublicationReader(RtpsReliableReaderProtocol p) noexcept
    {
        publicationReader_ = p;
    }
    /**
     * @brief RTPS protocol-related configuration settings for the RTPS reliable
     * reader associated to a built-in subscription reader.
     */
    void SubscriptionReader(RtpsReliableReaderProtocol p) noexcept
    {
        subscriptionReader_ = p;
    }
    /**
     * @brief RTPS protocol-related configuration settings for SPDP announcements
     */
    void AnnouncementConfig(RtpsSPDPProtocol p) noexcept
    {
        announcementConfig_ = p;
    }

    /**
     * @brief The time period after which other DomainParticipants can consider
     * this one dead if they do not receive a liveliness packet from this
     * DomainParticipant.
     * @param t
     */
    void SetParticipantLivelinessLeaseDuration(dds::core::Duration t) noexcept
    {
        participantLivelinessLeaseDuration_ = t;
    }

    /**
     * @brief The time period after which other DomainParticipants can consider
     * this one dead if they do not receive a liveliness packet from this
     * DomainParticipant.
     * @param t
     */
    void SetParticipantLivelinessAssertPeriod(dds::core::Duration t) noexcept
    {
        participantLivelinessAssertPeriod_ = t;
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    const RtpsReliableWriterProtocol &PublicationWriter() const noexcept
    {
        return publicationWriter_;
    }
    /**
     * @brief Getter (see setter with the same name)
     */
    const RtpsReliableWriterProtocol &SubscriptionWriter() const noexcept
    {
        return subscriptionWriter_;
    }
    /**
     * @brief Getter (see setter with the same name)
     */
    const RtpsReliableReaderProtocol &PublicationReader() const noexcept
    {
        return publicationReader_;
    }
    /**
     * @brief Getter (see setter with the same name)
     */
    const RtpsReliableReaderProtocol &SubscriptionReader() const noexcept
    {
        return subscriptionReader_;
    }
    /**
     * @brief Getter (see setter with the same name)
     */
    const RtpsSPDPProtocol &AnnouncementConfig() const noexcept
    {
        return announcementConfig_;
    }

    /**
     * @brief Getter (see setter with the same name)
     * @return
     */
    const dds::core::Duration GetParticipantLivelinessLeaseDuration() const noexcept
    {
        return participantLivelinessLeaseDuration_;
    }

    /**
     * @brief Getter (see setter with the same name)
     * @return
     */
    const dds::core::Duration GetParticipantLivelinessAssertPeriod() const noexcept
    {
        return participantLivelinessAssertPeriod_;
    }
private:
    RtpsReliableWriterProtocol publicationWriter_{};
    RtpsReliableWriterProtocol subscriptionWriter_{};
    RtpsReliableReaderProtocol publicationReader_{};
    RtpsReliableReaderProtocol subscriptionReader_{};
    RtpsSPDPProtocol announcementConfig_{};

    dds::core::Duration participantLivelinessLeaseDuration_{60U, 0U};
    dds::core::Duration participantLivelinessAssertPeriod_{24U, 0U};
};
}
}
}

#endif /* DDS_CORE_POLICY_DISCOVERY_CONFIG_HPP */

