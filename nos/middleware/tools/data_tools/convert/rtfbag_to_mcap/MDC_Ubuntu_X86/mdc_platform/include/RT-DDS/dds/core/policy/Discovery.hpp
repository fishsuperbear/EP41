/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Discovery.hpp
 */

#ifndef DDS_CORE_POLICY_DISCOVERY_HPP
#define DDS_CORE_POLICY_DISCOVERY_HPP

#include <RT-DDS/dds/core/Types.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Configures entity discovery
 */
class Discovery {
public:
    /**
     * @brief Sets the initial list of peers that the discovery mechanism will
     * contact to announce this DomainParticipant.
     *
     * As part of the participant discovery phase, the
     * dds::domain::DomainParticipant will announce itself to the domain by
     * sending participant DATA messages. The initial_peers specifies the
     * initial list of peers that will be contacted. A remote
     * dds::domain::DomainParticipant is discovered by receiving participant
     * announcements from a remote peer. When the new remote
     * dds::domain::DomainParticipant has been added to the participant's
     * database, the endpoint discovery phase commences and information about
     * the DataWriters and DataReaders is exchanged.
     *
     * Each element of this list must be a peer descriptor in the proper format.
     */
    void InitialPeers(dds::core::StringSeq s) noexcept
    {
        initialPeers_ = std::move(s);
    }

    /**
     * @brief Specifies the multicast group addresses on which discovery-related
     * meta-traffic can be received by the DomainParticipant.
     *
     * The multicast group addresses on which the Discovery mechanism will
     * listen for meta-traffic.
     *
     * Each element of this list must be a valid multicast address (IPv4 or
     * IPv6) in the proper format (see Address Format).
     *
     * The domain_id determines the multicast port on which the Discovery
     * mechanism can receive meta-data.
     */
    void MulticastReceiveAddresses(dds::core::StringSeq s) noexcept
    {
        multicastReceiveAddresses_ = std::move(s);
    }

    /**
     * @breif Getter (see setter with the same name)
     */
    const dds::core::StringSeq &InitialPeers() const noexcept
    {
        return initialPeers_;
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    const dds::core::StringSeq &MulticastReceiveAddresses() const noexcept
    {
        return multicastReceiveAddresses_;
    }

private:
    dds::core::StringSeq initialPeers_{"239.255.0.1", "127.0.0.1"};
    dds::core::StringSeq multicastReceiveAddresses_{"239.255.0.1"};
};
}
}
}

#endif /* DDS_CORE_POLICY_DISCOVERY_HPP */

