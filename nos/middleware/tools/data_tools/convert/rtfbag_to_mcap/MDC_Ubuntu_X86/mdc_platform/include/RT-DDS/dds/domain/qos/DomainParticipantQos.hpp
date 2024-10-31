/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DomainParticipantQos.hpp
 */

#ifndef DDS_DOMAIN_QOS_DOMAIN_PARTICIPANT_QOS_HPP
#define DDS_DOMAIN_QOS_DOMAIN_PARTICIPANT_QOS_HPP

#include <RT-DDS/dds/core/policy/Discovery.hpp>
#include <RT-DDS/dds/core/policy/DiscoveryConfig.hpp>
#include <RT-DDS/dds/core/policy/UserData.hpp>
#include <RT-DDS/dds/core/policy/Transport.hpp>
#include <RT-DDS/dds/core/policy/TransportInterfaces.hpp>
#include <RT-DDS/dds/core/policy/DiscoveryFilter.hpp>

namespace dds {
namespace domain {
namespace qos {
/**
 * @brief Container of the QoS policies that a dds::domain::DomainParticipant
 * supports.
 */
class DomainParticipantQos {
public:
    /**
     * @brief Set Discovery QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-02201,
     * DomainParticipantQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::Discovery policy) noexcept
    {
        discovery_ = std::move(policy);
    }

    /**
     * @brief Set DiscoveryConfig QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-02201,
     * DomainParticipantQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::DiscoveryConfig policy) noexcept
    {
        discoveryConfig_ = std::move(policy);
    }

    /**
     * @brief Set Transport QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-02203,
     * DomainParticipantQos shall support Transport policy,
     * QM,
     * DR-iAOS-RCS-DDS-00007, DR-iAOS-RCS-DDS-00008, DR-iAOS-RCS-DDS-00028
     * }
     */
    void Set(dds::core::policy::Transport policy) noexcept
    {
        transport_ = std::move(policy);
    }

    /**
     * @brief Set TransportInterfaces QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-02203,
     * DomainParticipantQos shall support Transport policy,
     * QM,
     * DR-iAOS-RCS-DDS-00007, DR-iAOS-RCS-DDS-00008, DR-iAOS-RCS-DDS-00028
     * }
     */
    void Set(dds::core::policy::TransportInterfaces policy) noexcept
    {
        transportInterfaces_ = std::move(policy);
    }

    /**
     * @brief Set UserData QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-02202,
     * DomainParticipantQos shall support UserData policy,
     * QM,
     * DR-iAOS-RCS-DDS-00007, DR-iAOS-RCS-DDS-00008, DR-iAOS-RCS-DDS-00028
     * }
     */
    void Set(dds::core::policy::UserData policy)
    {
        userData_ = std::move(policy);
    }

    /**
     * @brief Gets Discovery QoS policy by const reference.
     * @return dds::core::policy::Discovery
     * @req{AR-iAOS-RCS-DDS-02201,
     * DomainParticipantQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    const dds::core::policy::Discovery &Discovery(void) const noexcept
    {
        return discovery_;
    }

    /**
     * @brief Gets DiscoveryConfig QoS policy by const reference.
     * @return dds::core::policy::DiscoveryConfig
     * @req{AR-iAOS-RCS-DDS-02201,
     * DomainParticipantQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    const dds::core::policy::DiscoveryConfig &DiscoveryConfig(void) const noexcept
    {
        return discoveryConfig_;
    }

    /**
     * @brief Gets Transport QoS policy by const reference.
     * @return dds::core::policy::Transport
     * @req{AR-iAOS-RCS-DDS-02203,
     * DomainParticipantQos shall support Transport policy,
     * QM,
     * DR-iAOS-RCS-DDS-00007, DR-iAOS-RCS-DDS-00008, DR-iAOS-RCS-DDS-00028
     * }
     */
    const dds::core::policy::Transport &Transport(void) const noexcept
    {
        return transport_;
    }

    /**
     * @brief Gets TransportInterfaces QoS policy by const reference.
     * @return dds::core::policy::TransportInterfaces
     * @req{AR-iAOS-RCS-DDS-02203,
     * DomainParticipantQos shall support Transport policy,
     * QM,
     * DR-iAOS-RCS-DDS-00007, DR-iAOS-RCS-DDS-00008, DR-iAOS-RCS-DDS-00028
     * }
     */
    const dds::core::policy::TransportInterfaces &TransportInterfaces(void) const noexcept
    {
        return transportInterfaces_;
    }

    /**
     * @brief Gets UserData QoS policy by const reference.
     * @return dds::core::policy::UserData
     * @req{AR-iAOS-RCS-DDS-02202,
     * DomainParticipantQos shall support UserData policy,
     * QM,
     * DR-iAOS-RCS-DDS-00007, DR-iAOS-RCS-DDS-00008, DR-iAOS-RCS-DDS-00028
     * }
     */
    const dds::core::policy::UserData &UserData(void) const noexcept
    {
        return userData_;
    }

    void Set(dds::core::policy::DiscoveryFilter policy)
    {
        discoveryFilter_ = std::move(policy);
    }

    const dds::core::policy::DiscoveryFilter &DiscoveryFilter(void) const noexcept
    {
        return discoveryFilter_;
    }

private:
    dds::core::policy::Discovery discovery_{};
    dds::core::policy::DiscoveryConfig discoveryConfig_{};
    dds::core::policy::Transport transport_{dds::core::policy::Transport::UDP()};
    dds::core::policy::TransportInterfaces transportInterfaces_{};
    dds::core::policy::UserData userData_{};
    dds::core::policy::DiscoveryFilter discoveryFilter_{};
};
}
}
}

#endif /* DDS_DOMAIN_QOS_DOMAIN_PARTICIPANT_QOS_HPP */

