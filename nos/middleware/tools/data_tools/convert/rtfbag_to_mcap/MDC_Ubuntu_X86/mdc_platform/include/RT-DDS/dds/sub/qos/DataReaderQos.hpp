/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DataReaderQoS.hpp
 */

#ifndef DDS_SUB_QOS_DATA_READER_QOS_HPP
#define DDS_SUB_QOS_DATA_READER_QOS_HPP

#include <RT-DDS/dds/core/policy/DataReaderProtocol.hpp>
#include <RT-DDS/dds/core/policy/Durability.hpp>
#include <RT-DDS/dds/core/policy/Deadline.hpp>
#include <RT-DDS/dds/core/policy/DestinationOrder.hpp>
#include <RT-DDS/dds/core/policy/History.hpp>
#include <RT-DDS/dds/core/policy/IgnoreLocalParticipant.hpp>
#include <RT-DDS/dds/core/policy/Reliability.hpp>
#include <RT-DDS/dds/core/policy/Transport.hpp>
#include <RT-DDS/dds/core/policy/TransportChannel.hpp>
#include <RT-DDS/dds/core/policy/TimeBasedFilter.hpp>
#include <RT-DDS/dds/core/policy/DataProcessType.hpp>
#include <RT-DDS/dds/core/policy/PropertyList.hpp>

namespace dds {
namespace sub {
namespace qos {
/**
 * @brief Container of the QoS policies that a dds::sub::DataReader supports.
 */
class DataReaderQos {
public:
    /**
     * @brief Gets DataReaderProtocol QoS policy by const reference.
     * @return dds::core::policy::DataReaderProtocol
     * @req{AR-iAOS-RCS-DDS-05302,
     * DataReaderQos shall support DataReaderProtocol policy,
     * QM,
     * DR-iAOS-RCS-DDS-00030
     * }
     */
    const core::policy::DataReaderProtocol &DataReaderProtocol() const noexcept
    {
        return dataReaderProtocol_;
    }

    /**
     * @brief Set DataReaderProtocol QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::DataReaderProtocol policy) noexcept
    {
        dataReaderProtocol_ = std::move(policy);
    }

    /**
     * @brief Gets Durability QoS policy by const reference.
     * @return dds::core::policy::Durability
     * @req{AR-iAOS-RCS-DDS-05303,
     * DataReaderQos shall support Durability policy,
     * QM,
     * DR-iAOS-RCS-DDS-00030
     * }
     */
    const core::policy::Durability &Durability() const noexcept
    {
        return durability_;
    }

    /**
     * @brief Set Durability QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::Durability policy) noexcept
    {
        durability_ = std::move(policy);
    }

    /**
     * @brief Gets Deadline QoS policy by const reference.
     * @return dds::core::policy::Deadline
     * @req{AR-iAOS-RCS-DDS-05304,
     * DataReaderQos shall support Deadline policy,
     * QM,
     * DR-iAOS-RCS-DDS-00030
     * }
     */
    const core::policy::Deadline &Deadline() const noexcept
    {
        return deadline_;
    }

    /**
     * @brief Set Deadline QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::Deadline policy) noexcept
    {
        deadline_ = std::move(policy);
    }

    /**
     * @brief Gets DestinationOrder QoS policy by const reference.
     * @return dds::core::policy::DestinationOrder
     * @req{AR-iAOS-RCS-DDS-05305,
     * DataReaderQos shall support DestinationOrder policy,
     * QM,
     * DR-iAOS-RCS-DDS-00030
     * }
     */
    const core::policy::DestinationOrder &DestinationOrder() const noexcept
    {
        return destinationOrder_;
    }

    /**
     * @brief Set DestinationOrder QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::DestinationOrder policy) noexcept
    {
        destinationOrder_ = std::move(policy);
    }

    /**
     * @brief Gets History QoS policy by const reference.
     * @return dds::core::policy::History
     * @req{AR-iAOS-RCS-DDS-05306,
     * DataReaderQos shall support History policy,
     * QM,
     * DR-iAOS-RCS-DDS-00030
     * }
     */
    const core::policy::History &History() const noexcept
    {
        return history_;
    }

    /**
     * @brief Set History QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::History policy) noexcept
    {
        history_ = std::move(policy);
    }

    /**
     * @brief Set PropertyList QoS policy
     * @param policy the PropertyList policy to set
     * @return void
     */
    void Set(dds::core::policy::PropertyList propertyList)
    {
        propertyList_ = std::move(propertyList);
    }

    /**
     * @brief Gets IgnoreLocalParticipant QoS policy by const reference.
     * @return dds::core::policy::IgnoreLocalParticipant
     */
    const core::policy::IgnoreLocalParticipant &IgnoreLocalParticipant() const noexcept
    {
        return ignoreLocalParticipant_;
    }

    /**
     * @brief Set IgnoreLocalParticipant QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::IgnoreLocalParticipant policy) noexcept
    {
        ignoreLocalParticipant_ = std::move(policy);
    }

    /**
     * @brief Gets Reliability QoS policy by const reference.
     * @return dds::core::policy::Reliability
     * @req{AR-iAOS-RCS-DDS-05307,
     * DataReaderQos shall support Reliability policy,
     * QM,
     * DR-iAOS-RCS-DDS-00030
     * }
     */
    const core::policy::Reliability &Reliability() const noexcept
    {
        return reliability_;
    }

    /**
     * @brief Set Reliability QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::Reliability policy) noexcept
    {
        reliability_ = policy;
    }

    /**
     * @brief Gets Transport QoS policy by const reference.
     * @return dds::core::policy::Transport
     * @req{AR-iAOS-RCS-DDS-05308,
     * DataReaderQos shall support Transport policy,
     * QM,
     * DR-iAOS-RCS-DDS-00030
     * }
     */
    const core::policy::Transport &Transport() const noexcept
    {
        return transport_;
    }

    /**
     * @brief Set Transport QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::Transport policy) noexcept
    {
        transport_ = std::move(policy);
    }

    /**
     * @brief Gets TransportChannel QoS policy by const reference.
     * @return dds::core::policy::TransportChannel
     * @req{AR-iAOS-RCS-DDS-05309,
     * DataReaderQos shall support TransportChannel policy,
     * QM,
     * DR-iAOS-RCS-DDS-00030, DR-iAOS-RCS-DDS-00082, DR-iAOS-RCS-DDS-00086
     * }
     */
    const core::policy::TransportChannel &TransportChannel() const noexcept
    {
        return transportChannel_;
    }

    /**
     * @brief Set TransportChannel QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::TransportChannel policy)
    {
        transportChannel_ = std::move(policy);
    }

    const core::policy::TimeBasedFilter& TimeBasedFilter() const noexcept
    {
        return timeBasedFilter_;
    }

    /**
     * @brief Set DestinationOrder QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05301,
     * DataReaderQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(core::policy::TimeBasedFilter filter) noexcept
    {
        timeBasedFilter_ = std::move(filter);
    }

    /**
     * @ingroup
     * @brief gets the direct return flag by const reference
     * @param NONE
     * @return dds::core::policy::DataProcessType
     * @req{
     *
     * }
     */
    const core::policy::DataProcessType& DataProcessType() const noexcept
    {
        return dataProcessType_;
    }

    /**
     * @ingroup
     * @brief set direct return policy
     * @param policy direct data process policy to set
     * @return void
     * @req{
     *
     * }
     */
    void Set(core::policy::DataProcessType dataProcessType) noexcept
    {
        dataProcessType_ = dataProcessType;
    }

    /**
     * @brief Get property QoS policy by const reference
     * @param NONE
     * @return dds::core::policy::PropertyList
     */
    const dds::core::policy::PropertyList& PropertyList() const noexcept
    {
        return propertyList_;
    }


private:
    dds::core::policy::DataReaderProtocol dataReaderProtocol_{};
    dds::core::policy::Durability durability_{};
    dds::core::policy::Deadline deadline_{};
    dds::core::policy::DestinationOrder destinationOrder_{};
    dds::core::policy::History history_{};
    dds::core::policy::IgnoreLocalParticipant ignoreLocalParticipant_{};
    dds::core::policy::Reliability reliability_{dds::core::policy::ReliabilityKind::BEST_EFFORT};
    dds::core::policy::Transport transport_{};
    dds::core::policy::TransportChannel transportChannel_{};
    dds::core::policy::TimeBasedFilter timeBasedFilter_ {};
    dds::core::policy::DataProcessType dataProcessType_{};
    dds::core::policy::PropertyList propertyList_{};
};
}
}
}

#endif /* DDS_SUB_QOS_DATA_READER_QOS_HPP */

