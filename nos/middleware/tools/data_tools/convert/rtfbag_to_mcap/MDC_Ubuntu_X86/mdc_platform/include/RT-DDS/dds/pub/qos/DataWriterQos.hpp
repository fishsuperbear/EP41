/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DataWriterQos.hpp
 */

#ifndef DDS_PUB_QOS_DATA_WRITER_QOS_HPP
#define DDS_PUB_QOS_DATA_WRITER_QOS_HPP

#include <RT-DDS/dds/core/policy/DataWriterProtocol.hpp>
#include <RT-DDS/dds/core/policy/Deadline.hpp>
#include <RT-DDS/dds/core/policy/DestinationOrder.hpp>
#include <RT-DDS/dds/core/policy/Durability.hpp>
#include <RT-DDS/dds/core/policy/History.hpp>
#include <RT-DDS/dds/core/policy/Reliability.hpp>
#include <RT-DDS/dds/core/policy/Transport.hpp>
#include <RT-DDS/dds/core/policy/TransportChannel.hpp>
#include <RT-DDS/dds/core/policy/TransportPriority.hpp>
#include <RT-DDS/dds/core/policy/Authentication.hpp>
#include <RT-DDS/dds/core/policy/Extension.hpp>
#include <RT-DDS/dds/core/policy/DataProcessType.hpp>
#include <RT-DDS/dds/core/policy/PropertyList.hpp>
#include <RT-DDS/dds/core/policy/TransportMode.hpp>

namespace dds {
namespace pub {
namespace qos {
/**
 * @brief Container of the QoS policies that a dds::pub::DataWriter
 * supports.
 */
class DataWriterQos {
public:
    /**
     * @brief Set DataWriterProtocol QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04301,
     * DataWriterQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::DataWriterProtocol policy) noexcept
    {
        dataWriterProtocol_ = std::move(policy);
    }

    /**
     * @brief Set Deadline QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04301,
     * DataWriterQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::Deadline policy) noexcept
    {
        deadline_ = std::move(policy);
    }

    /**
     * @brief Set DestinationOrder QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04301,
     * DataWriterQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::DestinationOrder policy) noexcept
    {
        destinationOrder_ = std::move(policy);
    }

    /**
     * @brief Set Durability QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04301,
     * DataWriterQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::Durability policy) noexcept
    {
        durability_ = std::move(policy);
    }

    /**
     * @brief Set History QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04301,
     * DataWriterQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::History policy) noexcept
    {
        history_ = std::move(policy);
    }

    /**
     * @brief Set Reliability QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04301,
     * DataWriterQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::Reliability policy) noexcept
    {
        reliability_ = std::move(policy);
    }

    /**
     * @brief Set TransportPriority QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04301,
     * DataWriterQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::TransportPriority policy) noexcept
    {
        transportPriority_ = std::move(policy);
    }

    /**
     * @brief Set Transport QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04301,
     * DataWriterQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::Transport policy) noexcept
    {
        transport_ = std::move(policy);
    }

    /**
     * @brief Set TransportChannel QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04301,
     * DataWriterQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::TransportChannel policy)
    {
        transportChannel_ = std::move(policy);
    }
    /**
    * @brief Set Authentication QoS policy.
    * @return dds::core::policy::Authentication
    * DataWriterQos shall support setting of policies,
    */
    void Set(dds::core::policy::Authentication policy)
    {
        authentication_ = std::move(policy);
    }

    /**
    * @brief Set Extension QoS policy.
    * @return void
    * DataWriterQos shall support setting of policies,
    */
    void Set(dds::core::policy::Extension policy) noexcept
    {
        extension_ = std::move(policy);
    }

    /**
     * @brief Set dataProcessType QoS policy
     * @param policy the dataProcessType policy to set
     * @return void
     */
    void Set(dds::core::policy::DataProcessType dataProcessType) noexcept
    {
        dataProcessType_ = dataProcessType;
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
     * @brief Set transportMode QoS policy
     * @param policy the transportMode policy to set
     * @return void
     * @req {AR-iAOS-RCS-DDS-20028, AR20221205568478
     * DDS supports synchronous/asynchronous sending of data by setting TransportMode,
     * SR-iAOS3-RCS-DDS-00055
     */
    void Set(const dds::core::policy::TransportMode &transportMode) noexcept
    {
        transportMode_ = transportMode;
    }

    /**
     * @brief Gets DataWriterProtocol QoS policy by const reference.
     * @return dds::core::policy::DataWriterProtocol
     * @req{AR-iAOS-RCS-DDS-04302,
     * DataWriterQos shall support DataWriterProtocol policy,
     * QM,
     * DR-iAOS-RCS-DDS-00029
     * }
     */
    const dds::core::policy::DataWriterProtocol &DataWriterProtocol(void) const noexcept
    {
        return dataWriterProtocol_;
    }

    /**
     * @brief Gets Deadline QoS policy by const reference.
     * @return dds::core::policy::Deadline
     * @req{AR-iAOS-RCS-DDS-04303,
     * DataWriterQos shall support Deadline policy,
     * QM,
     * DR-iAOS-RCS-DDS-00029
     * }
     */
    const dds::core::policy::Deadline &Deadline(void) const noexcept
    {
        return deadline_;
    }

    /**
     * @brief Gets DestinationOrder QoS policy by const reference.
     * @return dds::core::policy::DestinationOrder
     * @req{AR-iAOS-RCS-DDS-04304,
     * DataWriterQos shall support DestinationOrder policy,
     * QM,
     * DR-iAOS-RCS-DDS-00029
     * }
     */
    const dds::core::policy::DestinationOrder &DestinationOrder(void) const noexcept
    {
        return destinationOrder_;
    }

    /**
     * @brief Gets Durability QoS policy by const reference.
     * @return dds::core::policy::Durability
     * @req{AR-iAOS-RCS-DDS-04305,
     * DataWriterQos shall support Durability policy,
     * QM,
     * DR-iAOS-RCS-DDS-00029
     * }
     */
    const dds::core::policy::Durability &Durability(void) const noexcept
    {
        return durability_;
    }

    /**
     * @brief Gets History QoS policy by const reference.
     * @return dds::core::policy::History
     * @req{AR-iAOS-RCS-DDS-04306,
     * DataWriterQos shall support History policy,
     * QM,
     * DR-iAOS-RCS-DDS-00029
     * }
     */
    const dds::core::policy::History &History(void) const noexcept
    {
        return history_;
    }

    /**
     * @brief Gets Reliability QoS policy by const reference.
     * @return dds::core::policy::Reliability
     * @req{AR-iAOS-RCS-DDS-04307,
     * DataWriterQos shall support Reliability policy,
     * QM,
     * DR-iAOS-RCS-DDS-00029
     * }
     */
    const dds::core::policy::Reliability &Reliability(void) const noexcept
    {
        return reliability_;
    }

    /**
     * @brief Gets Transport QoS policy by const reference.
     * @return dds::core::policy::Transport
     * @req{AR-iAOS-RCS-DDS-04308,
     * DataWriterQos shall support Transport policy,
     * QM,
     * DR-iAOS-RCS-DDS-00029
     * }
     */
    const dds::core::policy::Transport &Transport(void) const noexcept
    {
        return transport_;
    }

    /**
     * @brief Gets TransportChannel QoS policy by const reference.
     * @return dds::core::policy::TransportChannel
     * @req{AR-iAOS-RCS-DDS-04309,
     * DataWriterQos shall support TransportChannel policy,
     * QM,
     * DR-iAOS-RCS-DDS-00029, DR-iAOS-RCS-DDS-00082, DR-iAOS-RCS-DDS-00086
     * }
     */
    const dds::core::policy::TransportChannel &TransportChannel(void) const noexcept
    {
        return transportChannel_;
    }

    /**
     * @brief Gets TransportPriority QoS policy by const reference.
     * @return dds::core::policy::TransportPriority
     * @req{AR-iAOS-RCS-DDS-04310,
     * DataWriterQos shall support TransportPriority policy,
     * QM,
     * DR-iAOS-RCS-DDS-00029, DR-iAOS-RCS-DDS-00072
     * }
     */
    const dds::core::policy::TransportPriority &TransportPriority(void) const noexcept
    {
        return transportPriority_;
    }

    /**
     * @brief Gets Authentication QoS policy by const reference.
     * @return dds::core::policy::Authentication
     * DataWriterQos shall support Authentication policy,
     */
    const dds::core::policy::Authentication &Authentication(void) const noexcept
    {
        return authentication_;
    }

    /**
     * @brief Gets Extension QoS policy by const reference.
     * @return dds::core::policy::Extension
     * DataWriterQos shall support Extension policy,
     */
    const dds::core::policy::Extension &Extension(void) const noexcept
    {
        return extension_;
    }

    /**
     * @brief Get dataProcessType QoS policy by const reference
     * @param NONE
     * @return dds::core::policy::DataProcessType
     */
    const dds::core::policy::DataProcessType& DataProcessType() const noexcept
    {
        return dataProcessType_;
    }

    /**
     * @brief Get property QoS policy by const reference
     * @param NONE
     * @return dds::core::policy::PropertyList
     */
    dds::core::policy::PropertyList &PropertyList() noexcept
    {
        return propertyList_;
    }

    /**
     * @brief Get transport QoS policy by const reference
     * @param NONE
     * @return dds::core::policy::TransportMode
     * @req {AR-iAOS-RCS-DDS-20028, AR20221205568478
     * DDS supports synchronous/asynchronous sending of data by setting TransportMode,
     * SR-iAOS3-RCS-DDS-00055
     */
    const dds::core::policy::TransportMode &TransportMode() const noexcept
    {
        return transportMode_;
    }


private:
    dds::core::policy::DataWriterProtocol dataWriterProtocol_{};
    dds::core::policy::Deadline deadline_{};
    dds::core::policy::DestinationOrder destinationOrder_{};
    dds::core::policy::Durability durability_{};
    dds::core::policy::History history_{};
    dds::core::policy::Reliability reliability_{dds::core::policy::ReliabilityKind::RELIABLE};
    dds::core::policy::Transport transport_{};
    dds::core::policy::TransportChannel transportChannel_{};
    dds::core::policy::TransportPriority transportPriority_{};
    dds::core::policy::Authentication authentication_{};
    dds::core::policy::Extension extension_{};
    dds::core::policy::DataProcessType dataProcessType_{};
    dds::core::policy::PropertyList propertyList_{};
    dds::core::policy::TransportMode transportMode_{};
};
}
}
}

#endif /* DDS_PUB_QOS_DATA_WRITER_QOS_HPP */

