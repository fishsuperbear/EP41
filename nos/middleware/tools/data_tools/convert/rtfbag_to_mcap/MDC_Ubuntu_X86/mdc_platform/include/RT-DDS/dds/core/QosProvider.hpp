/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: QosProvider.hpp
 */

#ifndef DDS_CORE_QOS_PROVIDER_HPP
#define DDS_CORE_QOS_PROVIDER_HPP

#include <RT-DDS/dds/core/Reference.hpp>
#include <RT-DDS/dds/core/ReturnCode.hpp>
#include <RT-DDS/dds/domain/qos/DomainParticipantQos.hpp>
#include <RT-DDS/dds/topic/qos/TopicQos.hpp>
#include <RT-DDS/dds/pub/qos/PublisherQos.hpp>
#include <RT-DDS/dds/pub/qos/DataWriterQos.hpp>
#include <RT-DDS/dds/sub/qos/SubscriberQos.hpp>
#include <RT-DDS/dds/sub/qos/DataReaderQos.hpp>

namespace dds {
namespace core {
class QosProviderImpl;

/**
 * @brief The QosProvider class provides a way for a user to control and access
 * the XML QoS profiles.
 * @details
 * A QosProvider is created with a URI that identifies a file location
 * containing QoS profiles, a string representation of the QoS profiles that you
 * want the QosProvider to load, or a URL group that has a combination of files
 * and strings from which to load profiles.
 */
class QosProvider : public dds::core::Reference<QosProviderImpl> {
public:
    /**
     * @brief Create a QosProvider fetching QoS configuration from the specified URI.
     * @param[in] uri The URI describing the location of the QoS profiles to load.
     * @param[in] profile The QoS profile
     * @req{AR-iAOS-RCS-DDS-01201,
     * QosProvider shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00028
     * DR-iAOS-RCS-DDS-00029
     * DR-iAOS-RCS-DDS-00030
     * DR-iAOS-RCS-DDS-00031
     * DR-iAOS-RCS-DDS-00032
     * DR-iAOS-RCS-DDS-00033
     * }
     */
    QosProvider(std::string uri, std::string profile) noexcept;

    /**
     * @brief Default Destructor.
     * @req{AR-iAOS-RCS-DDS-01202,
     * QosProvider shall support destruction process.,
     * QM,
     * DR-iAOS-RCS-DDS-00028
     * DR-iAOS-RCS-DDS-00029
     * DR-iAOS-RCS-DDS-00030
     * DR-iAOS-RCS-DDS-00031
     * DR-iAOS-RCS-DDS-00032
     * DR-iAOS-RCS-DDS-00033
     * }
     */
    ~QosProvider() override = default;

    /**
     * @brief Initializing the QosProvider.
     * @return dds::core::ReturnCode
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @req{AR-iAOS-RCS-DDS-01201,
     * QosProvider shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00028
     * DR-iAOS-RCS-DDS-00029
     * DR-iAOS-RCS-DDS-00030
     * DR-iAOS-RCS-DDS-00031
     * DR-iAOS-RCS-DDS-00032
     * DR-iAOS-RCS-DDS-00033
     * }
     */
    dds::core::ReturnCode Init() const;

    /**
     * @brief Get the dds::domain::qos::DomainParticipantQos with a specified
     *      name associated with this QosProvider.
     * @param[out] qos  DomainParticipantQos to get.
     * @param[in] name  Profile name.
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::UNSUPPORTED
     * @req{AR-iAOS-RCS-DDS-01203,
     * QosProvider shall support getting DomainParticipantQos.,
     * QM,
     * DR-iAOS-RCS-DDS-00028
     * }
     */
    dds::core::ReturnCode GetParticipantQos(
        dds::domain::qos::DomainParticipantQos &qos,
        const std::string &name) const;

    /**
     * @brief Get the dds::topic::qos::TopicQos with a specified name
     *      associated with this QosProvider.
     * @param[out] qos  TopicQos to get.
     * @param[in] name  Profile name.
     * @retval dds::core::ReturnCode::UNSUPPORTED
     * @req{AR-iAOS-RCS-DDS-01204,
     * QosProvider shall support getting TopicQos.,
     * QM,
     * DR-iAOS-RCS-DDS-00032
     * }
     */
    dds::core::ReturnCode GetTopicQos(
        dds::topic::qos::TopicQos &qos,
        const std::string &name) const;

    /**
     * @brief Get the dds::pub::qos::PublisherQos with a specified name
     *      associated with this QosProvider.
     * @param[out] qos  PublisherQos to get.
     * @param[in] name  Profile name.
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::UNSUPPORTED
     * @req{AR-iAOS-RCS-DDS-01205,
     * QosProvider shall support getting PublisherQos.,
     * QM,
     * DR-iAOS-RCS-DDS-00032
     * }
     */
    dds::core::ReturnCode GetPublisherQos(
        dds::pub::qos::PublisherQos &qos,
        const std::string &name) const;

    /**
     * @brief Get the dds::sub::qos::SubscriberQos with a specified name
     *      associated with this QosProvider.
     * @param[out] qos  SubscriberQos to get.
     * @param[in] name  Profile name.
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::UNSUPPORTED
     * @req{AR-iAOS-RCS-DDS-01206,
     * QosProvider shall support getting SubscriberQos.,
     * QM,
     * DR-iAOS-RCS-DDS-00033
     * }
     */
    dds::core::ReturnCode GetSubscriberQos(
        dds::sub::qos::SubscriberQos &qos,
        const std::string &name) const;

    /**
     * @brief Get the dds::pub::qos::DataWriterQos with a specified name
     *      associated with this QosProvider.
     * @param[out] qos  DataWriterQos to get.
     * @param[in] name  Profile name.
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::UNSUPPORTED
     * @req{AR-iAOS-RCS-DDS-01207,
     * QosProvider shall support getting DataWriterQos.,
     * QM,
     * DR-iAOS-RCS-DDS-00029
     * DR-iAOS-RCS-DDS-00072
     * DR-iAOS-RCS-DDS-00082
     * }
     */
    dds::core::ReturnCode GetDataWriterQos(
        dds::pub::qos::DataWriterQos &qos,
        const std::string &name) const;

    /**
     * @brief Get the dds::sub::qos::DataReaderQos with a specified name
     *      associated with this QosProvider.
     * @param[out] qos  DataReaderQos to get.
     * @param[in] name  Profile name.
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::UNSUPPORTED
     * @req{AR-iAOS-RCS-DDS-01208,
     * QosProvider shall support getting DataReaderQos.,
     * QM,
     * DR-iAOS-RCS-DDS-00030
     * DR-iAOS-RCS-DDS-00082
     * }
     */
    dds::core::ReturnCode GetDataReaderQos(
        dds::sub::qos::DataReaderQos &qos,
        const std::string &name) const;
};
}
}

#endif /* DDS_CORE_QOS_PROVIDER_HPP */

