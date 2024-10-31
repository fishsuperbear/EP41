/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DomainParticipant.hpp
 */

#ifndef DDS_DOMAIN_DOMAIN_PARTICIPANT_HPP
#define DDS_DOMAIN_DOMAIN_PARTICIPANT_HPP

#include <RT-DDS/dds/core/Entity.hpp>
#include <RT-DDS/dds/core/Types.hpp>
#include <RT-DDS/dds/core/ReturnCode.hpp>
#include <RT-DDS/dds/core/status/StatusMask.hpp>
#include <RT-DDS/dds/domain/qos/DomainParticipantQos.hpp>
#include <RT-DDS/dds/topic/ParticipantBuiltinTopicData.hpp>

namespace dds {
namespace topic {
class AnyTopic;
}
}

namespace dds {
namespace pub {
class Publisher;
class AnyDataWriter;
}
}

namespace dds {
namespace sub {
class Subscriber;
class AnyDataReader;
template<class T>
class DataReader;
}
}

namespace dds {
namespace domain {
class DomainParticipantImpl;

class DomainParticipant final : public dds::core::Entity {
public:
    /**
     * @brief Create a new DomainParticipant.
     * @param domainId[in] The id of the domain joined by this DomainParticipant.
     * @param qos[in] The QoS settings for this DomainParticipant.
     * @details Create a new DomainParticipant. If DomainParticipantQos is not transferred,
     * the default value is used.
     * @req{AR-iAOS-RCS-DDS-02101,
     * DomainParticipant shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00009
     * }
     */
    explicit DomainParticipant(
        int32_t domainId,
        const dds::domain::qos::DomainParticipantQos &qos = dds::domain::qos::DomainParticipantQos()) noexcept;

    /**
     * @brief Default Destructor.
     * @req{AR-iAOS-RCS-DDS-02102,
     * DomainParticipant shall support destruction process.,
     * QM,
     * DR-iAOS-RCS-DDS-00009
     * }
     */
    ~DomainParticipant() override = default;

    /**
     * @brief Set the DomainParticipantQos setting for this DomainParticipant
     * instance.
     * @param[in] qos Set of policies to be applied to the DomainParticipant.
     * Immutable policies cannot be changed after this entity has been enabled.
     * @note for now, the only changeable qos after Enable is UserDataQos
     * @return dds::core::ReturnCode
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::BAD_PARAMETER
     * @retval dds::core::ReturnCode::NOT_ENABLED
     * @retval dds::core::ReturnCode::ALREADY_DELETED
     * @req{AR-iAOS-RCS-DDS-02103,
     * DomainParticipant shall support setting DomainParticipantQos.,
     * QM,
     * DR-iAOS-RCS-DDS-00008
     * }
     */
    dds::core::ReturnCode SetQos(
        dds::domain::qos::DomainParticipantQos qos);

    /**
     * @brief Gets the current QoS policies of this DomainParticipant.
     * @return The current QoS policies.
     * @req{AR-iAOS-RCS-DDS-02104,
     * DomainParticipant shall support getting DomainParticipantQos.,
     * QM,
     * DR-iAOS-RCS-DDS-00008
     * }
     */
    dds::domain::qos::DomainParticipantQos GetQos() const;

    /**
     * @brief find a topic related to this participant, by topic name
     * @param topicName the name of the topic you want to find
     * @return the topic ptr, which could be null if not found
     */
    std::shared_ptr<topic::AnyTopic> FindTopic(std::string const& topicName) const;

    static std::string GetGlobalStatString(void);

    /**
     * @brief trigger participant to send extra SPDP
     * @param period send SPDP in period
     * @return dds::core::ReturnCode return code
     */
    dds::core::ReturnCode Announce(uint16_t period);

private:
    std::shared_ptr<DomainParticipantImpl> impl_;
    friend class dds::topic::AnyTopic;
    friend class dds::pub::Publisher;
    friend class dds::pub::AnyDataWriter;
    friend class dds::sub::Subscriber;
    friend class dds::sub::AnyDataReader;
    friend class dds::sub::DataReader<dds::topic::ParticipantBuiltinTopicData>;
};
}
}

#endif /* DDS_DOMAIN_DOMAIN_PARTICIPANT_HPP */

