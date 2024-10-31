/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Topic.hpp
 */

#ifndef DDS_TOPIC_ANY_TOPIC_HPP
#define DDS_TOPIC_ANY_TOPIC_HPP

#include <RT-DDS/dds/core/Entity.hpp>
#include <RT-DDS/dds/topic/qos/TopicQos.hpp>

namespace dds {
namespace domain {
class DomainParticipant;
}

namespace pub {
class AnyDataWriter;
}

namespace sub {
class AnyDataReader;
}

namespace topic {
class AnyTopicImpl;

class AnyTopic : public dds::core::Entity {
public:
    /**
     * @brief Get name of specific topic.
     * @return std::string name of specific topic.
     * @req{AR-iAOS-RCS-DDS-03103,
     * Topic<T> shall support getting TopicName,
     * QM,
     * DR-iAOS-RCS-DDS-00015
     * }
     */
    std::string Name(void) const;

    ~AnyTopic(void) override = default;

    const std::string& TypeID(void) const noexcept;

protected:
    explicit AnyTopic(
        const dds::domain::DomainParticipant &participant,
        const std::string &topicName,
        const std::string &typeName,
        const dds::topic::qos::TopicQos &qos) noexcept;

    explicit AnyTopic(std::shared_ptr<AnyTopicImpl> impl) noexcept;

private:
    std::shared_ptr<AnyTopicImpl> impl_;
    friend class dds::pub::AnyDataWriter;
    friend class dds::sub::AnyDataReader;
    friend class dds::domain::DomainParticipant;
};
}
}

#endif /* DDS_TOPIC_ANY_TOPIC_HPP */

