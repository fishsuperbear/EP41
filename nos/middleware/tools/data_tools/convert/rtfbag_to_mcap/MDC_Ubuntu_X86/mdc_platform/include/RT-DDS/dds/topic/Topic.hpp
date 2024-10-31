/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Topic.hpp
 */

#ifndef DDS_TOPIC_TOPIC_HPP
#define DDS_TOPIC_TOPIC_HPP

#include <RT-DDS/dds/topic/AnyTopic.hpp>

namespace dds {
namespace topic {
template<typename T>
class Topic : public AnyTopic {
public:
    /**
     * @brief Constructor of topic.
     * @param[in] participant Participant that the topic will be attached to
     * @param[in] topicName   name of the topic
     * @req{AR-iAOS-RCS-DDS-03101,
     * Topic<T> shall support creation process,
     * QM,
     * DR-iAOS-RCS-DDS-00015
     * }
     */
    Topic(const dds::domain::DomainParticipant &participant, const std::string &topicName,
          const dds::topic::qos::TopicQos &qos = dds::topic::qos::TopicQos{}) noexcept
        : AnyTopic(participant, topicName, Topic<T>::TypeName(), qos)
    {}

    /**
     * @brief Destructor of topic.
     * @req{AR-iAOS-RCS-DDS-03102,
     * Topic<T> shall support destruction process,
     * QM,
     * DR-iAOS-RCS-DDS-00015
     * }
     */
    ~Topic() override = default;

    /**
    * @brief Get dataType of specific topic.
    * @return std::string Type name of specific topic.
    * @req{AR-iAOS-RCS-DDS-03104,
    * Topic<T> shall support getting TypeName,
    * QM,
    * DR-iAOS-RCS-DDS-00015
    * }
    */
    static std::string TypeName(void) noexcept
    {
        return T::TypeName();
    }

    static std::shared_ptr<Topic<T>> CastFromAnyTopic(const AnyTopic& anotherTopic)
    {
        struct MakeShared final : public Topic<T> {
            explicit MakeShared(const AnyTopic& another) noexcept : Topic<T>(another) {}
            ~MakeShared() final = default;
        };
        if (anotherTopic.TypeID() == Topic<T>::TypeName()) {
            return std::make_shared<MakeShared>(anotherTopic);
        }
        return nullptr;
    }

protected:
    explicit Topic(const AnyTopic& another) noexcept
        : AnyTopic(another)
    {}
};
}
}

#endif /* DDS_TOPIC_TOPIC_HPP */

