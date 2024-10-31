/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Subscriber.hpp
 */

#ifndef DDS_SUB_SUBSCRIBER_HPP
#define DDS_SUB_SUBSCRIBER_HPP

#include <memory>

#include <RT-DDS/dds/core/Entity.hpp>
#include <RT-DDS/dds/core/status/StatusMask.hpp>
#include <RT-DDS/dds/sub/qos/SubscriberQos.hpp>

namespace dds {
namespace domain {
class DomainParticipant;
}
}

namespace dds {
namespace sub {
class SubscriberImpl;

class Subscriber : public dds::core::Entity {
public:
    explicit Subscriber(
        dds::domain::DomainParticipant participant,
        dds::sub::qos::SubscriberQos qos = dds::sub::qos::SubscriberQos()) noexcept;

    ~Subscriber(void) override = default;

private:
    std::shared_ptr<SubscriberImpl> impl_;
    friend class AnyDataReader;
};
}
}

#endif /* DDS_SUB_SUBSCRIBER_HPP */

