/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Publisher.hpp
 */

#ifndef DDS_PUB_PUBLISHER_HPP
#define DDS_PUB_PUBLISHER_HPP

#include <RT-DDS/dds/core/Entity.hpp>
#include <RT-DDS/dds/pub/qos/PublisherQos.hpp>

#include <memory>

namespace dds {
namespace domain {
class DomainParticipant;
}
}

namespace dds {
namespace pub {
class PublisherImpl;

class Publisher : public dds::core::Entity {
public:
    /**
     * @brief Create a new Publisher.
     * @param[in] participant
     * @param[in] qos
     * @req{AR-iAOS-RCS-DDS-04401,
     * Publisher shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00017
     * }
     */
    explicit Publisher(
        dds::domain::DomainParticipant &participant,
        dds::pub::qos::PublisherQos qos = dds::pub::qos::PublisherQos()) noexcept;

    /**
     * @brief Default Destructor.
     * @req{AR-iAOS-RCS-DDS-04402,
     * Publisher shall support destruction process.,
     * QM,
     * DR-iAOS-RCS-DDS-00009
     * }
     */
    ~Publisher() override = default;

private:
    std::shared_ptr<PublisherImpl> impl_;
    friend class AnyDataWriter;
};
}
}

#endif /* DDS_PUB_PUBLISHER_HPP */

