/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: PublisherQoS.hpp
 */

#ifndef DDS_PUB_QOS_PUBLISHER_QOS_HPP
#define DDS_PUB_QOS_PUBLISHER_QOS_HPP

#include <RT-DDS/dds/core/policy/Partition.hpp>

namespace dds {
namespace pub {
namespace qos {
/**
 * @brief Container of the QoS policies that a dds::pub::Publisher
 * supports.
 */
class PublisherQos {
public:
    /**
     * @brief Set Partition QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-04501,
     * PublisherQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::Partition partition)
    {
        partition_ = std::move(partition);
    }

    /**
     * @brief Gets Partition QoS policy by const reference.
     * @return dds::core::policy::Partition
     * @req{AR-iAOS-RCS-DDS-04502,
     * PublisherQos shall support Partition policy,
     * QM,
     * DR-iAOS-RCS-DDS-00001, DR-iAOS-RCS-DDS-00032, DR-iAOS-RCS-DDS-00074
     * }
     */
    const dds::core::policy::Partition &Partition() const noexcept
    {
        return partition_;
    }

private:
    dds::core::policy::Partition partition_{};
};
}
}
}

#endif /* DDS_PUB_QOS_PUBLISHER_QOS_HPP */

