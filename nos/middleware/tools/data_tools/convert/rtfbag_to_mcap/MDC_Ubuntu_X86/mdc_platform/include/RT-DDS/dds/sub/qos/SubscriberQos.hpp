/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: SubscriberQoS.hpp
 */

#ifndef DDS_SUB_QOS_SUBSCRIBER_QOS
#define DDS_SUB_QOS_SUBSCRIBER_QOS

#include <RT-DDS/dds/core/policy/Partition.hpp>

namespace dds {
namespace sub {
namespace qos {
/**
 * @brief Container of the QoS policies that a dds::sub::Subscriber supports.
 */
class SubscriberQos {
public:
    /**
     * @brief Set Partition QoS policy.
     * @param[in] policy Qos policy to set.
     * @req{AR-iAOS-RCS-DDS-05201,
     * SubscriberQos shall support setting of policies,
     * QM,
     * DR-iAOS-RCS-DDS-00006
     * }
     */
    void Set(dds::core::policy::Partition partition)
    {
        partition_ = std::move(partition);
    }

    /**
     * @brief Gets Partition QoS policy.
     * @return dds::core::policy::Partition
     * @req{AR-iAOS-RCS-DDS-05202,
     * SubscriberQos shall support Partition policy,
     * QM,
     * DR-iAOS-RCS-DDS-00001, DR-iAOS-RCS-DDS-00033
     * }
     */
    dds::core::policy::Partition Partition() noexcept
    {
        return partition_;
    }

private:
    dds::core::policy::Partition partition_{};
};
}
}
}

#endif /* DDS_SUB_QOS_SUBSCRIBER_QOS */

