/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Reliability.hpp
 */

#ifndef DDS_CORE_POLICY_RELIABILITY_HPP
#define DDS_CORE_POLICY_RELIABILITY_HPP

#include <RT-DDS/dds/core/Duration.hpp>
#include <RT-DDS/dds/core/policy/ReliabilityKind.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Indicates the level of reliability in sample delivered that a
 * dds::pub::DataWriter offers or a dds::sub::DataReader requests.
 */
class Reliability {
public:
    Reliability()
        : Reliability(dds::core::policy::ReliabilityKind::BEST_EFFORT)
    {}

    /**
     * @brief Creates an instance with the specified reliability kind an
     * optionally a specific maximum blocking time.
     * The max blocking time only applies to reliable DataWriters.
     */
    explicit Reliability(
        dds::core::policy::ReliabilityKind kind,
        dds::core::Duration maxBlockingTime = dds::core::Duration::FromMilliSecs(DEFAULT_BLOCKING_TIME))
        : kind_(kind),
          maxBlockingTime_(maxBlockingTime)
    {}

    ~Reliability() = default;

    /**
     *
     * @param maxBlockingTime
     * @return
     */
    static Reliability Reliable(
        dds::core::Duration maxBlockingTime = dds::core::Duration::FromMilliSecs(DEFAULT_BLOCKING_TIME))
    {
        return Reliability(dds::core::policy::ReliabilityKind::RELIABLE, maxBlockingTime);
    }

    static Reliability BestEffort()
    {
        return Reliability(dds::core::policy::ReliabilityKind::BEST_EFFORT);
    }

    /**
     * @brief Sets the reliability kind
     * @details \dref_details_ReliabilityQosPolicy_kind
     */
    void Kind(dds::core::policy::ReliabilityKind kind) noexcept
    {
        kind_ = kind;
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    dds::core::policy::ReliabilityKind Kind() const noexcept
    {
        return kind_;
    }

    /**
     * @brief Sets the maximum time a DataWriter may block on a call to write().
     *
     * @details \dref_details_ReliabilityQosPolicy_max_blocking_time
     */
    void MaxBlockingTime(dds::core::Duration maxBlockingTime) noexcept
    {
        maxBlockingTime_ = maxBlockingTime;
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    dds::core::Duration MaxBlockingTime() const noexcept
    {
        return maxBlockingTime_;
    }

private:
    ReliabilityKind kind_;
    Duration maxBlockingTime_{dds::core::Duration::FromMilliSecs(DEFAULT_BLOCKING_TIME)};
    static const uint32_t DEFAULT_BLOCKING_TIME{100U};
};
}
}
}

#endif /* DDS_CORE_POLICY_RELIABILITY_HPP */

