/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: RtpsReliableReaderProtocol.hpp
 */

#ifndef DDS_CORE_POLICY_TIME_BASED_FILTER_HPP
#define DDS_CORE_POLICY_TIME_BASED_FILTER_HPP

#include <RT-DDS/dds/core/Duration.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Configures aspects of the RTPS protocol related to a reliable
 * DataReader.
 *
 * It is used to config reliable reader according to RTPS protocol.
 */
class TimeBasedFilter {
public:
    /**
     * @brief Creates the default TimeBasedFilter, with an zero separation.
     */
    TimeBasedFilter() = default;

    ~TimeBasedFilter() = default;

    /**
     * @brief The minimum separation for a reader.
     *
     * When a reader wants to receive only one packet at a given minimum separation.
     */
    void MinimumSeparation(const dds::core::Duration& d) noexcept
    {
        minimumSeparation_ = d;
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    const dds::core::Duration &MinimumSeparation() const noexcept
    {
        return minimumSeparation_;
    }

private:
    dds::core::Duration minimumSeparation_{dds::core::Duration::Zero()};
};
}
}
}

#endif /* DDS_CORE_POLICY_TIME_BASED_FILTER_HPP */

