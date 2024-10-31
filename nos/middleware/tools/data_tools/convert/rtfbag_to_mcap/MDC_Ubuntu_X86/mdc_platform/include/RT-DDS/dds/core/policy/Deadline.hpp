/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Deadline.hpp
 */

#ifndef DDS_CORE_POLICY_DEADLINE_HPP
#define DDS_CORE_POLICY_DEADLINE_HPP

#include <RT-DDS/dds/core/Duration.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Expresses the maximum duration (deadline) within which an instance is
 * expected to be updated.
 *
 * Expresses the maximum duration (deadline) within which an instance is
 * expected to be updated.
 */
class Deadline {
public:
    /**
     * @brief Creates the default deadline, with an infinite period.
     */
    Deadline() noexcept
    {}

    /**
     * @brief Creates a deadline policy with the specified period.
     */
    explicit Deadline(dds::core::Duration d) noexcept
        : period_(d)
    {}

    ~Deadline() = default;

    /**
     * @brief Sets the duration of the deadline period.
     *
     * <b>[default]</b> dds::core::Duration::infinite()
     */
    void Period(dds::core::Duration d) noexcept
    {
        period_ = d;
    }

    /**
     * @brief Getter the duration of the deadline period.
     */
    const dds::core::Duration &Period() const noexcept
    {
        return period_;
    }

private:
    dds::core::Duration period_{dds::core::Duration::Infinite()};
};
}
}
}

#endif /* DDS_CORE_POLICY_DEADLINE_HPP */

