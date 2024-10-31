/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: TransportPriority.hpp
 */

#ifndef DDS_CORE_POLICY_TRANSPORT_PRIORITY_HPP
#define DDS_CORE_POLICY_TRANSPORT_PRIORITY_HPP

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Allows applications to take advantage of transports capable of sending
 * messages with different priorities.
 */
class TransportPriority {
public:
    /**
     * @brief Creates a policy with value 0.
     */
    TransportPriority() = default;

    /**
     * @brief Creates a policy with the specified value.
     */
    explicit TransportPriority(uint8_t v) noexcept
        : value_(v)
    {}

    ~TransportPriority() = default;

    /**
     * @brief Sets the value.
     */
    void Value(uint8_t v) noexcept
    {
        value_ = v;
    }
    /**
     * @brief Getter (see setter with the same name)
     */
    uint8_t Value() const noexcept
    {
        return value_;
    }

private:
    uint8_t value_{0U};
};
}
}
}

#endif /* DDS_CORE_POLICY_TRANSPORT_PRIORITY_HPP */

