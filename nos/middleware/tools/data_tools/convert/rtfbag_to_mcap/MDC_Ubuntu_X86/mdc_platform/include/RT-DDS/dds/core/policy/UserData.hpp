/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: UserData.hpp
 */
#ifndef DDS_CORE_POLICY_USER_DATA_HPP
#define DDS_CORE_POLICY_USER_DATA_HPP

#include <RT-DDS/dds/core/Types.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Attaches a buffer of opaque data that is distributed by Built-in
 * Topics during discovery.
 */
class UserData {
public:
    /**
     * @brief Creates an instance with an empty sequence of bytes.
     */
    UserData() = default;

    /**
     * @brief Creates an instance with a sequence of bytes
     *
     * @param seq A vector containing the bytes to create this UserData
     */
    explicit UserData(dds::core::OctetSeq value) noexcept
        : value_(std::move(value))
    {}

    ~UserData() = default;

    /**
     * @brief Setter
     * @param value
     */
    void Value(dds::core::OctetSeq value) noexcept
    {
        value_ = std::move(value);
    }

    /**
     * @brief Gets the user data.
     * @return the sequence of bytes representing the user data
     */
    const dds::core::OctetSeq &Value() const noexcept
    {
        return value_;
    }

private:
    dds::core::OctetSeq value_{};
};
}
}
}

#endif /* DDS_CORE_POLICY_USER_DATA_HPP */

