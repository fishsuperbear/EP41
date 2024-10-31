/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: TransportQos.hpp
 */

#ifndef DDS_CORE_POLICY_TRANSPORT_QOS_HPP
#define DDS_CORE_POLICY_TRANSPORT_QOS_HPP

#include <bitset>

namespace dds {
namespace core {
namespace policy {
/* 16 bits of BitSet */
using TransportType = std::bitset<static_cast<size_t>(16)>;

class Transport : public TransportType {
public:
    Transport() noexcept : TransportType(UDP_TRANSPORT)
    {}

    explicit Transport(const TransportType& baseType) noexcept: TransportType(baseType)
    {}

    Transport& operator=(const Transport& rhs) & = default;
    Transport& operator=(Transport&& rhs) & noexcept = default;
    Transport(const Transport& rhs) = default;
    Transport(Transport&& rhs) = default;

    ~Transport() = default;

    using TransportType::TransportType;

    static Transport UDP() noexcept
    {
        return Transport(UDP_TRANSPORT);
    }

    static Transport ICC() noexcept
    {
        return Transport(ICC_TRANSPORT);
    }

    static Transport DataPlain() noexcept
    {
        return Transport(DATA_PLAIN_TRANSPORT);
    }

    Transport& operator=(const TransportType& another) noexcept
    {
        TransportType::operator=(another);
        return *this;
    }

    Transport operator|(const Transport& another) const noexcept
    {
        TransportType myBits = *this;
        Transport res{myBits | another};
        return res;
    }

    Transport& operator|=(const Transport& another) noexcept
    {
        this->TransportType::operator|=(another);
        return *this;
    }

private:
    static const uint16_t UDP_TRANSPORT{0x01U};              /* Set bit 0 */
    static const uint16_t ICC_TRANSPORT{static_cast<uint16_t>(static_cast<uint16_t>(0x01U) << 1U)};         /* Set bit 1 */
    static const uint16_t DATA_PLAIN_TRANSPORT{static_cast<uint16_t>(static_cast<uint16_t>(0x01U) << 2U)};  /* Set bit 2 */
};
}
}
}

#endif /* DDS_CORE_POLICY_TRANSPORT_QOS_HPP */

