/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Octets.hpp
 */

#ifndef DDS_CORE_OCTETS_HPP
#define DDS_CORE_OCTETS_HPP

#include <cstdint>
#include <cstring>

namespace dds {
namespace core {
class Octets {
public:
    Octets() noexcept: Octets(0U)
    {}

    explicit Octets(uint32_t size) noexcept
        : size_(size), buffer_(nullptr)
    {}

    ~Octets() = default;

    void Size(uint32_t s) noexcept
    {
        size_ = s;
    }

    uint32_t Size() const noexcept
    {
        return size_;
    }

    void Buffer(uint8_t *buf) const noexcept
    {
        buffer_ = buf;
    }

    uint8_t *Buffer() const noexcept
    {
        return buffer_;
    }

    bool operator==(const Octets &rhs) const noexcept
    {
        if (size_ != rhs.size_) {
            return false;
        }
        return memcmp(buffer_, rhs.buffer_, static_cast<std::size_t>(size_)) == 0;
    }

private:
    uint32_t size_{0U};
    mutable uint8_t *buffer_{nullptr};
};
}
}

#endif /* DDS_CORE_OCTETS_HPP */

