/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_CORE_INSTANCEHANDLE_HPP
#define SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_CORE_INSTANCEHANDLE_HPP

#include <cstdint>
#include <array>
#include <ostream>

#include <RT-DDS/dds/core/Guid.hpp>

namespace dds {
namespace core {
/**
 * @brief A class that used to represent entities in the DDS, for now it is only used to represent the instance
 * @details InstanceHandle support stream to string, value compare and std::hash, the value is unchangeable
 * @req{AR-iAOS-RCS-DDS-01003,
 * Support basic operations of InstanceHandle,
 * QM,
 * DR-iAOS3-RCS-DDS-00153
 * }
 */
class InstanceHandle {
public:
    InstanceHandle() = default;
    InstanceHandle(const InstanceHandle& rhs) = default;
    InstanceHandle(InstanceHandle&& rhs) noexcept = default;
    InstanceHandle& operator=(const InstanceHandle& rhs) = default;
    InstanceHandle& operator=(InstanceHandle&& rhs) noexcept = default;
    ~InstanceHandle() = default;

    /** instance handle is a 16byte value */
    using ValueType = std::array<uint8_t, 16U>;

    InstanceHandle(const uint8_t *rawData, uint64_t len);

    explicit InstanceHandle(const ValueType& value) noexcept;

    explicit InstanceHandle(const Guid& guid);

    const ValueType& GetValue() const noexcept;

    static InstanceHandle Nil() noexcept;

    bool operator<(const InstanceHandle& that) const noexcept;

    bool operator>(const InstanceHandle& that) const noexcept;

    bool operator<=(const InstanceHandle& that) const noexcept;

    bool operator>=(const InstanceHandle& that) const noexcept;

    bool operator==(const InstanceHandle& that) const noexcept;

    bool operator!=(const InstanceHandle& that) const noexcept;

    friend std::ostream& operator<<(std::ostream& os, const InstanceHandle& handle);

    std::string ToString() const;

    Guid ToRtpsGuid() const;

    bool IsNil() const noexcept;

    friend struct std::hash<dds::core::InstanceHandle>;
private:
    ValueType value_{};
    bool isValid_{false};
};
}
}

namespace std {
template<>
struct hash<dds::core::InstanceHandle> {
    std::size_t operator()(dds::core::InstanceHandle const& handle) const noexcept;
};
}

#endif // SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_CORE_INSTANCEHANDLE_HPP
