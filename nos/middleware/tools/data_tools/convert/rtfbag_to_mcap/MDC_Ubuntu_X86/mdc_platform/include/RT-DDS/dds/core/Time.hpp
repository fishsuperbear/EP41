/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_CORE_TIME_HPP
#define SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_CORE_TIME_HPP

#include <cstdint>
#include <ostream>

namespace dds {
namespace core {
class Time {
public:
    Time() = default;

    explicit Time(int32_t sec, uint32_t nanoSec = 0U) noexcept;

    ~Time() = default;

    Time(const Time& rhs) = default;
    Time& operator=(const Time& rhs) = default;
    Time(Time&& rhs) noexcept = default;
    Time& operator=(Time&& rhs) noexcept = default;

    int32_t GetSec() const noexcept;

    void SetSec(int32_t sec) noexcept;

    uint32_t GetNanoSec() const noexcept;

    void SetNanoSec(uint32_t nanoSec) noexcept;

    /**
     * @brief Set nano by fraction, a fraction is 1 / 2^32 sec
     */
    void SetNanoSecByFraction(uint32_t fraction) noexcept;

    uint32_t GetNanoSecByFraction() const noexcept;

    static Time Invalid() noexcept;

    bool operator==(const Time& rhs) const noexcept;

    bool operator!=(const Time& rhs) const noexcept;

    bool operator<(const Time& rhs) const noexcept;

    bool operator>(const Time& rhs) const noexcept;

    bool operator<=(const Time& rhs) const noexcept;

    bool operator>=(const Time& rhs) const noexcept;

    friend std::ostream& operator<<(std::ostream& os, const Time& time1);

    std::string ToString() const;

    friend struct std::hash<dds::core::Time>;
private:
    int32_t sec_{};
    uint32_t nanoSec_{};
};
}
}

namespace std {
template<>
struct hash<dds::core::Time> {
    std::size_t operator()(dds::core::Time const& time) const noexcept;
};
}


#endif // SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_CORE_TIME_HPP
