/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Duration.hpp
 */

#ifndef DDS_CORE_DURATION_HPP
#define DDS_CORE_DURATION_HPP

#include <chrono>
#include <sstream>
#include <iomanip>

namespace dds {
namespace core {
/**
 * @brief Represents a time interval
 */
class Duration {
public:
    /**
     * @brief Create a duration elapsing a specific amount of time.
     * @param[in] sec The number of seconds to represent.
     * @param[in] nanoSec The number of nanoseconds to represent.
     */
    explicit Duration(uint32_t sec = 0U, uint32_t nanoSec = 0U) noexcept
        : Duration(DurationType{std::chrono::duration<uint32_t>{sec}} +
        DurationType{std::chrono::duration<uint32_t, std::nano>{std::min(uint32_t{999'999'999U}, nanoSec)}})
    {}

    ~Duration() = default;

    /**
     * @brief Create a Duration elapsing a specific number of seconds.
     * @param[in] sec The number of seconds to construct the object from.
     * @return A newly constructed Duration object.
     */
    static Duration FromSecs(uint32_t sec) noexcept
    {
        return Duration(DurationType{std::chrono::duration<uint32_t>{sec}});
    }

    /**
     * @brief Create a Duration elapsing a specific number of milliseconds.
     * @param[in] milliSecs The number of milliseconds to construct the object from.
     * @return A newly constructed Duration object.
     */
    static Duration FromMilliSecs(uint32_t milliSecs) noexcept
    {
        return Duration(DurationType{std::chrono::duration<uint32_t, std::milli>{milliSecs}});
    }

    /**
     * @brief Create a Duration elapsing a specific number of microseconds.
     * @param[in] microSecs The number of microseconds to construct the object from.
     * @return A newly constructed Duration object.
     */
    static Duration FromMicroSecs(uint32_t microSecs) noexcept
    {
        return Duration(DurationType{std::chrono::duration<uint32_t, std::micro>{microSecs}});
    }

    /**
     * @brief Returns a zero duration.
     * @return A zero duration.
     */
    static Duration Zero() noexcept
    {
        return Duration(DurationType{0});
    }

    /**
     * @brief Special value that represents an infinite Duration.
     * @return An infinite Duration.
     */
    static Duration Infinite() noexcept
    {
        return Duration(DurationType{UINT64_MAX});
    }

    /**
     * @brief Get the number of seconds represented by this Duration object.
     * @return The number of seconds (excluding the nanoseconds).
     */
    uint32_t Sec() const noexcept
    {
        if (value_.count() == INFINITE_VALUE) {
            return INFINITE_SEC;
        }
        return static_cast<uint32_t>(value_.count() / NANO_SEC);
    }

    /**
     * @brief Get the number of nanoseconds represented by this Duration object.
     * @return The number of nanoseconds (excluding the seconds).
     */
    uint32_t NanoSec() const noexcept
    {
        if (value_.count() == INFINITE_VALUE) {
            return INFINITE_NANO_SEC;
        }
        return static_cast<uint32_t>(value_.count() - static_cast<uint64_t>(Sec() * NANO_SEC));
    }

    /**
     * @brief Special value that indicates that records the values of second and
     * nanosecond.
     * @return The special value.
     */
    uint64_t Value() const noexcept
    {
        return value_.count();
    }

    /**
     * @brief Check if this Duration is equal to another.
     * @param[in] rhs The Duration to compare with this Duration.
     * @return
     * * true - if this Duration is equal to the other object.
     * * false - otherwise.
     */
    bool operator==(Duration rhs) const noexcept
    {
        return value_ == rhs.value_;
    }

    /**
     * @brief Check if this Duration is greater than another.
     * @param[in] rhs The Duration to compare with this Duration.
     * @return
     * * true - if this Duration is greater than the other object.
     * * false - otherwise.
     */
    bool operator>(Duration rhs) const noexcept
    {
        return value_.count() > rhs.value_.count();
    }

    /**
     * @brief Check if this Duration is smaller than another.
     * @param[in] rhs The Duration to compare with this Duration.
     * @return
     * * true - if this Duration is smaller than the other object.
     * * false - otherwise.
     */
    bool operator<(Duration rhs) const noexcept
    {
        return value_.count() < rhs.value_.count();
    }

    friend std::ostream& operator<<(std::ostream& os, const dds::core::Duration &duration1);

    std::string ToString() const;

private:
    using DurationType = std::chrono::duration<uint64_t, std::nano>;

    /**
      * @brief Create a duration elapsing a specific value.
      * @param[in] value The specific value.
      */
    explicit Duration(DurationType value) noexcept
        : value_(value)
    {}

    DurationType value_;
    static const uint32_t NANO_SEC{1000000000U};
    static const uint64_t INFINITE_VALUE{UINT64_MAX};
    static const uint32_t INFINITE_SEC{0x7FFFFFFFU};
    static const uint32_t INFINITE_NANO_SEC{999999999U};
};
}
}

#endif /* DDS_CORE_DURATION_HPP */
