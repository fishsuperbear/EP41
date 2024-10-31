/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: SizeCounterCDR.hpp
 */

#ifndef DDS_CDR_SIZE_COUNTER_CDR_HPP
#define DDS_CDR_SIZE_COUNTER_CDR_HPP

#include <cstdint>
#include <vector>
#include <array>
#include <RT-DDS/dds/core/Octets.hpp>

namespace dds {
namespace cdr {
class SerializePayload {
public:
    uint32_t size;
    uint8_t *buffer;
};

inline std::size_t Alignment(std::size_t index, std::size_t bytes) noexcept
{
    std::size_t res = (((index + bytes) - 1U) & ~(bytes - 1U));
    if (res >= index) {
        return res;
    } else {
        return index;
    }
}

/**
 * @brief Size Counter for CDR.
 */
class SizeCounterCDR {
public:
    /**
     * @brief Size calculation of std::is_integral types.
     * @tparam T Type that matches std::is_integral traits.
     * @param value     data to be calculated.
     * @req{AR-iAOS-RCS-DDS-06401,
     * SizeCounterCDR shall support size calculation of integer types.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    void operator()(const T &value) noexcept
    {
        static_cast<void>(value);
        size_ = Alignment(size_, sizeof(T));
        AddSize(sizeof(T));
    }

    /**
     * @brief Size calculation of std::string.
     * @param[in] value     data to be calculated.
     * @req{AR-iAOS-RCS-DDS-06402,
     * SizeCounterCDR shall support size calculation of std::string.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    void operator()(const std::string &value) noexcept
    {
        operator()(static_cast<uint32_t>(sizeof(uint32_t)));
        AddSize(value.length() + 1U);
    }

    /**
     * @brief Size calculation of std::vector<T> types.
     * @param[in] value     data to be calculated.
     * @req{AR-iAOS-RCS-DDS-06403,
     * SizeCounterCDR shall support size calculation of std::vector<T> types.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    void operator()(const std::vector<T> &value) noexcept
    {
        operator()(static_cast<uint32_t>(sizeof(uint32_t)));
        for (const T &v : value) {
            operator()(v);
        }
    }

    /**
     * @brief Size calculation of std::array<T> types.
     * @param[in] value     data to be calculated.
     * @req{AR-iAOS-RCS-DDS-06404,
     * SizeCounterCDR shall support size calculation of of std::array<T> types.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    template<typename T, std::size_t size, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    void operator()(const std::array<T, size> &value) noexcept
    {
        for (const T &v : value) {
            operator()(v);
        }
    }

    /**
     * @brief Size calculation of dds::core::Octets.
     * @param[in] value     data to be calculated.
     * @req{AR-iAOS-RCS-DDS-06405,
     * SizeCounterCDR shall support size calculation of dds::core::Octets.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    void operator()(dds::core::Octets value) noexcept
    {
        operator()(value.Size());
        AddSize(static_cast<std::size_t>(value.Size()));
    }

    /**
     * @brief This function aligns the buffer to the end with 4 bytes.
     * @req{AR-iAOS-RCS-DDS-06406,
     * SizeCounterCDR shall support alignment of buffer with 4 bytes.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    void AlignmentEnd() noexcept
    {
        size_ = Alignment(size_, sizeof(uint32_t));
    }

    /**
     * @brief Get the size of the sample.
     * @return the size of the sample.
     * @req{AR-iAOS-RCS-DDS-06407,
     * SizeCounterCDR shall support getting size of the sample.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    std::size_t GetSize() const noexcept
    {
        return size_;
    }

private:
    void AddSize(std::size_t inc) noexcept
    {
        if ((SIZE_MAX - size_) < inc) {
            size_ = SIZE_MAX;
        } else {
            size_ += inc;
        }
    }

    std::size_t size_{sizeof(uint32_t)};
};
}
}

#endif /* DDS_CDR_SIZE_COUNTER_CDR_HPP */

