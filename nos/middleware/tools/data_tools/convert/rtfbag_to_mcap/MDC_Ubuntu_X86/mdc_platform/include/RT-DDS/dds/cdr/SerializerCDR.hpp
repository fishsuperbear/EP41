/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: SerializerCDR.hpp
 */

#ifndef DDS_CDR_SERIALIZER_CDR_HPP
#define DDS_CDR_SERIALIZER_CDR_HPP

#include <cstdint>
#include <vector>
#include <securec.h>
#include <RT-DDS/dds/core/Octets.hpp>
#include <RT-DDS/dds/cdr/SizeCounterCDR.hpp>

namespace dds {
namespace cdr {
/**
 * @brief Serializer for CDR.
 */
class SerializerCDR {
public:
    /**
     * @brief Constructor for SerializerCDR.
     * @param[in] size      Size of the buffer to serialize.
     * @param[in] buffer    Buffer to serialize.
     */
    SerializerCDR(const std::size_t size, uint8_t *buffer) noexcept
        : size_(size), buffer_(buffer)
    {}

    /**
     * @brief Default destructor.
     */
    ~SerializerCDR() = default;

    /**
     * @brief Serialization of Identifier.
     * @details All data encapsulation schemes must start with an encapsulation
     * scheme identifier. The identifier occupies the first two octets of the
     * serialized data-stream. The remaining part of the serialized data stream
     * either contains the actual data.
     * @return boolean
     * @retval true     Succeed to serialize.
     * @retval false    Fail to serialize.
     * @req{AR-iAOS-RCS-DDS-06201,
     * SerializerCDR shall support serialization of Identifier Header.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    bool IdentifierHeader() noexcept
    {
        bool ret{operator()(static_cast<uint8_t>(0))};
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        static const uint8_t cdrBe{0x0};
        ret = (operator()(cdrBe)) && ret;
#else
        static const uint8_t cdrLe{0x1U};
        ret = (operator()(cdrLe)) && ret;
#endif
        ret = (operator()(static_cast<uint16_t>(0))) && ret;

        return ret;
    }

    /**
     * @brief Serialization method of std::is_integral types.
     * @tparam T Type that matches std::is_integral traits.
     * @param[in] value     data to be serialized.
     * @return boolean
     * @retval true     Succeed to serialize.
     * @retval false    Fail to serialize.
     * @req{AR-iAOS-RCS-DDS-06202,
     * SerializerCDR shall support serialization of integer types.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator()(const T &value) noexcept
    {
        const std::size_t s{sizeof(T)};
        bool ret{false};

        pos_ = Alignment(pos_, s);
        if ((pos_ + s) <= size_) {
            /* AXIVION Next Line AutosarC++19_03-M5.0.15 : the special usage for serialization, the len is checked */
            *reinterpret_cast<T *>(&buffer_[pos_]) = value;
            pos_ += s;
            ret = true;
        }

        return ret;
    }

    /**
     * @brief Serialization method of std::string.
     * @param[in] value     data to be serialized.
     * @return boolean
     * @retval true     Succeed to serialize.
     * @retval false    Fail to serialize.
     * @req{AR-iAOS-RCS-DDS-06203,
     * SerializerCDR shall support serialization of std::string.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    bool operator()(const std::string &value) noexcept
    {
        bool ret{true};

        ret = (operator()(static_cast<uint32_t>(value.length() + 1U))) && ret;
        if (ret && !value.empty()) {
            ret = false;
            if ((pos_ < size_) &&
                (memcpy_s(&buffer_[pos_], size_ - pos_, value.data(), value.length()) == EOK)) {
                pos_ += value.length();
                ret = true;
            }
        }
        ret = (operator()(static_cast<int8_t>(0))) && ret;

        return ret;
    }

    /**
     * @brief Serialization method of std::vector<T> types.
     * @tparam T Type that matches std::is_integral traits.
     * @param[in] value     data to be serialized.
     * @return boolean
     * @retval true     Succeed to serialize.
     * @retval false    Fail to serialize.
     * @req{AR-iAOS-RCS-DDS-06204,
     * SerializerCDR shall support serialization of vector with integer types.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator()(const std::vector<T> &value) noexcept
    {
        bool ret{operator()(static_cast<uint32_t>(value.size()))};

        for (const T &v : value) {
            ret = (operator()(v)) && ret;
        }

        return ret;
    }

    template<typename T, std::size_t Nm, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator()(const std::array<T, Nm> &value) noexcept
    {
        bool ret{true};

        for (const T &v : value) {
            ret = (operator()(v)) && ret;
        }

        return ret;
    }

    /**
     * @brief Serialization method of dds::core::Octets.
     * @param[in] value     data to be serialized.
     * @return boolean
     * @retval true     Succeed to serialize.
     * @retval false    Fail to serialize.
     * @req{AR-iAOS-RCS-DDS-06205,
     * SerializerCDR shall support serialization of dds::core::Octets.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    bool operator()(dds::core::Octets &value) noexcept
    {
        bool ret{operator()(value.Size())};

        if (ret && ((pos_ + value.Size()) <= size_)) {
            /* AXIVION Next Line AutosarC++19_03-M5.0.15 : the special usage for serialization, the len is checked */
            value.Buffer(&buffer_[pos_]);
            pos_ += value.Size();
        } else {
            ret = false;
        }

        return ret;
    }

private:
    const std::size_t size_;
    uint8_t *buffer_;
    std::size_t pos_{0U};
};
}
}

#endif /* DDS_CDR_SERIALIZER_CDR_HPP */

