/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DeserializerCDR.hpp
 */

#ifndef DDS_CDR_DESERIALIZER_CDR_HPP
#define DDS_CDR_DESERIALIZER_CDR_HPP

#include <securec.h>

#include <RT-DDS/dds/cdr/SizeCounterCDR.hpp>

namespace dds {
namespace cdr {
/**
 * @brief Deserializer for CDR.
 */
class DeserializerCDR {
public:
    /**
     * @brief Constructor for DeserializerCDR.
     * @param[in] size      Size of the buffer to be deserialized.
     * @param[in] buffer    Buffer to be deserialized.
     */
    DeserializerCDR(const std::size_t size, const uint8_t *buffer) noexcept
        : DeserializerCDR{size, static_cast<std::size_t>(0), buffer}
    {}

    /**
     * @brief Default destructor.
     */
    ~DeserializerCDR() = default;

    /**
     * @brief Deserialization of Identifier.
     * @details All data encapsulation schemes must start with an encapsulation
     * scheme identifier. The identifier occupies the first two octets of the
     * serialized data-stream. The remaining part of the serialized data stream
     * either contains the actual data.
     * @return boolean
     * @retval true     Succeed to deserialize.
     * @retval false    Fail to deserialize.
     * @req{AR-iAOS-RCS-DDS-06301,
     * DeserializerCDR shall support deserialization of Identifier Header.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    bool IdentifierHeader() noexcept
    {
        pos_ += sizeof(uint32_t);
        return true;
    }

    /**
     * @brief Deserialization method of std::is_integral types.
     * @tparam T Type that matches std::is_integral traits.
     * @param[out] value     data deserialized.
     * @return boolean
     * @retval true     Succeed to deserialize.
     * @retval false    Fail to deserialize.
     * @req{AR-iAOS-RCS-DDS-06302,
     * DeserializerCDR shall support deserialization of integer types.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator()(T &value) noexcept /* AXIVION Same Line AutosarC++19_03-A8.4.8 : to compact dynamic Enumerate */
    {
        const std::size_t s{sizeof(T)};
        bool ret{false};

        pos_ = Alignment(pos_, s);
        if ((pos_ + s) <= size_) {
            /* AXIVION Next Line AutosarC++19_03-M5.0.15 : the special usage for deserialization, the len is checked */
            value = *reinterpret_cast<const T *>(&buffer_[pos_]);
            pos_ += s;
            ret = true;
        }

        return ret;
    }

    /**
     * @brief Deserialization method of std::string.
     * @param[in] value     data deserialized.
     * @return boolean
     * @retval true     Succeed to deserialize.
     * @retval false    Fail to deserialize.
     * @req{AR-iAOS-RCS-DDS-06303,
     * DeserializerCDR shall support deserialization of std::string.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    bool operator()(std::string &value) noexcept
    {
        bool ret{true};
        uint32_t s = 0U;

        ret = (operator()(s)) && ret;
        if (ret) {
            ret = false;
            /** if the str is empty, the s should be 1 not 0 */
            if (((pos_ + s) <= size_) && (s != 0)) {
                value.resize(static_cast<std::size_t>(s) - 1U);
                if (!value.empty()) {
                    /* AXIVION Next Line AutosarC++19_03-M5.0.15 : the special usage for deserial, the len is checked */
                    ret = memcpy_s(&value[0U], value.size(), &buffer_[pos_], static_cast<std::size_t>(s - 1U)) == EOK;
                } else {
                    ret = true;
                }
                pos_ += s;
            }
        }

        return ret;
    }

    /**
     * @brief Deserialization method of std::vector<T> types.
     * @tparam T Type that matches std::is_integral traits.
     * @param[in] value     data deserialized.
     * @return boolean
     * @retval true     Succeed to deserialize.
     * @retval false    Fail to deserialize.
     * @req{AR-iAOS-RCS-DDS-06304,
     * DeserializerCDR shall support deserialization of vector with integer types.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator()(std::vector<T> &value) noexcept
    {
        bool ret{true};
        uint32_t size;

        ret = (operator()(size)) && ret;
        if (ret) {
            value.resize(size);
            for (T &v: value) {
                ret = (operator()(v)) && ret;
                if (!ret) {
                    break;
                }
            }
        }

        return ret;
    }

    template<typename T, std::size_t Nm, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator()(std::array<T, Nm> &value) noexcept
    {
        bool ret{true};

        for (T &v: value) {
            ret = (operator()(v)) && ret;
        }

        return ret;
    }

    /**
     * @brief Deserialization method of dds::core::Octets.
     * @param[in] value     data deserialized.
     * @return boolean
     * @retval true     Succeed to deserialize.
     * @retval false    Fail to deserialize.
     * @req{AR-iAOS-RCS-DDS-06305,
     * DeserializerCDR shall support deserialization of dds::core::Octets.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034
     * }
     */
    bool operator()(dds::core::Octets &value) noexcept
    {
        uint32_t size{0U};
        bool ret{operator()(size)};

        if (ret && ((pos_ + size) <= size_)) {
            value.Size(size);
            /* AXIVION Next Line AutosarC++19_03-M5.0.15 : the special usage for deserialization, the len is checked */
            value.Buffer(const_cast<uint8_t *>(&buffer_[pos_]));
            pos_ += size;
        }

        return ret;
    }

private:
    DeserializerCDR(std::size_t size, std::size_t pos, const uint8_t *buffer) noexcept
        : size_{size}, pos_{pos}, buffer_{buffer}
    {}

    const std::size_t size_;
    std::size_t pos_;
    const uint8_t *buffer_;
};
}
}

#endif /* DDS_CDR_DESERIALIZER_CDR_HPP */

