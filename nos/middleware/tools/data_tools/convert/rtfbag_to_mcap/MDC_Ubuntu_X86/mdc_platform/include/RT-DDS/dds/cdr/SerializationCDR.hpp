/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: SerializerCDR.hpp
 */

#ifndef DDS_CDR_SERIALIZATION_CDR_HPP
#define DDS_CDR_SERIALIZATION_CDR_HPP

#include <RT-DDS/dds/cdr/SizeCounterCDR.hpp>
#include <RT-DDS/dds/cdr/SerializerCDR.hpp>
#include <RT-DDS/dds/cdr/DeserializerCDR.hpp>

namespace dds {
namespace cdr {
/**
 * @brief Class which supports Serialization with CDR method.
 * @tparam T Data types to serialize.
 */
template<typename T>
class SerializationCDR {
public:
    explicit SerializationCDR(T &value) noexcept
        : value_(value)
    {}

    ~SerializationCDR() = default;

    /**
     * @brief Get size of specific CDR entity.
     * @return Size of the CDR entity.
     * @req{AR-iAOS-RCS-DDS-06101,
     * SerializationCDR shall support calculating sample size based on CDR specification,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034, DR-iAOS-RCS-DDS-00071
     * }
     */
    std::size_t GetSize() const noexcept
    {
        SizeCounterCDR sizeCounter;
        value_.GetSize(sizeCounter);
        sizeCounter.AlignmentEnd();
        return sizeCounter.GetSize();
    }

    /**
     * @brief Serialize specific CDR to payload.
     * @param[in,out] payload Payload that the CDR will be serialized to.
     * @return bool if success true, else false.
     * @req{AR-iAOS-RCS-DDS-06102,
     * SerializationCDR shall support serializing sample based on CDR specification,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034, DR-iAOS-RCS-DDS-00071
     * }
     */
    bool Serialize(const SerializePayload &payload) noexcept
    {
        SerializerCDR s{static_cast<std::size_t>(payload.size), payload.buffer};
        return value_.Enumerate(s);
    }

    /**
     * @brief Get struct value of templete T from payload.
     * @param[in] payload Payload to be deserialized.
     * @return bool if success true, else false.
     * @req{AR-iAOS-RCS-DDS-06103,
     * SerializationCDR shall support deserializing sample based on CDR specification.,
     * ASL-D,
     * DR-iAOS-RCS-DDS-00034, DR-iAOS-RCS-DDS-00071
     * }
     */
    bool Deserialize(const SerializePayload &payload) noexcept
    {
        DeserializerCDR d{static_cast<std::size_t>(payload.size), payload.buffer};
        return value_.Enumerate(d);
    }

private:
    T &value_;
};
}
}

#endif /* DDS_CDR_SERIALIZATION_CDR_HPP */

