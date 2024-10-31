/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: provided RawBuffer dataType
 * Create: 2020-08-10
 */
#ifndef VRTF_VCC_API_INTERNAL_RAWBUFFER_H
#define VRTF_VCC_API_INTERNAL_RAWBUFFER_H
#include "securec.h"
#include "vrtf/vcc/api/types.h"
namespace vrtf {
namespace vcc {
namespace api {
namespace types {
template<typename T>
struct IsRawBufferSupport {
    static const bool value = false;
};

template<>
struct IsRawBufferSupport<uint16_t> {
    static const bool value = true;
};

template<>
struct IsRawBufferSupport<uint8_t> {
    static const bool value = true;
};

template<>
struct IsRawBufferSupport<uint32_t> {
    static const bool value = true;
};

template<>
struct IsRawBufferSupport<uint64_t> {
    static const bool value = true;
};

template<>
struct IsRawBufferSupport<std::vector<uint8_t>> {
    static const bool value = true;
};

template<>
struct IsRawBufferSupport<std::vector<uint16_t>> {
    static const bool value = true;
};

template<>
struct IsRawBufferSupport<std::vector<uint64_t>> {
    static const bool value = true;
};

template<>
struct IsRawBufferSupport<std::vector<uint32_t>> {
    static const bool value = true;
};

template<typename T>
struct IsVector {
    static const bool value = false;
};

template<typename T>
struct IsVector<std::vector<T>> {
    static const bool value = true;
};

class RawBuffer {
public:
    RawBuffer(uint8_t* data, const uint64_t& size, Mbuf* mbuf = nullptr);
    ~RawBuffer();
    uint8_t* GetRawBufferPtr();
    std::uint64_t GetRawBufferSize() const;
    Mbuf* GetMbufPtr() const noexcept;
    template<typename T>
    typename std::enable_if<IsRawBufferSupport<T>::value && !IsVector<T>::value, RawBuffer>::type& operator<<(
        const T& data)
    {
        size_t size = sizeof(T);
        if (rawDataSize < pos_ + size) {
            size_t const remainSize = rawDataSize - pos_;
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            auto logInstance = ara::godel::common::log::Log::GetLog("CM");
            logInstance->warn("vrtf::vcc::api::types::RawBuffer::Rawbuffer is full",
                {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                    "[RawBuffer][Rawbuffer is full, cannot add data][remainSize=" << remainSize << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return *this;
        }
        bool rst = false;
        rst = CpyUserDataToRawBuffer(size, data, size);
        if (!rst) {
            return *this;
        }
        pos_ += size;
        return *this;
    }

    template<typename T>
    typename std::enable_if<IsRawBufferSupport<T>::value && IsVector<T>::value, RawBuffer>::type& operator<<(T& data)
    {
        size_t const vecSize = data.size();
        size_t const size = sizeof(typename T::value_type);
        size_t const actualNumber = ((rawDataSize - pos_) / (size) > vecSize) ? vecSize : (rawDataSize - pos_) / size;
        if (vecSize == 0 || actualNumber == 0) {
            return *this;
        }
        if (rawDataSize < pos_ + size * vecSize) {
            size_t const remainSize = (rawDataSize - (size * actualNumber)) - pos_ ;
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            auto logInstance = ara::godel::common::log::Log::GetLog("CM");
            logInstance->warn("vrtf::vcc::api::types::RawBuffer::Rawbuffer is full",
                {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                "[RawBuffer][Rawbuffer is full, cannot add vector data][remainSize=" << remainSize << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        }
        size_t cpySizeCount = actualNumber * size;
        bool rst = CpyUserDataToRawBuffer(size, data[0], cpySizeCount);
        if (!rst) {
            return *this;
        }
        pos_ += cpySizeCount;
        return *this;
    }

    template<typename T>
    typename std::enable_if<!IsRawBufferSupport<T>::value>::type operator<<(T& data)
    {
        static_cast<void>(data);
        /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
        /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
        auto logInstance = ara::godel::common::log::Log::GetLog("CM");
        logInstance->error("vrtf::vcc::api::types::RawBuffer::Rawbuffer not support this data type",
        {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
            "[Rawbuffer][Rawbuffer not support this data type]";
        /* AXIVION enable style AutosarC++19_03-A5.0.1 */
        /* AXIVION enable style AutosarC++19_03-A5.1.1 */
    }
    uint8_t* rawDataPtr;
private:

    template<typename T>
    bool CpyUserDataToRawBuffer(size_t inlineTypeSize, T const &data, size_t const cpySizeCount) const
    {
        bool result = false;
        if (inlineTypeSize == 1 || inlineTypeSize == 2 || inlineTypeSize == 4 || inlineTypeSize == 8) {
            errno_t memcpyResult = memcpy_s(rawDataPtr + pos_, rawDataSize - pos_, &data, cpySizeCount);
            if (memcpyResult != 0) {
                /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
                /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
                auto logInstance = ara::godel::common::log::Log::GetLog("CM");
                logInstance->error("vrtf::vcc::api::types::RawBuffer::Rawbuffer memcpy failed",
                {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT})
                    << "Rawbuffer memcpy failed";
                /* AXIVION enable style AutosarC++19_03-A5.0.1 */
                /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            } else {
                result = true;
            }
        }
        return result;
    }
    uint64_t rawDataSize;
    uint64_t pos_;
    Mbuf *mbuf_ = nullptr;
};
}
}
}
}
#endif
