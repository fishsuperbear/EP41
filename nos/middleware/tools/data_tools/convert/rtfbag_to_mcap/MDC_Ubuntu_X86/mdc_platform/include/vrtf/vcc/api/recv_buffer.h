/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Define types in communication mannger
 * Create: 2020-08-10
 */
#ifndef VRTF_VCC_API_INTERNAL_RECVBUFFER_H
#define VRTF_VCC_API_INTERNAL_RECVBUFFER_H
#include "vrtf/vcc/api/types.h"
#include "vrtf/vcc/api/raw_buffer.h"
#include "vrtf/vcc/driver/event_handler.h"
#include "vrtf/vcc/utils/dp_adapter_handler.h"
#include "securec.h"
namespace vrtf {
namespace vcc {
namespace utils {
class DpAdapterHandler;
}
namespace api {
namespace types {
class RecvBuffer;
using RecvBufferHandler = std::function<void(RecvBuffer)>;
template<typename T>
struct IsRecvBuffer {
    static const bool value = false;
};

template<>
struct IsRecvBuffer<RecvBuffer> {
    static const bool value = true;
};
class ReturnLoanControl {
public:
    ReturnLoanControl(const std::shared_ptr<vrtf::vcc::driver::EventHandler>& driver, const uint8_t* data);
    ~ReturnLoanControl();
    ReturnLoanControl(const ReturnLoanControl& other) = delete;
    ReturnLoanControl& operator=(const ReturnLoanControl& other) = delete;
private:
    std::shared_ptr<vrtf::vcc::driver::EventHandler> driver_;
    const uint8_t* rawDataPtr;
};

class RecvBuffer {
public:
    RecvBuffer(
        const uint8_t* data, const uint64_t& size, const std::shared_ptr<vrtf::vcc::driver::EventHandler>& driver);
    RecvBuffer(
        const uint8_t* data, const uint64_t& size, const std::shared_ptr<ReturnLoanControl>& returnLoanPtr);
    RecvBuffer(const uint8_t* data, uint64_t size, Mbuf *mbuf, utils::MbufFreeFunc freeFunc = nullptr);
    RecvBuffer() = default;
    RecvBuffer(const RecvBuffer& other) = delete;
    RecvBuffer& operator=(const RecvBuffer& other) = delete;
    RecvBuffer(RecvBuffer && recvBuffer) noexcept;
    RecvBuffer& operator=(RecvBuffer && recvBuffer) noexcept;
    ~RecvBuffer();
    template<typename T>
    typename std::enable_if<IsRawBufferSupport<T>::value && !IsVector<T>::value, RecvBuffer>::type& operator>>(T& data)
    {
        size_t size = sizeof(T);
        if (rawDataSize < pos_ + size) {
            size_t remainSize = rawDataSize - pos_;
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            auto logInstance = ara::godel::common::log::Log::GetLog("CM");
            logInstance->warn("vrtf::vcc::api::types::RecvBuffer::Recvbuffer is to end",
            {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                "[RecvBuffer][Recvbuffer is to end, none ostream][remainSize=" << remainSize << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return *this;
        }
        bool rst = CpyRecvBufferToUserData(size, &data, size);
        if (!rst) {
            return *this;
        }
        pos_ += size;
        return *this;
    }

    template<typename T>
    typename std::enable_if<IsRawBufferSupport<T>::value && IsVector<T>::value, RecvBuffer>::type& operator>>(T& data)
    {
        size_t size = sizeof(typename T::value_type);
        size_t vecSize = (rawDataSize - pos_) / size;
        if (vecSize == 0) {
            size_t remainSize = (rawDataSize - pos_) % size;
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            auto logInstance = ara::godel::common::log::Log::GetLog("CM");
            logInstance->warn("vrtf::vcc::api::types::RecvBuffer::Recvbuffer is to end",
            {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                "[RecvBuffer][Recvbuffer is to end, none ostream][remainSize=" << remainSize << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return *this;
        }
        data.resize(vecSize);
        size_t cpySizeCount =  vecSize * size;
        bool rst = CpyRecvBufferToUserData(size, &data[0], cpySizeCount);
        if (!rst) {
            return *this;
        }
        pos_ += cpySizeCount;
        return *this;
    }
    template<typename T>
    typename std::enable_if<!IsRawBufferSupport<T>::value>::type operator>>(T& data)
    {
        static_cast<void>(data);
        /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
        /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
        auto logInstance = ara::godel::common::log::Log::GetLog("CM");
        logInstance->error(
            "vrtf::vcc::api::types::RecvBuffer::Recvbuffer not support this data type",
            {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT})
            << "[Recvbuffer][Recvbuffer not support this data type]";
        /* AXIVION enable style AutosarC++19_03-A5.0.1 */
        /* AXIVION enable style AutosarC++19_03-A5.1.1 */
    }

    const std::uint8_t* Get() const;
    std::uint64_t GetSize() const;
    const std::shared_ptr<ReturnLoanControl> GetReturnLoan() const;
    Mbuf* GetMbufPtr() const { return mbuf_; }
private:
    template<typename T>
    bool CpyRecvBufferToUserData(size_t inlineTypeSize, T* data, size_t const cpySizeCount) const
    {
        bool result = false;
        if (inlineTypeSize == 1 || inlineTypeSize == 2 || inlineTypeSize == 4 || inlineTypeSize == 8) {
            errno_t memcpyResult = memcpy_s(data, cpySizeCount, rawDataPtr + pos_, cpySizeCount);
            if (memcpyResult != 0) {
                /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
                /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
                auto logInstance = ara::godel::common::log::Log::GetLog("CM");
                logInstance->error("vrtf::vcc::api::types::RecvBuffer::Recvbuffer memcpy failed",
                {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                    "Recvbuffer memcpy failed";
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
    const uint8_t* rawDataPtr;
    std::shared_ptr<ReturnLoanControl> returnLoanPtr_;
    Mbuf *mbuf_;
    utils::MbufFreeFunc mbufFree_;
};
}
}
}
}
#endif
