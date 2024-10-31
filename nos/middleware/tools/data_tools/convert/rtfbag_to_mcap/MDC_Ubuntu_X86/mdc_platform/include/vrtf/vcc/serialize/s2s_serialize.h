/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Service to signal serialize util file
 * Create: 2021-10-05
 */
#ifndef S2S_SERIALIZE_H
#define S2S_SERIALIZE_H

#include <securec.h>
#include <type_traits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <arpa/inet.h>
#include "ara/core/vector.h"
#include "ara/core/array.h"
#include "ara/hwcommon/log/log.h"
#include "vrtf/vcc/serialize/someip_serialize_helper.h"
#include "vrtf/vcc/api/types.h"
#include "vrtf/vcc/api/raw_buffer_for_raw_data.h"

/*
Note: Serialize max size is 16M (equal to 0x100000), cannot be result in integer overflow with type of size_t
*/
namespace vrtf {
namespace serialize {
namespace s2s {
template <typename T, typename Tag = void>
struct IsStruct {
    static const bool value = false;
};

template <typename T>
struct IsStruct<T, typename T::IsEnumerableTag> {
    static const bool value = true;
};

template <typename T>
constexpr bool IsUint8()
{
    return std::is_same<std::uint8_t, typename std::decay<T>::type>::value;
}

template <typename T>
constexpr bool IsS2SSerializable()
{
  return IsStruct<T>::value || std::is_signed<T>::value ||
         std::is_unsigned<T>::value || std::is_floating_point<T>::value;
}

template <typename T>
class Serializer;

class StructSerializeHelper {
public:
    StructSerializeHelper(
        std::uint8_t* payload, std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu, std::size_t curIdx = 0)
        : payload_(payload), pdu_(std::move(pdu)), curIdx_(curIdx)
    {
    }

    ~StructSerializeHelper() = default;
    template <typename T>
    void operator()(const T& value)
    {
        Serializer<T> serializer(value, pdu_, curIdx_);
        serializer.Serialize(payload_);
        curIdx_ = serializer.GetPostIndex();
    }

private:
    std::uint8_t* payload_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    std::size_t curIdx_;
};

class SerializeSignalCounter {
public:
    explicit SerializeSignalCounter(std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu)
        : pdu_(std::move(pdu)), count_(0), dynamicLength_(0)
    {}

    ~SerializeSignalCounter() = default;

    template <typename T>
    void operator()(const T& value)
    {
        Serializer<T> serializer(value, pdu_);
        std::size_t count = serializer.GetSignalCount();
        count_ += count;
        dynamicLength_ += serializer.GetDynamicLength();
    }

    std::size_t GetSignalCount() const { return count_; }

    std::size_t GetDynamicLength() const { return dynamicLength_; }

private:
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    std::size_t count_;
    std::size_t dynamicLength_;
};

template<typename T>
class Serializer {
public:
    using value_type = typename std::decay<T>::type;

    Serializer(const value_type& value, std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu, std::size_t curIdx = 0)
        : value_(value), pdu_(std::move(pdu)), curIdx_(curIdx), dynamicLength_(0), signalCount_(GetCountHelper()),
          postIdx_(curIdx_ + signalCount_)
    {
    }

    void Serialize(std::uint8_t *c) { SerializeHelper(c); }

    std::size_t GetSignalCount() { return signalCount_; }

    std::size_t GetPostIndex() { return postIdx_; }

    std::size_t GetDynamicLength() { return dynamicLength_; }

    ~Serializer() = default;

private:
    const value_type& value_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    std::size_t curIdx_;
    std::size_t dynamicLength_; // bytes
    std::size_t signalCount_;
    std::size_t postIdx_;

    template <typename U = value_type>
    void SerializeHelper(std::uint8_t* c,
        typename std::enable_if<!IsStruct<U>::value && (std::is_signed<U>::value ||
            std::is_unsigned<U>::value || std::is_floating_point<U>::value)>::type* = nullptr)
    {
        std::size_t size = sizeof(T);
        value_type valueTemp = value_;
        if (curIdx_ >= pdu_->ISignalCount()) {
            std::string const ctxId {"CM"};
            std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Serializer] Out of the pdu range, invalid serialization[pdu=" << pdu_->Name() << "]";
            return;
        }
        const std::shared_ptr<const vrtf::vcc::api::types::ISignal>& isignal {(*pdu_)[curIdx_]};
        std::size_t bitSize = size * 8;
        if (bitSize >= isignal->length) {
            const auto* data = reinterpret_cast<const std::uint8_t*>(&valueTemp);
            if (isignal->byteOrder == ByteOrder::LITTLEENDIAN) {
                vrtf::serialize::s2s::BitcpyLsb(c, isignal->startPos, data, 0, isignal->length);
            } else {
                vrtf::serialize::s2s::BitcpyMsb(c, isignal->startPos, data, 0, isignal->length);
            }
        } else {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            std::string const ctxId {"CM"};
            std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Serializer] Invalid iSignal length[pdu=" << pdu_->Name() <<
                ", isignal=" << isignal->name << ", dataTypeLen=" << bitSize << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        }
    }

    template <typename U = value_type>
    void SerializeHelper(std::uint8_t* c, typename std::enable_if<IsStruct<U>::value>::type* = nullptr)
    {
        StructSerializeHelper structSerializer(c, pdu_, curIdx_);
        (const_cast<value_type&>(value_)).enumerate(structSerializer);
    }

    template <typename U = value_type>
    void SerializeHelper(const std::uint8_t* c, typename std::enable_if<!IsS2SSerializable<U>()>::type* = nullptr)
    {
        static_cast<void>(c);
        std::string const ctxId {"CM"};
        std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
        logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
            "[S2S Serializer] The type is not supported!";
    }

    template <typename U = value_type>
    typename std::enable_if<!IsStruct<U>::value && (std::is_signed<U>::value ||
            std::is_unsigned<U>::value || std::is_floating_point<U>::value), std::size_t>::type
    GetCountHelper() const
    {
        return 1;
    }

    template <typename U = value_type>
    typename std::enable_if<IsStruct<U>::value, std::size_t>::type GetCountHelper()
    {
        SerializeSignalCounter serializeSignalCounter(pdu_);
        (const_cast<value_type&>(value_)).enumerate(serializeSignalCounter);
        dynamicLength_ += serializeSignalCounter.GetDynamicLength();
        return serializeSignalCounter.GetSignalCount();
    }

    template <typename U = value_type>
    typename std::enable_if<!IsS2SSerializable<U>(), std::size_t>::type GetCountHelper() const
    {
        return 0;
    }
};

template <typename T, std::size_t N>
class Serializer<ara::core::Array<T, N>> {
public:
    using value_type = ara::core::Array<T, N>;

    Serializer(const value_type& value, std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu, std::size_t curIdx = 0)
        : value_(value), pdu_(std::move(pdu)), curIdx_(curIdx), signalCount_(1), postIdx_(curIdx_ + signalCount_)
    {
    }

    void Serialize(std::uint8_t *c) { SerializeHelper(c); }

    std::size_t GetSignalCount() const { return signalCount_; }

    std::size_t GetPostIndex() const { return postIdx_; }

    std::size_t GetDynamicLength() const { return 0; }

private:
    const value_type& value_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    const std::size_t curIdx_;
    const std::size_t signalCount_;
    const std::size_t postIdx_;

    template <typename U = T>
    void SerializeHelper(std::uint8_t* c, typename std::enable_if<IsUint8<U>()>::type* = nullptr)
    {
        using namespace ara::godel::common::log;

        if (curIdx_ >= pdu_->ISignalCount()) {
            std::string const ctxId {"CM"};
            std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Serializer] Out of the pdu range, invalid serialization[pdu=" << pdu_->Name() << "]";
            return;
        }
        const std::shared_ptr<const vrtf::vcc::api::types::ISignal>& isignal {(*pdu_)[curIdx_]};
        const auto* data = reinterpret_cast<const std::uint8_t*>(value_.data());
        std::size_t copySize {N};
        if (copySize * 8 == isignal->length) {
            auto memcpySuccess = memcpy_s(c + isignal->startPos / 8, copySize, data, copySize);
            if (memcpySuccess != 0) {
                std::string const ctxId {"CM"};
                std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
                logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                    "[Someip Serializer Array] Memory copy return error, Invalid serialization.";
                return;
            }
        } else {
            std::string const ctxId {"CM"};
            std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Serializer Array] Invalid iSignal length[pdu=" <<
                pdu_->Name() << ", isignal=" << isignal->name  << ", dataTypeLen=" << copySize << "bytes]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        }
    }

    template <typename U = T>
    void SerializeHelper(const std::uint8_t* c, typename std::enable_if<!IsUint8<U>()>::type* = nullptr)
    {
        static_cast<void>(c);
        std::string const ctxId {"CM"};
        std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
        logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
            "[S2S Serializer Array] The type is not supported!";
    }
};

template <>
class Serializer<vrtf::core::RawBuffer> {
public:
    using ValueType = vrtf::core::RawBuffer;
    explicit Serializer(const ValueType& value,
                        const std::shared_ptr<vrtf::vcc::api::types::Pdu>& pdu, std::size_t curIdx = 0)
        : value_(value), pdu_(pdu)
    {
        static_cast<void>(curIdx);
    }

    void Serialize(std::uint8_t* c) const
    {
        SerializeHelper(c);
    }

    std::size_t GetDynamicLength() const
    {
        return sizeof(std::uint8_t) * value_.size();
    }

    std::size_t GetSignalCount() const { return 0; }

private:
    const ValueType& value_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    void SerializeHelper(std::uint8_t* c) const
    {
        using namespace ara::godel::common::log;
        std::size_t cpySize = sizeof(std::uint8_t) * value_.size();
        const std::uint8_t* data = reinterpret_cast<const std::uint8_t*>(value_.data());
        auto memcpySuccess = memcpy_s(c, cpySize, data, cpySize);
        if (memcpySuccess != 0 && cpySize != 0) {
            std::shared_ptr<Log> logInstance = Log::GetLog("CM");
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Serializer Rawdata] Memory copy return error, Invalid serialization.";
        }
    }
};

template <>
class Serializer<vrtf::core::RawBufferForRawData> {
public:
    using ValueType = vrtf::core::RawBufferForRawData;
    Serializer(const ValueType& value,
                        const std::shared_ptr<vrtf::vcc::api::types::Pdu>& pdu, std::size_t curIdx = 0)
    {
        static_cast<void>(value);
        static_cast<void>(pdu);
        static_cast<void>(curIdx);
    }

    void Serialize(std::uint8_t *) const { return; }

    std::size_t GetDynamicLength() const { return 0; }

    std::size_t GetSignalCount() const { return 0; }
};

template <typename T>
class Serializer<ara::core::Vector<T>> {
public:
    using value_type = ara::core::Vector<T>;

    Serializer(const value_type& value, std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu, std::size_t curIdx = 0)
        : value_(value),  pdu_(std::move(pdu)), curIdx_(curIdx), signalCount_(1), postIdx_(curIdx_ + signalCount_)
    {
    }
    ~Serializer() = default;
    void Serialize(std::uint8_t *c) { SerializeHelper(c); }

    std::size_t GetSignalCount() { return signalCount_; }

    std::size_t GetPostIndex() { return postIdx_; }

    std::size_t GetDynamicLength()
    {
        return sizeof(std::uint8_t) * value_.size();
    }

private:
    const value_type& value_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    const std::size_t curIdx_;
    const std::size_t signalCount_;
    const std::size_t postIdx_;

    template <typename U = T>
    void SerializeHelper(std::uint8_t* c, typename std::enable_if<IsUint8<U>()>::type* = nullptr)
    {
        using namespace ara::godel::common::log;
        size_t copySize = sizeof(std::uint8_t) * value_.size();
        const auto* data = reinterpret_cast<const std::uint8_t*>(value_.data());
        if (curIdx_ >= pdu_->ISignalCount()) {
            std::string const ctxId {"CM"};
            std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Serializer] Out of the pdu range, invalid serialization[pdu=" << pdu_->Name() << "]";
            return;
        }
        const std::shared_ptr<const vrtf::vcc::api::types::ISignal>& isignal {(*pdu_)[curIdx_]};
        if (isignal->length >= copySize * 8) {
            auto memcpySuccess = memcpy_s(c + isignal->startPos / 8, copySize, data, copySize);
            if (memcpySuccess != 0) {
                std::string const ctxId {"CM"};
                std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
                logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                    "[Someip Serializer Vector] Memory copy return error, Invalid serialization[pdu=" << pdu_->Name() <<
                    ", iSignal=" << isignal->name << "]";
                return;
            }
        } else {
            std::string const ctxId {"CM"};
            std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Serializer Vector] Invalid iSignal length[pdu=" <<
                pdu_->Name() << ", isignal=" << isignal->name << ", dataTypeLen=" << copySize << "bytes]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        }
    }

    template <typename U = T>
    void SerializeHelper(const std::uint8_t* c, typename std::enable_if<!IsUint8<U>()>::type* = nullptr)
    {
        static_cast<void>(c);
        std::string const ctxId {"CM"};
        std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
        logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
            "[S2S Serializer Vector] The type is not supported!";
    }
};

template <typename T>
class Deserializer;

class StructDeserializeHelper;

/**
 * @brief Template specialization for ara::core::Vector<std::uint8_t>.
 */
template<typename T>
class Deserializer<ara::core::Vector<T>> {
public:
    using ResultType = ara::core::Vector<T>;

    Deserializer(const std::uint8_t* data, std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu, std::size_t curIdx = 0)
        : data_(data), pdu_(std::move(pdu)), signalCount_(1), curIdx_(curIdx)
    {
    }
    ~Deserializer() {
        data_ = nullptr;
    }

    ResultType GetValue() const { return GetValueHelper(); }

    std::size_t GetPostIndex() const { return curIdx_ + signalCount_; }

    std::size_t GetSignalCount() const { return signalCount_; }

    bool GetIsValid() const { return isValid_; }

private:
    const std::uint8_t* data_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    std::size_t signalCount_;
    std::size_t curIdx_;
    mutable bool isValid_ {true};

    template <typename U = T>
    ResultType GetValueHelper(typename std::enable_if<IsUint8<U>()>::type* = nullptr) const
    {
        using namespace ara::godel::common::log;
        ResultType result {0};
        if (curIdx_ >= pdu_->ISignalCount()) {
            std::string const ctxIdentifier {"CM"};
            std::shared_ptr<Log> logInstance {Log::GetLog(ctxIdentifier)};
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Serializer] Out of the pdu range, invalid serialization[pdu=" << pdu_->Name() << "]";
            isValid_ = false;
            return result;
        }
        const std::shared_ptr<const vrtf::vcc::api::types::ISignal>& iSignal = (*pdu_)[curIdx_];
        std::size_t copySize {pdu_->GetRealDynamicLength() / 8}; // bits to bytes
        result.resize(copySize);
        auto memcpySuccess = memcpy_s(result.data(), copySize, data_ + iSignal->startPos / 8, copySize);
        if (memcpySuccess != 0) {
            std::string const ctxId {"CM"};
            std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Deserializer Vector] Memory copy return error, Invalid deserialization.";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            isValid_ = false;
            return result;
        }
        return result;
    }

    template <typename U = T>
    ResultType GetValueHelper(typename std::enable_if<!IsUint8<U>()>::type* = nullptr) const
    {
        using namespace ara::godel::common::log;
        std::string const ctxId {"CM"};
        std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
        ResultType result;
        /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
        logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
            "[S2S Deserializer Vector] The type is not supported!";
        isValid_ = false;
        return result;
    }
};

template <typename T, std::size_t N>
class Deserializer<ara::core::Array<T, N>> {
public:
    using ResultType = ara::core::Array<T, N>;

    Deserializer(
        const std::uint8_t* data, std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu, const std::size_t curIdx = 0)
        : data_(data), pdu_(std::move(pdu)), curIdx_(curIdx), signalCount_(1)
    {
    }

    std::size_t GetSignalCount() const { return signalCount_; }

    ResultType GetValue() const { return GetValueHelper(); }

    std::size_t GetPostIndex() const { return curIdx_ + signalCount_; }

    bool GetIsValid() const { return isValid_; }

private:
    const std::uint8_t* data_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    std::size_t curIdx_;
    std::size_t signalCount_;
    mutable bool isValid_ {true};

    template <typename U = T>
    ResultType GetValueHelper(typename std::enable_if<IsUint8<U>()>::type* = nullptr) const
    {
        using namespace ara::godel::common::log;
        ResultType result {0};
        if (curIdx_ >= pdu_->ISignalCount()) {
            std::string const ctxId {"CM"};
            std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Serializer] Out of the pdu range, invalid serialization[pdu=" << pdu_->Name() << "]";
            isValid_ = false;
            return result;
        }
        const std::shared_ptr<const vrtf::vcc::api::types::ISignal>& iSignal = (*pdu_)[curIdx_];
        const std::size_t copySize = N;
        if (iSignal->length == copySize * 8) {
            auto memcpySuccess = memcpy_s(result.data(), copySize, data_ + iSignal->startPos / 8, copySize);
            if (memcpySuccess != 0) {
                std::string const ctxId {"CM"};
                std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
                /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
                /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
                logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                    "[S2S Deserializer Array] Memory copy return error, invalid deserialization.";
                /* AXIVION enable style AutosarC++19_03-A5.0.1 */
                /* AXIVION enable style AutosarC++19_03-A5.1.1 */
                isValid_ = false;
                return result;
            }
        } else {
            std::string const ctxId {"CM"};
            std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Deserializer Array] Invalid iSignal length[pdu=" <<
                pdu_->Name() << ", isignal=" << iSignal->name << ", dataTypeLen=" << copySize << "bytes]";
            isValid_ = false;
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        }
        return result;
    }

    template <typename U = T>
    ResultType GetValueHelper(typename std::enable_if<!IsUint8<U>()>::type* = nullptr) const
    {
        using namespace ara::godel::common::log;
        std::string const ctxId {"CM"};
        std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
        ResultType result;
        logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
            "[S2S Deserializer Array] The type is not supported!";
        isValid_ = false;
        return result;
    }
};

class DeserializeSignalCounter {
public:
    DeserializeSignalCounter(const std::uint8_t* data, std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu,
        std::size_t curIdx = 0, std::size_t count = 0)
        : data_(data), pdu_(std::move(pdu)), curIdx_(curIdx), count_(count)
    {
    }
    ~DeserializeSignalCounter() {
        data_ = nullptr;
    }

    template <typename T>
    void operator()(const T&)
    {
        using namespace ara::godel::common::log;
        Deserializer<T> deserializer(data_, pdu_, curIdx_);
        std::size_t count = deserializer.GetSignalCount();
        count_ += count;
        curIdx_ += count;
    }

    std::size_t GetSignalCount() const { return count_; }

private:
    const std::uint8_t* data_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    std::size_t curIdx_;
    std::size_t count_;
};

class StructDeserializeHelper {
public:
    StructDeserializeHelper(
        const std::uint8_t* data, std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu, std::size_t curIdx = 0)
        : data_(data), pdu_(std::move(pdu)), curIdx_(curIdx)
    {
    }
    ~StructDeserializeHelper() {
        data_ = nullptr;
    }

    bool GetIsValid() const { return isValid_; }

    template <typename T>
    void operator()(T& value)
    {
        using namespace ara::godel::common::log;
        if (curIdx_ >= pdu_->ISignalCount()) {
            std::string const ctxId {"CM"};
            std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Deserializer] Out of the pdu range, invalid serialization[pdu=" << pdu_->Name() << "]";
            isValid_ = false;
        } else {
            Deserializer<T> deserializer(data_, pdu_, curIdx_);
            value = deserializer.GetValue();
            if (!deserializer.GetIsValid()) {
                isValid_ = false;
            }
            curIdx_ = deserializer.GetPostIndex();
        }
    }

private:
    const std::uint8_t* data_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    std::size_t curIdx_;
    bool isValid_ {true};
};

template <>
class Deserializer<vrtf::core::RawBuffer> {
public:
    using ResultType = vrtf::core::RawBuffer;
    Deserializer(
        const std::uint8_t* data, const std::shared_ptr<vrtf::vcc::api::types::Pdu>& pdu, std::size_t curIdx = 0)
        : data_(data), pdu_(pdu)
    {
        static_cast<void>(curIdx);
    }
    ~Deserializer() = default;
    ResultType GetValue() const { return GetValueHelper(); }

    std::size_t GetSignalCount() const { return 0; }

    bool GetIsValid() const { return isValid_; }

private:
    const std::uint8_t* data_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    mutable bool isValid_ {true};
    ResultType GetValueHelper() const
    {
        ResultType result;
        result.reserve(pdu_->PayloadSize());
        static_cast<void>(result.insert(result.cbegin(), data_, data_ + pdu_->PayloadSize()));
        return result;
    }
};

template <typename T>
class Deserializer {
public:
    using ValueType = typename std::decay<T>::type;
    Deserializer(
        const std::uint8_t* data, std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu, std::size_t curIdx = 0)
        : data_(data), pdu_(std::move(pdu)),
          curIdx_(curIdx), signalCount_(GetCountHelper()), postIdx_(curIdx_ + signalCount_) {}
    ~Deserializer() {
        data_ = nullptr;
    }

    ValueType GetValue() const { return GetValueHelper(); }

    std::size_t GetSignalCount() const { return signalCount_; }

    std::size_t GetPostIndex() const { return postIdx_; }

    bool GetIsValid() const { return isValid_; }

private:
    const std::uint8_t* data_;
    const std::shared_ptr<vrtf::vcc::api::types::Pdu> pdu_;
    std::size_t curIdx_;
    std::size_t signalCount_;
    std::size_t postIdx_;
    mutable bool isValid_ {true};

    template <typename U = ValueType>
    ValueType GetValueHelper(
        typename std::enable_if<!IsStruct<U>::value && (std::is_signed<U>::value ||
            std::is_unsigned<U>::value || std::is_floating_point<U>::value)>::type* = nullptr) const
    {
        ValueType result {0};
        if (curIdx_ >= pdu_->ISignalCount()) {
            std::string const ctxId {"CM"};
            std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Deserializer] Out of the pdu range, invalid deserialization[pdu=" << pdu_->Name() << "]";
            isValid_ = false;
            return result;
        }
        const std::shared_ptr<const vrtf::vcc::api::types::ISignal>& iSignal = (*pdu_)[curIdx_];
        std::size_t retBitSize = sizeof(U) * 8;
        if (retBitSize < iSignal->length) {
            std::string const ctxId {"CM"};
            std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                "[S2S Deserializer] Invalid iSignal length[pdu=" << pdu_->Name() <<
                ", iSignal=" << iSignal->name << ", dataTypeLen=" << retBitSize << "bits]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            isValid_ = false;
            return result;
        }
        if (iSignal->byteOrder == ByteOrder::LITTLEENDIAN) {
            vrtf::serialize::s2s::DesrBitcpyLsb(reinterpret_cast<std::uint8_t*>(&result),
                                                        0, data_, iSignal->startPos, iSignal->length);
        } else {
            vrtf::serialize::s2s::DesrBitcpyMsb(reinterpret_cast<std::uint8_t*>(&result),
                                                        0, data_, iSignal->startPos, iSignal->length);
        }
        FillProcess(result, iSignal);
        return result;
    }

    template <typename U = ValueType>
    void FillProcess(U& result, const std::shared_ptr<const vrtf::vcc::api::types::ISignal>& iSignal,
        typename std::enable_if<std::is_signed<U>::value && !std::is_floating_point<U>::value>::type* = nullptr) const
    {
        if (iSignal->length > 0) {
            size_t msb {iSignal->length - 1};
            vrtf::serialize::s2s::FillWithMsb(&result, msb);
        }
    }

    template <typename U = ValueType>
    void FillProcess(U& result, const std::shared_ptr<const vrtf::vcc::api::types::ISignal>& iSignal,
        typename std::enable_if<!std::is_signed<U>::value || std::is_floating_point<U>::value>::type* = nullptr) const
    {
        static_cast<void>(result);
        static_cast<void>(iSignal);
    }

    template <typename U = ValueType>
    ValueType GetValueHelper(typename std::enable_if<IsStruct<U>::value>::type* = nullptr) const
    {
        ValueType result = ValueType();
        StructDeserializeHelper deserializer(data_, pdu_, curIdx_);
        result.enumerate(deserializer);
        isValid_ = deserializer.GetIsValid();
        return result;
    }

    template <typename U = ValueType>
    ValueType GetValueHelper(typename std::enable_if<!IsS2SSerializable<U>()>::type* = nullptr) const
    {
        using namespace ara::godel::common::log;
        std::string const ctxId {"CM"};
        std::shared_ptr<Log> logInstance {Log::GetLog(ctxId)};
        logInstance->error("S2SSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
            "[S2S Deserializer] The type is not supported!";
        isValid_ = false;
        return {};
    }

    template <typename U = ValueType>
    typename std::enable_if<!IsStruct<U>::value && (std::is_signed<U>::value ||
        std::is_unsigned<U>::value || std::is_floating_point<U>::value), std::size_t>::type
    GetCountHelper() const
    {
        return 1;
    }

    template <typename U = ValueType>
    typename std::enable_if<IsStruct<U>::value, std::size_t>::type GetCountHelper() const
    {
        DeserializeSignalCounter sizeCounter(data_, pdu_, curIdx_);
        ValueType* x = nullptr;
        x->enumerate(sizeCounter);
        return sizeCounter.GetSignalCount();
    }

    template <typename U = ValueType>
    typename std::enable_if<!IsS2SSerializable<U>(), std::size_t>::type GetCountHelper() const
    {
        return 0;
    }
};
} // s2s
} // serialize
} // vrtf

#endif
