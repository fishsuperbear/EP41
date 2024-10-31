/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Someip tlv serialize util file
 * Create: 2021-05-05
 */
#ifndef VRTF_TLV_SERIALIZE_HELPER_H
#define VRTF_TLV_SERIALIZE_HELPER_H
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <arpa/inet.h>
#include "vrtf/vcc/serialize/serialize_config.h"
#include "vrtf/vcc/serialize/someip_serialize_helper.h"
#include "vrtf/vcc/utils/template_deduction_helper.h"
#include "vrtf/vcc/utils/log.h"
#include "securec.h"
namespace tlv {
namespace serialize {
const std::uint16_t WIRETYPE_ZERO {0};
const std::uint16_t WIRETYPE_ONE {1};
const std::uint16_t WIRETYPE_TWO {2};
const std::uint16_t WIRETYPE_THREE {3};
const std::uint16_t WIRETYPE_FOUR {4};
const std::uint16_t WIRETYPE_FIVE {5};
const std::uint16_t WIRETYPE_SIX {6};
const std::uint16_t WIRETYPE_SEVEN {7};
const std::uint16_t DATAID_BIT {0x0FFFU};
const std::uint16_t WIRETYPE_BIT {0x7000U};
const std::uint16_t WIRETYPE_POS {12};
template <typename T>
class IsSmartPointerHelper : public std::false_type {};

template <typename T>
class IsSmartPointerHelper<std::shared_ptr<T> > : public std::true_type {};

struct GetSizeResult {
    bool isSkip;
    size_t size;
    size_t tagPos;
};

struct DataIdParams {
    size_t totalSize; // Point after each tag
    size_t paramsSize; // Sum of length field size and data size
    std::uint8_t lengthFieldSize; // Length field byte
};

struct RecordDataIdResult {
    std::unordered_map<std::uint16_t, DataIdParams> dataIdMap;
    size_t size;
};
class TlvSerializeHelper {
public:
    static void CopyLengthField(std::uint8_t* data, const size_t dataLength, const size_t lengthField,
        const vrtf::serialize::ByteOrder order)
    {
        if (lengthField == 0) {
            return;
        }
        errno_t memcpySuccess {1};
        switch (lengthField) {
            case vrtf::serialize::someip::ONE_LENGTH_FIELD: {
                const std::uint8_t length {static_cast<std::uint8_t>(dataLength)};
                memcpySuccess = memcpy_s(data, dataLength + lengthField, &length, lengthField);
                break;
            }

            case vrtf::serialize::someip::TWO_LENGTH_FIELD: {
                std::uint16_t length {static_cast<std::uint16_t>(dataLength)};
                length  = vrtf::serialize::someip::HtonsEx(length, order);
                memcpySuccess = memcpy_s(data, dataLength + lengthField, &length, lengthField);
                break;
            }

            case vrtf::serialize::someip::FOUR_LENGTH_FIELD: {
                std::uint32_t length {static_cast<std::uint32_t>(dataLength)};
                length  = vrtf::serialize::someip::HtonlEx(length, order);
                memcpySuccess = memcpy_s(data, dataLength + lengthField, &length, lengthField);
                break;
            }
            default: {
                break;
            }
        }
        if (memcpySuccess != 0) {
            std::string const ctxId {"CM"};
            std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance->error("TlvSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
            "[Someip Serializer] Memory copy struct static length fail.";
        }
    }

    // template use for future CM support serialize config
    static size_t GetLengthFieldLength(const vrtf::serialize::SerializeConfig& config)
    {
        if ((config.someipSerializeType == vrtf::serialize::SomeipSerializeType::ENABLETLV) &&
            (config.wireType == vrtf::serialize::WireType::STATIC)) {
            return config.staticLengthField;
        }
        return sizeof(std::uint32_t); // dynamic and not tlv serialzie length
    }

    static size_t GetDeserializeLength(const vrtf::serialize::SerializeConfig& config, const std::uint8_t* data)
    {
        std::size_t size {vrtf::serialize::someip::MAX_UINT32_BUFFER_T};
        if ((config.someipSerializeType == vrtf::serialize::SomeipSerializeType::ENABLETLV) &&
            (config.wireType == vrtf::serialize::WireType::STATIC)) {
            switch (config.staticLengthField) {
                case vrtf::serialize::someip::ONE_LENGTH_FIELD: {
                    size = *reinterpret_cast<const std::uint8_t*>(data);
                    break;
                }

                case vrtf::serialize::someip::TWO_LENGTH_FIELD: {
                    std::uint16_t length {*reinterpret_cast<const std::uint16_t*>(data)};
                    length = vrtf::serialize::someip::NtohsEx(length, config.byteOrder);
                    size = length;
                    break;
                }

                case vrtf::serialize::someip::FOUR_LENGTH_FIELD: {
                    std::uint32_t length {*reinterpret_cast<const std::uint32_t*>(data)};
                    length = vrtf::serialize::someip::NtohlEx(length, config.byteOrder);
                    size = length;
                    break;
                }
                default: {
                    break;
                }
            }
        } else {
            std::uint32_t length {*reinterpret_cast<const std::uint32_t*>(data)};
            length = vrtf::serialize::someip::NtohlEx(length, config.byteOrder);
            size = length;
        }
        return size;
    }

    static size_t GetShiftLength(const std::uint16_t wireType,
        const vrtf::serialize::ByteOrder byteOrder, std::uint8_t lengthFieldSize, const std::uint8_t* tlvPtr)
    {
        using SomeipHelper = vrtf::serialize::someip::SerializateHelper;
        size_t pos {0};
        switch (wireType) {
            case WIRETYPE_ZERO: {
                pos += sizeof(std::uint8_t);
                break;
            }
            case WIRETYPE_ONE: {
                pos += sizeof(std::uint16_t);
                break;
            }
            case WIRETYPE_TWO: {
                pos += sizeof(std::uint32_t);
                break;
            }
            case WIRETYPE_THREE: {
                pos += sizeof(std::uint64_t);
                break;
            }
            case WIRETYPE_FOUR: {
                pos = SomeipHelper::GetDataLengthByLengthField(lengthFieldSize, byteOrder, tlvPtr) + lengthFieldSize;
                break;
            }
            case WIRETYPE_FIVE: {
                pos = SomeipHelper::GetDataLengthByLengthField(lengthFieldSize, byteOrder, tlvPtr) + lengthFieldSize;
                break;
            }
            case WIRETYPE_SIX: {
                pos = SomeipHelper::GetDataLengthByLengthField(lengthFieldSize, byteOrder, tlvPtr) + lengthFieldSize;
                break;
            }
            case WIRETYPE_SEVEN: {
                pos = SomeipHelper::GetDataLengthByLengthField(lengthFieldSize, byteOrder, tlvPtr) + lengthFieldSize;
                break;
            }
            default: {
                break;
            }
        }
        return pos;
    }
    static std::uint8_t GetLengthFieldByWireType(const std::uint16_t wireType, std::uint8_t staticLengthFieldSize)
    {
        std::uint8_t lengthFieldSize {0};
        switch (wireType) {
            case WIRETYPE_FOUR: {
                lengthFieldSize = staticLengthFieldSize;
                break;
            }
            case WIRETYPE_FIVE: {
                lengthFieldSize = sizeof(std::uint8_t);
                break;
            }
            case WIRETYPE_SIX: {
                lengthFieldSize = sizeof(std::uint16_t);
                break;
            }
            case WIRETYPE_SEVEN: {
                lengthFieldSize = sizeof(std::uint32_t);
                break;
            }
            default: {
                break;
            }
        }
        return lengthFieldSize;
    }
    static std::uint16_t GetInlineTypeTlvTag(uint8_t lengthField, const std::uint16_t dataId)
    {
        std::uint16_t wireTypeValue {0U};
        switch (lengthField) {
            case vrtf::serialize::someip::ONE_BYTES_LENGTH: {
                wireTypeValue = tlv::serialize::WIRETYPE_ZERO;
                break;
            }
            case vrtf::serialize::someip::TWO_BYTES_LENGTH: {
                wireTypeValue = tlv::serialize::WIRETYPE_ONE;
                break;
            }
            case vrtf::serialize::someip::FOUR_BYTES_LENGTH: {
                wireTypeValue = tlv::serialize::WIRETYPE_TWO;
                break;
            }
            case vrtf::serialize::someip::EIGHT_BYTES_LENGTH: {
                wireTypeValue = tlv::serialize::WIRETYPE_THREE;
                break;
            }
            default: {}
        }
        std::uint16_t tag {dataId};
        tag = static_cast<std::uint16_t>(
            tag | static_cast<std::uint16_t>(wireTypeValue << tlv::serialize::WIRETYPE_POS));
        return tag;
    }
    static std::uint16_t GetComplexTypeTlvTag(vrtf::serialize::WireType wireType, const uint8_t lengthField,
        const std::uint16_t dataId)
    {
        std::uint8_t wireTypeValue {0};
        if (wireType == vrtf::serialize::WireType::STATIC) {
            wireTypeValue = tlv::serialize::WIRETYPE_FOUR;
        } else {
            switch (lengthField) {
                case vrtf::serialize::someip::ONE_BYTES_LENGTH: {
                    wireTypeValue = tlv::serialize::WIRETYPE_FIVE;
                    break;
                }
                case vrtf::serialize::someip::TWO_BYTES_LENGTH: {
                    wireTypeValue = tlv::serialize::WIRETYPE_SIX;
                    break;
                }
                case vrtf::serialize::someip::FOUR_BYTES_LENGTH: {
                    wireTypeValue = tlv::serialize::WIRETYPE_SEVEN;
                    break;
                }
                default: {}
            }
        }
        std::uint16_t tag {dataId};
        tag = static_cast<std::uint16_t>(tag | static_cast<std::uint16_t>(
            wireTypeValue << tlv::serialize::WIRETYPE_POS));
        return tag;
    }
    static std::uint8_t GetLengthFieldByDataLength(std::size_t dataLength)
    {
        using namespace vrtf::serialize::someip;
        std::uint8_t lengthField {0};
        if (dataLength <= MAX_UINT8_T) {
            lengthField = vrtf::serialize::someip::ONE_BYTES_LENGTH;
        } else if (dataLength > MAX_UINT8_T && dataLength <= MAX_UINT16_T) {
            lengthField = vrtf::serialize::someip::TWO_BYTES_LENGTH;
        } else {
            lengthField = vrtf::serialize::someip::FOUR_BYTES_LENGTH;
        }
        return lengthField;
    }
    template <typename T>
    static void CopyTagData(std::uint8_t* dataPtr, std::size_t destSize,
        std::shared_ptr<vrtf::serialize::SerializationNode> const &currentNodeConfig, std::uint8_t lengthFiledSize)
    {
        using namespace ara::godel::common::log;
        std::uint16_t tag {GetOptionalTlvTagOr<T>(currentNodeConfig->serializationConfig.wireType,
            currentNodeConfig->dataId, lengthFiledSize)};
        tag = vrtf::serialize::someip::HtonsEx(tag, currentNodeConfig->serializationConfig.byteOrder);
        auto memcpyResult = memcpy_s(dataPtr, destSize, &tag, sizeof(std::uint16_t));
        if (memcpyResult != 0) {
            std::shared_ptr<Log> logInstance {Log::GetLog("CM")};
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            const size_t logLimit {500};
            logInstance->error("TLV_CopyTagData", {logLimit, LogLimitConfig::LimitType::TIMELIMIT}) <<
                "[Serialization]Copy TLV tag failed[dataId=" << currentNodeConfig->dataId << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        }
    }
    template <typename T>
    static std::uint16_t GetOptionalTlvTagOr(vrtf::serialize::WireType wiretype, const std::uint16_t dataId,
        std::uint8_t lengthFiledSize,
        typename std::enable_if<vrtf::vcc::utils::TemplateDeduction::IsOptional<T>::value>::type* = nullptr) noexcept
    {
        std::uint16_t tag {GetTlvTag<typename T::value_type>(wiretype, dataId, lengthFiledSize)};
        return tag;
    }
    template <typename T>
    static std::uint16_t GetOptionalTlvTagOr(vrtf::serialize::WireType wiretype, const std::uint16_t dataId,
        std::uint8_t lengthFiledSize,
        typename std::enable_if<!vrtf::vcc::utils::TemplateDeduction::IsOptional<T>::value>::type* = nullptr) noexcept
    {
        std::uint16_t tag {GetTlvTag<T>(wiretype, dataId, lengthFiledSize)};
        return tag;
    }
    template <typename T>
    static std::uint16_t GetTlvTag(vrtf::serialize::WireType, const std::uint16_t dataId, std::uint8_t,
        typename std::enable_if<!vrtf::vcc::utils::TemplateDeduction::IsStruct<T>::value &&
        (std::is_trivially_copyable<T>::value &&
        !vrtf::vcc::utils::TemplateDeduction::IsArray<T>::value)>::type* = nullptr) noexcept
    {
        std::uint16_t tag {tlv::serialize::TlvSerializeHelper::GetInlineTypeTlvTag(sizeof(T), dataId)};
        return tag;
    }

    template <typename T>
    static std::uint16_t GetTlvTag(vrtf::serialize::WireType wiretype, const std::uint16_t dataId,
        std::uint8_t lengthFiledSize,
        typename std::enable_if<vrtf::vcc::utils::TemplateDeduction::IsStruct<T>::value ||
            !(std::is_trivially_copyable<T>::value) ||
            vrtf::vcc::utils::TemplateDeduction::IsArray<T>::value>::type* = nullptr) noexcept
    {
        std::uint16_t tag {tlv::serialize::TlvSerializeHelper::GetComplexTypeTlvTag(
            wiretype, lengthFiledSize, dataId)};
        return tag;
    }
    static RecordDataIdResult RecordDataId(const vrtf::serialize::ByteOrder byteOrder, const uint8_t staticLengthField,
        const uint8_t* initAddress, const size_t initLength)
    {
        RecordDataIdResult result;
        const uint8_t* tlvPtr {initAddress};
        size_t tlvLength {0};
        while (tlvLength < initLength) {
            if (initLength - tlvLength < sizeof(std::uint16_t)) {
                std::string const ctxId {"CM"};
                std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
                logInstance->error("TlvSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                    "[TLV] Lack of enough code stream to get tag";
                return RecordDataIdResult({{}, vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE});
            }
            std::uint16_t tag {*reinterpret_cast<const std::uint16_t*>(tlvPtr)};
            tag = vrtf::serialize::someip::NtohsEx(tag, byteOrder);
            std::uint16_t nowId {static_cast<std::uint16_t>(tag & DATAID_BIT)}; // dataId is low 12 bits
            const std::uint16_t wireType {static_cast<std::uint16_t>((tag & WIRETYPE_BIT) >> WIRETYPE_POS)};
            if (wireType > WIRETYPE_SEVEN) {
                std::string const ctxId {"CM"};
                std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
                logInstance->error("TlvSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                    "[TLV] Wrong wire type[wireType=" << wireType << " dataId=" << nowId << "]";
                return RecordDataIdResult({{}, vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE});
            }
            tlvLength += sizeof(std::uint16_t);
            tlvPtr += sizeof(std::uint16_t); // pos dataId length
            const std::uint8_t lengthFieldSize = GetLengthFieldByWireType(wireType, staticLengthField);
            if (initLength - tlvLength < lengthFieldSize) {
                std::string const ctxId {"CM"};
                std::shared_ptr<ara::godel::common::log::Log> logInstance {ara::godel::common::log::Log::GetLog(ctxId)};
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
                logInstance->error("TlvSerialize", vrtf::vcc::api::types::LOG_LIMIT_CONFIG) <<
                    "[TLV] Lack of enough code stream to get pos length[dataId" << nowId << "]";
                return RecordDataIdResult({{}, vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE});
            }
            const size_t posLength {GetShiftLength(wireType, byteOrder, lengthFieldSize, tlvPtr)};
            if (posLength == 0) {
                return RecordDataIdResult({{}, vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE});
            }
            DataIdParams params {tlvLength, posLength, lengthFieldSize};
            static_cast<void>(result.dataIdMap.emplace(nowId, params));
            tlvLength += posLength;
            tlvPtr += posLength;
        }
        result.size = tlvLength;
        return result;
    }
    template <typename DataType>
    static typename std::enable_if<!vrtf::vcc::utils::TemplateDeduction::IsOptional<DataType>::value, bool>::type
    IsHasValue(DataType const &) { return true; }
    template <typename DataType>
    static typename std::enable_if<vrtf::vcc::utils::TemplateDeduction::IsOptional<DataType>::value, bool>::type
    IsHasValue(DataType const &optional) { return optional.has_value(); }

};
}
}
#endif
