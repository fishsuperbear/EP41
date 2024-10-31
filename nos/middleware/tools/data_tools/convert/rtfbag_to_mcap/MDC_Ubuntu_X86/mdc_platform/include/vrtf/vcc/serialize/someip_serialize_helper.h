/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Someip Serialize help util file
 * Create: 2021-05-05
 */
#ifndef VRTF_SOMEIP_SERIALIZE_HELPER_H
#define VRTF_SOMEIP_SERIALIZE_HELPER_H
#include <cstddef>
#include <cstdint>
#include <map>
#include <arpa/inet.h>
#include <bitset>

#include "vrtf/vcc/serialize/serialize_config.h"
namespace vrtf {
namespace serialize {
namespace someip {
const int FOURCE_BYTE_POS {32};
const int THREE_BYTE_POS {24};
const int ONE_BYTE_POS {8};
const size_t ONE_LENGTH_FIELD {1};
const size_t TWO_LENGTH_FIELD {2};
const size_t FOUR_LENGTH_FIELD {4};
const size_t ONE_BYTES_LENGTH {1};
const size_t TWO_BYTES_LENGTH {2};
const size_t FOUR_BYTES_LENGTH {4};
const size_t EIGHT_BYTES_LENGTH {8};
constexpr uint8_t LENGTH_OF_TWO_BYTES {16};
const std::uint32_t FIRST_BYTE_VALUE32 {0xFF000000U};
const std::uint32_t SECOND_BYTE_VALUE32 {0x00FF0000U};
const std::uint32_t THIRD_BYTE_VALUE32 {0x0000FF00U};
const std::uint32_t FOURTH_BYTE_VALUE32 {0x000000FFU};
const std::uint16_t FIRST_BYTE_VALUE16 {0xFF00U};
const std::uint16_t SECOND_BYTE_VALUE16 {0x00FFU};
const std::uint64_t MAX_UINT32_T {0xFFFFFFFFU};
const std::uint64_t MAX_UINT32_BUFFER_T {0xFFFFFFF0U};
const std::uint64_t MAX_UINT8_T {0xFFU};
const std::uint64_t MAX_UINT16_T {0xFFFFU};
const std::size_t MAX_SOMEIP_SERIALIZE_SIZE {0xFFFFFFFFU};

inline bool CheckStructLengthField(const vrtf::serialize::SerializeConfig &config)
{
    if (!(config.wireType == vrtf::serialize::WireType::DYNAMIC && config.isTopStruct) &&
        !config.isIngoreOutLength) {
        return true;
    }
    return false;
}

inline std::uint32_t HtonlEx(const std::uint32_t host, const vrtf::serialize::ByteOrder order)
{
    std::uint32_t result {host};
    if (order == vrtf::serialize::ByteOrder::LITTLEENDIAN) {
        if (!IsLittleEndian()) {
            result = ((host & FIRST_BYTE_VALUE32) >> THREE_BYTE_POS) | ((host & SECOND_BYTE_VALUE32) >> ONE_BYTE_POS) |
                ((host & THIRD_BYTE_VALUE32) << ONE_BYTE_POS) | ((host & FOURTH_BYTE_VALUE32) << THREE_BYTE_POS);
        }
    } else {
        result = htonl(host);
    }
    return result;
}

inline std::uint32_t NtohlEx(const std::uint32_t host, const vrtf::serialize::ByteOrder order)
{
    std::uint32_t result {host};
    if (order == vrtf::serialize::ByteOrder::LITTLEENDIAN) {
        if (!IsLittleEndian()) {
            result = ((host & FIRST_BYTE_VALUE32) >> THREE_BYTE_POS) | ((host & SECOND_BYTE_VALUE32) >> ONE_BYTE_POS) |
                ((host & THIRD_BYTE_VALUE32) << ONE_BYTE_POS) | ((host & FOURTH_BYTE_VALUE32) << THREE_BYTE_POS);
        }
    } else {
        result = ntohl(host);
    }
    return result;
}

inline std::uint16_t HtonsEx(const std::uint16_t host, const vrtf::serialize::ByteOrder order)
{
    std::uint16_t result {host};
    if (order == vrtf::serialize::ByteOrder::LITTLEENDIAN) {
        if (!IsLittleEndian()) {
            result = (static_cast<std::uint16_t>(host & FIRST_BYTE_VALUE16) >> ONE_BYTE_POS) |
                (static_cast<std::uint16_t>(host & SECOND_BYTE_VALUE16) << ONE_BYTE_POS);
        }
    } else {
        result = htons(host);
    }
    return result;
}

inline std::uint16_t NtohsEx(const std::uint16_t host, const vrtf::serialize::ByteOrder order)
{
    std::uint16_t result {host};
    if (order == vrtf::serialize::ByteOrder::LITTLEENDIAN) {
        if (!IsLittleEndian()) {
            result = (static_cast<std::uint16_t>(host & FIRST_BYTE_VALUE16) >> ONE_BYTE_POS) |
                (static_cast<std::uint16_t>(host & SECOND_BYTE_VALUE16) << ONE_BYTE_POS);
        }
    } else {
        result = ntohs(host);
    }
    return result;
}

template<typename T>
T Htonl64(const T& host, const vrtf::serialize::ByteOrder byteOrder) noexcept
{
    T result {host};
    if (((byteOrder == vrtf::serialize::ByteOrder::LITTLEENDIAN) && IsLittleEndian()) ||
        ((byteOrder == vrtf::serialize::ByteOrder::BIGENDIAN) && !IsLittleEndian())) {
        return result;
    } else {
        std::uint32_t low {static_cast<std::uint32_t>(host & MAX_UINT32_T)};
        std::uint32_t high {static_cast<std::uint32_t>((host >> FOURCE_BYTE_POS) & MAX_UINT32_T)};
        low = HtonlEx(low, byteOrder);
        high = HtonlEx(high, byteOrder);
        result = low;
        result <<= FOURCE_BYTE_POS; // 32: half of 64 bits data
        result |= static_cast<std::uint64_t>(high);
    }
    return result;
}

template<typename T>
T Ntohl64(const T& host, const vrtf::serialize::ByteOrder byteOrder) noexcept
{
    T result {host};
    if (((byteOrder == vrtf::serialize::ByteOrder::LITTLEENDIAN) && IsLittleEndian()) ||
        ((byteOrder == vrtf::serialize::ByteOrder::BIGENDIAN) && !IsLittleEndian())) {
        return result;
    } else {
        std::uint32_t low {static_cast<std::uint32_t>(host & MAX_UINT32_T)};
        std::uint32_t high {static_cast<std::uint32_t>((host >> FOURCE_BYTE_POS) & MAX_UINT32_T)};
        low = NtohlEx(low, byteOrder);
        high = NtohlEx(high, byteOrder);
        result = low;
        result <<= FOURCE_BYTE_POS; // 32: half of 64 bits data
        result |= static_cast<std::uint64_t>(high);
    }
    return result;
}
class SerializateHelper {
public:
    static ByteOrder GetByteOrderByConfig(const SerializeConfig& config,
        const std::shared_ptr<SerializationNode> &serializateConfig)
    {
        ByteOrder byteOrder {ByteOrder::BIGENDIAN};
        if (serializateConfig == nullptr) {
            byteOrder = config.byteOrder;
        } else {
            byteOrder = serializateConfig->serializationConfig.byteOrder;
        }
        return byteOrder;
    }
    static bool GetIsAddBomByConfig(const SerializeConfig& config,
        const std::shared_ptr<SerializationNode> &serializateConfig)
    {
        bool implementsLegencyStringSerialization = false;
        if (serializateConfig != nullptr) {
            implementsLegencyStringSerialization =
                serializateConfig->serializationConfig.implementsLegencyStringSerialization;
        } else {
            implementsLegencyStringSerialization = config.implementsLegencyStringSerialization;
        }
        return !implementsLegencyStringSerialization;
    }
    static size_t GetMaxValueOfLength(const uint8_t lengthFieldSize)
    {
        size_t maxDataValue = 0;
        switch (lengthFieldSize) {
            case vrtf::serialize::someip::ONE_LENGTH_FIELD: {
                maxDataValue = MAX_UINT8_T;
                break;
            }
            case vrtf::serialize::someip::TWO_LENGTH_FIELD: {
                maxDataValue = MAX_UINT16_T;
                break;
            }
            case vrtf::serialize::someip::FOUR_LENGTH_FIELD: {
                maxDataValue = MAX_UINT32_BUFFER_T;
                break;
            }
            default: {
                maxDataValue = MAX_SOMEIP_SERIALIZE_SIZE;
                break;
            }
        }
        return maxDataValue;
    }
    static size_t GetDataLengthByLengthField(const std::size_t lengthField, const vrtf::serialize::ByteOrder byteOrder,
        const std::uint8_t* data)
    {
        std::uint32_t lenTmp = 0;
        if (lengthField == vrtf::serialize::someip::ONE_LENGTH_FIELD) {
            lenTmp = *reinterpret_cast<const std::uint8_t*>(data);
        } else if (lengthField == vrtf::serialize::someip::TWO_LENGTH_FIELD) {
            lenTmp = *reinterpret_cast<const std::uint16_t*>(data);
            lenTmp = NtohsEx(static_cast<std::uint16_t>(lenTmp), byteOrder);
        } else if (lengthField == vrtf::serialize::someip::FOUR_LENGTH_FIELD) {
            lenTmp = *reinterpret_cast<const std::uint32_t*>(data);
            lenTmp = NtohlEx(lenTmp, byteOrder);
        } else {
            // do nothing
        }
        return lenTmp;
    }
    static std::uint8_t GetAlignmentPaddingSize(const size_t currentPos, const uint8_t alignmentByte)
    {
        std::uint8_t paddingSize {0};
        if (alignmentByte > 1) { // Less than 1 means no need for byte alignment
            const std::uint8_t someipHeaderSize = 16; // someip headder is 16 byte
            paddingSize = (someipHeaderSize + currentPos) % alignmentByte;
            if (paddingSize != 0) {
                paddingSize = static_cast<std::uint8_t>(alignmentByte - paddingSize);
            }
        }
        return paddingSize;
    }
    static std::size_t GetDeserializeRealSize(size_t remainSize, size_t deserializeDataSize, size_t paddingSize,
        bool isLastSerializeNode)
    {
        if (remainSize < deserializeDataSize) {
            return vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE;
        }
        size_t size {deserializeDataSize};
        if (remainSize - deserializeDataSize >= paddingSize) {
            size += paddingSize;
        } else {
            if (!isLastSerializeNode) {
                size = vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE;
            }
        }
        return size;
    }
    static bool CompareTransportAndLocalEndian(ByteOrder byteOrder)
    {
        return ((!IsLittleEndian() && byteOrder == ByteOrder::BIGENDIAN) ||
            (IsLittleEndian() && byteOrder == ByteOrder::LITTLEENDIAN));
    }
};

inline bool IsUseSerializationNode(const std::shared_ptr<SerializationNode> &currentNodeConfig)
{
    return (currentNodeConfig != nullptr) && (currentNodeConfig->childNodeList != nullptr);
}
}

namespace s2s {
inline size_t Lsb2msb(size_t startBit, size_t signalLen)
{
    startBit = ((startBit - startBit % someip::EIGHT_BYTES_LENGTH) + 7) - startBit % someip::EIGHT_BYTES_LENGTH;
    startBit = (startBit + signalLen) - 1;
    startBit = ((startBit - startBit % someip::EIGHT_BYTES_LENGTH) + 7) - startBit % someip::EIGHT_BYTES_LENGTH;
    return startBit;
}

inline void BitcpyLsb(uint8_t* dest, std::size_t destpos, const std::uint8_t* src, std::size_t srcpos, uint32_t length)
{
    int64_t len {static_cast<int64_t>(length)};
    while (len > 0) {
        std::size_t destbytepos {destpos / someip::EIGHT_BYTES_LENGTH};
        std::uint8_t indent = destpos % someip::EIGHT_BYTES_LENGTH;
        std::uint8_t curSrcByte = src[srcpos / someip::EIGHT_BYTES_LENGTH];
        std::uint8_t byte1 = curSrcByte;
        if (static_cast<uint32_t>(len) <= someip::EIGHT_BYTES_LENGTH - indent) {
            byte1 <<= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
            byte1 >>= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
            byte1 <<= indent;
            dest[destbytepos] |= byte1;
        } else if (static_cast<uint32_t>(len) > someip::EIGHT_BYTES_LENGTH - indent) {
            byte1 <<= indent;
            dest[destbytepos] |= byte1;
            std::uint8_t byte2 = curSrcByte;
            if (static_cast<uint32_t>(len) < someip::EIGHT_BYTES_LENGTH) {
                byte2 <<= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
                byte2 >>= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
            }
            byte2 >>= someip::EIGHT_BYTES_LENGTH - indent;
            dest[destbytepos + 1] |= byte2;
        } else {}

        len -= someip::EIGHT_BYTES_LENGTH;
        destpos += someip::EIGHT_BYTES_LENGTH;
        srcpos += someip::EIGHT_BYTES_LENGTH;
    }
}

inline void DesrBitcpyLsb(uint8_t* dest, size_t destpos, const uint8_t* src, size_t srcpos, uint32_t length)
{
    int64_t len{length};
    while (len > 0) {
        std::size_t srcbytepos {srcpos / someip::EIGHT_BYTES_LENGTH};
        std::uint8_t indent = srcpos % someip::EIGHT_BYTES_LENGTH;
        std::uint8_t byte1 = src[srcbytepos];
        if (static_cast<uint32_t>(len) <= someip::EIGHT_BYTES_LENGTH - indent) {
            byte1 <<= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - indent - static_cast<std::uint8_t>(len));
            byte1 >>= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
        } else if (static_cast<uint32_t>(len) > someip::EIGHT_BYTES_LENGTH - indent) {
            byte1 >>= indent;
            std::uint8_t byte2 = src[srcbytepos + 1];
            if (static_cast<uint32_t>(len) < someip::EIGHT_BYTES_LENGTH) {
                byte2 <<= static_cast<std::uint8_t>(
                    someip::LENGTH_OF_TWO_BYTES - (static_cast<std::uint8_t>(len) + indent));
                byte2 >>= static_cast<std::uint8_t>(
                    someip::LENGTH_OF_TWO_BYTES - (static_cast<std::uint8_t>(len) + indent));
            }
            byte2 <<= someip::EIGHT_BYTES_LENGTH - indent;
            byte1 |= byte2;
        } else {}
        dest[destpos / someip::EIGHT_BYTES_LENGTH] = byte1;
        len -= someip::EIGHT_BYTES_LENGTH;
        destpos += someip::EIGHT_BYTES_LENGTH;
        srcpos += someip::EIGHT_BYTES_LENGTH;
    }
}

inline void BitcpyMsb(uint8_t* dest, std::size_t destpos, const std::uint8_t* src, std::size_t srcpos, uint32_t length)
{
    destpos = Lsb2msb(destpos, length);
    int64_t len {static_cast<int64_t>(length)};
    while (len > 0) {
        std::size_t destbytepos {destpos / someip::EIGHT_BYTES_LENGTH};
        std::uint8_t indent = destpos % someip::EIGHT_BYTES_LENGTH;
        std::uint8_t curSrcByte = src[srcpos / someip::EIGHT_BYTES_LENGTH];
        std::uint8_t byte1 = curSrcByte;
        if (static_cast<uint32_t>(len) <= someip::EIGHT_BYTES_LENGTH - indent) {
            byte1 <<= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
            byte1 >>= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
            byte1 <<= indent;
            dest[destbytepos] |= byte1;
        } else if (static_cast<uint32_t>(len) > someip::EIGHT_BYTES_LENGTH - indent) {
            byte1 <<= indent;
            dest[destbytepos] |= byte1;
            std::uint8_t byte2 = curSrcByte;
            if (static_cast<uint32_t>(len) < someip::EIGHT_BYTES_LENGTH) {
                byte2 <<= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
                byte2 >>= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
            }
            byte2 >>= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - indent);
            dest[destbytepos - 1] |= byte2;
        } else {}
        len -= someip::EIGHT_BYTES_LENGTH;
        destpos -= someip::EIGHT_BYTES_LENGTH;
        srcpos += someip::EIGHT_BYTES_LENGTH;
    }
}

inline void DesrBitcpyMsb(uint8_t* dest, size_t destpos, const uint8_t* src, size_t srcpos, uint32_t length)
{
    srcpos = Lsb2msb(srcpos, length);
    int64_t len{length};
    while (len > 0) {
        std::size_t srcbytepos{srcpos / someip::EIGHT_BYTES_LENGTH};
        std::uint8_t indent = srcpos % someip::EIGHT_BYTES_LENGTH;
        std::uint8_t byte1 = src[srcbytepos];
        if (static_cast<uint32_t>(len) <= someip::EIGHT_BYTES_LENGTH - indent) {
            byte1 <<= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - indent - static_cast<std::uint8_t>(len));
            byte1 >>= static_cast<std::uint8_t>(someip::EIGHT_BYTES_LENGTH - static_cast<std::uint8_t>(len));
        } else if (static_cast<uint32_t>(len) > someip::EIGHT_BYTES_LENGTH - indent) {
            byte1 >>= indent;
            std::uint8_t byte2 = src[srcbytepos - 1];
            if (static_cast<uint32_t>(len) < someip::EIGHT_BYTES_LENGTH) {
                byte2 <<= static_cast<std::uint8_t>(
                    someip::LENGTH_OF_TWO_BYTES - (static_cast<std::uint8_t>(len) + indent));
                byte2 >>= static_cast<std::uint8_t>(
                    someip::LENGTH_OF_TWO_BYTES - (static_cast<std::uint8_t>(len) + indent));
            }
            byte2 <<= someip::EIGHT_BYTES_LENGTH - indent;
            byte1 |= byte2;
        } else {}
        dest[destpos / someip::EIGHT_BYTES_LENGTH] = byte1;
        len -= someip::EIGHT_BYTES_LENGTH;
        destpos += someip::EIGHT_BYTES_LENGTH;
        srcpos -= someip::EIGHT_BYTES_LENGTH;
    }
}

template<typename signedInt>
inline void FillWithMsb(signedInt* src, size_t mostSignificantBit)
{
    std::bitset<sizeof(signedInt) * 8> bs(*src);
    if (bs.test(mostSignificantBit)) {
        signedInt mask = 0x0;
        mask = (~mask) << (mostSignificantBit + 1);
        *src |= mask;
    }
}
}
}
}
#endif
