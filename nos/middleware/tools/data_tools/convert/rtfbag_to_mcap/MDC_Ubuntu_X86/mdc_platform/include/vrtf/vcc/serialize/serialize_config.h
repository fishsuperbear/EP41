/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: This file provides an an interface to modify serialization configuration.
 * Create: 2021-02-19
 */
#ifndef VRTF_SERIALIZE_SERIALIZE_CONFIG_H
#define VRTF_SERIALIZE_SERIALIZE_CONFIG_H
#include <cstdint>
#include <list>
#include <memory>
namespace vrtf {
namespace serialize {
enum class SerializationType: uint32_t {
    ROS = 0U,
    CM = 1U,
    SIGNAL_BASED = 2U
};

enum class SerializeType: uint8_t {
    SHM = 0,
    DDS = 1,
    SOMEIP = 2,
    PROLOC = 3,
    SIGNAL_BASED = 4
};

enum class StructSerializationPolicy: uint8_t {
    OUT_LAYER_ALIGNMENT,
    DISABLE_OUT_LAYER_ALIGNMENT
};

enum class WireType: uint8_t {
    STATIC,
    DYNAMIC
};

enum class SomeipSerializeType: uint8_t {
    ENABLETLV,
    DISABLETLV
};

enum class ByteOrder: uint8_t {
    BIGENDIAN,
    LITTLEENDIAN
};
inline bool IsLittleEndian()
{
    union {
        int a;
        char b;
    } c;
    c.a = 1;
    return (c.b == 1);
}
const size_t DEFAULT_DYNAMIC_LENGTH = 4;
class SerializeConfig {
public:
SerializeConfig()
    : type(SerializationType::CM), structPolicy(StructSerializationPolicy::OUT_LAYER_ALIGNMENT),
      someipSerializeType(SomeipSerializeType::DISABLETLV), wireType(WireType::STATIC), byteOrder(ByteOrder::BIGENDIAN),
      staticLengthField(DEFAULT_DYNAMIC_LENGTH), stringLength(DEFAULT_DYNAMIC_LENGTH),
      vectorLength(DEFAULT_DYNAMIC_LENGTH), arrayLength(0), structLength(0),
      implementsLegencyStringSerialization(false), isTopStruct(true), structDeserializeLength(0),
      isIngoreOutLength(false) {}
virtual ~SerializeConfig() = default;
SerializationType type;
StructSerializationPolicy structPolicy;
SomeipSerializeType someipSerializeType =  SomeipSerializeType::DISABLETLV;
WireType wireType = WireType::STATIC;
ByteOrder byteOrder = ByteOrder::BIGENDIAN;
size_t staticLengthField;
size_t stringLength;
size_t vectorLength;
size_t arrayLength;
size_t structLength;
bool implementsLegencyStringSerialization;
bool isTopStruct;
size_t structDeserializeLength;
bool isIngoreOutLength;
void SyncTlvLength()
{
    if (someipSerializeType == SomeipSerializeType::ENABLETLV) {
        if (wireType == WireType::STATIC) {
            stringLength = staticLengthField;
            vectorLength = staticLengthField;
            arrayLength = staticLengthField;
            structLength = staticLengthField;
        } else {
            stringLength = DEFAULT_DYNAMIC_LENGTH;
            vectorLength = DEFAULT_DYNAMIC_LENGTH;
            arrayLength = DEFAULT_DYNAMIC_LENGTH;
            structLength = DEFAULT_DYNAMIC_LENGTH;
        }
    }
}
};

class ApSomeipTransformationProps {
public:
// All length unit is byte
uint8_t alignment = 1; // SWS_CM_10037, default value is 8 bit
ByteOrder byteOrder = ByteOrder::BIGENDIAN; // SWS_CM_10270, default byte order is BIGENDIAN
bool implementsLegencyStringSerialization = false; //SWS_CM_10058, default add BOM
WireType wireType = WireType::STATIC;   // SWS_CM_90443, default use wiretype 4
uint8_t arrayLengthField = 0;   // SWS_CM_00258, default array length is 0 byte ????
uint8_t mapLengthField = 4;    // SWS_CM_10267, default map length is 4 byte
uint8_t vectorLengthField = 4; // SWS_CM_10258, default vector length is 4 byte
uint8_t stringLengthField = 4; // SWS_CM_10275, default string length is 4 byte
uint8_t structLengthField = 0; // SWS_CM_00255, default struct length is 0 byte
uint8_t unionLengthField = 4;  // PRS_SOMEIP_00119, default union length is 4 byte
uint8_t unionTypeSelectorLength = 4; // PRS_SOMEIP_00119, default union type selector length is 4 byte
};
class SerializationNode;
using SerializationList = std::list<std::shared_ptr<SerializationNode>>;
class SerializationNode {
public:
ApSomeipTransformationProps serializationConfig;
uint16_t dataId = 0xffff;
uint8_t tlvLengthFieldSize = 0;
bool isChildNodeEnableTlv = false;
bool isLastSerializeNode = false;
bool isUnion = false; // Helper info
std::shared_ptr<SerializationList> childNodeList = nullptr;
};
}
}
#endif
