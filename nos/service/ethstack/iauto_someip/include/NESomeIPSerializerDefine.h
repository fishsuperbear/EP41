/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/

#ifndef INCLUDE_NE_SOMEIP_SERIALIZER_DEFINE_H
#define INCLUDE_NE_SOMEIP_SERIALIZER_DEFINE_H

#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <stdint.h>
#include <string>

// Define for payload enumeration serialization
#define NESomeIPPayloadEnumToInteger(T, V) (static_cast<std::underlying_type<T>::type>(V))
#define NESomeIPPayloadIntegerToEnum(T, V) (static_cast<T>(V))

typedef uint8_t byte_t;

// define for string serialize
const uint16_t UNICODE_MAX          = 0xffff;
const uint16_t HIGH_SURROGATE_LEAD  = 0xd800;  // 1101 1000 0000 0000
const uint16_t LOW_SURROGATE_LEAD   = 0xdc00;  // 1101 1100 0000 0000
const uint32_t CODE_POINT_MAX       = 0x0010ffff;
const uint16_t SURROGATE_MIN        = 0xd800;
const uint16_t SURROGATE_MAX        = 0xdfff;

const uint8_t NESomeIPDefine_static_tlv_Length_field_size = 4;

// Define enum for message payload byte order
enum NESomeIPPayloadByteOrder {
    NESomeIPPayloadByteOrder_BE = 0x00,  // big endian
    NESomeIPPayloadByteOrder_LE = 0x01  // little endian
};

// Define enum for message payload alignment
enum NESomeIPPayloadAlignment {
    NESomeIPPayloadAlignment_8 = 0x01,
    NESomeIPPayloadAlignment_16 = 0x02,
    NESomeIPPayloadAlignment_32 = 0x04,
    NESomeIPPayloadAlignment_64 = 0x08
};

// Define enum for message payload length field size of variable data
enum NESomeIPPayloadLengthFieldSize {
    NESomeIPPayloadLengthFieldSize_0 = 0x00,
    NESomeIPPayloadLengthFieldSize_8 = 0x01,
    NESomeIPPayloadLengthFieldSize_16 = 0x02,
    NESomeIPPayloadLengthFieldSize_32 = 0x04
};

// Define enum for message payload type field size of union
enum NESomeIPPayloadUnionTypeFieldSize {
    NESomeIPPayloadUnionTypeFieldSize_0 = 0x00,
    NESomeIPPayloadUnionTypeFieldSize_8 = 0x01,
    NESomeIPPayloadUnionTypeFieldSize_16 = 0x02,
    NESomeIPPayloadUnionTypeFieldSize_32 = 0x04
};

// Define enum for struct wire type in tag
enum class NESomeIPPayloadStructWireType : uint16_t {
    NESomeIPPayloadStructWireType_8BitBase = 0x00 << 12,
    NESomeIPPayloadStructWireType_16BitBase = 0x01 << 12,
    NESomeIPPayloadStructWireType_32BitBase = 0x02 << 12,
    NESomeIPPayloadStructWireType_64BitBase = 0x03 << 12,
    NESomeIPPayloadStructWireType_nByteComplex = 0x04 << 12,
    NESomeIPPayloadStructWireType_1ByteComplex = 0x05 << 12,
    NESomeIPPayloadStructWireType_2ByteComplex = 0x06 << 12,
    NESomeIPPayloadStructWireType_4ByteComplex = 0x07 << 12
};

// Define enum for char bit
enum NESomeIPPayloadCharBit
{
    NESomeIPPayloadChar_8Bit = 0,
    NESomeIPPayloadChar_16Bit = 1,
};

// Define enum for payload
enum NESomeIPPayloadErrorCode {
    NESomeIPPayloadErrorCode_Ok                      = 0x00,  // Synchronous success
    NESomeIPPayloadErrorCode_Failed                  = 0x01,  // Synchronous failed
    NESomeIPPayloadErrorCode_Empty                   = 0x02,
    NESomeIPPayloadErrorCode_end                     = 0x03,
    NESomeIPPayloadErrorCode_Unknown                 = 0xFF,
};

enum NESomeIPStringSerializeStatus {
    NESomeIPStringSerialize_Unknown = 0,
    NESomeIPStringSerialize_InvalidLead = 1,
    NESomeIPStringSerialize_InvalidCodePoint = 2,
    NESomeIPStringSerialize_InvalidUtf16 = 3,
    NESomeIPStringSerialize_SequenceTooLong = 4,
    NESomeIPStringSerialize_NotEnoughRoom = 5,
    NESomeIPStringSerialize_IncompleteSquence = 6,
    NESomeIPStringSerialize_Success = 7,
};

enum NESomeIPElementType {
    element_type_basic = 0x00,
    element_type_array = 0x01,
    element_type_string = 0x02,
    element_type_struct = 0x03,
    element_type_union = 0x04,
    element_type_vector = 0x05
};

struct NESomeIPExtendAttr {
    NESomeIPPayloadLengthFieldSize array_length_field_size;   // default 4byte
    NESomeIPPayloadLengthFieldSize vector_length_field_size;  // default 4byte
    NESomeIPPayloadLengthFieldSize struct_length_field_size;  // default 4byte
    NESomeIPPayloadLengthFieldSize string_length_field_size;  // default 4byte
    NESomeIPPayloadLengthFieldSize union_length_field_size;   // default 4byte
    NESomeIPPayloadAlignment alignment; //default 1byte
    NESomeIPPayloadByteOrder byte_order; //default little endiness
    NESomeIPPayloadUnionTypeFieldSize union_type_field_size; //default 4byte
    NESomeIPPayloadCharBit string_utf; //default utf_8
    bool is_tlv_dynamic_length_field_size; //default true ,false:NESomeIPDefine_static_tlv_Length_field_size
    uint32_t length_field_value;        // not for user, the value of length field, for deserialize check
    uint32_t has_read_length;           // not for user, the read total length, for deserialize check

    NESomeIPExtendAttr()
    : array_length_field_size(NESomeIPPayloadLengthFieldSize_32)
    , vector_length_field_size(NESomeIPPayloadLengthFieldSize_32)
    , struct_length_field_size(NESomeIPPayloadLengthFieldSize_32)
    , string_length_field_size(NESomeIPPayloadLengthFieldSize_32)
    , union_length_field_size(NESomeIPPayloadLengthFieldSize_32)
    , alignment(NESomeIPPayloadAlignment_8)
    , byte_order(NESomeIPPayloadByteOrder_LE)
    , union_type_field_size(NESomeIPPayloadUnionTypeFieldSize_32)
    , string_utf(NESomeIPPayloadChar_8Bit)
    , is_tlv_dynamic_length_field_size(true)
    , length_field_value(0)
    , has_read_length(0) {
    }

    NESomeIPExtendAttr(const NESomeIPExtendAttr& other) {
        array_length_field_size = other.array_length_field_size;
        vector_length_field_size = other.vector_length_field_size;
        struct_length_field_size = other.struct_length_field_size;
        string_length_field_size = other.string_length_field_size;
        union_length_field_size = other.union_length_field_size;
        alignment = other.alignment;
        byte_order = other.byte_order;
        union_type_field_size = other.union_type_field_size;
        string_utf = other.string_utf;
        is_tlv_dynamic_length_field_size = other.is_tlv_dynamic_length_field_size;
        length_field_value = other.length_field_value;
        has_read_length = other.has_read_length;
    }

    NESomeIPExtendAttr& operator=(const NESomeIPExtendAttr& other) {
        if (this == &other) {
            return *this;
        }

        array_length_field_size = other.array_length_field_size;
        vector_length_field_size = other.vector_length_field_size;
        struct_length_field_size = other.struct_length_field_size;
        string_length_field_size = other.string_length_field_size;
        union_length_field_size = other.union_length_field_size;
        alignment = other.alignment;
        byte_order = other.byte_order;
        union_type_field_size = other.union_type_field_size;
        string_utf = other.string_utf;
        is_tlv_dynamic_length_field_size = other.is_tlv_dynamic_length_field_size;
        length_field_value = other.length_field_value;
        has_read_length = other.has_read_length;
        return *this;
    }
};

struct NESomeIPInternalAttr {
    NESomeIPElementType element_type;
    NESomeIPExtendAttr extend_attr;
    uint32_t begin_index;
    bool need_padding;

    NESomeIPInternalAttr()
    : element_type(element_type_basic)
    , extend_attr()
    , begin_index()
    , need_padding(false) {
    }

    NESomeIPInternalAttr(const NESomeIPInternalAttr& other) {
        element_type = other.element_type;
        extend_attr = other.extend_attr;
        begin_index = other.begin_index;
        need_padding = other.need_padding;
    }

    NESomeIPInternalAttr& operator=(const NESomeIPInternalAttr& other) {
        if (this == &other) {
            return *this;
        }

        element_type = other.element_type;
        extend_attr = other.extend_attr;
        begin_index = other.begin_index;
        need_padding = other.need_padding;
        return *this;
    }
};

#endif  // INCLUDE_NE_SOMEIP_SERIALIZER_DEFINE_H
/* EOF */
