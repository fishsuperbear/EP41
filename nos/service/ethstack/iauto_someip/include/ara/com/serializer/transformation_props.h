#ifndef TRANSFORMATION_PROPS_H_
#define TRANSFORMATION_PROPS_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <stdint.h>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <ara/core/optional.h>
#include <ara/core/error_code.h>
#include <ara/core/error_domain.h>

namespace ara {
namespace com {

// Define enum for payload
enum SerializeErrorCode {
    SerializeErrorCode_Ok      = 0x00,  // Synchronous success
    SerializeErrorCode_Failed  = 0x01,  // Synchronous failed
    SerializeErrorCode_Empty   = 0x02,
    SerializeErrorCode_Unknown = 0xFF,
};

enum class WireTypes : std::uint16_t
{
    base_8bit               = 0U,
    base_16bit              = 1U,
    base_32bit              = 2U,
    base_64bit              = 3U,
    complex_with_length     = 4U,
    complex_force_length_8  = 5U,
    complex_force_length_16 = 6U,
    complex_force_length_32 = 7U
};

/**
 * @brief Byte Order
 *
 * @uptrace{SWS_CM_10272}
 * @uptrace{SWS_CM_10274}
 * @uptrace{SWS_CM_10276}
 * @uptrace{SWS_CM_10279}
 * @uptrace{SWS_CM_10280}
 * @uptrace{SWS_CM_10281}
 */
enum ByteOrderEnum : uint8_t {
    BYTE_ORDER_BIG_ENDIAN    = 0x00U,  // MOST-SIGNIFICANT-BYTE-FIRST
    BYTE_ORDER_LITTLE_ENDIAN = 0x01U,  // MOST-SIGNIFICANT-BYTE-LAST
    BYTE_ORDER_OPAQUE        = 0x02U   // OPAQUE
};

/**
 * @brief String Encoding.
 *
 * @uptrace{SWS_CM_10242}
 */
enum StringEncoding : uint8_t {
    UTF8  = 0x01U,
    UTF16 = 0x02U
};

/**
 * @brief Define enum for alignment.
 *
 */
enum DataAlignment : uint8_t {
    DataAlignment_8  = 0x01U,
    DataAlignment_16 = 0x02U,
    DataAlignment_32 = 0x04U,
    DataAlignment_64 = 0x08U
};

/**
 * @brief Define enum for length field size.
 *
 */
enum LengthFieldSize : uint8_t {
    LengthFieldSize_0  = 0x00U,  // string,vector,map,set,union
    LengthFieldSize_8  = 0x01U,
    LengthFieldSize_16 = 0x02U,
    LengthFieldSize_32 = 0x04U
};

/**
 * @brief Define enum for type selector field size.
 *
 */
enum TypeSelectorFieldSize : uint8_t {
    TypeSelectorFieldSize_0  = 0x00U,
    TypeSelectorFieldSize_8  = 0x01U,
    TypeSelectorFieldSize_16 = 0x02U,
    TypeSelectorFieldSize_32 = 0x04U
};

/**
 * @brief Define enum for length field type.
 *
 */
enum LengthFieldDataType : uint8_t {
    LengthFieldDataType_Basic  = 0x00U,
    LengthFieldDataType_String = 0x01U,
    LengthFieldDataType_Vector = 0x02U,
    LengthFieldDataType_Array  = 0x03U,
    LengthFieldDataType_Map    = 0x04U,
    LengthFieldDataType_Pair   = 0x05U,
    LengthFieldDataType_Set    = 0x06U,
    LengthFieldDataType_Struct = 0x07U,
    LengthFieldDataType_Union  = 0x08U
};

/**
 * @brief define the serialize props
 */
struct SomeipTransformationProps {
    DataAlignment  alignment{DataAlignment_8};
    ByteOrderEnum  byteOrder{BYTE_ORDER_BIG_ENDIAN};
    bool           implementsLegacyString{false};
    bool           isDynamicLengthFieldSize{false};
    bool           isSessionHandlingActive{false};
    /**
     * @brief Array length.
     *
     * @uptrace{SWS_CM_00256}
     * @uptrace{SWS_CM_00257}
     * @uptrace{SWS_CM_00258}
     * @uptrace{SWS_CM_00259}
     * @uptrace{SWS_CM_00260}
     * @uptrace{SWS_CM_10222}
     * @uptrace{SWS_CM_10257}
     * @uptrace{SWS_CM_10258}
     */
    LengthFieldSize  sizeOfArrayLengthField{LengthFieldSize_32};
    LengthFieldSize  sizeOfVectorLengthField{LengthFieldSize_32};
    /**
     * @brief length field shall be inserted in front of the serialized string.
     *
     * @uptrace{SWS_CM_10271}
     * @uptrace{SWS_CM_10273}
     * @uptrace{SWS_CM_10275}
     * @uptrace{SWS_CM_10277}
     * @uptrace{SWS_CM_10278}
     */
    LengthFieldSize       sizeOfStringLengthField{LengthFieldSize_32};
    LengthFieldSize       sizeOfStructLengthField{LengthFieldSize_0};
    LengthFieldSize       sizeOfUnionLengthField{LengthFieldSize_32};
    TypeSelectorFieldSize sizeOfUnionTypeSelectorField{TypeSelectorFieldSize_32};
    StringEncoding        stringEncoding{UTF8};

    SomeipTransformationProps() = default;

    SomeipTransformationProps(DataAlignment alignment_,
                   ByteOrderEnum  byteOrder_,
                   bool implementsLegacyString_,
                   bool isDynamicLengthFieldSize_,
                   bool isSessionHandlingActive_,
                   LengthFieldSize sizeOfArrayLengthField_,
                   LengthFieldSize sizeOfVecotrLengthField_,
                   LengthFieldSize sizeOfStringLengthField_,
                   LengthFieldSize sizeOfStructLengthField_,
                   LengthFieldSize sizeOfUnionLengthField_,
                   TypeSelectorFieldSize sizeOfUnionTypeSelectorField_,
                   StringEncoding stringEncoding_)
    : alignment(alignment_)
    , byteOrder(byteOrder_)
    , implementsLegacyString(implementsLegacyString_)
    , isDynamicLengthFieldSize(isDynamicLengthFieldSize_)
    , isSessionHandlingActive(isSessionHandlingActive_)
    , sizeOfArrayLengthField(sizeOfArrayLengthField_)
    , sizeOfVectorLengthField(sizeOfVecotrLengthField_)
    , sizeOfStringLengthField(sizeOfStringLengthField_)
    , sizeOfStructLengthField(sizeOfStructLengthField_)
    , sizeOfUnionLengthField(sizeOfUnionLengthField_)
    , sizeOfUnionTypeSelectorField(sizeOfUnionTypeSelectorField_)
    , stringEncoding(stringEncoding_)
    {

    }

    SomeipTransformationProps(DataAlignment alignment_,
                   ByteOrderEnum  byteOrder_,
                   bool implementsLegacyString_,
                   bool isDynamicLengthFieldSize_,
                   bool isSessionHandlingActive_,
                   LengthFieldSize sizeOfArrayLengthField_,
                   LengthFieldSize sizeOfStringLengthField_,
                   LengthFieldSize sizeOfStructLengthField_,
                   LengthFieldSize sizeOfUnionLengthField_,
                   TypeSelectorFieldSize sizeOfUnionTypeSelectorField_,
                   StringEncoding stringEncoding_)
    : alignment(alignment_)
    , byteOrder(byteOrder_)
    , implementsLegacyString(implementsLegacyString_)
    , isDynamicLengthFieldSize(isDynamicLengthFieldSize_)
    , isSessionHandlingActive(isSessionHandlingActive_)
    , sizeOfArrayLengthField(sizeOfArrayLengthField_)
    , sizeOfVectorLengthField(sizeOfArrayLengthField_)
    , sizeOfStringLengthField(sizeOfStringLengthField_)
    , sizeOfStructLengthField(sizeOfStructLengthField_)
    , sizeOfUnionLengthField(sizeOfUnionLengthField_)
    , sizeOfUnionTypeSelectorField(sizeOfUnionTypeSelectorField_)
    , stringEncoding(stringEncoding_)
    {

    }

};

}  // namespace com
}  // namespace ara

#endif  // TRANSFORMATION_PROPS_H_
/* EOF */
