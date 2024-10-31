/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file NCTypesSerializer.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTYPESSERIALIZER_H_
#define INCLUDE_NCORE_NCTYPESSERIALIZER_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <string>
#include <vector>

#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE
// Define enum for message payload byte order
enum NCTypesByteOrder : UINT32 {
    NCTypesByteOrder_First = 0x00U,  // big endian
    NCTypesByteOrder_Last  = 0x01U   // little endian
};

// Define enum for message payload alignment
enum NCTypesAlignment : UINT32 {
    NCTypesAlignment_8  = 0x01U,
    NCTypesAlignment_16 = 0x02U,
    NCTypesAlignment_32 = 0x04U,
    NCTypesAlignment_64 = 0x08U
};

// Define enum for message payload length field size of variable data
enum NCTypesLengthFieldSize : UINT32 {
    NCTypesLengthFieldSize_0  = 0x00U,
    NCTypesLengthFieldSize_8  = 0x01U,
    NCTypesLengthFieldSize_16 = 0x02U,
    NCTypesLengthFieldSize_32 = 0x04U
};

// Define enum for message payload type field size of union
enum NCTypesUnionTypeFieldSize : UINT32 {
    NCTypesUnionTypeFieldSize_0  = 0x00U,
    NCTypesUnionTypeFieldSize_8  = 0x01U,
    NCTypesUnionTypeFieldSize_16 = 0x02U,
    NCTypesUnionTypeFieldSize_32 = 0x04U
};

// Define enum for struct wire type in tag
enum class NCTypesStructWireType : UINT32 {
    NCTypesStructWireType_8BitBase     = 0x00U,
    NCTypesStructWireType_16BitBase    = 0x01U << 12U,
    NCTypesStructWireType_32BitBase    = 0x02U << 12U,
    NCTypesStructWireType_64BitBase    = 0x03U << 12U,
    NCTypesStructWireType_nByteComplex = 0x04U << 12U,
    NCTypesStructWireType_1ByteComplex = 0x05U << 12U,
    NCTypesStructWireType_2ByteComplex = 0x06U << 12U,
    NCTypesStructWireType_4ByteComplex = 0x07U << 12U
};

class NCTypesArraySerializer;
class NCTypesStructSerializer;
class NCTypesUnionSerializer;
class NCTypesSerializerUtilsByteOrder;

/**
 * @brief This class contains the API that can be used to serialization
 *                                        of message payload.
 *
 * This class contains the API that can be used to serialization
 *                                        of message payload.
 */
class NCTypesSerializer {
   public:
    NCTypesSerializer();
    virtual ~NCTypesSerializer();

    /**
     * @brief Set payload byte order.
     *
     * @param [in] value Byte order, big endian or little endian.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL set_byte_order( NCTypesByteOrder value );

    /**
     * @brief Set payload alignment.
     *
     * @param [in] value Alignment of memory.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL set_alignment( NCTypesAlignment value );

    /**
     * @brief Set payload length field size.
     *
     * @param [in] value Size of length field for variable data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL set_length_field_size( NCTypesLengthFieldSize value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Bool data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( NC_BOOL value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Unsigned integer 8-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( UINT8 value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Unsigned integer 16-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( UINT16 value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Unsigned integer 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( UINT32 value );

    /**
     * @brief Serialization for basic datatype.

     * @param [in] value Unsigned integer 64-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( UINT64 value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Signed integer 8-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( INT8 value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Signed integer 16-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( INT16 value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Signed integer 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( INT32 value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Signed integer 64-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( INT64 value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Float 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( FLOAT value );

    /**
     * @brief Serialization for basic datatype.
     *
     * @param [in] value Float 64-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_basic( DOUBLE value );

    /**
     * @brief Serialization for struct datatype.
     *
     * @param [in] serializer Struct data serializer pointer.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_struct( NCTypesStructSerializer *ser );

    /**
     * @brief Serialization for string utf8 datatype.
     *
     * @param [in] value String data pointer.
     * @param [in] length String data length.
     * @param [in] is_variable String data is variable or not.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_string( const CHAR *const value, UINT32 length, NC_BOOL is_variable );

    /**
     * @brief Serialization for string utf16 datatype.
     *
     * @param [in] value String data pointer.
     * @param [in] length String data length.
     * @param [in] is_variable String data is variable or not.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_string( const CHAR16 *const value, UINT32 length, NC_BOOL is_variable );

    /**
     * @brief Serialization for array datatype.
     *
     * @param [in] serializer Array data serializer pointer.
     * @param [in] is_variable Array data is variable or not.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_array( NCTypesArraySerializer *ser, NC_BOOL is_variable );

    /**
     * @brief Serialization for enumeration datatype.
     *
     * @param [in] value Unsigned integer 8-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_enum( UINT8 value );

    /**
     * @brief Serialization for enumeration datatype.
     *
     * @param [in] value Unsigned integer 16-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_enum( UINT16 value );

    /**
     * @brief Serialization for enumeration datatype.
     *
     * @param [in] value Unsigned integer 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_enum( UINT32 value );

    /**
     * @brief Serialization for bit field datatype.
     *
     * @param [in] value Unsigned integer 8-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_bits( UINT8 value );

    /**
     * @brief Serialization for bit field datatype.
     *
     * @param [in] value Unsigned integer 16-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_bits( UINT16 value );

    /**
     * @brief Serialization for bit field datatype.
     *
     * @param [in] value Unsigned integer 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_bits( UINT32 value );

    /**
     * @brief Serialization for union datatype.
     *
     * @param [in] serializer Union data serializer pointer.
     * @param [in] length Size of union data.
     * @param [in] type Index defined by user.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_union( NCTypesUnionSerializer *ser, UINT32 length, UINT32 type_index );

    /**
     * @brief Get serialization data size.
     *
     * @return The buffer size of serialization data.
     *
     * @attention Synchronous I/F.
     */

    UINT32 get_buffer_size() const;

    /**
     * @brief Get serialization data size.
     *
     * @param [out] buffer Point to buffer restore serialization data,
     *              size must be more than get_buffer_size().
     *
     * @return The buffer size of serialization data.
     *
     * @attention Synchronous I/F.
     */
    UINT32 to_buffer( UINT8 *buffer );

   protected:
    virtual NC_BOOL clear();
    NC_BOOL         write_length_field( UINT32 length );
    NC_BOOL         write_payload( const VOID *const value, UINT32 size );

   protected:
    NCTypesSerializerUtilsByteOrder *m_byte_order;

   private:
    virtual UINT32  calc_size( UINT32 size, NC_BOOL is_variable );
    virtual NC_BOOL prepare_or_check( UINT32 element_type );
    NC_BOOL         write_padding( UINT32 size, CHAR padding_value );
    NC_BOOL         write_type_field( UINT32 type_field );

   private:
    std::vector<std::string> m_list;
    UINT32                   m_size;
    UINT32                   m_alignment;
    UINT32                   m_length_field_size;
    NC_BOOL                  m_need_padding;
};

/**
 * @brief This class contains the API that can be used to
 *        serialization of array extension.
 *
 * This class contains the API that can be used to
 *        serialization of array extension.
 */
class NCTypesArraySerializer : public NCTypesSerializer {
   public:
    /**
     * @brief This class contains the API that can be used to
     *        serialization of array.
     *
     * This class contains the API that can be used to serialization of array.
     */
    NCTypesArraySerializer();

   private:
    virtual NC_BOOL clear();
    virtual NC_BOOL prepare_or_check( UINT32 element_type );

   private:
    UINT32 m_element_type;
};

/**
 * @brief This class contains the API that can be used to
 *        serialization of struct extension.
 *
 * This class contains the API that can be used to serialization of
 *        struct extension.
 */
class NCTypesStructSerializer : public NCTypesSerializer {
   public:
    /**
     * @brief This class contains the API that can be used to
     *        serialization of struct.
     *
     * This class contains the API that can be used to
     *        serialization of struct.
     */
    NCTypesStructSerializer();

    /**
     * @brief Set payload structured datatype and arguments with
     *        identifier and optional members.
     *
     * @param [in] value Flag of optional members or not.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL set_optional_member_flag( const NC_BOOL value );

    /**
     * @brief Set payload structured datatype tag layout.
     *
     * @param [in] wire_type Wire type of member in 3 bits.
     * @param [in] data_id Data id defined by user in 12 bits.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL write_data_id( NCTypesStructWireType wire_type, UINT16 data_id );

   private:
    virtual UINT32 calc_size( UINT32 size, NC_BOOL is_variable );

   private:
    NC_BOOL               m_optional_member_flag;
    NCTypesStructWireType m_wire_type;
};

/**
 * @brief This class contains the API that can be used to
 *        serialization of union extension.
 *
 * This class contains the API that can be used to serialization of
 *        union extension.
 */
class NCTypesUnionSerializer : public NCTypesSerializer {
   public:
    /**
     * @brief This class contains the API that can be used to
     *             serialization of union.
     *
     * This class contains the API that can be used to serialization of union.
     */
    NCTypesUnionSerializer();

   private:
    virtual NC_BOOL clear();
    virtual UINT32  calc_size( UINT32 size, NC_BOOL is_variable );
    virtual NC_BOOL prepare_or_check( UINT32 element_type );

   private:
    UINT32 m_element_type;
};

class NCTypesArrayDeserializer;
class NCTypesStructDeserializer;
class NCTypesUnionDeserializer;
class NCTypesSerializerUtilsByteOrder;

/**
 * @brief This class contains the API that can be used to
 *        deserialization of message payload.
 *
 * This class contains the API that can be used to deserialization of
 *        message payload.
 */
class NCTypesDeserializer {
   public:
    NCTypesDeserializer();
    virtual ~NCTypesDeserializer();

    /**
     * @brief Set payload byte order.
     *
     * @param [in] value Byte order, big endian or little endian.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL set_byte_order( NCTypesByteOrder value );

    /**
     * @brief Set payload alignment.
     *
     * @param [in] value Alignment of memory.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL set_alignment( NCTypesAlignment value );

    /**
     * @brief Set payload length field size.
     *
     * @param [in] value Size of length field for variable data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL set_length_field_size( NCTypesLengthFieldSize value );

    /**
     * @brief Set serialization data.
     *
     * @param [in] buffer Point to buffer restore serialization data.
     * @param [in] size Buffer size of serialization data.
     *
     * @return The buffer size of serialization data.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL from_buffer( const UINT8 *const buffer, UINT32 size );

    /**
     * @brief Check buffer is all read.
     *
     * @return The buffer is empty or not.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL is_empty() const;

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to NC_BOOL data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( NC_BOOL *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to unsigned integer 8-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( UINT8 *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to unsigned integer 16-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( UINT16 *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to unsigned integer 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( UINT32 *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to unsigned integer 64-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( UINT64 *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to signed integer 8-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( INT8 *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to signed integer 16-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( INT16 *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to signed integer 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( INT32 *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to signed integer 64-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( INT64 *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to float 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( FLOAT *const value );

    /**
     * @brief Deserialization for basic datatype.
     *
     * @param [out] value Reference to float 64-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_basic( DOUBLE *const value );

    /**
     * @brief Deserialization for struct datatype.
     *
     * @param [out] serializer Struct data serializer pointer.
     * @param [in] size Struct serialization data size.
     * @param [in] with_tag Serialization of struct data with tag or not.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_struct( NCTypesStructDeserializer *const ser, UINT32 size );
    /**
     * @brief Check string bom is utf8.
     *
     * @return The string bom is utf8 or not.
     *
     * @attention Synchronous I/F.
     */

    NC_BOOL string_bom_is_utf8() const;
    /**
     * @brief Deserialization for string utf8 datatype.
     *
     * @param [out] value String data pointer.
     * @param [in] size String data length.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_string( CHAR *const value, UINT32 size );

    /**
     * @brief Deserialization for string utf16 datatype.
     *
     * @param [out] value String data pointer.
     * @param [in] size String data length.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_string( CHAR16 *const value, UINT32 size );

    /**
     * @brief Deserialization for array datatype.
     *
     * @param [out] serializer Array data serializer pointer.
     * @param [in] size Array data Size.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_array( NCTypesArrayDeserializer *const ser, UINT32 size );

    /**
     * @brief Deserialization for enumeration datatype.
     *
     * @param [out] value Reference to unsigned integer 8-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_enum( UINT8 *const value );

    /**
     * @brief Deserialization for enumeration datatype.
     *
     * @param [out] value Reference to unsigned integer 16-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_enum( UINT16 *const value );

    /**
     * @brief Deserialization for enumeration datatype.
     *
     * @param [out] value Reference to unsigned integer 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_enum( UINT32 *const value );

    /**
     * @brief Deserialization for bit field datatype.
     *
     * @param [out] value Reference to unsigned integer 8-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_bits( UINT8 *const value );

    /**
     * @brief Deserialization for bit field datatype.
     *
     * @param [out] value Reference to unsigned integer 16-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_bits( UINT16 *const value );

    /**
     * @brief Deserialization for bit field datatype.
     *
     * @param [out] value Reference to unsigned integer 32-bit data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_bits( UINT32 *const value );

    /**
     * @brief Deserialization for union datatype.
     *
     * @param [out] serializer Union data serializer pointer.
     * @param [in] size Size of union data.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_union( NCTypesUnionDeserializer *const ser, UINT32 size );

    /**
     * @brief Get serialization data length field value.
     *
     * @param [out] value Reference to variable data length.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    virtual NC_BOOL get_length_field( UINT32 *const length );

    /**
     * @brief Get serialization data type for union data.
     *
     * @param [out] value Reference to union type defined by user.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL get_type_field( UINT32 *const data_type );

   protected:
    virtual NC_BOOL clear();
    UINT32          calc_size( UINT32 size, NC_BOOL is_variable );
    NC_BOOL         read_payload( VOID *const value, UINT32 size );
    NC_BOOL         read_length_field( UINT32 *const length );

   protected:
    NCTypesSerializerUtilsByteOrder *m_byte_order;
    UINT32                           m_length_field_size;
    NC_BOOL                          m_reading_variable;

   private:
    virtual NC_BOOL prepare_or_check( UINT32 element_type );

   private:
    UINT8 * m_buffer;
    UINT32  m_index;
    UINT32  m_size;
    UINT32  m_alignment;
    NC_BOOL m_need_padding;
};

/**
 * @brief This class contains the API that can be used to deserialization of
 *        array extension.
 *
 * This class contains the API that can be used to deserialization of
 *        array extension.
 */
class NCTypesArrayDeserializer : public NCTypesDeserializer {
   public:
    NCTypesArrayDeserializer();

   private:
    virtual NC_BOOL clear();
    virtual NC_BOOL prepare_or_check( UINT32 element_type );

   private:
    UINT32 m_element_type;
};

/**
 * @brief This class contains the API that can be used to deserialization of
 *        struct extension.
 *
 * This class contains the API that can be used to deserialization of
 *        struct extension.
 */
class NCTypesStructDeserializer : public NCTypesDeserializer {
   public:
    /**
     * @brief This class contains the API that can be used to
     *        serialization of struct.
     *
     * This class contains the API that can be used to
     *        serialization of struct.
     */
    NCTypesStructDeserializer();

    /**
     * @brief Set payload structured datatype and arguments with
     *      identifier and optional members.
     *
     * @param [in] value Flag of optional members or not.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL set_optional_member_flag( const NC_BOOL value );

    /**
     * @brief Set payload structured datatype tag layout.
     *
     * @param [out] wire_type Wire type of member in 3 bits.
     * @param [out] data_id Data id defined by user in 12 bits.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    NC_BOOL read_data_id( NCTypesStructWireType *const wire_type, UINT16 *const data_id );

    /**
     * @brief Get serialization data length field value.
     *
     * @param [out] value Reference to variable data length.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */
    virtual NC_BOOL get_length_field( UINT32 *const length );

   private:
    NC_BOOL               m_optional_member_flag;
    NCTypesStructWireType m_wire_type;
};

/**
 * @brief This class contains the API that can be used to
 *        deserialization of union extension.
 *
 * This class contains the API that can be used to
 *        deserialization of union extension.
 */
class NCTypesUnionDeserializer : public NCTypesDeserializer {
   public:
    NCTypesUnionDeserializer();

    /**
     * @brief Get serialization data length field value.
     *
     * @param [out] value Reference to variable data length.
     *
     * @return The result of api made, success or fail.
     *
     * @attention Synchronous I/F.
     */

    virtual NC_BOOL get_length_field( UINT32 *const length ) override;

   private:
    virtual NC_BOOL clear() override;
    virtual NC_BOOL prepare_or_check( UINT32 element_type ) override;

   private:
    UINT32 m_element_type;
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCTYPESSERIALIZER_H_
/* EOF */
