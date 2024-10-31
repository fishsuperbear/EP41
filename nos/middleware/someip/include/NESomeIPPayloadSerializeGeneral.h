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

#ifndef INCLUDE_NE_SOMEIP_SERIALIZER_GENERAL_H
#define INCLUDE_NE_SOMEIP_SERIALIZER_GENERAL_H

#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <stdint.h>
#if 0
#include <ara/core/string.h>
#endif
#include <vector>
#include <array>
#include "NESomeIPSerializerDefine.h"

    class NESomeIPPayloadUtilsByteOrder;

    /**
     * @brief This class contains the API that can be used to serialization of message payload.
     *
     * This class contains the API that can be used to serialization of message payload.
     */
    class NESomeIPPayloadSerializerGeneral {
    public:
        NESomeIPPayloadSerializerGeneral(const NESomeIPExtendAttr* attr);
        virtual ~NESomeIPPayloadSerializerGeneral();

        /**
         * @brief Set the union data begin.
         *
         * @param [in] attr attribution, can be NULL.
         * @param [in] type date type.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode union_data_begin(const NESomeIPExtendAttr* attr, uint32_t type);

        /**
         * @brief Set the tlv data begin.
         *
         * @param [in] attr attribution, can be NULL.
         * @param [in] data_id tlv date id.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode tlv_data_begin(const NESomeIPExtendAttr* attr, uint32_t data_id);

        /**
         * @brief Set the array data begin.
         *
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode array_data_begin(const NESomeIPExtendAttr* attr);

        /**
         * @brief Set the vector data begin.
         *
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode vector_data_begin(const NESomeIPExtendAttr* attr);

        /**
         * @brief Set the struct data begin.
         *
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode struct_data_begin(const NESomeIPExtendAttr* attr);

        /**
         * @brief Set the data end.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode data_end();

        /**
         * @brief buffer serialization data.
         *
         * @return The serialization data and clear the all attribution.
         *
         * @attention Synchronous I/F.
         */
        std::vector<uint8_t>&& to_buffer();

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Bool data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(bool value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Unsigned integer 8-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(uint8_t value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Unsigned integer 16-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(uint16_t value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Unsigned integer 32-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(uint32_t value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.

         * @param [in] value Unsigned integer 64-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(uint64_t value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Signed integer 8-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(int8_t value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Signed integer 16-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(int16_t value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Signed integer 32-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(int32_t value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Signed integer 64-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(int64_t value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Float 32-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(float value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for basic datatype.
         *
         * @param [in] value Float 64-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(double value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for enum datatype.
         *
         * @param [in] value enum data.
         * @param [in] size the number of bytes occupied by th enum.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write_enum(uint32_t value, uint32_t size, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for std::string datatype.
         *
         * @param [in] data the data of std::string.
         * @param [in] attr attribution, can be NULL.
         * @param [in] is_dynamic true: Strings with dynamic length; false: Strings with fixed length.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(const std::string& data, const NESomeIPExtendAttr* attr, bool is_dynamic = false);

        /**
         * @brief Serialization for ara::core::String datatype.
         *
         * @param [in] data the data of ara::core::String.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
#if 0
        NESomeIPPayloadErrorCode write(const ara::core::String& data, const NESomeIPExtendAttr* attr);
#endif

        /**
         * @brief Serialization for array datatype.
         *
         * @param [in] data Array data pointer.
         * @param [in] size Array data size.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(const uint8_t* data, uint32_t size, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for array datatype.
         *
         * @param [in] data Array data pointer.
         * @param [in] size Array data size.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(const int8_t* data, uint32_t size, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for std::vector<uint8_t> datatype.
         *
         * @param [in] data the data of std::vector<uint8_t>.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(const std::vector<uint8_t>& data, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for std::vector<int8_t> datatype.
         *
         * @param [in] data the data of std::vector<int8_t>.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode write(const std::vector<int8_t>& data, const NESomeIPExtendAttr* attr);

        /**
         * @brief set the error code, if the error code is set ,then the serilization will be stopped.
         *
         * @param [in] error code
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode set_error_code(NESomeIPPayloadErrorCode err_code);

        /**
         * @brief get the error code.
         *
         * @return The last error code.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode get_last_error();

    private:
        bool clear();
        NESomeIPPayloadErrorCode write_length_field(uint32_t length);
        void write_payload(const uint8_t *value, uint32_t size);
        void update_payload(uint32_t begin_index, const uint8_t *value, uint32_t size);
        NESomeIPPayloadErrorCode write_type_field(uint32_t length, uint32_t type);
        NESomeIPPayloadErrorCode write_data(uint32_t length, uint32_t value);
        void calc_size(const NESomeIPPayloadAlignment alignment, const uint32_t begin_index);
        bool prepare_or_check(uint32_t element_type);
        NESomeIPPayloadErrorCode write_union_padding(NESomeIPInternalAttr* write_attr);
        void write_padding(uint32_t size, uint8_t padding_value);
        NESomeIPInternalAttr get_write_attr(const NESomeIPExtendAttr* attr);
        NESomeIPExtendAttr get_begin_extend_attr(const NESomeIPExtendAttr* attr);
        uint32_t get_begin_index();
        NESomeIPPayloadErrorCode write_string(const std::string& data, const NESomeIPInternalAttr& write_attr, bool is_dynamic);

    private:
        NESomeIPPayloadLengthFieldSize m_array_length_field_size;
        NESomeIPPayloadLengthFieldSize m_vector_length_field_size;
        NESomeIPPayloadLengthFieldSize m_struct_length_field_size;
        NESomeIPPayloadLengthFieldSize m_string_length_field_size;
        NESomeIPPayloadLengthFieldSize m_union_length_field_size;
        bool m_is_tlv_dynamic_length_field_size;
        NESomeIPPayloadUtilsByteOrder *m_byte_order;
        NESomeIPPayloadAlignment m_alignment;
        NESomeIPPayloadUnionTypeFieldSize m_union_type_field_size;
        NESomeIPPayloadCharBit m_string_utf;

        uint32_t m_size;
        bool m_need_padding;    // m_attr_list为NULL时使用
        std::vector<uint8_t> m_list;
        NESomeIPPayloadErrorCode m_error_code;
        NESomeIPPayloadStructWireType m_struct_wire_type;
        std::vector<NESomeIPInternalAttr> m_attr_list;
    };

#endif  // INCLUDE_NE_SOMEIP_SERIALIZER_GENERAL_H
/* EOF */
