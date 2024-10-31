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

#ifndef INCLUDE_NE_SOMEIP_DESERIALIZER_GENERAL_H
#define INCLUDE_NE_SOMEIP_DESERIALIZER_GENERAL_H

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
     * @brief This class contains the API that can be used to deserialization of message payload.
     *
     * This class contains the API that can be used to deserialization of message payload.
     */
    class NESomeIPPayloadDeserializerGeneral {
    public:
        NESomeIPPayloadDeserializerGeneral(const NESomeIPExtendAttr* attr);
        virtual ~NESomeIPPayloadDeserializerGeneral();

        /**
         * @brief Set union data begin.
         *
         * @param [in] attr attribution, can be NULL.
         * @param [out] type date type.
         *
         * @return success or failed
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode union_data_begin(const NESomeIPExtendAttr* attr, uint32_t& type);

        /**
         * @brief Set tlv data begin.
         *
         * @param [in] attr attribution, can be NULL.
         * @param [out] data_id tlv date id.
         *
         * @return success or failed
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode tlv_data_begin(const NESomeIPExtendAttr* attr, uint32_t& data_id);

        /**
         * @brief Set array data begin.
         *
         * @param [in] attr attribution, can be NULL.
         *
         * @return success or failed
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode array_data_begin(const NESomeIPExtendAttr* attr);

        /**
         * @brief Set vector data begin.
         *
         * @param [in] attr attribution, can be NULL.
         *
         * @return success or failed
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode vector_data_begin(const NESomeIPExtendAttr* attr);

        /**
         * @brief Set struct data begin.
         *
         * @param [in] attr attribution, can be NULL.
         *
         * @return success or failed
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
         * @brief Set deserialization data.
         *
         * @param [in] buffer Point to buffer restore serialization data.
         * @param [in] size Buffer size of serialization data.
         *
         * @return success or failed
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode from_buffer(const uint8_t* buffer, uint32_t size);

        /**
         * @brief Set deserialization data.
         *
         * @param [in] buffer buffer vector to buffer restore serialization data.
         *
         * @return success or failed
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode from_buffer(const std::vector<uint8_t>& buffer);

        /**
         * @brief Set deserialization data.
         *
         * @param [in] buffer buffer vector to buffer restore serialization data.
         *
         * @return success or failed
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode from_buffer(std::vector<uint8_t>&& buffer);

        /**
         * @brief Check buffer is all read.
         *
         * @return The buffer is empty or not.
         *
         * @attention Synchronous I/F.
         */
        bool is_empty();

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to bool data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(bool& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to char data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(char& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to unsigned integer 8-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(uint8_t& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to unsigned integer 16-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(uint16_t& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to unsigned integer 32-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(uint32_t& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to unsigned integer 64-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(uint64_t& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to signed integer 8-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(int8_t& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to signed integer 16-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(int16_t& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to signed integer 32-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(int32_t& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to signed integer 64-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(int64_t& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to float 32-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(float& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for basic datatype.
         *
         * @param [out] value Reference to float 64-bit data.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(double& value, const NESomeIPExtendAttr* attr);

        /**
         * @brief Serialization for enum datatype.
         *
         * @param [out] value enum data.
         * @param [in] size the number of bytes occupied by th enum.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, NESomeIPPayloadErrorCode, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read_enum(uint32_t& value, uint32_t size, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for std::string datatype.
         *
         * @param [out] data std::string data pointer.
         * @param [in] size std::string data length.
         * @param [in] attr attribution, can be NULL.
         * @param [in] is_dynamic true: Strings with dynamic length; false: Strings with fixed length.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
        */
        NESomeIPPayloadErrorCode read(std::string& data, uint32_t size, const NESomeIPExtendAttr* attr, bool is_dynamic = false);

        /**
         * @brief Deserialization for ara::core::String datatype.
         *
         * @param [out] data ara::core::String data pointer.
         * @param [in] size ara::core::String data length.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
        */
#if 0
        NESomeIPPayloadErrorCode read(ara::core::String& data, uint32_t size, const NESomeIPExtendAttr* attr);
#endif

        /**
         * @brief Deserialization for array datatype.
         *
         * @param [out] data Array data pointer.
         * @param [out] size Array data size, if the data is dynamic, the size is out param, else is in param
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention User must NOT free data pointer.
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(uint8_t** data, uint32_t* size, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for array datatype.
         *
         * @param [out] data Array data pointer.
         * @param [out] size Array data size, if the data is dynamic, the size is out param, else is in param
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention User must NOT free data pointer.
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(int8_t** data, uint32_t* size, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for array datatype.
         *
         * @param [out] data Array data pointer.
         * @param [in] size data size
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(uint8_t* data, uint32_t size, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for array datatype.
         *
         * @param [out] data array data vector<uint8_t>.
         * @param [in] size Array data Size, the data length is fixed
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(std::vector<uint8_t>& data, uint32_t size, const NESomeIPExtendAttr* attr);

        /**
         * @brief Deserialization for std::vector<int8_t> datatype.
         *
         * @param [out] data the data of std::vector<int8_t>.
         * @param [in] size the data length is fixed.
         * @param [in] attr attribution, can be NULL.
         *
         * @return The result of api made, success or fail.
         *
         * @attention Synchronous I/F.
         */
        NESomeIPPayloadErrorCode read(std::vector<int8_t>& data, uint32_t size, const NESomeIPExtendAttr* attr);

        /**
         * @brief set the error code, if the error code is set ,then the deserilization will be stopped.
         *
         * @param [in] err_code error code
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

        // for ets test
        uint32_t get_length_field_value();

    private:
        bool clear();
        NESomeIPPayloadErrorCode calc_size(const NESomeIPPayloadAlignment alignment, const uint32_t begin_index);
        NESomeIPPayloadErrorCode read_payload(void *value, uint32_t size);
        NESomeIPPayloadErrorCode read_type_field(const uint32_t length, uint32_t *type);
        NESomeIPPayloadErrorCode read_data(const uint32_t length, uint32_t *size);
        NESomeIPPayloadErrorCode read_length_field(const uint32_t length, uint32_t *size);
        bool prepare_or_check(uint32_t element_type);
        NESomeIPInternalAttr get_write_attr(const NESomeIPExtendAttr* attr);
        NESomeIPExtendAttr get_begin_extend_attr(const NESomeIPExtendAttr* attr);
        NESomeIPPayloadErrorCode check_length_field();
        NESomeIPPayloadErrorCode modify_length_field_read(uint32_t size);
        NESomeIPPayloadLengthFieldSize get_length_field_size(const NESomeIPInternalAttr* attr);
        uint32_t get_begin_index();
        NESomeIPPayloadErrorCode read_string(std::string& data, uint32_t size, NESomeIPInternalAttr& write_attr, bool is_dynamic);

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

        std::vector<uint8_t> m_buffer;
        uint32_t m_size;
        uint32_t m_index;
        bool m_need_padding;    // m_attr_list为NULL时使用
        NESomeIPPayloadErrorCode m_error_code;
        NESomeIPPayloadStructWireType m_struct_wire_type;
        std::vector<NESomeIPInternalAttr> m_attr_list;
    };

#endif  // INCLUDE_NE_SOMEIP_DESERIALIZER_GENERAL_H
/* EOF */

