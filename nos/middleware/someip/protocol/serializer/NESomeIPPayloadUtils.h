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
#ifndef SERIALIZER_NESOMEIPPAYLOADUTILS_H_
#define SERIALIZER_NESOMEIPPAYLOADUTILS_H_

#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "NESomeIPSerializerDefine.h"

#ifndef UNUSED
#define UNUSED(expr) (void)expr
#endif

    const uint8_t BOOLEAN_BIT = 0b00000001;
    const uint8_t BOM_UTF8[3] = { 0xEF, 0xBB, 0xBF };
    const uint8_t BOM_UTF16BE[2] = { 0xFE, 0xFF };
    const uint8_t BOM_UTF16LE[2] = { 0xFF, 0xFE };

    enum NESomeIPPayloadElement {
        NESomeIPPayloadElement_Uint8,
        NESomeIPPayloadElement_Uint16,
        NESomeIPPayloadElement_Uint32,
        NESomeIPPayloadElement_Uint64,
        NESomeIPPayloadElement_Float,
        NESomeIPPayloadElement_Double,
        NESomeIPPayloadElement_Struct,
        NESomeIPPayloadElement_String,
        NESomeIPPayloadElement_Array,
        NESomeIPPayloadElement_Union,
        NESomeIPPayloadElement_Unknown
    };

    /**
     * @brief This class contains the utility that can be used to serialization of message payload.
     *
     * This class contains the utility that can be used to serialization of message payload.
     */
    class NESomeIPPayloadUtilsByteOrder {
    public:
        NESomeIPPayloadUtilsByteOrder();

        NESomeIPPayloadErrorCode set_network_byte_order(NESomeIPPayloadByteOrder value);
        NESomeIPPayloadByteOrder get_host_byte_order();
        NESomeIPPayloadByteOrder get_network_byte_order();

    public:
        uint16_t (*headers_short)(uint16_t s);
        uint32_t (*headers_long)(uint32_t l);
        uint16_t (*convert_short)(uint16_t s);
        uint32_t (*convert_long)(uint32_t l);
        uint64_t (*convert_longlong)(uint64_t ll);
        float (*convert_float)(float f);
        double (*convert_double)(double d);

    private:
        NESomeIPPayloadUtilsByteOrder(const NESomeIPPayloadUtilsByteOrder&);
        NESomeIPPayloadUtilsByteOrder& operator=(const NESomeIPPayloadUtilsByteOrder&);

    private:
        NESomeIPPayloadByteOrder m_host_byte_order;
        NESomeIPPayloadByteOrder m_network_byte_order;
    };

#endif  // SERIALIZER_NESOMEIPPAYLOADUTILS_H_
/* EOF */
