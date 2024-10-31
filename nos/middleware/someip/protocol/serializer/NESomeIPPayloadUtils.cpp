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

#include <arpa/inet.h>
#include "NESomeIPPayloadUtils.h"
#include "ne_someip_define.h"

namespace {
    uint16_t
    do_nothing(uint16_t s) {
        return s;
    }

    uint32_t
    do_nothing(uint32_t l) {
        return l;
    }

    uint64_t
    do_nothing(uint64_t ll) {
        return ll;
    }

    float
    do_nothing(float f) {
        return f;
    }

    double
    do_nothing(double d) {
        return d;
    }

    uint16_t
    reverse_byte_order(uint16_t value) {
        return (value & 0xFF) << 8 | (value & 0xFF00) >> 8;
    }

    uint32_t
    reverse_byte_order(uint32_t value) {
        return (value & 0xFF) << 24 | (value & 0xFF00) << 8
            | (value & 0xFF0000) >> 8 | (value & 0xFF000000) >> 24;
    }

    uint64_t
    reverse_byte_order(uint64_t value) {
        return (value & 0xFF) << 56 | (value & 0xFF00) << 40
            | (value & 0xFF0000) << 24 | (value & 0xFF000000) << 8
            | (value & 0xFF00000000) >> 8 | (value & 0xFF0000000000) >> 24
            | (value & 0xFF000000000000) >> 40 | (value & 0xFF00000000000000) >> 56;
    }

    float
    reverse_byte_order(float value) {
        float retVal;
        char *floatToConvert = reinterpret_cast<char *>(&value);
        char *returnFloat = reinterpret_cast<char *>(&retVal);

        // swap the bytes into a temporary buffer
        int size = sizeof(value);
        for (int i = 0; i < size; ++i) {
            returnFloat[i] = floatToConvert[size - 1 - i];
        }

        return retVal;
    }

    double
    reverse_byte_order(double value) {
        double retVal;
        char *doubleToConvert = reinterpret_cast<char *>(&value);
        char *returnDouble = reinterpret_cast<char *>(&retVal);

        // swap the bytes into a temporary buffer
        int size = sizeof(value);
        for (int i = 0; i < size; ++i) {
            returnDouble[i] = doubleToConvert[size - 1 - i];
        }

        return retVal;
    }
}  // namespace

    NESomeIPPayloadUtilsByteOrder::NESomeIPPayloadUtilsByteOrder()
    : headers_short(do_nothing)
    , headers_long(do_nothing)
    , convert_short(do_nothing)
    , convert_long(do_nothing)
    , convert_longlong(do_nothing)
    , convert_float(do_nothing)
    , convert_double(do_nothing)
    , m_host_byte_order((1 == htonl(1)) ? NESomeIPPayloadByteOrder_BE : NESomeIPPayloadByteOrder_LE)
    , m_network_byte_order() {
        set_network_byte_order(NESomeIPPayloadByteOrder_BE);
    }

    NESomeIPPayloadErrorCode
    NESomeIPPayloadUtilsByteOrder::set_network_byte_order(NESomeIPPayloadByteOrder value) {
        if (m_host_byte_order == NESomeIPPayloadByteOrder_LE) {
            headers_short = reverse_byte_order;
            headers_long = reverse_byte_order;
        } else {
            headers_short = do_nothing;
            headers_long = do_nothing;
        }

        if (m_host_byte_order != value) {
            convert_short = reverse_byte_order;
            convert_long = reverse_byte_order;
            convert_longlong = reverse_byte_order;
            convert_float = reverse_byte_order;
            convert_double = reverse_byte_order;
        } else {
            convert_short = do_nothing;
            convert_long = do_nothing;
            convert_longlong = do_nothing;
            convert_float = do_nothing;
            convert_double = do_nothing;
        }
        m_network_byte_order = value;
        return NESomeIPPayloadErrorCode_Ok;
    }

    NESomeIPPayloadByteOrder
    NESomeIPPayloadUtilsByteOrder::get_host_byte_order() {
        return m_host_byte_order;
    }

    NESomeIPPayloadByteOrder
    NESomeIPPayloadUtilsByteOrder::get_network_byte_order() {
        return m_network_byte_order;
    }

/* EOF */
