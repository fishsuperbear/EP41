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
#include <stddef.h>
#include "ne_someip_serializer.h"
#include "ne_someip_log.h"

bool ne_someip_ser_uint8(uint8_t** data, uint8_t value)
{
	if (NULL == data) {
        return false;
    }
    *((*data)++) = value;
    return true;
}

bool ne_someip_ser_uint16(uint8_t** data, uint16_t value)
{
	if (NULL == data) {
        return false;
    }
    *((*data)++) = value >> 8;
    *((*data)++) = value & 0xFF;
    return true;
}

bool ne_someip_ser_uint32(uint8_t** data, uint32_t value, uint8_t omit_last_byte)
{
	if (NULL == data) {
        return false;
    }

    if (!omit_last_byte) {
        *((*data)++) = (value >> 24) & 0xFF;
    } else {
        (*data)++;
    }
    *((*data)++) = (value >> 16) & 0xFF;
    *((*data)++) = (value >> 8) & 0xFF;
    *((*data)++) = value & 0xFF;

    return true;
}

bool ne_someip_ser_data_with_len(uint8_t** data, const uint8_t *value, uint32_t length)
{
	if (NULL == data) {
        return false;
    }
    memcpy(*data, value, length);
    *data = *data + length;
    return true;
}

