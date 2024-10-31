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
#include "ne_someip_deserializer.h"
#include <stddef.h>

bool ne_someip_deser_uint8(uint8_t** data, uint8_t* value)
{
	if (NULL == data || NULL == value) {
        return false;
    }
    *value = *((*data)++);
    return true;
}

bool ne_someip_deser_uint16(uint8_t** data, uint16_t* value)
{
	if (NULL == data || NULL == value) {
        return false;
    }
    uint8_t byte0 = *((*data)++);
    uint8_t byte1 = *((*data)++);
    *value = (byte0 << 8) | byte1;
    return true;
}

bool ne_someip_deser_uint32(uint8_t** data, uint32_t* value, uint8_t omit_last_byte)
{
	if (NULL == data || NULL == value) {
        return false;
    }
    uint8_t byte0 = 0;
    if (!omit_last_byte) {
       byte0 = *((*data)++);
    } else {
        (*data)++;
    }

    uint8_t byte1 = *((*data)++);
    uint8_t byte2 = *((*data)++);
    uint8_t byte3 = *((*data)++);
    *value = (byte0 << 24) | (byte1 << 16) | (byte2 << 8) | byte3;
    return true;
}

bool ne_someip_deser_data_with_len(uint8_t** data, uint8_t* value, uint32_t length)
{
	if (NULL == data || NULL == value) {
        return false;
    }
    memcpy(value, *data, length);
    *data = *data + length;
    return true;
}