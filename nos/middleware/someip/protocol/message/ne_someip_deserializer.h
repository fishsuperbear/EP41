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
#ifndef MESSAGE_NE_SOMEIP_DESERIALIZER_H
#define MESSAGE_NE_SOMEIP_DESERIALIZER_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include "ne_someip_define.h"

bool ne_someip_deser_uint8(uint8_t** data, uint8_t* value);
bool ne_someip_deser_uint16(uint8_t** data, uint16_t* value);
bool ne_someip_deser_uint32(uint8_t** data, uint32_t* value, uint8_t omit_last_byte);
bool ne_someip_deser_data_with_len(uint8_t** data, uint8_t* value, uint32_t length);

#ifdef __cplusplus
}
#endif
#endif // MESSAGE_NE_SOMEIP_DESERIALIZER_H
/* EOF */