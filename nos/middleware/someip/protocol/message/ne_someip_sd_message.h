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
#ifndef MESSAGE_NE_SOMEIP_SD_MESSAGE_H
#define MESSAGE_NE_SOMEIP_SD_MESSAGE_H

#ifdef __cplusplus
extern "C" {
#endif
#include "ne_someip_sd_define.h"

//define sd message serialize
bool ne_someip_sd_msg_ser(const ne_someip_sd_msg_t* in_data, uint8_t* out_data);

//define sd message deserialize
bool ne_someip_sd_msg_deser(ne_someip_sd_msg_t* out_data, const uint8_t* in_header_data, const uint8_t* in_sd_data,
    uint32_t header_length, uint32_t payload_length);

#ifdef __cplusplus
}
#endif
#endif  // MESSAGE_NE_SOMEIP_SD_MESSAGE_H
/* EOF */