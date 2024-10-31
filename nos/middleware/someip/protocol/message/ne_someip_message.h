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
#ifndef MESSAGE_NE_SOMEIP_MESSAGE_H
#define MESSAGE_NE_SOMEIP_MESSAGE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include "ne_someip_define.h"
#include "ne_someip_list.h"
#include "ne_someip_internal_define.h"

bool ne_someip_msg_ser(ne_someip_list_t* buffer_list, const ne_someip_header_t* header,
	const ne_someip_payload_t* payload);
bool ne_someip_msg_header_ser(uint8_t** data, const ne_someip_header_t* message);
// bool ne_someip_msg_header_ser_new(uint8_t* data, uint32_t* length, const ne_someip_header_t* message);
bool ne_someip_msg_deser(const ne_someip_list_t* buffer_list, ne_someip_header_t* header,
	ne_someip_payload_t* payload);
bool ne_someip_msg_header_deser(const uint8_t** data, ne_someip_header_t* message);
bool ne_someip_msg_header_deser_new(uint8_t* data, uint32_t length, ne_someip_header_t* message);
bool ne_someip_msg_len_deser(const uint8_t* data, ne_someip_message_length_t* length);
bool ne_someip_msg_methodid_deser(const uint8_t* data, ne_someip_method_id_t* method_id);
bool ne_someip_msg_serviceid_deser(const uint8_t* data, ne_someip_service_id_t* service_id);
bool ne_someip_msg_type_deser(const uint8_t* data, ne_someip_message_type_t* message_type);

#ifdef __cplusplus
}
#endif
#endif // MESSAGE_NE_SOMEIP_MESSAGE_H
/* EOF */