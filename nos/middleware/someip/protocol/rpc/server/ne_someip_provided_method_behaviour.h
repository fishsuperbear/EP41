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
#ifndef MANAGER_SERVER_NE_SOMEIP_PROVIDED_METHOD_BEHAVIOUR_H
#define MANAGER_SERVER_NE_SOMEIP_PROVIDED_METHOD_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_handler.h"
#include "ne_someip_config_define.h"
#include "ne_someip_endpoint_define.h"

typedef struct ne_someip_provided_method_behaviour ne_someip_provided_method_t;

ne_someip_provided_method_t*
ne_someip_provided_method_create(const ne_someip_method_config_t* config);

ne_someip_provided_method_t*
ne_someip_provided_method_ref(ne_someip_provided_method_t* method_behaviour);

void ne_someip_provided_method_unref(ne_someip_provided_method_t* method_behaviour);

ne_someip_error_code_t
ne_someip_provided_method_send(ne_someip_provided_method_t* method_behaviour,
	ne_someip_header_t* header, ne_someip_payload_t* payload,
	ne_someip_remote_client_info_t* remote_addr, void* endpoint, void* seq_id,
	ne_someip_endpoint_send_policy_t send_policy);

// void ne_someip_provided_method_recv_req(ne_someip_provided_method_t* method_behaviour,
// 	const ne_someip_message_t* message, const ne_someip_remote_client_info_t* remote_addr);

// ne_someip_error_code_t ne_someip_provided_method_reg_method_handler(ne_someip_provided_method_t* method_behaviour,
// 	ne_someip_recv_request_handler handler);

// ne_someip_error_code_t ne_someip_provided_method_unreg_method_handler(ne_someip_provided_method_t* method_behaviour);

void ne_someip_provided_method_reset(ne_someip_provided_method_t* method_behaviour);

bool ne_someip_provided_method_set_udp_collection_time_out(ne_someip_provided_method_t* method_behaviour,
	uint32_t time);

bool ne_someip_provided_method_set_udp_trigger_mode(ne_someip_provided_method_t* method_behaviour,
	ne_someip_udp_collection_trigger_t trigger);

ne_someip_error_code_t ne_someip_provided_method_set_request_permission(ne_someip_provided_method_t* method_behaviour,
	const ne_someip_remote_client_info_t* remote_addr, ne_someip_permission_t priority);

ne_someip_permission_t ne_someip_provided_method_get_request_permission(ne_someip_provided_method_t* method_behaviour,
	const ne_someip_remote_client_info_t* remote_addr);

/****************************************************get config*****************************************************/

uint32_t ne_someip_provided_method_get_udp_collection_time_out(const ne_someip_provided_method_t* method_behaviour);

ne_someip_udp_collection_trigger_t
ne_someip_provided_method_get_udp_trigger_mode(const ne_someip_provided_method_t* method_behaviour);

ne_someip_l4_protocol_t
ne_someip_provided_get_method_comm_type(const ne_someip_provided_method_t* method_behaviour);

uint32_t ne_someip_provided_method_get_segment_len(const ne_someip_provided_method_t* method_behaviour);
uint32_t ne_someip_provided_method_get_separation_time(const ne_someip_provided_method_t* method_behaviour);

bool ne_someip_provided_method_get_message_type(const ne_someip_provided_method_t* method_behaviour);

#ifdef __cplusplus
}
#endif
#endif // MANAGER_SERVER_NE_SOMEIP_PROVIDED_METHOD_BEHAVIOUR_H
/* EOF */