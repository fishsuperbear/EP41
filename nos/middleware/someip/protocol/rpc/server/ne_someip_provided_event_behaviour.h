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
#ifndef MANAGER_SERVER_NE_SOMEIP_PROVIDED_EVENT_BEHAVIOUR_H
#define MANAGER_SERVER_NE_SOMEIP_PROVIDED_EVENT_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_define.h"
#include "ne_someip_config_define.h"
#include "ne_someip_endpoint_define.h"
#include "ne_someip_provided_eventgroup_behaviour.h"
#include "ne_someip_server_define.h"

typedef struct ne_someip_provided_event_behaviour ne_someip_provided_event_t;

ne_someip_provided_event_t*
ne_someip_provided_event_create(const ne_someip_event_config_t* config);

ne_someip_provided_event_t*
ne_someip_provided_event_ref(ne_someip_provided_event_t* event_behaviour);

void ne_someip_provided_event_unref(ne_someip_provided_event_t* event_behaviour);

// ne_someip_error_code_t ne_someip_provided_event_add_subscriber(ne_someip_provided_event_t* event_behaviour,
// 	char* ip_addr, uint16_t port);
// ne_someip_error_code_t ne_someip_provided_event_remove_subscriber(ne_someip_provided_event_t* event_behaviour,
// 	char* ip_addr, uint16_t port);
// ne_someip_error_code_t ne_someip_provided_event_change_to_multicast(ne_someip_provided_event_t* event_behaviour,
// 	char* ip_addr, uint16_t port);
// ne_someip_error_code_t ne_someip_provided_event_change_to_unicast(ne_someip_provided_event_t* event_behaviour,
// 	char* ip_addr, uint16_t port);
ne_someip_error_code_t
ne_someip_provided_event_send(ne_someip_provided_event_t* event_behaviour,
	ne_someip_header_t* header, ne_someip_payload_t* payload,
	void* endpoint, ne_someip_send_spec_t* seq_id, ne_someip_endpoint_send_policy_t send_policy);

void ne_someip_provided_event_reset(ne_someip_provided_event_t* event_behaviour);

bool ne_someip_provided_event_set_udp_collection_time_out(ne_someip_provided_event_t* event_behaviour,
	uint32_t time);

bool ne_someip_provided_event_set_udp_trigger_mode(ne_someip_provided_event_t* event_behaviour,
	ne_someip_udp_collection_trigger_t trigger);

bool ne_someip_provided_event_set_eg_ref(ne_someip_provided_event_t* event_behaviour,
	const ne_someip_provided_eg_t* eg_ref);

void ne_someip_provided_event_change_field_payload(ne_someip_provided_event_t* event_behaviour,
	ne_someip_payload_t* payload);

ne_someip_error_code_t
ne_someip_provided_event_send_initial_value(ne_someip_provided_event_t* event_behaviour,
	ne_someip_header_t* header, void* endpoint, void* seq_id, ne_someip_endpoint_send_policy_t send_policy,
	uint32_t ip_addr, uint16_t port);

/******************************************************find config**********************************************************/
ne_someip_l4_protocol_t
ne_someip_provided_get_event_comm_type(ne_someip_provided_event_t* event_behaviour);

uint32_t
ne_someip_provided_event_get_udp_collection_time_out(ne_someip_provided_event_t* event_behaviour);

ne_someip_udp_collection_trigger_t
ne_someip_provided_event_get_udp_trgger_mode(ne_someip_provided_event_t* event_behaviour);

uint32_t
ne_someip_provided_event_get_segment_len(ne_someip_provided_event_t* event_behaviour);

uint32_t 
ne_someip_provided_event_get_separation_time(ne_someip_provided_event_t* event_behaviour);

ne_someip_serializer_type_t
ne_someip_provided_event_get_ser_type(ne_someip_provided_event_t* event_behaviour);

bool
ne_someip_provided_event_get_field_flag(ne_someip_provided_event_t* event_behaviour);

ne_someip_list_t*
ne_someip_provided_event_get_eg_list(ne_someip_provided_event_t* event_behaviour);

#ifdef __cplusplus
}
#endif
#endif // MANAGER_SERVER_NE_SOMEIP_PROVIDED_EVENT_BEHAVIOUR_H
/* EOF */