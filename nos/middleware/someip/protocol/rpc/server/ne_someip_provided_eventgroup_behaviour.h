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
#ifndef MANAGER_SERVER_NE_SOMEIP_PROVIDED_EVENTGROUP_BEHAVIOUR_H
#define MANAGER_SERVER_NE_SOMEIP_PROVIDED_EVENTGROUP_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_config_define.h"
#include "ne_someip_handler.h"
#include "ne_someip_sd_define.h"

typedef struct ne_someip_provided_eventgroup_behaviour ne_someip_provided_eg_t;

ne_someip_provided_eg_t*
ne_someip_provided_eg_create(const ne_someip_eventgroup_config_t* config);

ne_someip_provided_eg_t*
ne_someip_provided_eg_ref(ne_someip_provided_eg_t* subscribe_behaviour);

void ne_someip_provided_eg_unref(ne_someip_provided_eg_t* subscribe_behaviour);

ne_someip_error_code_t
ne_someip_provided_eg_add_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	uint32_t ip_addr, ne_someip_address_type_t type, uint16_t tcp_port, uint16_t udp_port,
	bool reliable_flag, bool unreliable_flag, uint32_t ttl, bool* first_flag);

ne_someip_error_code_t
ne_someip_provided_eg_remove_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	ne_someip_sd_recv_subscribe_t* subscriber_info);

bool
ne_someip_provided_eg_get_subscribers(ne_someip_provided_eg_t* subscribe_behaviour, ne_someip_l4_protocol_t comm_type,
	ne_someip_list_t** subscriber_list, uint32_t* multicast_ip, uint16_t* multicast_port);

void ne_someip_provided_eg_clear_subscribers(ne_someip_provided_eg_t* subscribe_behaviour);

ne_someip_error_code_t
ne_someip_provided_eg_set_subscriber_permission(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr, ne_someip_permission_t priority);

ne_someip_permission_t
ne_someip_provided_eg_get_subscriber_permission(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr);

void ne_someip_provided_eg_add_wait_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr);

bool ne_someip_provided_eg_find_wait_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr);

void ne_someip_provided_eg_delete_wait_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr);

ne_someip_error_code_t
ne_someip_provided_eg_remote_reboot(ne_someip_provided_eg_t* subscribe_behaviour,
	uint32_t ip_addr);

void ne_someip_provided_eg_reset(ne_someip_provided_eg_t* subscribe_behaviour);

void ne_someip_provided_eg_wait_priority_expired(void* data);

ne_someip_list_t*
ne_someip_provided_eg_get_tcp_subscribers(ne_someip_provided_eg_t* subscribe_behaviour);
ne_someip_list_t*
ne_someip_provided_eg_get_udp_subscribers(ne_someip_provided_eg_t* subscribe_behaviour);

bool ne_someip_provided_eg_get_unicast_to_multicast_flag(ne_someip_provided_eg_t* subscribe_behaviour);

bool ne_someip_provided_eg_set_multicast_ip_addr(ne_someip_provided_eg_t* subscribe_behaviour, uint32_t ip_addr);

bool ne_someip_provided_eg_set_multicast_port(ne_someip_provided_eg_t* subscribe_behaviour, uint16_t port);

bool ne_someip_provided_eg_set_threshold(ne_someip_provided_eg_t* subscribe_behaviour, uint16_t threshold);

bool ne_someip_provided_eg_set_time(ne_someip_provided_eg_t* subscribe_behaviour,
	ne_someip_server_subscribe_time_config_t* time);

ne_someip_l4_protocol_t
ne_someip_provided_eg_get_comm_type(const ne_someip_provided_eg_t* subscribe_behaviour);

bool ne_someip_provided_eg_get_multicast_flag(const ne_someip_provided_eg_t* subscribe_behaviour);

uint32_t ne_someip_provided_eg_get_multicast_addr(const ne_someip_provided_eg_t* subscribe_behaviour);
uint16_t ne_someip_provided_eg_get_multicast_port(const ne_someip_provided_eg_t* subscribe_behaviour);

ne_someip_eventgroup_id_t
ne_someip_provided_eg_get_eg_id(const ne_someip_provided_eg_t* subscribe_behaviour);

ne_someip_eventgroup_config_t*
ne_someip_provided_eg_get_config(const ne_someip_provided_eg_t* subscribe_behaviour);

#ifdef __cplusplus
}
#endif
#endif // MANAGER_SERVER_NE_SOMEIP_PROVIDED_EVENTGROUP_BEHAVIOUR_H
/* EOF */