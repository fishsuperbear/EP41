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
#include "ne_someip_provided_eventgroup_behaviour.h"
#include "ne_someip_list.h"
#include "ne_someip_map.h"
#include "ne_someip_object.h"
#include "ne_someip_log.h"
#include "ne_someip_ipc_behaviour.h"
#include "ne_someip_sd_tool.h"
#include "ne_someip_server_tool.h"

struct ne_someip_provided_eventgroup_behaviour
{
	const ne_someip_eventgroup_config_t* eventgroup_config;
    uint32_t multicast_ip_addr;
    uint16_t multicast_port;
    uint16_t threshold;
    ne_someip_server_subscribe_time_config_t* time;
	// ne_someip_register_unregister_status_t register_status;
	ne_someip_list_t* tcp_subscriber_list;//ne_someip_endpoint_net_addr_t*
	ne_someip_list_t* udp_subscriber_list;//ne_someip_endpoint_net_addr_t*
	bool unicast_to_multicast_flag;
	bool is_subscribed;
	ne_someip_map_t* subscriber_permission;//<ne_someip_remote_client_info_t*, ne_someip_permission_t*>
	ne_someip_permission_t default_permission;
	ne_someip_list_t* wait_ack_list;//ne_someip_remote_client_info_t*
	ne_someip_map_t* subscriber_timer_map;//<ne_someip_subscriber_addr_t*, ne_someip_looper_timer_t*>
	NEOBJECT_MEMBER
};

void ne_someip_provided_eg_t_free(ne_someip_provided_eg_t* subscribe_behaviour);
bool ne_someip_provided_eg_add_timer(ne_someip_provided_eg_t* subscribe_behaviour,
	ne_someip_subscriber_addr_t* sub_addr, uint32_t ttl);
bool ne_someip_provided_eg_remove_timer(ne_someip_provided_eg_t* subscribe_behaviour,
	ne_someip_subscriber_addr_t* sub_addr);
bool ne_someip_provided_eg_refresh_timer(ne_someip_provided_eg_t* subscribe_behaviour,
	ne_someip_subscriber_addr_t* sub_addr, uint32_t ttl);
void ne_someip_provided_eg_subscriber_timer_expired(ne_someip_subscriber_usr_data_t* usr_data);
void ne_someip_provided_eg_update_unicast_to_multicast_flag(ne_someip_provided_eg_t* subscribe_behaviour);
void ne_soemip_provided_eg_timer_free(ne_someip_looper_timer_t* timer);

NEOBJECT_FUNCTION(ne_someip_provided_eg_t);
ne_someip_provided_eg_t*
ne_someip_provided_eg_ref(ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("eventgroup is null");
		return NULL;
	}

	return ne_someip_provided_eg_t_ref(subscribe_behaviour);
}

void ne_someip_provided_eg_unref(ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		return;
	}

	ne_someip_provided_eg_t_unref(subscribe_behaviour);
}

ne_someip_provided_eg_t*
ne_someip_provided_eg_create(const ne_someip_eventgroup_config_t* config)
{
	ne_someip_provided_eg_t* p_eventgroup = (ne_someip_provided_eg_t*)malloc(sizeof(ne_someip_provided_eg_t));
	if (NULL == p_eventgroup) {
		ne_someip_log_error("malloc error");
		return NULL;
	}
	memset(p_eventgroup, 0, sizeof(ne_someip_provided_eg_t));

	ne_someip_provided_eg_t_ref_count_init(p_eventgroup);
	p_eventgroup->eventgroup_config = config;
	p_eventgroup->multicast_ip_addr = 0;
	p_eventgroup->multicast_port = 0;
	p_eventgroup->threshold = 0;
	p_eventgroup->time = NULL;
	// p_eventgroup->register_status = ne_someip_register_unregister_status_initial;
	p_eventgroup->tcp_subscriber_list = NULL;
	p_eventgroup->udp_subscriber_list = NULL;
	p_eventgroup->wait_ack_list = NULL;

	p_eventgroup->unicast_to_multicast_flag = false;
	p_eventgroup->is_subscribed = false;

	if (NULL == p_eventgroup->subscriber_permission) {
		//new map
	}

	if (NULL == p_eventgroup->subscriber_timer_map) {
		p_eventgroup->subscriber_timer_map
			= ne_someip_map_new(ne_someip_server_subscriber_addr_hash_fun, ne_someip_server_subscriber_addr_cmp,
			ne_someip_sd_free, ne_soemip_provided_eg_timer_free);
	}

	p_eventgroup->default_permission = ne_someip_permission_allow;

	return p_eventgroup;
}

void ne_someip_provided_eg_t_free(ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		return;
	}

	//TODO stop all behaviour
	// ne_someip_server_free_eventgroup_config(subscribe_behaviour->eventgroup_config);
	if (NULL != subscribe_behaviour->tcp_subscriber_list) {
		ne_someip_list_destroy(subscribe_behaviour->tcp_subscriber_list, ne_someip_server_subscriber_free);
		subscribe_behaviour->tcp_subscriber_list = NULL;
	}

	if (NULL != subscribe_behaviour->udp_subscriber_list) {
		ne_someip_list_destroy(subscribe_behaviour->udp_subscriber_list, ne_someip_server_subscriber_free);
		subscribe_behaviour->udp_subscriber_list = NULL;
	}

	if (NULL != subscribe_behaviour->wait_ack_list) {
		ne_someip_list_destroy(subscribe_behaviour->wait_ack_list, ne_someip_server_remote_info_free);
		subscribe_behaviour->wait_ack_list = NULL;
	}
	if (NULL != subscribe_behaviour->subscriber_timer_map) {
		ne_someip_map_unref(subscribe_behaviour->subscriber_timer_map);
		subscribe_behaviour->subscriber_timer_map = NULL;
	}
	ne_someip_provided_eg_t_ref_count_deinit(subscribe_behaviour);
	free(subscribe_behaviour);
	subscribe_behaviour = NULL;
}

ne_someip_error_code_t
ne_someip_provided_eg_add_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	uint32_t ip_addr, ne_someip_address_type_t type, uint16_t tcp_port, uint16_t udp_port,
	bool reliable_flag, bool unreliable_flag, uint32_t ttl, bool* first_flag)
{
	ne_someip_log_debug("start");
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("subscribe_behaviour is null");
		return ne_someip_error_code_failed;
	}

	*first_flag = false;
	if (reliable_flag) {
		ne_someip_endpoint_net_addr_t* sub_addr = (ne_someip_endpoint_net_addr_t*)malloc(sizeof(ne_someip_endpoint_net_addr_t));
		if (NULL == sub_addr) {
			ne_someip_log_error("malloc error");
			return ne_someip_error_code_failed;
		}
		memset(sub_addr, 0, sizeof(ne_someip_endpoint_net_addr_t));
		sub_addr->ip_addr = ip_addr;
		sub_addr->port = tcp_port;
		sub_addr->type = type;
		if (NULL == subscribe_behaviour->tcp_subscriber_list) {
			ne_someip_log_debug("tcp_subscriber_list is null");
			subscribe_behaviour->tcp_subscriber_list = ne_someip_list_create();
			ne_someip_list_append(subscribe_behaviour->tcp_subscriber_list, sub_addr);
			*first_flag = true;
		} else {
			if (NULL == ne_someip_list_find(subscribe_behaviour->tcp_subscriber_list, sub_addr, ne_someip_subscriber_cmp, NULL)) {
				ne_someip_log_debug("tcp_subscriber_list has not the subscriber");
				ne_someip_list_append(subscribe_behaviour->tcp_subscriber_list, sub_addr);
				*first_flag = true;
			} else {
				ne_someip_sd_free(sub_addr);
			}
		}
	}

	if (unreliable_flag) {
		ne_someip_endpoint_net_addr_t* sub_addr = (ne_someip_endpoint_net_addr_t*)malloc(sizeof(ne_someip_endpoint_net_addr_t));
		if (NULL == sub_addr) {
			ne_someip_log_error("malloc error");
			return ne_someip_error_code_failed;
		}
		memset(sub_addr, 0, sizeof(ne_someip_endpoint_net_addr_t));

		sub_addr->ip_addr = ip_addr;
		sub_addr->port = udp_port;
		sub_addr->type = type;

		if (NULL == subscribe_behaviour->udp_subscriber_list) {
			ne_someip_log_debug("udp_subscriber_list is null");
			subscribe_behaviour->udp_subscriber_list = ne_someip_list_create();
			ne_someip_list_append(subscribe_behaviour->udp_subscriber_list, sub_addr);
			*first_flag = true;
		} else {
			if (NULL == ne_someip_list_find(subscribe_behaviour->udp_subscriber_list, sub_addr, ne_someip_subscriber_cmp, NULL)) {
				ne_someip_log_debug("udp_subscriber_list has not the subscriber");
				ne_someip_list_append(subscribe_behaviour->udp_subscriber_list, sub_addr);
				*first_flag = true;
			} else {
				ne_someip_sd_free(sub_addr);
			}
		}
	}

	//start timer for subscriber,but the ip+port is not same as other client
	ne_someip_subscriber_addr_t tmp_sub_addr;
	tmp_sub_addr.ip_addr = ip_addr;
	tmp_sub_addr.udp_port = udp_port;
	tmp_sub_addr.tcp_port = tcp_port;
	tmp_sub_addr.type = type;
	if (*first_flag && 0xFFFFFF != ttl) {
		ne_someip_provided_eg_add_timer(subscribe_behaviour, &tmp_sub_addr, ttl);
	} else if (0xFFFFFF != ttl){
		ne_someip_provided_eg_refresh_timer(subscribe_behaviour, &tmp_sub_addr, ttl);
	} else {
		ne_someip_log_debug("ttl is 0xFFFFFF or first_flag is false");
	}

	ne_someip_provided_eg_update_unicast_to_multicast_flag(subscribe_behaviour);

	return ne_someip_error_code_ok;
}

ne_someip_error_code_t
ne_someip_provided_eg_remove_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	ne_someip_sd_recv_subscribe_t* subscriber_info)
{
	ne_someip_log_debug("start");
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("subscribe_behaviour is null");
		return ne_someip_error_code_failed;
	}

	if (NULL == subscriber_info) {
		ne_someip_log_error("subscriber_info is null");
		return ne_someip_error_code_failed;
	}

	if (subscriber_info->reliable_flag) {
		ne_someip_endpoint_net_addr_t sub_addr;
		sub_addr.ip_addr = subscriber_info->client_addr;
		sub_addr.port = subscriber_info->tcp_port;
		sub_addr.type = subscriber_info->addr_type;

		ne_someip_list_element_t* tmp_sub
			= ne_someip_list_find(subscribe_behaviour->tcp_subscriber_list, &sub_addr, ne_someip_subscriber_cmp, NULL);
		if (NULL != tmp_sub) {
			ne_someip_list_remove_by_elem(subscribe_behaviour->tcp_subscriber_list, tmp_sub, free);
		}
	}

	if (subscriber_info->unreliable_flag) {
		ne_someip_endpoint_net_addr_t sub_addr;
		sub_addr.ip_addr = subscriber_info->client_addr;
		sub_addr.port = subscriber_info->udp_port;
		sub_addr.type = subscriber_info->addr_type;

		ne_someip_list_element_t* tmp_sub
			= ne_someip_list_find(subscribe_behaviour->udp_subscriber_list, &sub_addr, ne_someip_subscriber_cmp, NULL);
		if (NULL != tmp_sub) {
			ne_someip_list_remove_by_elem(subscribe_behaviour->udp_subscriber_list, tmp_sub, free);
		}
	}

	ne_someip_subscriber_addr_t tmp_sub_addr;
	tmp_sub_addr.ip_addr = subscriber_info->client_addr;
	tmp_sub_addr.tcp_port = subscriber_info->tcp_port;
	tmp_sub_addr.udp_port = subscriber_info->udp_port;
	tmp_sub_addr.type = subscriber_info->addr_type;
	ne_someip_provided_eg_remove_timer(subscribe_behaviour, &tmp_sub_addr);

	ne_someip_provided_eg_update_unicast_to_multicast_flag(subscribe_behaviour);

	return ne_someip_error_code_ok;
}

bool
ne_someip_provided_eg_get_subscribers(ne_someip_provided_eg_t* subscribe_behaviour, ne_someip_l4_protocol_t comm_type,
	ne_someip_list_t** subscriber_list, uint32_t* multicast_ip, uint16_t* multicast_port)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("subscriber is NULL");
		*subscriber_list = NULL;
		return false;
	}

	if (!(comm_type & subscribe_behaviour->eventgroup_config->comm_type)) {
		*subscriber_list = NULL;
		return false;
	}

	if (subscribe_behaviour->unicast_to_multicast_flag && ne_someip_protocol_udp == comm_type) {
		*subscriber_list = NULL;
		*multicast_ip = subscribe_behaviour->multicast_ip_addr;
		*multicast_port = subscribe_behaviour->multicast_port;
		return true;
	}

	if (ne_someip_protocol_udp == comm_type) {
		*subscriber_list = subscribe_behaviour->udp_subscriber_list;
	} else {
		*subscriber_list = subscribe_behaviour->tcp_subscriber_list;
	}

	return false;
}

void ne_someip_provided_eg_clear_subscribers(ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("subscriber is NULL");
		return;
	}

	if (NULL != subscribe_behaviour->tcp_subscriber_list) {
		ne_someip_list_remove_all(subscribe_behaviour->tcp_subscriber_list, ne_someip_server_subscriber_free);
	}

	if (NULL != subscribe_behaviour->udp_subscriber_list) {
		ne_someip_list_remove_all(subscribe_behaviour->udp_subscriber_list, ne_someip_server_subscriber_free);
	}

	if (NULL != subscribe_behaviour->wait_ack_list) {
		ne_someip_list_remove_all(subscribe_behaviour->wait_ack_list, ne_someip_server_remote_info_free);
	}

	subscribe_behaviour->is_subscribed = false;
	ne_someip_provided_eg_update_unicast_to_multicast_flag(subscribe_behaviour);
}

ne_someip_error_code_t
ne_someip_provided_eg_set_subscriber_permission(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr, ne_someip_permission_t priority)
{
	//TODO, should check subscriber_priority, if unkown ,should save the permission
	return ne_someip_error_code_ok;
}

ne_someip_permission_t
ne_someip_provided_eg_get_subscriber_permission(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("subscribe is null");
		return ne_someip_permission_reject;
	}
	//TODO if default is allow,return allow; if default is unkown, check the pemission_map;
	return subscribe_behaviour->default_permission;
}

void ne_someip_provided_eg_add_wait_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr)
{
	//TODO
}

bool ne_someip_provided_eg_find_wait_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr)
{
	//TODO
    return true;
}

void ne_someip_provided_eg_delete_wait_subscriber(ne_someip_provided_eg_t* subscribe_behaviour,
	const ne_someip_remote_client_info_t* remote_addr)
{
	//TODO
}

ne_someip_error_code_t
ne_someip_provided_eg_remote_reboot(ne_someip_provided_eg_t* subscribe_behaviour,
	uint32_t ip_addr)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("subscribe_behaviour is null");
		return ne_someip_error_code_failed;
	}

	ne_someip_list_iterator_t* tcp_it = ne_someip_list_iterator_create(subscribe_behaviour->tcp_subscriber_list);
	while (ne_someip_list_iterator_valid(tcp_it)) {
		ne_someip_endpoint_net_addr_t* subscriber = (ne_someip_endpoint_net_addr_t*)ne_someip_list_iterator_data(tcp_it);
		if (NULL == subscriber) {
			ne_someip_list_iterator_remove(tcp_it, ne_someip_sd_free);
			ne_someip_list_iterator_next(tcp_it);
			continue;
		}
		if (ip_addr == subscriber->ip_addr) {
			ne_someip_list_iterator_remove(tcp_it, ne_someip_sd_free);
		}
		ne_someip_list_iterator_next(tcp_it);
	}
	ne_someip_list_iterator_destroy(tcp_it);

	ne_someip_list_iterator_t* udp_it = ne_someip_list_iterator_create(subscribe_behaviour->udp_subscriber_list);
	while (ne_someip_list_iterator_valid(udp_it)) {
		ne_someip_endpoint_net_addr_t* subscriber = (ne_someip_endpoint_net_addr_t*)ne_someip_list_iterator_data(udp_it);
		if (NULL == subscriber) {
			ne_someip_list_iterator_remove(udp_it, ne_someip_sd_free);
			ne_someip_list_iterator_next(udp_it);
			continue;
		}
		if (ip_addr == subscriber->ip_addr) {
			ne_someip_list_iterator_remove(udp_it, ne_someip_sd_free);
		}
		ne_someip_list_iterator_next(udp_it);
	}
	ne_someip_list_iterator_destroy(udp_it);

	ne_someip_list_iterator_t* wait_ack_it = ne_someip_list_iterator_create(subscribe_behaviour->wait_ack_list);
	while (ne_someip_list_iterator_valid(wait_ack_it)) {
		ne_someip_remote_client_info_t* subscriber = (ne_someip_remote_client_info_t*)ne_someip_list_iterator_data(wait_ack_it);
		if (NULL == subscriber) {
			ne_someip_list_iterator_remove(wait_ack_it, ne_someip_sd_free);
			ne_someip_list_iterator_next(wait_ack_it);
			continue;
		}
		if (ip_addr == subscriber->ipv4) {
			ne_someip_list_iterator_remove(wait_ack_it, ne_someip_sd_free);
		}
		ne_someip_list_iterator_next(wait_ack_it);
	}
	ne_someip_list_iterator_destroy(wait_ack_it);

	ne_someip_provided_eg_update_unicast_to_multicast_flag(subscribe_behaviour);
    return ne_someip_error_code_ok;
}

void ne_someip_provided_eg_reset(ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("subscribe_behaviour is null");
	}

	if (NULL != subscribe_behaviour->tcp_subscriber_list) {
		ne_someip_list_destroy(subscribe_behaviour->tcp_subscriber_list, ne_someip_server_subscriber_free);
	}

	if (NULL != subscribe_behaviour->udp_subscriber_list) {
		ne_someip_list_destroy(subscribe_behaviour->udp_subscriber_list, ne_someip_server_subscriber_free);
	}

	subscribe_behaviour->unicast_to_multicast_flag = false;
	subscribe_behaviour->is_subscribed = false;
}

void ne_someip_provided_eg_wait_priority_expired(void* data)
{
	//TODO
}

bool ne_someip_provided_eg_get_unicast_to_multicast_flag(ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		return false;
	}

	return subscribe_behaviour->unicast_to_multicast_flag;
}

ne_someip_list_t*
ne_someip_provided_eg_get_tcp_subscribers(ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("subscribe_behaviour is null");
		return NULL;
	}
	return subscribe_behaviour->tcp_subscriber_list;
}

ne_someip_list_t*
ne_someip_provided_eg_get_udp_subscribers(ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("subscribe_behaviour is null");
		return NULL;
	}
	return subscribe_behaviour->udp_subscriber_list;
}

bool ne_someip_provided_eg_set_multicast_ip_addr(ne_someip_provided_eg_t* subscribe_behaviour, uint32_t ip_addr)
{
	if (NULL == subscribe_behaviour) {
		return false;
	}

	subscribe_behaviour->multicast_ip_addr = ip_addr;
	return true;
}

bool ne_someip_provided_eg_set_multicast_port(ne_someip_provided_eg_t* subscribe_behaviour, uint16_t port)
{
	if (NULL == subscribe_behaviour) {
		return false;
	}

	subscribe_behaviour->multicast_port = port;
	return true;
}

bool ne_someip_provided_eg_set_threshold(ne_someip_provided_eg_t* subscribe_behaviour, uint16_t threshold)
{
	if (NULL == subscribe_behaviour) {
		return false;
	}

	subscribe_behaviour->threshold = threshold;
	return true;
}

bool ne_someip_provided_eg_set_time(ne_someip_provided_eg_t* subscribe_behaviour,
	ne_someip_server_subscribe_time_config_t* time)
{
	if (NULL == subscribe_behaviour) {
		return false;
	}

	subscribe_behaviour->time = time;
	return true;
}

ne_someip_l4_protocol_t
ne_someip_provided_eg_get_comm_type(const ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour || NULL == subscribe_behaviour->eventgroup_config) {
		ne_someip_log_error("eventgroup is null");
		return ne_someip_protocol_unknown;
	}

	return subscribe_behaviour->eventgroup_config->comm_type;
}

bool ne_someip_provided_eg_get_multicast_flag(const ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("eventgroup is null");
		return false;
	}

	if (0 == subscribe_behaviour->multicast_ip_addr) {
		return false;
	}
	return true;
}

uint32_t ne_someip_provided_eg_get_multicast_addr(const ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("eventgroup is null");
		return 0;
	}

	return subscribe_behaviour->multicast_ip_addr;
}

uint16_t ne_someip_provided_eg_get_multicast_port(const ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("eventgroup is null");
		return 0;
	}

	return subscribe_behaviour->multicast_port;
}

ne_someip_eventgroup_id_t
ne_someip_provided_eg_get_eg_id(const ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour || NULL == subscribe_behaviour->eventgroup_config) {
		ne_someip_log_error("eventgroup is null");
		return 0;
	}

	return subscribe_behaviour->eventgroup_config->eventgroup_id;
}

ne_someip_eventgroup_config_t*
ne_someip_provided_eg_get_config(const ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour) {
		ne_someip_log_error("eventgroup is null");
		return 0;
	}

	return subscribe_behaviour->eventgroup_config;
}

bool ne_someip_provided_eg_add_timer(ne_someip_provided_eg_t* subscribe_behaviour,
	ne_someip_subscriber_addr_t* sub_addr, uint32_t ttl)
{
	ne_someip_log_debug("start");
	if (NULL == subscribe_behaviour || NULL == sub_addr) {
		ne_someip_log_error("subscribe_behaviour or sub_addr is null");
		return false;
	}

	ne_someip_looper_timer_runnable_t* timer_runnable
		= (ne_someip_looper_timer_runnable_t*)malloc(sizeof(ne_someip_looper_timer_runnable_t));
	if (NULL == timer_runnable) {
		ne_someip_log_error("malloc error");
		return false;
	}
	memset(timer_runnable, 0, sizeof(ne_someip_looper_timer_runnable_t));

	ne_someip_subscriber_addr_t* tmp_sub_addr = (ne_someip_subscriber_addr_t*)malloc(sizeof(ne_someip_subscriber_addr_t));
	if (NULL == tmp_sub_addr) {
		ne_someip_log_error("malloc error");
		ne_someip_sd_free(timer_runnable);
		return false;
	}
	memset(tmp_sub_addr, 0, sizeof(ne_someip_subscriber_addr_t));
	memcpy(tmp_sub_addr, sub_addr, sizeof(ne_someip_subscriber_addr_t));

	ne_someip_subscriber_usr_data_t* usr_data = (ne_someip_subscriber_usr_data_t*)malloc(sizeof(ne_someip_subscriber_usr_data_t));
	if (NULL == usr_data) {
		ne_someip_log_error("malloc error");
		ne_someip_sd_free(timer_runnable);
		ne_someip_sd_free(tmp_sub_addr);
		return false;
	}
	memset(usr_data, 0, sizeof(ne_someip_subscriber_usr_data_t));
	memcpy(&usr_data->sub_addr, tmp_sub_addr, sizeof(ne_someip_subscriber_addr_t));
	usr_data->eg_behaviour = subscribe_behaviour;

	timer_runnable->free = ne_someip_sd_free;
	timer_runnable->user_data = usr_data;
	timer_runnable->run = ne_someip_provided_eg_subscriber_timer_expired;

	ne_someip_looper_timer_t* timer = ne_someip_looper_timer_new(timer_runnable);
	if (NULL == timer) {
		ne_someip_log_error("timer new error");
		ne_someip_sd_free(timer_runnable);
		ne_someip_sd_free(usr_data);
		ne_someip_sd_free(tmp_sub_addr);
		return false;
	}

	int tmp_ret = ne_someip_looper_timer_start(timer, NE_LOOPER_TIMER_TYPE_INTERVAL_ONE_SHOT, ttl * 1000);
	if (-1 == tmp_ret) {
		ne_someip_log_error("timer start error");
		if (NULL != timer) {
			ne_someip_looper_timer_unref(timer);
		}
		ne_someip_sd_free(tmp_sub_addr);
		return false;
	}
	ne_someip_looper_timer_unref(timer);

	ne_someip_map_insert(subscribe_behaviour->subscriber_timer_map, tmp_sub_addr, timer);

	return true;
}

bool ne_someip_provided_eg_remove_timer(ne_someip_provided_eg_t* eg_behaviour,
	ne_someip_subscriber_addr_t* sub_addr)
{
	ne_someip_log_debug("start");
	if (NULL == eg_behaviour || NULL == sub_addr) {
		ne_someip_log_error("eg_behaviour or sub_addr is null");
		return false;
	}

	uint32_t hash_ret;
	ne_someip_looper_timer_t* timer = ne_someip_map_find(eg_behaviour->subscriber_timer_map, sub_addr, &hash_ret);
	if (NULL != timer) {
		ne_someip_looper_timer_stop(timer);
	}
	ne_someip_map_remove(eg_behaviour->subscriber_timer_map, sub_addr, true);

	return true;
}

bool ne_someip_provided_eg_refresh_timer(ne_someip_provided_eg_t* eg_behaviour,
	ne_someip_subscriber_addr_t* sub_addr, uint32_t ttl)
{
	ne_someip_log_debug("start");
	if (NULL == eg_behaviour || NULL == sub_addr) {
		ne_someip_log_error("eg_behaviour or sub_addr is null");
		return false;
	}

	uint32_t hash_ret;
	ne_someip_looper_timer_t* timer = ne_someip_map_find(eg_behaviour->subscriber_timer_map, sub_addr, &hash_ret);
	ne_someip_looper_timer_refresh(timer, NE_LOOPER_TIMER_TYPE_INTERVAL_ONE_SHOT, ttl * 1000);

	return true;
}

void ne_someip_provided_eg_subscriber_timer_expired(ne_someip_subscriber_usr_data_t* usr_data)
{
	ne_someip_log_debug("start");
	if (NULL == usr_data) {
		ne_someip_log_error("usr_data is null");
		return;
	}

	ne_someip_provided_eg_t* eg_behaviour = usr_data->eg_behaviour;
	if (NULL == eg_behaviour) {
		ne_someip_log_error("eg_behaviour is null");
		return;
	}
	ne_someip_subscriber_addr_t subscriber_addr;
	memcpy(&subscriber_addr, &usr_data->sub_addr, sizeof(ne_someip_subscriber_addr_t));
	ne_someip_map_remove(eg_behaviour->subscriber_timer_map, &subscriber_addr, true);

	if (NULL != eg_behaviour->tcp_subscriber_list) {
		ne_someip_endpoint_net_addr_t tcp_subscriber_addr;
		tcp_subscriber_addr.ip_addr = subscriber_addr.ip_addr;
		tcp_subscriber_addr.port = subscriber_addr.tcp_port;
		tcp_subscriber_addr.type = subscriber_addr.type;
		ne_someip_list_iterator_t* tcp_subscriber_list_it = ne_someip_list_iterator_create(eg_behaviour->tcp_subscriber_list);
		while (ne_someip_list_iterator_valid(tcp_subscriber_list_it)) {
			ne_someip_endpoint_net_addr_t* tmp_subscriber = (ne_someip_endpoint_net_addr_t*)ne_someip_list_iterator_data(tcp_subscriber_list_it);
			if (NULL == tmp_subscriber) {
				ne_someip_list_iterator_next(tcp_subscriber_list_it);
				continue;
			}
			if (0 == memcmp(tmp_subscriber, &tcp_subscriber_addr, sizeof(ne_someip_endpoint_net_addr_t))) {
				ne_someip_list_iterator_remove(tcp_subscriber_list_it, ne_someip_sd_free);
				break;
			}
			ne_someip_list_iterator_next(tcp_subscriber_list_it);
		}
		ne_someip_list_iterator_destroy(tcp_subscriber_list_it);
	}

	if (NULL != eg_behaviour->udp_subscriber_list) {
		ne_someip_endpoint_net_addr_t udp_subscriber_addr;
		udp_subscriber_addr.ip_addr = subscriber_addr.ip_addr;
		udp_subscriber_addr.port = subscriber_addr.udp_port;
		udp_subscriber_addr.type = subscriber_addr.type;
		ne_someip_list_iterator_t* udp_subscriber_list_it = ne_someip_list_iterator_create(eg_behaviour->udp_subscriber_list);
		while (ne_someip_list_iterator_valid(udp_subscriber_list_it)) {
			ne_someip_endpoint_net_addr_t* tmp_subscriber = (ne_someip_endpoint_net_addr_t*)ne_someip_list_iterator_data(udp_subscriber_list_it);
			if (NULL == tmp_subscriber) {
				ne_someip_list_iterator_next(udp_subscriber_list_it);
				continue;
			}
			if (0 == memcmp(tmp_subscriber, &udp_subscriber_addr, sizeof(ne_someip_endpoint_net_addr_t))) {
				ne_someip_list_iterator_remove(udp_subscriber_list_it, ne_someip_sd_free);
				break;
			}
			ne_someip_list_iterator_next(udp_subscriber_list_it);
		}
		ne_someip_list_iterator_destroy(udp_subscriber_list_it);
	}

	ne_someip_provided_eg_update_unicast_to_multicast_flag(eg_behaviour);
}

void ne_someip_provided_eg_update_unicast_to_multicast_flag(ne_someip_provided_eg_t* subscribe_behaviour)
{
	if (NULL == subscribe_behaviour || 0 == subscribe_behaviour->threshold) {
		return;
	}

	if (subscribe_behaviour->eventgroup_config->comm_type & ne_someip_protocol_udp) {
		if (subscribe_behaviour->threshold > ne_someip_list_length(subscribe_behaviour->udp_subscriber_list)) {
			subscribe_behaviour->unicast_to_multicast_flag = false;
		} else if (subscribe_behaviour->threshold <= ne_someip_list_length(subscribe_behaviour->udp_subscriber_list)) {
			subscribe_behaviour->unicast_to_multicast_flag = true;
		}
	}
}

void ne_soemip_provided_eg_timer_free(ne_someip_looper_timer_t* timer)
{
	if (NULL != timer) {
        ne_someip_looper_timer_stop(timer);
	}
}