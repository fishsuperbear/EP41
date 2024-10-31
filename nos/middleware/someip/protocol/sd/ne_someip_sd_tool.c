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
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "ne_someip_sd_tool.h"
#include "ne_someip_sd_define.h"
#include "ne_someip_list.h"
#include "ne_someip_map.h"
#include "ne_someip_ipc_behaviour.h"
#include "ne_someip_log.h"
#include "ne_someip_endpoint_udp_sd.h"

/*****************************************************free*********************************************/
void ne_someip_sd_free(void* data)
{
	if (data) {
		free(data);
		data = NULL;
	}
}

void ne_someip_sd_service_free(void* data)
{
	if (data) {
		ne_someip_list_t* data_list = (ne_someip_list_t*)data;
		if (NULL == data_list) {
			ne_someip_log_error("type conversion error");
			return;
		}

		ne_someip_list_remove_all(data_list, ne_someip_sd_free);
		data = NULL;
	}
}

void ne_someip_sd_timer_value_free(void* data)
{
	if (data) {
		ne_someip_looper_timer_t* timer = (ne_someip_looper_timer_t*)data;
		if (NULL != timer) {
			ne_someip_looper_timer_unref(timer);
			timer = NULL;
		}
	}
}

void ne_someip_sd_subscribe_value_free(void* data)
{
	ne_someip_ipc_send_subscribe_t* sub = (ne_someip_ipc_send_subscribe_t*)data;
	if (NULL == sub) {
		return;
	}

	ne_someip_list_iterator_t* iter = ne_someip_list_iterator_create(sub->eventgroup_list);
	while (ne_someip_list_iterator_valid(iter)) {
		ne_someip_list_iterator_remove(iter, ne_someip_sd_free);
		ne_someip_list_iterator_next(iter);
	}
	ne_someip_list_iterator_destroy(iter);

	free(sub);
	sub = NULL;
}

void ne_someip_sd_endpoint_free(void* data)
{
	if (data) {
		ne_someip_endpoint_udp_sd_t* endpoint = (ne_someip_endpoint_udp_sd_t*)data;
		if (NULL != endpoint) {
			ne_someip_endpoint_udp_sd_unref(endpoint);
			endpoint = NULL;
		}
	}
}

void ne_someip_sd_remote_service_free(void* data)
{
	if (data) {
		ne_someip_sd_recv_offer_t* service = (ne_someip_sd_recv_offer_t*)data;
		if (NULL != service) {
			free(service);
			service = NULL;
		}
	}
}

void ne_someip_sd_remote_service_list_free(void* data)
{
	if (data) {
		ne_someip_list_t* services = (ne_someip_list_t*)data;
		if (NULL != services) {
			ne_someip_list_destroy(services, ne_someip_sd_remote_service_free);
			services = NULL;
		}
	}
}

void ne_someip_sd_ins_spec_free(void* data)
{
	if (data) {
		ne_someip_service_instance_spec_t* ins_spec = (ne_someip_service_instance_spec_t*)data;
		if (NULL != ins_spec) {
			free(ins_spec);
			ins_spec = NULL;
		}
	}
}

void ne_someip_sd_ins_sepc_ip_free(void* data)
{
	if (data) {
		ne_someip_sd_ins_ip_spec_t* ins_ip_spec = (ne_someip_sd_ins_ip_spec_t*)data;
		if (NULL != ins_ip_spec) {
			free(ins_ip_spec);
			ins_ip_spec = NULL;
		}
	}
}

void ne_someip_sd_ip_list_free(void* data)
{
	if (data) {
		ne_someip_list_t* ip_list_spec = (ne_someip_list_t*)data;
		if (NULL != ip_list_spec) {
			ne_someip_list_destroy(ip_list_spec, ne_someip_sd_free);
			ip_list_spec = NULL;
		}
	}
}

void ne_someip_sd_subscribe_list_free(void* data)
{
	if (data) {
        ne_someip_list_destroy((ne_someip_list_t*)data, NULL);
	}
}

void ne_someip_daemon_client_id_list_free(void* data)
{
	if (data) {
		ne_someip_list_t* client_is_list = (ne_someip_list_t*)data;
		if (NULL != client_is_list) {
			ne_someip_list_destroy(client_is_list, ne_someip_sd_free);
			client_is_list = NULL;
		}
	}
}

/*****************************************************cmp*******************************************************/
int ne_someip_sd_offer_key_cmp(void* key1, void* key2)
{
	ne_someip_sd_offer_key_t* tmp_key = (ne_someip_sd_offer_key_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_sd_offer_key_t* temp_key = (ne_someip_sd_offer_key_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (tmp_key->base.type == temp_key->base.type
		&& tmp_key->src_addr == temp_key->src_addr
		&& tmp_key->src_port == temp_key->src_port
		&& tmp_key->dst_addr == temp_key->dst_addr
		&& tmp_key->dst_port == temp_key->dst_port
		&& 0 == memcmp(&(tmp_key->offer_timer), &(temp_key->offer_timer), sizeof(ne_someip_server_offer_time_config_t))) {
		return 0;
	}

	return -1;
}

int ne_someip_sd_offer_r_key_cmp(void* key1, void* key2)
{
	ne_someip_sd_offer_key_r_t* tmp_key = (ne_someip_sd_offer_key_r_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_sd_offer_key_r_t* temp_key = (ne_someip_sd_offer_key_r_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (tmp_key->base.type == temp_key->base.type
		&& tmp_key->sequence_id == temp_key->sequence_id
		&& tmp_key->counter == temp_key->counter
		&& tmp_key->src_addr == temp_key->src_addr
		&& tmp_key->src_port == temp_key->src_port
		&& tmp_key->dst_addr == temp_key->dst_addr
		&& tmp_key->dst_port == temp_key->dst_port
		&& 0 == memcmp(&(tmp_key->offer_timer), &(temp_key->offer_timer), sizeof(ne_someip_server_offer_time_config_t))) {
		return 0;
	}

	return -1;
}

int ne_someip_sd_find_key_cmp(void* key1, void* key2)
{
	ne_someip_sd_find_key_t* tmp_key = (ne_someip_sd_find_key_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_sd_find_key_t* temp_key = (ne_someip_sd_find_key_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (0 == memcmp(tmp_key, temp_key, sizeof(ne_someip_sd_find_key_t))) {
		return 0;
	}

	return -1;
}

int ne_someip_sd_find_r_key_cmp(void* key1, void* key2)
{
	ne_someip_sd_find_key_r_t* tmp_key = (ne_someip_sd_find_key_r_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_sd_find_key_r_t* temp_key = (ne_someip_sd_find_key_r_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (0 == memcmp(tmp_key, temp_key, sizeof(ne_someip_sd_find_key_r_t))) {
		return 0;
	}

	return -1;
}

int ne_someip_sd_subscribe_key_cmp(void* key1, void* key2)
{
	ne_someip_sd_subscribe_key_t* tmp_key = (ne_someip_sd_subscribe_key_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_sd_subscribe_key_t* temp_key = (ne_someip_sd_subscribe_key_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (tmp_key->base.type == temp_key->base.type
		&& tmp_key->client_id == temp_key->client_id
		&& tmp_key->service_id == temp_key->service_id
		&& tmp_key->instance_id == temp_key->instance_id
		&& tmp_key->major_version == temp_key->major_version
		&& tmp_key->local_addr == temp_key->local_addr
		&& tmp_key->local_port == temp_key->local_port
		&& tmp_key->counter == temp_key->counter) {
		bool is_equal = true;
		ne_someip_list_element_t* tmp_element = ne_someip_list_first(tmp_key->eventgroup_list);
		ne_someip_list_element_t* temp_element = ne_someip_list_first(temp_key->eventgroup_list);
		if (!tmp_element && !temp_element) {
			return 0;
		}

		if ((!tmp_element && temp_element) || (tmp_element && !temp_element)) {
			return -1;
		}

		while (tmp_element) {
			ne_someip_ipc_subscribe_eg_t* tmp_subscribe_eg = (ne_someip_ipc_subscribe_eg_t*)(tmp_element->data);
			if (NULL == tmp_subscribe_eg) {
				ne_someip_log_error("type conversion error");
				return -1;
			}
			bool tmp_is_equal = false;
			while (temp_element) {
				ne_someip_ipc_subscribe_eg_t* temp_subscribe_eg = (ne_someip_ipc_subscribe_eg_t*)(temp_element->data);
				if (NULL == temp_subscribe_eg) {
					ne_someip_log_error("type conversion error");
					return -1;
				}
				if (tmp_subscribe_eg->eventgroup_id == temp_subscribe_eg->eventgroup_id) {
					tmp_is_equal = true;
					break;
				}
				temp_element = temp_element->next;
			}
			if (!tmp_is_equal) {
				is_equal = false;
				break;
			}
			tmp_element = tmp_element->next;
		}

		if (is_equal) {
			return 0;
		}
	}

	return -1;
}

int ne_someip_sd_session_key_cmp(void* key1, void* key2)
{
	ne_someip_sd_session_key_t* tmp_key = (ne_someip_sd_session_key_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_sd_session_key_t* temp_key = (ne_someip_sd_session_key_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (0 == memcmp(tmp_key, temp_key, sizeof(ne_someip_sd_session_key_t))) {
		return 0;
	}

	return -1;
}

int ne_someip_sd_endpoint_key_cmp(void* key1, void* key2)
{
	ne_someip_endpoint_net_addr_t* tmp_key = (ne_someip_endpoint_net_addr_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_endpoint_net_addr_t* temp_key = (ne_someip_endpoint_net_addr_t*)key2;
	if (!temp_key) {
		return -1;
	}

    if (tmp_key->ip_addr == temp_key->ip_addr && tmp_key->port == temp_key->port
	    && tmp_key->type == temp_key->type) {
        return 0;
	}
	// if (0 == memcmp(tmp_key, temp_key, sizeof(ne_someip_endpoint_net_addr_t))) {
	// 	return 0;
	// }

	return -1;
}

int ne_someip_sd_finished_subscribe_key_cmp(void* key1, void* key2)
{
	ne_someip_service_instance_spec_t* tmp_key = (ne_someip_service_instance_spec_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_service_instance_spec_t* temp_key = (ne_someip_service_instance_spec_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (0 == memcmp(tmp_key, temp_key, sizeof(ne_someip_service_instance_spec_t))) {
		return 0;
	}

	return -1;
}

int ne_someip_sd_service_key_cmp(void* key1, void* key2)
{
	ne_someip_ipc_reg_unreg_service_handler_t* tmp_key = (ne_someip_ipc_reg_unreg_service_handler_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_ipc_reg_unreg_service_handler_t* temp_key = (ne_someip_ipc_reg_unreg_service_handler_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (0 == memcmp(tmp_key, temp_key, sizeof(ne_someip_ipc_reg_unreg_service_handler_t))) {
		return 0;
	}

	return -1;
}

int ne_someip_sd_uint32_key_cmp(void* key1, void* key2)
{
	uint32_t* tmp_key = (uint32_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	uint32_t* temp_key = (uint32_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (*tmp_key == *temp_key) {
		return 0;
	}

	return -1;
}

int ne_someip_sd_ins_spec_cmp(void* key1, void* key2)
{
	ne_someip_service_instance_spec_t* tmp_key = (ne_someip_service_instance_spec_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_service_instance_spec_t* temp_key = (ne_someip_service_instance_spec_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if ((NE_SOMEIP_ANY_SERVICE == tmp_key->service_id || tmp_key->service_id == temp_key->service_id)
		&& (NE_SOMEIP_ANY_INSTANCE == tmp_key->instance_id || tmp_key->instance_id == temp_key->instance_id)
		&& (NE_SOMEIP_ANY_MAJOR == tmp_key->major_version || tmp_key->major_version == temp_key->major_version)) {
			return 0;
	}

	return -1;
}

int ne_someip_sd_ins_sepc_ip_cmp(void* key1, void* key2)
{
	ne_someip_sd_ins_ip_spec_t* tmp_key = (ne_someip_sd_ins_ip_spec_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	ne_someip_sd_ins_ip_spec_t* temp_key = (ne_someip_sd_ins_ip_spec_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if ((tmp_key->ins_spec.service_id == temp_key->ins_spec.service_id)
		&& (tmp_key->ins_spec.instance_id == temp_key->ins_spec.instance_id)
		&& (tmp_key->ins_spec.major_version == temp_key->ins_spec.major_version)
		&& (tmp_key->ip_addr == temp_key->ip_addr)) {
		return 0;
	}

	return -1;
}

int ne_someip_sd_uint16_cmp(void* key1, void* key2)
{
	uint16_t* tmp_key = (uint16_t*)key1;
	if (!tmp_key) {
		return -1;
	}
	uint16_t* temp_key = (uint16_t*)key2;
	if (!temp_key) {
		return -1;
	}

	if (*tmp_key == *temp_key) {
		return 0;
	}

	return -1;
}

/*******************************hash func********************************************/
uint32_t ne_someip_sd_offer_key_hash_fun(const void* key)
{
	ne_someip_sd_offer_key_t* tmp_key = (ne_someip_sd_offer_key_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)((uint32_t)tmp_key->base.type + tmp_key->offer_timer.ttl + tmp_key->offer_timer.request_response_delay_min
		+ tmp_key->offer_timer.request_response_delay_max + tmp_key->offer_timer.repetition_max + tmp_key->offer_timer.repetition_base_delay
		+ tmp_key->offer_timer.initial_delay_min + tmp_key->offer_timer.initial_delay_max + tmp_key->offer_timer.cyclic_offer_delay
		+ tmp_key->src_addr + tmp_key->src_port + tmp_key->dst_addr + tmp_key->dst_port);
}
uint32_t ne_someip_sd_offer_r_key_hash_fun(const void* key)
{
	ne_someip_sd_offer_key_r_t* tmp_key = (ne_someip_sd_offer_key_r_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)((uint32_t)tmp_key->base.type + tmp_key->offer_timer.ttl + tmp_key->offer_timer.request_response_delay_min
		+ tmp_key->offer_timer.request_response_delay_max + tmp_key->offer_timer.repetition_max + tmp_key->offer_timer.repetition_base_delay
		+ tmp_key->offer_timer.initial_delay_min + tmp_key->offer_timer.initial_delay_max + tmp_key->offer_timer.cyclic_offer_delay
		+ tmp_key->src_addr + tmp_key->src_port + tmp_key->dst_addr + tmp_key->dst_port + tmp_key->counter);
}
uint32_t ne_someip_sd_find_key_hash_fun(const void* key)
{
	ne_someip_sd_find_key_t* tmp_key = (ne_someip_sd_find_key_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)((uint32_t)tmp_key->base.type + tmp_key->find_timer.initial_delay_max + tmp_key->find_timer.initial_delay_min
		+ tmp_key->find_timer.repetition_base_delay + tmp_key->find_timer.repetition_max
		+ tmp_key->find_timer.ttl + tmp_key->dst_addr + tmp_key->dst_port + tmp_key->src_addr + tmp_key->src_port);
}
uint32_t ne_someip_sd_find_r_key_hash_fun(const void* key)
{
	ne_someip_sd_find_key_r_t* tmp_key = (ne_someip_sd_find_key_r_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)((uint32_t)tmp_key->base.type + tmp_key->find_timer.initial_delay_max + tmp_key->find_timer.initial_delay_min
		+ tmp_key->find_timer.repetition_base_delay + tmp_key->find_timer.repetition_max
		+ tmp_key->find_timer.ttl + tmp_key->dst_addr + tmp_key->dst_port + tmp_key->src_addr + tmp_key->src_port + tmp_key->counter);
}
uint32_t ne_someip_sd_subsribe_key_hash_fun(const void* key)
{
	ne_someip_sd_subscribe_key_t* tmp_key = (ne_someip_sd_subscribe_key_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)((uint32_t)tmp_key->base.type + tmp_key->client_id + tmp_key->counter + tmp_key->instance_id
		+ tmp_key->local_addr + tmp_key->local_port + tmp_key->major_version + tmp_key->service_id);
}
uint32_t ne_someip_sd_ins_spec_key_hash_fun(const void* key)
{
	ne_someip_service_instance_spec_t* tmp_key = (ne_someip_service_instance_spec_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)(tmp_key->service_id + tmp_key->instance_id + tmp_key->major_version);
}
uint32_t ne_someip_sd_session_key_hash_fun(const void* key)
{
	ne_someip_sd_session_key_t* tmp_key = (ne_someip_sd_session_key_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)(tmp_key->local_addr + tmp_key->remote_addr);
}

uint32_t ne_someip_sd_service_handler_key_hash_fun(const void* key)
{
	ne_someip_ipc_reg_unreg_service_handler_t* tmp_key = (ne_someip_ipc_reg_unreg_service_handler_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)((uint32_t)tmp_key->type + tmp_key->service_id + tmp_key->major_version
		+ tmp_key->length + tmp_key->instance_id + tmp_key->client_id);
}

uint32_t ne_someip_sd_uint32_key_hash_fun(const void* key)
{
	uint32_t* tmp_key = (uint32_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return *tmp_key;
}

uint32_t ne_someip_sd_ins_sepc_ip_hash_fun(const void* key)
{
	ne_someip_sd_ins_ip_spec_t* tmp_key = (ne_someip_sd_ins_ip_spec_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)(tmp_key->ip_addr + tmp_key->ins_spec.service_id + tmp_key->ins_spec.instance_id + tmp_key->ins_spec.major_version);
}

uint32_t ne_someip_sd_uint16_hash_fun(const void* key)
{
	uint16_t* tmp_key = (uint16_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	return (uint32_t)(*tmp_key);
}

/************************************uint32 to char***************************/
void ne_someip_sd_convert_uint32_to_ip(const char* description, uint32_t ip, const char* file, uint16_t line,
	const char* fun)
{
	char* tmp_ip = malloc(NESOMEIP_IP_ADDR_LENGTH);
	if (NULL == tmp_ip) {
		return;
	}
	memset(tmp_ip, 0, NESOMEIP_IP_ADDR_LENGTH);
	strcpy(tmp_ip, inet_ntoa(*((struct in_addr*)&ip)));
	ne_someip_log_info("%s, ip [%s]", description, tmp_ip);
	if (NULL != tmp_ip) {
		free(tmp_ip);
		tmp_ip = NULL;
	}
}

void ne_someip_sd_convert_uint32_to_ip_t(const char* description, uint32_t src_ip, uint32_t dst_ip,
	const char* file, uint16_t line, const char* fun)
{
	char* s_tmp_ip = malloc(NESOMEIP_IP_ADDR_LENGTH);
	char* d_tmp_ip = malloc(NESOMEIP_IP_ADDR_LENGTH);
	if (NULL == s_tmp_ip || NULL == d_tmp_ip) {
		free(s_tmp_ip);
		free(d_tmp_ip);
		s_tmp_ip = NULL;
		d_tmp_ip = NULL;
		return;
	}
	memset(s_tmp_ip, 0, NESOMEIP_IP_ADDR_LENGTH);
	memset(d_tmp_ip, 0, NESOMEIP_IP_ADDR_LENGTH);
	strcpy(s_tmp_ip, inet_ntoa(*((struct in_addr*)&src_ip)));
	strcpy(d_tmp_ip, inet_ntoa(*((struct in_addr*)&dst_ip)));
	ne_someip_log_info("%s, src ip [%s], dst ip [%s]",
		description, s_tmp_ip, d_tmp_ip);
	if (NULL != s_tmp_ip) {
		free(s_tmp_ip);
		s_tmp_ip = NULL;
	}
	if (NULL != d_tmp_ip) {
		free(d_tmp_ip);
		d_tmp_ip = NULL;
	}
}
