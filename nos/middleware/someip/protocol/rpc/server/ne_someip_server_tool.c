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
#include "ne_someip_server_tool.h"
#include "ne_someip_log.h"
#include "ne_someip_provided_service_instance.h"

/*************************************************free**************************************/
void ne_someip_server_network_info_free(void* data)
{
	if (NULL == data) {
		return;
	}

	ne_someip_network_info_t* tmp_data = (ne_someip_network_info_t*)data;
	if (NULL != tmp_data) {
		free(tmp_data);
	}
}

void ne_someip_server_subscriber_free(void* data)
{
	ne_someip_endpoint_net_addr_t* tmp_data = (ne_someip_endpoint_net_addr_t*)data;

	if (NULL == tmp_data) {
		free(tmp_data);
	}
}

void ne_someip_server_remote_info_free(void* data)
{
	ne_someip_remote_client_info_t* tmp_data = (ne_someip_remote_client_info_t*)data;

	if (NULL == tmp_data) {
		free(tmp_data);
	}
}

/*************************************************cmp********************************************/
int ne_someip_uint16_cmp(void* key1, void* key2)
{
	if (NULL == key1 || NULL == key2) {
		return 0;
	}

	if (*(uint16_t*)key1 == *(uint16_t*)key2) {
		return 0;
	}

	return -1;
}

int ne_someip_ins_spec_cmp(void* key1, void* key2)
{
	if (NULL == key1 || NULL == key2) {
		return 0;
	}

	ne_someip_service_instance_spec_t* tmp_key = (ne_someip_service_instance_spec_t*)key1;
	ne_someip_service_instance_spec_t* temp_key = (ne_someip_service_instance_spec_t*)key2;

	if (0 == memcmp(tmp_key, temp_key, sizeof(ne_someip_service_instance_spec_t))) {
		return 0;
	}

	return -1;
}

uint32_t ne_someip_server_subscriber_addr_cmp(void* key1, void* key2)
{
	ne_someip_subscriber_addr_t* tmp_key = (ne_someip_subscriber_addr_t*)key1;
	ne_someip_subscriber_addr_t* temp_key = (ne_someip_subscriber_addr_t*)key2;

	if (0 == memcmp(tmp_key, temp_key, sizeof(ne_someip_subscriber_addr_t))) {
		return 0;
	}

	return -1;
}

int ne_someip_subscriber_cmp(ne_someip_endpoint_net_addr_t* first_subsriber, ne_someip_endpoint_net_addr_t* second_subsriber)
{
	if (NULL == first_subsriber || NULL == second_subsriber) {
		return 0;
	}

	if (first_subsriber->ip_addr == second_subsriber->ip_addr
		&& first_subsriber->port == second_subsriber->port
		&& first_subsriber->type == second_subsriber->type) {
		return 0;
	}

	return -1;
}

int ne_someip_saved_event_resp_seq_cmp(void* key1, void* key2) {
    if (NULL == key1 || NULL == key2) {
        return -1;
    }

    ne_someip_saved_event_resp_seq_info_t* k1 = key1;
    ne_someip_saved_event_resp_seq_info_t* k2 = key2;
    if ((k1->method_id == k2->method_id) && (k1->session_id == k2->session_id)) {
        return 0;
    }
    return -1;
}

int ne_someip_forward_link_cmp(void* key1, void* key2) {
    if (NULL == key1 || NULL == key2) {
        return -1;
    }

    ne_someip_server_forward_link_t* k1 = key1;
    ne_someip_server_forward_link_t* k2 = key2;
    if ((k1->addr_type == k2->addr_type) && \
        (k1->remote_addr == k2->remote_addr) && \
        (k1->remote_port == k2->remote_port)) {
        return 0;
    }
    return -1;
}

/*********************************************hash fun*********************************************/
uint32_t ne_someip_p_ins_uint16_key_hash_fun(const void* key)
{
	uint16_t* tmp_key = (uint16_t*)key;
	return (uint32_t)(*tmp_key & 0x0000FFFF);
}

uint32_t ne_someip_server_subscriber_addr_hash_fun(const void* key)
{
	ne_someip_subscriber_addr_t* tmp_key = (ne_someip_subscriber_addr_t*)key;
	if (NULL == tmp_key) {
		return 0;
	}
	uint32_t length = tmp_key->ip_addr + tmp_key->tcp_port + tmp_key->udp_port + tmp_key->type;

	return length;
}

uint32_t ne_someip_p_ins_saved_event_resp_seq_hasn_func(const void* key) {
    if (NULL == key) {
        return 0;
    }
    ne_someip_saved_event_resp_seq_info_t* tmp_key = key;
    uint32_t value = tmp_key->method_id + tmp_key->session_id;
    return value;
}

uint32_t ne_someip_p_ins_forward_link_hasn_func(const void* key) {
    if (NULL == key) {
        return 0;
    }
    ne_someip_server_forward_link_t* k = key;
    uint32_t value = k->addr_type + k->remote_addr + k->remote_port;
    return value;
}

/*********************************************handle map*******************************************/
void ne_someip_remove_all_map(ne_someip_map_t* map)
{
	ne_someip_map_iter_t* it = ne_someip_map_iter_new(map);
    void* key;
    void* value;
    while (ne_someip_map_iter_next(it, &key, &value)) {
        uint32_t hash_ret;
        ne_someip_map_remove(map, key, false);
    }
    ne_someip_map_iter_destroy(it);
}

/*****************************************log print***************************************/

uint32_t ne_someip_server_get_payload_len(const ne_someip_payload_t* payload)
{
	if (NULL == payload) {
		return 0;
	}

	uint32_t total_len = 0;
	uint32_t tmp_num = payload->num;
	while (tmp_num) {
		ne_someip_payload_slice_t* tmp_slice = *(payload->buffer_list) + tmp_num - 1;
		if (NULL == tmp_slice) {
			continue;
		}
		total_len = total_len + tmp_slice->length;
		tmp_num--;
	}

	return total_len;
}
