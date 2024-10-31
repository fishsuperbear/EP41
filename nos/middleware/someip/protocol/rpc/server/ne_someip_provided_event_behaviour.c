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
#include "ne_someip_provided_event_behaviour.h"
#include "ne_someip_list.h"
#include "ne_someip_map.h"
#include "ne_someip_server_define.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_provided_eventgroup_behaviour.h"
#include "ne_someip_object.h"
#include "ne_someip_log.h"
#include "ne_someip_endpoint_tool.h"
#include "ne_someip_message.h"
#include "ne_someip_endpoint_udp_data.h"
#include "ne_someip_endpoint_tcp_data.h"
#include "ne_someip_sd_tool.h"
#include "ne_someip_server_tool.h"

struct ne_someip_provided_event_behaviour
{
	const ne_someip_event_config_t* event_config;
	// ne_someip_map_t* subscriber_map; //<bool, ne_someip_list_t*(ne_someip_endpoint_net_addr_t*)>
	uint32_t udp_collection_buffer_timeout;
    ne_someip_udp_collection_trigger_t udp_collection_trigger;
	ne_someip_list_t* eventgroup_ref_list;//ne_someip_provided_eg_t*
	ne_someip_payload_t* initial_value;
	NEOBJECT_MEMBER
};

void ne_someip_provided_event_t_free(ne_someip_provided_event_t* event_behaviour);

void ne_someip_provided_event_find_subscriber(ne_someip_provided_event_t* event_behaviour,
	ne_someip_list_t* subscriber_list);
ne_someip_trans_buffer_struct_t*
ne_someip_provided_event_create_trans_buffer(ne_someip_header_t* header, ne_someip_payload_t* payload);

NEOBJECT_FUNCTION(ne_someip_provided_event_t);
ne_someip_provided_event_t*
ne_someip_provided_event_ref(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour) {
		ne_someip_log_error("event is null");
		return NULL;
	}

	return ne_someip_provided_event_t_ref(event_behaviour);
}

void ne_someip_provided_event_unref(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour) {
		return;
	}

	ne_someip_provided_event_t_unref(event_behaviour);
}

ne_someip_provided_event_t*
ne_someip_provided_event_create(const ne_someip_event_config_t* config)
{
	if (NULL == config) {
		ne_someip_log_error("config is null");
		return NULL;
	}
	ne_someip_provided_event_t* p_event
		= (ne_someip_provided_event_t*)malloc(sizeof(ne_someip_provided_event_t));
	if (NULL == p_event) {
		ne_someip_log_error("malloc error");
		return NULL;
	}
	memset(p_event, 0, sizeof(ne_someip_provided_event_t));

	ne_someip_provided_event_t_ref_count_init(p_event);
	p_event->event_config = config;
	p_event->udp_collection_buffer_timeout = 0;
	p_event->udp_collection_trigger = ne_someip_udp_collection_trigger_unknown;
	p_event->eventgroup_ref_list = ne_someip_list_create();
	if (NULL == p_event->eventgroup_ref_list) {
		ne_someip_provided_event_unref(p_event);
		return NULL;
	}

	p_event->initial_value = NULL;

	return p_event;
}

void ne_someip_provided_event_t_free(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour) {
		return;
	}

	//TODO stop all behaviour
	// ne_someip_server_free_event_config(event_behaviour->event_config);
	ne_someip_list_destroy(event_behaviour->eventgroup_ref_list, NULL);
	event_behaviour->eventgroup_ref_list = NULL;
	if (NULL != event_behaviour->initial_value) {
		ne_someip_payload_unref(event_behaviour->initial_value);
		event_behaviour->initial_value = NULL;
	}
	ne_someip_provided_event_t_ref_count_deinit(event_behaviour);
	free(event_behaviour);
	event_behaviour = NULL;
}

ne_someip_error_code_t
ne_someip_provided_event_send(ne_someip_provided_event_t* event_behaviour,
	ne_someip_header_t* header, ne_someip_payload_t* payload,
	void* endpoint, ne_someip_send_spec_t* seq_id, ne_someip_endpoint_send_policy_t send_policy)
{
	if (NULL == event_behaviour || NULL == header || NULL == endpoint || NULL == event_behaviour->event_config) {
		ne_someip_log_error("method_behaviour or header or endpoint or config is null");
		return ne_someip_error_code_failed;
	}
	ne_someip_log_debug("start event id [%d]", header->method_id);

	ne_someip_l4_protocol_t protocol = ne_someip_provided_get_event_comm_type(event_behaviour);
	if ((ne_someip_protocol_udp != protocol) && (ne_someip_protocol_tcp != protocol)) {
		ne_someip_log_error("transport protocol is error");
		return ne_someip_error_code_failed;
	}

	bool is_send = false;
	ne_someip_list_iterator_t* eventgroup_ref_iter = ne_someip_list_iterator_create(event_behaviour->eventgroup_ref_list);
	while (ne_someip_list_iterator_valid(eventgroup_ref_iter)) {
		ne_someip_list_t* subscriber_list = NULL;
		uint32_t multicast_ip = 0;
		uint16_t multicast_port = 0;
		bool is_send_by_multicast
			= ne_someip_provided_eg_get_subscribers((ne_someip_provided_eg_t*)(ne_someip_list_iterator_data(eventgroup_ref_iter)), protocol,
				&subscriber_list, &multicast_ip, &multicast_port);
		if (is_send_by_multicast) {
			ne_someip_trans_buffer_struct_t* trans_buffer = ne_someip_provided_event_create_trans_buffer(header, payload);
			if (NULL == trans_buffer) {
				ne_someip_list_iterator_next(eventgroup_ref_iter);
				continue;
			}
			if (0 == multicast_ip || 0 == multicast_port) {
				ne_someip_ep_tool_trans_buffer_free(trans_buffer);
				ne_someip_list_iterator_next(eventgroup_ref_iter);
				continue;
			}
			ne_someip_sd_convert_uint32_to_ip("send event to", multicast_ip, __FILE__, __LINE__, __FUNCTION__);
			ne_someip_log_debug("port [%d]", multicast_port);
			ne_someip_endpoint_net_addr_t peer_addr;
			peer_addr.ip_addr = multicast_ip;
			peer_addr.port = multicast_port;
			peer_addr.type = ne_someip_address_type_ipv4;
			++(seq_id->send_num);
			ne_someip_endpoint_udp_data_send_async((ne_someip_endpoint_udp_data_t*)endpoint, trans_buffer,
				&peer_addr, &send_policy, seq_id);
			is_send = true;
		} else {
			if (NULL == subscriber_list || 0 == ne_someip_list_length(subscriber_list)) {
				ne_someip_list_iterator_next(eventgroup_ref_iter);
				continue;
			}

			is_send = true;
			ne_someip_list_iterator_t* subscriber_list_iter = ne_someip_list_iterator_create(subscriber_list);
			while (ne_someip_list_iterator_valid(subscriber_list_iter)) {
				//serialize header and payload
				ne_someip_trans_buffer_struct_t* trans_buffer = ne_someip_provided_event_create_trans_buffer(header, payload);
				if (NULL == trans_buffer) {
					ne_someip_list_iterator_next(subscriber_list_iter);
					continue;
				}
				//send
				ne_someip_endpoint_net_addr_t* peer_addr
					= (ne_someip_endpoint_net_addr_t*)(ne_someip_list_iterator_data(subscriber_list_iter));
				if (NULL == peer_addr) {
					ne_someip_log_error("peer addr is null");
					ne_someip_ep_tool_trans_buffer_free(trans_buffer);
					ne_someip_list_iterator_next(subscriber_list_iter);
					continue;
				}
				ne_someip_sd_convert_uint32_to_ip("send event to", peer_addr->ip_addr, __FILE__, __LINE__, __FUNCTION__);
				ne_someip_log_debug("port [%d]", peer_addr->port);
				if (ne_someip_protocol_udp == protocol) {
					++(seq_id->send_num);
					ne_someip_endpoint_udp_data_send_async((ne_someip_endpoint_udp_data_t*)endpoint, trans_buffer,
						peer_addr, &send_policy, seq_id);
				} else if (ne_someip_protocol_tcp == protocol) {
					++(seq_id->send_num);
					ne_someip_endpoint_tcp_data_send_async((ne_someip_endpoint_tcp_data_t*)endpoint, trans_buffer,
						peer_addr, &send_policy, seq_id);
				}
				ne_someip_list_iterator_next(subscriber_list_iter);
			}
			ne_someip_list_iterator_destroy(subscriber_list_iter);
		}
		ne_someip_list_iterator_next(eventgroup_ref_iter);
	}
	ne_someip_list_iterator_destroy(eventgroup_ref_iter);
	seq_id->is_notify = true;

	if (!is_send) {
		return ne_someip_error_code_failed;
	}

	return ne_someip_error_code_ok;
}

ne_someip_error_code_t
ne_someip_provided_event_send_initial_value(ne_someip_provided_event_t* event_behaviour,
	ne_someip_header_t* header, void* endpoint, void* seq_id, ne_someip_endpoint_send_policy_t send_policy,
	uint32_t ip_addr, uint16_t port)
{
	if (NULL == event_behaviour || NULL == header || NULL == endpoint || NULL == event_behaviour->event_config) {
		ne_someip_log_error("method_behaviour or header or endpoint or config is null");
		return ne_someip_error_code_failed;
	}
	ne_someip_log_debug("start event id [%d]", header->method_id);

    header->method_id = header->method_id | NESOMEIP_EVENT_H_VALUE;
    header->message_length = NE_SOMEIP_HEADER_LEN_IN_MSG_LEN + ne_someip_server_get_payload_len(event_behaviour->initial_value);
    ne_someip_l4_protocol_t protocol = ne_someip_provided_get_event_comm_type(event_behaviour);

	//serialize header and payload
	uint8_t* someip_header = (uint8_t*)malloc(NE_SOMEIP_HEADER_LENGTH);
	if (NULL == someip_header) {
		ne_someip_log_error("malloc error");
		return ne_someip_error_code_failed;
	}
	memset(someip_header, 0, NE_SOMEIP_HEADER_LENGTH);

	uint8_t* tmp_data = someip_header;
	bool ret = ne_someip_msg_header_ser(&tmp_data, header);
	if (!ret) {
		ne_someip_log_error("serialize error");
		if (NULL != someip_header) {
			free(someip_header);
			someip_header = NULL;
		}
		return ne_someip_error_code_failed;
	} else {
		ne_someip_trans_buffer_struct_t* trans_buffer =
			(ne_someip_trans_buffer_struct_t*)malloc(sizeof(ne_someip_trans_buffer_struct_t));
		if (NULL == trans_buffer) {
			ne_someip_log_error("malloc ne_someip_trans_buffer_struct_t failed.");
			if (NULL != someip_header) {
				free(someip_header);
				someip_header = NULL;
			}
			return ne_someip_error_code_failed;
		}
		memset(trans_buffer, 0, sizeof(ne_someip_trans_buffer_struct_t));
		trans_buffer->ipc_data = NULL;
		trans_buffer->someip_header = NULL;
		trans_buffer->payload = NULL;

		//header
		ne_someip_endpoint_buffer_t* ep_header
			= (ne_someip_endpoint_buffer_t*)malloc(sizeof(ne_someip_endpoint_buffer_t));
		if (NULL == ep_header) {
			ne_someip_log_error("malloc error");
			if (NULL != someip_header) {
				free(someip_header);
				someip_header = NULL;
			}
			if (NULL != trans_buffer) {
				free(trans_buffer);
				trans_buffer = NULL;
			}
			return ne_someip_error_code_failed;
		}
		memset(ep_header, 0, sizeof(ne_someip_endpoint_buffer_t));
		ep_header->iov_buffer = (char*)someip_header;
		ep_header->size = NE_SOMEIP_HEADER_LENGTH;
		trans_buffer->someip_header = ep_header;

		// for (int i = 0; i < ep_header->size; i++) {
		// 	ne_someip_log_info("+++++++++++++++++++event data[%d] = %d", i, (ep_header->iov_buffer)[i]);
		// }

		// payload
		trans_buffer->payload = ne_someip_payload_ref(event_behaviour->initial_value);

		//send
		ne_someip_endpoint_net_addr_t peer_addr;
		peer_addr.ip_addr = ip_addr;
		peer_addr.port = port;
		peer_addr.type = ne_someip_address_type_ipv4;

		ne_someip_sd_convert_uint32_to_ip("send event to", ip_addr, __FILE__, __LINE__, __FUNCTION__);
		ne_someip_log_info("port [%d]", port);
		if (ne_someip_protocol_udp == protocol) {
			ne_someip_endpoint_udp_data_send_async((ne_someip_endpoint_udp_data_t*)endpoint, trans_buffer,
				&peer_addr, &send_policy, seq_id);
		} else if (ne_someip_protocol_tcp == protocol) {
			ne_someip_endpoint_tcp_data_send_async((ne_someip_endpoint_tcp_data_t*)endpoint, trans_buffer,
				&peer_addr, &send_policy, seq_id);
		} else {
			ne_someip_log_error("transport protocol is error");
			ne_someip_ep_tool_trans_buffer_free(trans_buffer);
			return ne_someip_error_code_failed;
		}
	}

	return ne_someip_error_code_ok;
}

/*void ne_someip_provided_event_find_subscriber(ne_someip_provided_event_t* event_behaviour,
	bool* is_multicast, ne_someip_list_t* subscriber_list, uint32_t* multicast_ip, uint16_t* multicast_port)
{
	if (NULL == event_behaviour) {
		ne_someip_log_error("event is null");
		return;
	}

	ne_someip_l4_protocol_t protocol = ne_someip_provided_get_event_comm_type(event_behaviour);
	ne_someip_list_iterator_t* iter = ne_someip_list_iterator_create(event_behaviour->eventgroup_ref_list);
	while (ne_someip_list_iterator_valid(iter)) {
		ne_someip_list_t* sub_list = NULL;
		uint32_t multicast_ip = 0;
		uint16_t multicast_port = 0;
		bool is_send_by_multicast
			= ne_someip_provided_eg_get_subscribers((ne_someip_provided_eg_t*)(ne_someip_list_iterator_data(iter)), protocol,
				subscriber_list, &multicast_ip, &multicast_port);

		if (is_send_by_multicast) {

		} else {

		}
		if (NULL == sub_list || 0 == ne_someip_list_length(sub_list)) {
			ne_someip_list_iterator_next(iter);
			continue;
		}
		ne_someip_list_iterator_t* sub_list_iter = ne_someip_list_iterator_create(sub_list);
		while (ne_someip_list_iterator_valid(sub_list_iter)) {
			ne_someip_endpoint_net_addr_t* net_addr = (ne_someip_endpoint_net_addr_t*)ne_someip_list_iterator_data(sub_list_iter);
			if (NULL == net_addr) {
				ne_someip_log_error("net addr is null");
				ne_someip_list_iterator_next(sub_list_iter);
				continue;
			}

			ne_someip_list_element_t* element = ne_someip_list_find(subscriber_list, net_addr, ne_someip_subscriber_cmp, NULL);
			if (NULL == element) {
				ne_someip_list_append(subscriber_list, net_addr);
			}
			ne_someip_list_iterator_next(sub_list_iter);
		}
		ne_someip_list_iterator_destroy(sub_list_iter);

		ne_someip_list_iterator_next(iter);
	}
	ne_someip_list_iterator_destroy(iter);
}*/

void ne_someip_provided_event_reset(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour) {
		return;
	}

	//TODO
}

bool ne_someip_provided_event_set_udp_collection_time_out(ne_someip_provided_event_t* event_behaviour,
	uint32_t time)
{
	if (NULL == event_behaviour) {
		return false;
	}

	event_behaviour->udp_collection_buffer_timeout = time;
	return true;
}

bool ne_someip_provided_event_set_udp_trigger_mode(ne_someip_provided_event_t* event_behaviour,
	ne_someip_udp_collection_trigger_t trigger)
{
	if (NULL == event_behaviour) {
		return false;
	}

	event_behaviour->udp_collection_trigger = trigger;
	return true;
}

bool ne_someip_provided_event_set_eg_ref(ne_someip_provided_event_t* event_behaviour,
	const ne_someip_provided_eg_t* eg_ref)
{
	if (NULL == event_behaviour) {
		ne_someip_log_error("event hehaviour is null");
		return false;
	}

	ne_someip_list_append(event_behaviour->eventgroup_ref_list, eg_ref);
	return true;
}

void ne_someip_provided_event_change_field_payload(ne_someip_provided_event_t* event_behaviour,
	ne_someip_payload_t* payload)
{
	ne_someip_log_debug("start");
	if (NULL == event_behaviour || NULL == event_behaviour->event_config) {
		ne_someip_log_error("event hehaviour is null or config is null");
		return;
	}

	if (false == event_behaviour->event_config->is_field) {
		ne_someip_log_debug("is_field is false");
		return;
	}

	if (NULL != event_behaviour->initial_value) {
		ne_someip_payload_unref(event_behaviour->initial_value);
		event_behaviour->initial_value = NULL;
	}
	event_behaviour->initial_value = ne_someip_payload_ref(payload);
}

ne_someip_trans_buffer_struct_t*
ne_someip_provided_event_create_trans_buffer(ne_someip_header_t* header, ne_someip_payload_t* payload)
{
	header->method_id = header->method_id | NESOMEIP_EVENT_H_VALUE;
	uint8_t* someip_header = (uint8_t*)malloc(NE_SOMEIP_HEADER_LENGTH);
	if (NULL == someip_header) {
		ne_someip_log_error("malloc error");
		//ne_someip_list_destroy(subscriber_list, NULL);
		return NULL;
	}
	memset(someip_header, 0, NE_SOMEIP_HEADER_LENGTH);

	uint8_t* tmp_data = someip_header;
	bool ret = ne_someip_msg_header_ser(&tmp_data, header);
	if (!ret) {
		ne_someip_log_error("serialize error");
		if (NULL != someip_header) {
			free(someip_header);
		}
		return NULL;
	}

	ne_someip_trans_buffer_struct_t* trans_buffer =
		(ne_someip_trans_buffer_struct_t*)malloc(sizeof(ne_someip_trans_buffer_struct_t));
	if (NULL == trans_buffer) {
		ne_someip_log_error("malloc ne_someip_trans_buffer_struct_t failed.");
		if (NULL != someip_header) {
			free(someip_header);
		}
		return NULL;
	}
	memset(trans_buffer, 0, sizeof(ne_someip_trans_buffer_struct_t));
	trans_buffer->ipc_data = NULL;
	trans_buffer->someip_header = NULL;
	trans_buffer->payload = NULL;

	//header
	ne_someip_endpoint_buffer_t* ep_header
		= (ne_someip_endpoint_buffer_t*)malloc(sizeof(ne_someip_endpoint_buffer_t));
	if (NULL == ep_header) {
		ne_someip_log_error("malloc error");
		if (NULL != someip_header) {
			free(someip_header);
			someip_header = NULL;
		}
		if (NULL != trans_buffer) {
			free(trans_buffer);
			trans_buffer = NULL;
		}
		return NULL;
	}
	memset(ep_header, 0, sizeof(ne_someip_endpoint_buffer_t));
	ep_header->iov_buffer = (char*)someip_header;
	ep_header->size = NE_SOMEIP_HEADER_LENGTH;
	trans_buffer->someip_header = ep_header;
	trans_buffer->payload = payload;

	return trans_buffer;
}

ne_someip_l4_protocol_t
ne_someip_provided_get_event_comm_type(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour || NULL == event_behaviour->event_config) {
		ne_someip_log_error("event_behaviour is null");
		return 0;
	}

	return event_behaviour->event_config->comm_type;
}

uint32_t ne_someip_provided_event_get_udp_collection_time_out(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour) {
		ne_someip_log_error("event_behaviour is null");
		return 0;
	}

	return event_behaviour->udp_collection_buffer_timeout;
}

ne_someip_udp_collection_trigger_t
ne_someip_provided_event_get_udp_trgger_mode(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour) {
		ne_someip_log_error("event_behaviour is null");
		return 0;
	}

	return event_behaviour->udp_collection_trigger;
}

uint32_t ne_someip_provided_event_get_segment_len(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour || NULL == event_behaviour->event_config) {
		ne_someip_log_error("event_behaviour or config is null");
		return 0;
	}

	return event_behaviour->event_config->max_segment_length;
}

uint32_t
ne_someip_provided_event_get_separation_time(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour || NULL == event_behaviour->event_config) {
		ne_someip_log_error("event_behaviour or config is null");
		return 0;
	}

	return event_behaviour->event_config->separation_time;
}

ne_someip_serializer_type_t
ne_someip_provided_event_get_ser_type(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour || NULL == event_behaviour->event_config) {
		ne_someip_log_error("event_behaviour or config is null");
		return 0;
	}

	return event_behaviour->event_config->serializer_type;
}

bool
ne_someip_provided_event_get_field_flag(ne_someip_provided_event_t* event_behaviour)
{
	if (NULL == event_behaviour || NULL == event_behaviour->event_config) {
		ne_someip_log_error("event_behaviour or config is null");
		return 0;
	}

	return event_behaviour->event_config->is_field;
}

ne_someip_list_t*
ne_someip_provided_event_get_eg_list(ne_someip_provided_event_t* event_behaviour) {
    if (NULL == event_behaviour || NULL == event_behaviour->eventgroup_ref_list) {
        ne_someip_log_error("event_behaviour or eventgroup_ref_list is null");
        return NULL;
    }

    return event_behaviour->eventgroup_ref_list;
}

/* EOF */
