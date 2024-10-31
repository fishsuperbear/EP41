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
#include "ne_someip_provided_method_behaviour.h"
#include "ne_someip_object.h"
#include "ne_someip_log.h"
#include "ne_someip_endpoint_udp_data.h"
#include "ne_someip_endpoint_tcp_data.h"
#include "ne_someip_message.h"
#include "ne_someip_endpoint_udp_data.h"
#include "ne_someip_endpoint_tcp_data.h"
#include "ne_someip_sd_tool.h"
#include "ne_someip_endpoint_tool.h"

struct ne_someip_provided_method_behaviour
{
	const ne_someip_method_config_t* method_config;
	uint32_t udp_collection_buffer_timeout;
    ne_someip_udp_collection_trigger_t udp_collection_trigger;
	ne_someip_map_t* request_permission;//<ne_someip_remote_client_info_t*, ne_someip_permission_t*>
	ne_someip_permission_t default_permission;
	NEOBJECT_MEMBER
};

void ne_someip_provided_method_t_free(ne_someip_provided_method_t* method_behaviour);

NEOBJECT_FUNCTION(ne_someip_provided_method_t);

ne_someip_provided_method_t* ne_someip_provided_method_ref(ne_someip_provided_method_t* method_behaviour)
{
	if (NULL == method_behaviour) {
		ne_someip_log_error("method is null");
		return NULL;
	}

	return ne_someip_provided_method_t_ref(method_behaviour);
}

void ne_someip_provided_method_unref(ne_someip_provided_method_t* method_behaviour)
{
	if (NULL == method_behaviour) {
		return;
	}

	ne_someip_provided_method_t_unref(method_behaviour);
}

ne_someip_provided_method_t*
ne_someip_provided_method_create(const ne_someip_method_config_t* config)
{
	ne_someip_provided_method_t* p_method = (ne_someip_provided_method_t*)malloc(sizeof(ne_someip_provided_method_t));
	if (NULL == p_method) {
		ne_someip_log_error("malloc error");
		return NULL;
	}
	memset(p_method, 0, sizeof(ne_someip_provided_method_t));

	ne_someip_provided_method_t_ref_count_init(p_method);
	p_method->method_config = config;
	p_method->udp_collection_buffer_timeout = 0;
	p_method->udp_collection_trigger = ne_someip_udp_collection_trigger_unknown;

	return p_method;
}

void ne_someip_provided_method_t_free(ne_someip_provided_method_t* method_behaviour)
{
	if (NULL == method_behaviour) {
		return;
	}
	//TODO stop all behaviour
	// ne_someip_server_free_method_config(method_behaviour->method_config);
	if (NULL != method_behaviour->request_permission) {
		ne_someip_map_remove_all(method_behaviour->request_permission);
		method_behaviour->request_permission = NULL;
	}
	ne_someip_provided_method_t_ref_count_deinit(method_behaviour);
	free(method_behaviour);
	method_behaviour = NULL;
}

ne_someip_error_code_t ne_someip_provided_method_send(ne_someip_provided_method_t* method_behaviour,
	ne_someip_header_t* header, ne_someip_payload_t* payload
	, ne_someip_remote_client_info_t* remote_addr, void* endpoint, void* seq_id,
	ne_someip_endpoint_send_policy_t send_policy)
{
	if (NULL == method_behaviour) {
		ne_someip_log_info("send response for null method");
	}
	if (NULL == header || NULL == remote_addr || NULL == endpoint) {
		ne_someip_log_error("method_behaviour or message or remote_addr or endpoint is null");
		return ne_someip_error_code_failed;
	}
	ne_someip_sd_convert_uint32_to_ip("send response to", remote_addr->ipv4, __FILE__, __LINE__, __FUNCTION__);
	ne_someip_log_info("port [%d]", remote_addr->port);

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
		}
		return ne_someip_error_code_failed;
	} else {
		ne_someip_trans_buffer_struct_t* trans_buffer =
		    (ne_someip_trans_buffer_struct_t*)malloc(sizeof(ne_someip_trans_buffer_struct_t));
	    if (NULL == trans_buffer) {
	        ne_someip_log_error("malloc ne_someip_trans_buffer_struct_t failed.");
	        if (NULL != someip_header) {
				free(someip_header);
			}
	        return ne_someip_error_code_failed;
	    }
		memset(trans_buffer, 0, sizeof(ne_someip_trans_buffer_struct_t));
		trans_buffer->ipc_data = NULL;
		trans_buffer->someip_header = NULL;
		trans_buffer->payload = NULL;

		//header
		ne_someip_endpoint_buffer_t* tmp_header = (ne_someip_endpoint_buffer_t*)malloc(sizeof(ne_someip_endpoint_buffer_t));
		if (NULL == tmp_header) {
			ne_someip_log_error("malloc error");
	    	if (NULL != someip_header) {
				free(someip_header);
			}
			if (NULL != trans_buffer) {
				free(trans_buffer);
			}
	    	return ne_someip_error_code_failed;
		}
		memset(tmp_header, 0, sizeof(ne_someip_endpoint_buffer_t));
		tmp_header->iov_buffer = (char*)someip_header;
		tmp_header->size = NE_SOMEIP_HEADER_LENGTH;
		trans_buffer->someip_header = tmp_header;

		// payload
		trans_buffer->payload = payload;

		//peer address
		ne_someip_endpoint_net_addr_t peer_addr;
		// memcpy(peer_addr.ip_addr, remote_addr->ip_addr， NESOMEIP_IP_ADDR_LENGTH);
		peer_addr.ip_addr = remote_addr->ipv4;
		peer_addr.port = remote_addr->port;
		peer_addr.type = remote_addr->type;

		//send
		if (ne_someip_protocol_udp == remote_addr->protocol) {
			ne_someip_endpoint_udp_data_send_async((ne_someip_endpoint_udp_data_t*)endpoint, trans_buffer,
                             &peer_addr, &send_policy, seq_id);
		} else if (ne_someip_protocol_tcp == remote_addr->protocol) {
			ne_someip_endpoint_tcp_data_send_async((ne_someip_endpoint_tcp_data_t*)endpoint, trans_buffer,
                             &peer_addr, &send_policy, seq_id);
		} else {
			ne_someip_log_error("transport protocol is error");
			ne_someip_ep_tool_trans_buffer_no_payload_free(trans_buffer);
			return ne_someip_error_code_failed;
		}
	}

	return ne_someip_error_code_ok;
}

// void ne_someip_provided_method_recv_req(ne_someip_provided_method_t* method_behaviour,
// 	const ne_someip_message_t* message, const ne_someip_remote_client_info_t* remote_addr)
// {
// 	if (NULL == method_behaviour) {
// 		ne_someip_log_error("method_behaviour is null");
// 		return;
// 	}
// 	ne_someip_recv_request_handler handler = method_behaviour->method_handler;
// 	handler(message, remote_addr);
// }

// ne_someip_error_code_t ne_someip_provided_method_reg_method_handler(ne_someip_provided_method_t* method_behaviour,
// 	ne_someip_recv_request_handler handler)
// {
// 	if (NULL == method_behaviour) {
// 		ne_someip_log_error("method_behaviour is null");
// 		return;
// 	}

// 	method_behaviour->method_handler = handler;
// }

// ne_someip_error_code_t ne_someip_provided_method_unreg_method_handler(ne_someip_provided_method_t* method_behaviour)
// {
// 	if (NULL == method_behaviour) {
// 		ne_someip_log_error("method_behaviour is null");
// 		return;
// 	}

// 	method_behaviour->method_handler = NULL;
// }

void ne_someip_provided_method_reset(ne_someip_provided_method_t* method_behaviour)
{
	if (NULL == method_behaviour) {
		return;
	}

	//TODO
}

bool ne_someip_provided_method_set_udp_collection_time_out(ne_someip_provided_method_t* method_behaviour,
	uint32_t time)
{
	if (NULL == method_behaviour) {
		ne_someip_log_error("method_behaviour is null");
		return false;
	}

	method_behaviour->udp_collection_buffer_timeout = time;
	return true;
}

bool ne_someip_provided_method_set_udp_trigger_mode(ne_someip_provided_method_t* method_behaviour,
	ne_someip_udp_collection_trigger_t trigger)
{
	if (NULL == method_behaviour) {
		ne_someip_log_error("method_behaviour is null");
		return false;
	}

	method_behaviour->udp_collection_trigger = trigger;
	return true;
}

ne_someip_error_code_t
ne_someip_provided_method_set_request_permission(ne_someip_provided_method_t* method_behaviour,
	const ne_someip_remote_client_info_t* remote_addr, ne_someip_permission_t priority)
{
	//TODO, should check requester permission, if unkown ,should save permission
	return ne_someip_error_code_ok;
}

ne_someip_permission_t ne_someip_provided_method_get_request_permission(ne_someip_provided_method_t* method_behaviour,
	const ne_someip_remote_client_info_t* remote_addr)
{
	if (NULL == method_behaviour) {
		ne_someip_log_error("method is null");
		return ne_someip_permission_reject;
	}
	//TODO if default is allow,return allow; if default is unkown, check the pemission_map;
	return method_behaviour->default_permission;
}

uint32_t
ne_someip_provided_method_get_udp_collection_time_out(const ne_someip_provided_method_t* method_behaviour)
{
	if (NULL != method_behaviour) {
		return method_behaviour->udp_collection_buffer_timeout;
	}

	return 0;
}

ne_someip_udp_collection_trigger_t
ne_someip_provided_method_get_udp_trigger_mode(const ne_someip_provided_method_t* method_behaviour)
{
	if (NULL != method_behaviour) {
		return method_behaviour->udp_collection_trigger;
	}

	return 0;
}

ne_someip_l4_protocol_t
ne_someip_provided_get_method_comm_type(const ne_someip_provided_method_t* method_behaviour)
{
	if (NULL != method_behaviour && NULL != method_behaviour->method_config) {
		return method_behaviour->method_config->comm_type;
	}

	return 0;
}

uint32_t
ne_someip_provided_method_get_segment_len(const ne_someip_provided_method_t* method_behaviour)
{
	if (NULL != method_behaviour && NULL != method_behaviour->method_config) {
		return method_behaviour->method_config->segment_length_response;
	}

	return 0;
}

uint32_t
ne_someip_provided_method_get_separation_time(const ne_someip_provided_method_t* method_behaviour)
{
	if (NULL != method_behaviour && NULL != method_behaviour->method_config) {
		return method_behaviour->method_config->sepration_time_response;
	}

	return 0;
}

bool
ne_someip_provided_method_get_message_type(const ne_someip_provided_method_t* method_behaviour)
{
	if (NULL != method_behaviour && NULL != method_behaviour->method_config) {
		return method_behaviour->method_config->fire_and_forget;
	}

	return 0;
}