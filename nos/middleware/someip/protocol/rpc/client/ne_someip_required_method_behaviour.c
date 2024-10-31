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
#include "ne_someip_required_method_behaviour.h"
#include "ne_someip_message.h"
#include "ne_someip_ipc_behaviour.h"
#include "ne_someip_log.h"
#include "ne_someip_endpoint_tool.h"
#include "ne_someip_sd_tool.h"

ne_someip_required_method_behaviour_t* ne_someip_req_method_behaviour_new(ne_someip_method_config_t* config)
{
    if (NULL == config) {
        ne_someip_log_error("config is NULL.");
        return NULL;
    }
    ne_someip_required_method_behaviour_t* method_behav =
        (ne_someip_required_method_behaviour_t*)malloc(sizeof(ne_someip_required_method_behaviour_t));
    if (NULL == method_behav) {
        ne_someip_log_error("malloc ne_someip_required_method_behaviour_t error.");
        return method_behav;
    }
    memset(method_behav, 0, sizeof(*method_behav));
    method_behav->config = config;
    method_behav->udp_collection_buffer_timeout = 0;
    method_behav->udp_collection_trigger = ne_someip_udp_collection_trigger_unknown;

    return method_behav;
}

void ne_someip_req_method_behaviour_free(ne_someip_required_method_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_info("behaviour is NULL.");
        return;
    }

    behaviour->config = NULL;
    free(behaviour);
    behaviour = NULL;
}

//  send to endpoint
ne_someip_error_code_t ne_someip_req_method_behav_send_request(void* endpoint, ne_someip_endpoint_net_addr_t* peer_addr,
    ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_endpoint_send_policy_t* policy, const void* seq_data)
{
    if (NULL == endpoint || NULL == peer_addr || NULL == header || NULL == policy) {
        ne_someip_log_error("endpoint or peer_addr or header or policy is NULL.");
        return ne_someip_error_code_failed;
    }

    // serialize someip message
    // ne_someip_list_t* buffer_list = ne_someip_list_create();
    // if (NULL == buffer_list) {
    //     ne_someip_log_error("ne_someip_list_create failed.");
    //     return ne_someip_error_code_failed;
    // }
    // bool ser_res = ne_someip_msg_ser(buffer_list, header, payload);
    // if (!ser_res) {
    //     ne_someip_log_error("serialize someip message error.");
    //     ne_someip_list_destroy(buffer_list, ne_someip_ep_tool_ep_buf_list_free);
    //     return ne_someip_error_code_failed;
    // }
    ne_someip_trans_buffer_struct_t* trans_buffer = (ne_someip_trans_buffer_struct_t*)malloc(sizeof(ne_someip_trans_buffer_struct_t));
    if (NULL == trans_buffer) {
        ne_someip_log_error("malloc ne_someip_trans_buffer_struct_t failed.");
        return ne_someip_error_code_failed;
    }
    trans_buffer->ipc_data = NULL;
    trans_buffer->someip_header = NULL;
    trans_buffer->payload = payload;
    uint8_t* data = (uint8_t*)malloc(NE_SOMEIP_HEADER_LENGTH);
    if (NULL == data) {
        ne_someip_log_error("malloc data error.");
        ne_someip_ep_tool_trans_buffer_no_payload_free(trans_buffer);
        return ne_someip_error_code_failed;
    }
    memset(data, 0, NE_SOMEIP_HEADER_LENGTH);
    uint32_t length = NE_SOMEIP_HEADER_LENGTH;
    uint8_t* tmp_data = data;
    bool ser_ret = ne_someip_msg_header_ser(&tmp_data, header);
    if (!ser_ret) {
        ne_someip_log_error("ne_someip_msg_header_ser failed.");
        free(data);
        ne_someip_ep_tool_trans_buffer_no_payload_free(trans_buffer);
        return ne_someip_error_code_failed;
    }
    trans_buffer->someip_header = (ne_someip_endpoint_buffer_t*)malloc(sizeof(ne_someip_endpoint_buffer_t));
    if (NULL == trans_buffer->someip_header) {
        ne_someip_log_error("ne_someip_msg_header_ser failed.");
        free(data);
        ne_someip_ep_tool_trans_buffer_no_payload_free(trans_buffer);
        return ne_someip_error_code_failed;
    }
    trans_buffer->someip_header->iov_buffer = (char*)data;
    trans_buffer->someip_header->size = length;

    ne_someip_client_send_seq_data_t* tmp_seq_data =
        (ne_someip_client_send_seq_data_t*)malloc(sizeof(ne_someip_client_send_seq_data_t));
    if (NULL == tmp_seq_data) {
        ne_someip_log_error("malloc ne_someip_client_send_seq_data_t error.");
        ne_someip_ep_tool_trans_buffer_no_payload_free(trans_buffer);
        // ne_someip_list_destroy(buffer_list, ne_someip_ep_tool_ep_buf_list_free);
        return ne_someip_error_code_failed;
    }
    tmp_seq_data->method_id = header->method_id;
    tmp_seq_data->seq_data = seq_data;

    ne_someip_error_code_t ret = ne_someip_error_code_unknown;
    ne_someip_endpoint_base_t* ep_base = (ne_someip_endpoint_base_t*)endpoint;
    switch (ep_base->endpoint_type)
    {
        case ne_someip_endpoint_type_tcp:
        {
            ne_someip_sd_convert_uint32_to_ip("tcp :", peer_addr->ip_addr, __FILE__, __LINE__, __FUNCTION__);
            ne_someip_log_debug("port [%d]", peer_addr->port);
            ret = ne_someip_endpoint_tcp_data_send_async((ne_someip_endpoint_tcp_data_t*)endpoint, trans_buffer,
                peer_addr, policy, tmp_seq_data);
            break;
        }
        case ne_someip_endpoint_type_udp:
        {
            ne_someip_sd_convert_uint32_to_ip("udp :", peer_addr->ip_addr, __FILE__, __LINE__, __FUNCTION__);
            ne_someip_log_debug("port [%d]", peer_addr->port);
            ret = ne_someip_endpoint_udp_data_send_async((ne_someip_endpoint_udp_data_t*)endpoint, trans_buffer,
                peer_addr, policy, tmp_seq_data);
            break;
        }
        default :
        {
           ne_someip_log_error("endpoint type is wrong. neither tcp nor udp.");
           ret = ne_someip_error_code_failed;
           break;
        }
    }

    if (ne_someip_error_code_ok != ret) {
        ne_someip_ep_tool_trans_buffer_no_payload_free(trans_buffer);
    }

	return ret;
}

ne_someip_error_code_t ne_someip_req_method_behav_send_request_to_daemon(ne_someip_header_t* header,
    ne_someip_payload_t* payload, void* required_instance, ne_someip_endpoint_send_policy_t* send_plicy)
{
    if (NULL == required_instance) {
        ne_someip_log_error("required_instance is NULL.");
        return ne_someip_error_code_failed;
    }

    ne_someip_error_code_t ret =
        ne_someip_ipc_behaviour_send_req_msg(header, payload,
        (ne_someip_required_service_instance_t*)required_instance, send_plicy);

    return ret;
}

//  when port reuse, send to daemon
ne_someip_error_code_t ne_someip_req_method_behaviour_reg_response_handler_to_daemon(void* required_instance)
{
    if (NULL == required_instance) {
        ne_someip_log_error("required_instance is NULL.");
        return ne_someip_error_code_failed;
    }

    ne_someip_error_code_t ret =
        ne_someip_ipc_behaviour_reg_resp_handler((ne_someip_required_service_instance_t*)required_instance);

    return ret;
}

ne_someip_error_code_t ne_someip_req_method_behaviour_unreg_response_handler_to_daemon(void* required_instance)
{
     if (NULL == required_instance) {
        ne_someip_log_error("required_instance is NULL.");
        return ne_someip_error_code_failed;
    }

    ne_someip_error_code_t ret =
        ne_someip_ipc_behaviour_unreg_resp_handler((ne_someip_required_service_instance_t*)required_instance);

    return ret;
}

//  receive from endpoint
void ne_someip_req_method_behav_recv_response(ne_someip_required_service_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload)
{
    if (NULL == instance || NULL == header) {
        ne_someip_log_error("instance or header is NULL.");
        return;
    }

    ne_someip_sync_obj_sync_start(instance->resp_handler_sync);
    ne_someip_list_iterator_t* iter = ne_someip_list_iterator_create(instance->resp_handler_list);
    while(ne_someip_list_iterator_valid(iter)) {
        if (NULL != ne_someip_list_iterator_data(iter)) {
            ((ne_someip_saved_recv_response_handler_t*)(ne_someip_list_iterator_data(iter)))->handler(instance,
                header, payload,
                ((ne_someip_saved_recv_response_handler_t*)(ne_someip_list_iterator_data(iter)))->user_data);
        }
        ne_someip_list_iterator_next(iter);
    }
    ne_someip_list_iterator_destroy(iter);
    ne_someip_sync_obj_sync_end(instance->resp_handler_sync);
}