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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_SERVICE_INSTANCE_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_SERVICE_INSTANCE_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_define.h"
#include "ne_someip_handler.h"
#include "ne_someip_config_define.h"
#include "ne_someip_client_define.h"
#include "ne_someip_endpoint_define.h"
#include "ne_someip_ipc_define.h"
#include "ne_someip_looper.h"

/********************************interface************************************/
// user thread
ne_someip_required_service_instance_t* ne_someip_required_service_instance_new(ne_someip_client_id_t client_id,
    ne_someip_looper_t* work_looper, ne_someip_looper_t* io_looper,
    const ne_someip_required_service_instance_config_t* config, ne_someip_required_service_connect_behaviour_t* behav,
    const ne_someip_service_instance_spec_t* spec, ne_someip_endpoint_unix_t* unix_endpoint);

ne_someip_required_service_instance_t*
    ne_someip_required_service_instance_ref(ne_someip_required_service_instance_t* instance);

void ne_someip_required_service_instance_unref(ne_someip_required_service_instance_t* instance);

ne_someip_error_code_t ne_someip_req_serv_inst_start(ne_someip_required_service_instance_t* instance);

ne_someip_error_code_t ne_someip_req_serv_inst_stop(ne_someip_required_service_instance_t* instance);

ne_someip_error_code_t ne_someip_req_serv_inst_reg_sub_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_subscribe_status_handler handler, const void* user_data);

ne_someip_error_code_t ne_someip_req_serv_inst_unreg_sub_handler(
    ne_someip_required_service_instance_t* instance, ne_someip_subscribe_status_handler handler);

ne_someip_error_code_t ne_someip_req_serv_inst_reg_send_status_handler(
    ne_someip_required_service_instance_t* instance, ne_someip_send_req_status_handler handler, const void* user_data);

ne_someip_error_code_t ne_someip_req_serv_inst_unreg_send_status_handler(
    ne_someip_required_service_instance_t* instance, ne_someip_send_req_status_handler handler);

ne_someip_error_code_t ne_someip_req_serv_inst_reg_event_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_recv_event_handler handler, const void* user_data);

ne_someip_error_code_t ne_someip_req_serv_inst_unreg_event_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_recv_event_handler handler);

ne_someip_error_code_t ne_someip_req_serv_inst_reg_response_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_recv_response_handler handler, const void* user_data);

ne_someip_error_code_t ne_someip_req_serv_inst_unreg_response_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_recv_response_handler handler);

ne_someip_error_code_t ne_someip_req_serv_inst_create_req_header_no_session_id(
    ne_someip_required_service_instance_t* instance, ne_someip_header_t* header, ne_someip_method_id_t method_id);

ne_someip_error_code_t ne_someip_req_serv_inst_create_req_header_with_session_id(
    ne_someip_required_service_instance_t* instance, ne_someip_header_t* header,
    ne_someip_method_id_t method_id, ne_someip_session_id_t session_id);

// work thread
ne_someip_error_code_t ne_someip_req_serv_inst_start_subscribe_eventgroup(ne_someip_required_service_instance_t* instance,
    ne_someip_eventgroup_id_t eventgroup_id);

ne_someip_error_code_t ne_someip_req_serv_inst_stop_subscribe_eventgroup(ne_someip_required_service_instance_t* instance,
    ne_someip_eventgroup_id_t eventgroup_id, bool* notify_obj, pthread_t tid);

ne_someip_error_code_t ne_someip_req_serv_inst_send_request(ne_someip_required_service_instance_t* instance,
    const void* sequence_id, ne_someip_header_t* header, ne_someip_payload_t* payload);

// notify the subscribe status to upper app
void ne_someip_req_serv_inst_notify_sub_status(ne_someip_required_service_instance_t* instance,
    ne_someip_eventgroup_id_t eventgroup_id, ne_someip_subscribe_status_t status, ne_someip_error_code_t code);

// notify the send status to upper app
void ne_someip_req_serv_inst_notify_send_status(ne_someip_required_service_instance_t* instance,
    const void* sequence_id, ne_someip_method_id_t method_id, ne_someip_error_code_t code);

// get info from required_instance config
ne_someip_client_id_t ne_someip_req_serv_inst_get_client_id(const ne_someip_required_service_instance_t* instance);
ne_someip_service_id_t ne_someip_req_serv_inst_get_service_id(const ne_someip_required_service_instance_t* instance);
ne_someip_instance_id_t ne_someip_req_serv_inst_get_instance_id(const ne_someip_required_service_instance_t* instance);
ne_someip_major_version_t ne_someip_req_serv_inst_get_major_version(const ne_someip_required_service_instance_t* instance);
ne_someip_minor_version_t ne_someip_req_serv_inst_get_minor_version(const ne_someip_required_service_instance_t* instance);
ne_someip_message_type_enum_t ne_someip_req_serv_inst_get_method_type(const ne_someip_required_service_instance_t* instance,
    ne_someip_method_id_t method_id);
ne_someip_l4_protocol_t
ne_someip_req_serv_inst_get_method_protocol(const ne_someip_required_service_instance_t* instance,
    ne_someip_method_id_t method_id);
uint32_t ne_someip_req_serv_inst_get_ip_addr(const ne_someip_required_service_instance_t* instance);
uint16_t ne_someip_req_serv_inst_get_tcp_port(const ne_someip_required_service_instance_t* instance);
uint16_t ne_someip_req_serv_inst_get_udp_port(const ne_someip_required_service_instance_t* instance);
uint16_t ne_someip_req_serv_inst_get_multicast_port(const ne_someip_required_service_instance_t* instance);
ne_someip_address_type_t ne_someip_req_serv_inst_get_addr_type(const ne_someip_required_service_instance_t* instance);
bool ne_someip_req_serv_inst_get_tcp_port_reuse_status(const ne_someip_required_service_instance_t* instance);
bool ne_someip_req_serv_inst_get_udp_port_reuse_status(const ne_someip_required_service_instance_t* instance);
uint32_t ne_someip_req_serv_inst_get_udp_collection(const ne_someip_required_service_instance_t* instance);
uint32_t ne_someip_req_serv_inst_get_remote_ip_addr(const ne_someip_required_service_instance_t* instance);
uint16_t ne_someip_req_serv_inst_get_remote_tcp_port(const ne_someip_required_service_instance_t* instance);
uint16_t ne_someip_req_serv_inst_get_remote_udp_port(const ne_someip_required_service_instance_t* instance);
bool ne_someip_req_serv_inst_is_tcp_tls_used(const ne_someip_required_service_instance_t* instance);
bool ne_someip_req_serv_inst_is_udp_tls_used(const ne_someip_required_service_instance_t* instance);
// for send policy
ne_someip_required_provided_method_config_t*
    ne_someip_req_serv_inst_get_method_send_config(const ne_someip_required_service_instance_t* instance, ne_someip_method_id_t method_id);
ne_someip_required_eventgroup_config_t*
    ne_someip_req_serv_inst_get_eventgroup_config(const ne_someip_required_service_instance_t* instance, ne_someip_eventgroup_id_t eg_id);
// for find timer(used by sd)
ne_someip_client_find_time_config_t*
    ne_someip_req_serv_inst_get_find_time_config(const ne_someip_required_service_instance_t* instance);
ne_someip_network_config_t*
    ne_someip_req_serv_inst_get_local_net_config(const ne_someip_required_service_instance_t* instance);
ne_someip_client_inter_config_t*
ne_someip_req_serv_inst_get_inter_config(const ne_someip_required_service_instance_t* instance);
ne_someip_ssl_key_info_t*
ne_someip_req_serv_inst_get_tcp_key_info(const ne_someip_required_service_instance_t* instance);
ne_someip_ssl_key_info_t*
ne_someip_req_serv_inst_get_udp_key_info(const ne_someip_required_service_instance_t* instance);
ne_someip_service_config_t*
ne_someip_req_serv_inst_get_service_config(const ne_someip_required_service_instance_t* instance);
ne_someip_list_t* 
ne_someip_req_serv_inst_get_event_eg_info(const ne_someip_required_service_instance_t* instance);
/********************************interface************************************/

/*********************************callback*************************************/
void ne_someip_req_serv_inst_avail_status_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_service_status_t status, ne_someip_instance_id_t instance_id);

// recv subscribe ack info (sd)
void ne_someip_req_serv_inst_subscribe_ack_handler(ne_someip_required_service_instance_t* instance,
    const ne_someip_ipc_recv_subscribe_ack_t* sub_ack);

void ne_someip_req_serv_inst_subscribe_reply_handler(ne_someip_required_service_instance_t* instance,
    const ne_someip_ipc_recv_local_subscribe_status_t* reply);

// recv remote reboot notify (sd)
void ne_someip_req_serv_inst_reboot_handler(ne_someip_required_service_instance_t* instance);

void ne_someip_req_serv_inst_unix_link_change(ne_someip_required_service_instance_t* instance,
    ne_someip_endpoint_transmit_link_state_t state);

// recv response/event data (forward)
void ne_someip_req_serv_inst_recv_forward_data_handler(ne_someip_trans_buffer_struct_t* trans_buffer,
    void* addr_pair, void* instance);

// create remote socket reply (forward)
// if return fail, try again
void ne_someip_req_serv_inst_create_socket_reply_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_l4_protocol_t type, bool status);
void ne_someip_req_serv_inst_recv_create_destory_socket_reply(ne_someip_required_service_instance_t* instance,
    ne_someip_ipc_create_socket_reply_t* reply);

// remote tcp connect reply (forward)
void ne_someip_req_serv_inst_tcp_connect_reply_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_endpoint_net_addr_t* local_tcp_addr, bool status);

// remote tcp/udp status reply (if return fail, can't send data) (forward)
void ne_someip_req_serv_inst_tcp_udp_status_changed_handler(ne_someip_required_service_instance_t* instance,
    const ne_someip_recv_tcp_udp_message_info_t* info, ne_someip_endpoint_transmit_link_state_t status);

// the reply of request sended to forward daemon (forward)
void ne_someip_req_serv_inst_recv_send_req_reply(ne_someip_required_service_instance_t* instance,
    ne_someip_ipc_send_rpc_msg_reply_t* reply_info);

// the reply of message handler(method/event) registed to forward deamon (forward)
void ne_someip_req_serv_inst_msg_handler_reg_reply_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_method_id_t method_id, ne_someip_error_code_t code);

// the reply of message handler(method/event) unregisted to forward deamon (forward)
void ne_someip_req_serv_inst_msg_handler_unreg_reply_handler(ne_someip_required_service_instance_t* instance,
    ne_someip_method_id_t method_id, ne_someip_error_code_t code);

// the reply of join/leave group
void ne_someip_req_serv_inst_recv_join_group_reply(ne_someip_required_service_instance_t* instance,
    ne_someip_ipc_join_group_reply_t* reply);
/*********************************callback*************************************/

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_SERVICE_INSTANCE_H
/* EOF */
