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
#ifndef MANAGER_SERVER_ne_someip_provided_ins_H
#define MANAGER_SERVER_ne_someip_provided_ins_H
#ifdef __cplusplus
extern "C"
{
#endif

#include "ne_someip_provided_method_behaviour.h"
#include "ne_someip_provided_event_behaviour.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_provided_eventgroup_behaviour.h"
#include "ne_someip_endpoint_tcp_data.h"
#include "ne_someip_endpoint_udp_data.h"
#include "ne_someip_endpoint_unix.h"
#include "ne_someip_looper.h"
#include "ne_someip_ipc_define.h"
#include "ne_someip_handler.h"

ne_someip_provided_instance_t* ne_someip_provided_ins_create(const ne_someip_provided_service_instance_config_t* instance_config,
	ne_someip_looper_t* work_looper, ne_someip_looper_t* io_looper, ne_someip_thread_t* work_thread,
	ne_someip_endpoint_unix_t* unix_endpoint);

ne_someip_provided_instance_t* ne_someip_provided_ins_ref(ne_someip_provided_instance_t* instance);
void ne_someip_provided_ins_unref(ne_someip_provided_instance_t* instance);

void ne_someip_provided_ins_offer(void* data);

void ne_someip_provided_ins_stop_offer(void* data);

void ne_someip_provided_ins_send_response(void* data);

void ne_someip_provided_ins_send_event(void* data);

void ne_someip_provided_ins_reg_req_handler_to_daemon(void* data);
void ne_someip_provided_ins_unreg_req_handler_to_daemon(void* data);

ne_someip_error_code_t ne_someip_provided_ins_reg_subscribe_handler(ne_someip_provided_instance_t* instance,
	ne_someip_recv_subscribe_handler handler, const void* user_data);

ne_someip_error_code_t ne_someip_provided_ins_unreg_subscribe_handler(ne_someip_provided_instance_t* instance);

ne_someip_error_code_t ne_someip_provided_ins_reg_req_handler(ne_someip_provided_instance_t* instance,
	ne_someip_recv_request_handler handler, const void* user_data);

ne_someip_error_code_t ne_someip_provided_ins_unreg_req_handler(ne_someip_provided_instance_t* instance);

ne_someip_error_code_t ne_someip_provided_ins_reg_ser_status_handler(ne_someip_provided_instance_t* instance,
	ne_someip_offer_status_handler handler, const void* user_data);

ne_someip_error_code_t ne_someip_provided_ins_unreg_ser_status_handler(ne_someip_provided_instance_t* instance);

ne_someip_error_code_t ne_someip_provided_ins_reg_event_status_handler(ne_someip_provided_instance_t* instance,
	ne_someip_send_event_status_handler handler, const void* user_data);

ne_someip_error_code_t ne_someip_provided_ins_unreg_event_status_handler(ne_someip_provided_instance_t* instance);

ne_someip_error_code_t ne_someip_provided_ins_reg_resp_status_handler(ne_someip_provided_instance_t* instance,
	ne_someip_send_resp_status_handler handler, const void* user_data);

ne_someip_error_code_t ne_someip_provided_ins_unreg_resp_status_handler(ne_someip_provided_instance_t* instance);

ne_someip_error_code_t ne_someip_provided_ins_set_eg_permission(ne_someip_provided_instance_t* instance,
	ne_someip_eventgroup_id_t eventgroup_id, const ne_someip_remote_client_info_t* remote_addr,
	ne_someip_permission_t* priority);

ne_someip_offer_status_t ne_someip_provided_ins_get_service_status(ne_someip_provided_instance_t* instance);

void ne_someip_provided_ins_recv_subscribe(ne_someip_provided_instance_t* instance, const ne_someip_ipc_recv_subscribe_t* subscribe_info);

void ne_someip_provided_ins_service_status_changed(ne_someip_provided_instance_t* instance, ne_someip_ipc_recv_local_offer_status_t* message);

ne_someip_session_id_t ne_someip_provided_ins_get_session_id(ne_someip_provided_instance_t* instance);

void ne_someip_provided_ins_stop_offer_complete(ne_someip_provided_instance_t* instance);

void ne_someip_provided_ins_network_changed(void* user_data);

/*********************************** get the value of config*********************************/
ne_someip_service_config_t* ne_someip_provided_ins_get_serv_config(const ne_someip_provided_instance_t* instance);
ne_someip_server_internal_config_t
ne_someip_provided_ins_get_internal_config(const ne_someip_provided_instance_t* instance);
ne_someip_service_id_t ne_someip_provided_ins_get_service_id(const ne_someip_provided_instance_t* instance);

ne_someip_instance_id_t ne_someip_provided_ins_get_instance_id(const ne_someip_provided_instance_t* instance);

ne_someip_major_version_t ne_someip_provided_ins_get_major_version(const ne_someip_provided_instance_t* instance);

ne_someip_minor_version_t ne_someip_provided_ins_get_minor_version(const ne_someip_provided_instance_t* instance);

ne_someip_address_type_t ne_someip_provided_ins_get_addr_type(const ne_someip_provided_instance_t* instance);

uint32_t ne_someip_provided_ins_get_ip_addr(const ne_someip_provided_instance_t* instance);

char* ne_someip_provided_ins_get_if_name(const ne_someip_provided_instance_t* instance);

bool ne_someip_provided_ins_get_reliable_flag(const ne_someip_provided_instance_t* instance);

bool ne_someip_provided_ins_get_unreliable_flag(const ne_someip_provided_instance_t* instance);

bool ne_someip_provided_ins_get_tcp_reuse_flag(const ne_someip_provided_instance_t* instance);

bool ne_someip_provided_ins_get_udp_reuse_flag(const ne_someip_provided_instance_t* instance);

uint16_t ne_someip_provided_ins_get_udp_port(const ne_someip_provided_instance_t* instance);

uint16_t ne_someip_provided_ins_get_tcp_port(const ne_someip_provided_instance_t* instance);

ne_someip_server_offer_time_config_t
ne_someip_provided_ins_get_time_config(const ne_someip_provided_instance_t* instance);

uint32_t ne_someip_provided_ins_get_sd_multicast_addr(const ne_someip_provided_instance_t* instance);

uint16_t ne_someip_provided_ins_get_sd_multicast_port(const ne_someip_provided_instance_t* instance);

uint32_t ne_someip_provided_ins_get_method_udp_collection_buffer_size(const ne_someip_provided_instance_t* instance);

ne_someip_looper_t* ne_someip_provided_ins_get_io_looper(const ne_someip_provided_instance_t* instance);
ne_someip_looper_t* ne_someip_provided_ins_get_work_looper(const ne_someip_provided_instance_t* instance);

ne_someip_endpoint_tcp_data_t* ne_someip_provided_ins_get_tcp_endpoint(const ne_someip_provided_instance_t* instance);

ne_someip_endpoint_udp_data_t* ne_someip_provided_ins_get_udp_endpoint(const ne_someip_provided_instance_t* instance);

ne_someip_endpoint_unix_t* ne_someip_provided_ins_get_unix_endpoint(const ne_someip_provided_instance_t* instance);

bool ne_someip_provided_ins_is_tcp_tls_used(const ne_someip_provided_instance_t* instance);
bool ne_someip_provided_ins_is_udp_tls_used(const ne_someip_provided_instance_t* instance);
ne_someip_ssl_key_info_t* ne_someip_provided_ins_get_tcp_key_info(const ne_someip_provided_instance_t* instance);
ne_someip_ssl_key_info_t* ne_someip_provided_ins_get_udp_key_info(const ne_someip_provided_instance_t* instance);
ne_someip_l4_protocol_t ne_someip_provided_ins_get_method_protocol(const ne_someip_provided_instance_t* instance,
    ne_someip_method_id_t method_id);

/**********************************************************receive the ipc reply**************************************************************/

void ne_someip_provided_ins_recv_ipc_send_event_resp_reply(ne_someip_provided_instance_t* instance,
    ne_someip_ipc_send_rpc_msg_reply_t* reply);

void ne_someip_provided_ins_recv_ipc_create_destory_socket_reply(ne_someip_provided_instance_t* instance,
    ne_someip_ipc_create_socket_reply_t* reply);

void ne_someip_provided_ins_recv_tcp_udp_status_changed(ne_someip_provided_instance_t* instance,
    ne_someip_recv_tcp_udp_message_info_t* info, ne_someip_endpoint_transmit_link_state_t status);

void ne_someip_provided_ins_recv_rpc_msg(ne_someip_trans_buffer_struct_t* trans_buffer, void* addr_pair, void* user_data);

void ne_someip_provided_ins_recv_forward_msg(ne_someip_trans_buffer_struct_t* trans_buffer, void* addr_pair, void* user_data);

void ne_someip_provided_ins_remote_reboot(ne_someip_provided_instance_t* instance, uint32_t ip_addr);

void ne_someip_provided_ins_unix_link_changed(ne_someip_provided_instance_t* instance,
	ne_someip_endpoint_transmit_link_state_t state);

#ifdef __cplusplus
}
#endif
#endif // MANAGER_SERVER_ne_someip_provided_ins_H
/* EOF */
