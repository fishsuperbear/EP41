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
#ifndef SRC_PROTOCOL_RPC_COMMON_REUSE_NE_SOMEIP_COMMON_REUSE_MANAGER_H
#define SRC_PROTOCOL_RPC_COMMON_REUSE_NE_SOMEIP_COMMON_REUSE_MANAGER_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_define.h"
#include "ne_someip_ipc_define.h"
#include "ne_someip_internal_define.h"

typedef struct ne_someip_common_reuse_manager ne_someip_common_reuse_manager_t;

ne_someip_error_code_t ne_someip_common_reuse_manager_init();
ne_someip_error_code_t ne_someip_common_reuse_manager_deinit();

ne_someip_endpoint_unix_t* ne_someip_comm_reuse_mgr_get_rpc_unix_endpoint();

// endpoint req
ne_someip_error_code_t ne_someip_comm_reuse_mgr_create_tcp_endpoint(ne_someip_ipc_msg_type_t type,
    const ne_someip_ipc_create_socket_t* create_endpoint, const ne_someip_endpoint_unix_addr_t* remote_addr,
    ne_someip_list_t* connect_list);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_create_udp_endpoint(const ne_someip_ipc_create_socket_t* create_endpoint,
    const ne_someip_endpoint_unix_addr_t* remote_addr);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_tcp_connect(const ne_someip_ipc_tcp_connect_t* tcp_connect,
    const ne_someip_endpoint_unix_addr_t* remote_addr, bool* is_saved_before);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_tcp_status_notify(const ne_someip_ipc_tcp_connect_t* tcp_connect,
    const ne_someip_endpoint_unix_addr_t* remote_addr, bool is_connect);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_server_tcp_link_notify(const ne_someip_ipc_create_socket_t* create_info,
    const ne_someip_endpoint_unix_addr_t* remote_addr, ne_someip_list_t* connect_list);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_tcp_disconnect(const ne_someip_ipc_tcp_connect_t* tcp_disconnect,
    const ne_someip_endpoint_unix_addr_t* remote_addr);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_udp_join_group(const ne_someip_ipc_join_group_t* join_group,
    const ne_someip_endpoint_unix_addr_t* remote_addr);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_udp_leave_group(const ne_someip_ipc_join_group_t* leave_group,
    const ne_someip_endpoint_unix_addr_t* remote_addr);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_destroy_tcp_endpoint(const ne_someip_ipc_create_socket_t* destroy_endpoint,
    const ne_someip_endpoint_unix_addr_t* remote_addr);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_destroy_udp_endpoint(const ne_someip_ipc_create_socket_t* destroy_endpoint,
    const ne_someip_endpoint_unix_addr_t* remote_addr);

// endpoint reply
bool ne_someip_comm_reuse_mgr_create_destory_tcp_udp_endpoint_reply(const ne_someip_ipc_create_socket_t* endpoint,
    ne_someip_endpoint_unix_addr_t* unix_path, bool result);
bool ne_someip_comm_reuse_mgr_tcp_connect_disconnect_reply(const ne_someip_ipc_tcp_connect_t* endpoint,
    ne_someip_endpoint_unix_addr_t* unix_path, bool result);
bool ne_someip_comm_reuse_mgr_udp_join_leave_group_reply(const ne_someip_ipc_join_group_t* join_group,
    ne_someip_endpoint_unix_addr_t* unix_path, bool result);

// send rpc msg
ne_someip_error_code_t ne_someip_comm_reuse_mgr_send_rpc_msg(ne_someip_ipc_rpc_msg_header_t* ipc_header,
    ne_someip_trans_buffer_struct_t* trans_buffer, ne_someip_endpoint_unix_addr_t* proxy_path);
bool ne_someip_comm_reuse_mgr_send_rpc_msg_reply(ne_someip_ipc_rpc_msg_header_t* ipc_header,
    ne_someip_endpoint_unix_addr_t* proxy_path, ne_someip_error_code_t result);
// ne_someip_error_code_t ne_someip_comm_reuse_mgr_send_resp_msg(ne_someip_ipc_rpc_msg_header_t* ipc_header,
//     ne_someip_trans_buffer_struct_t* trans_buffer);
// ne_someip_error_code_t ne_someip_comm_reuse_mgr_send_event_msg(ne_someip_ipc_rpc_msg_header_t* ipc_header,
//     ne_someip_trans_buffer_struct_t* trans_buffer);

// regisetr/unregister handler
ne_someip_error_code_t ne_someip_comm_reuse_mgr_reg_req_handler(ne_someip_ipc_reg_unreg_method_handler_t* req,
	ne_someip_endpoint_unix_addr_t* client_unix_path);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_reg_resp_handler(ne_someip_ipc_reg_unreg_method_handler_t* resp,
	ne_someip_endpoint_unix_addr_t* client_unix_path);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_reg_event_handler(ne_someip_ipc_reg_unreg_event_handler_t* event,
	ne_someip_endpoint_unix_addr_t* client_unix_path);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_unreg_req_handler(ne_someip_ipc_reg_unreg_method_handler_t* req,
	ne_someip_endpoint_unix_addr_t* client_unix_path);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_unreg_resp_handler(ne_someip_ipc_reg_unreg_method_handler_t* resp,
	ne_someip_endpoint_unix_addr_t* client_unix_path);
ne_someip_error_code_t ne_someip_comm_reuse_mgr_unreg_event_handler(ne_someip_ipc_reg_unreg_event_handler_t* event,
	ne_someip_endpoint_unix_addr_t* client_unix_path);
bool ne_someip_comm_reuse_mgr_reg_unreg_method_reply(ne_someip_ipc_reg_unreg_method_handler_t* req,
    ne_someip_endpoint_unix_addr_t* client_unix_path, bool result);
bool ne_someip_comm_reuse_mgr_reg_unreg_event_reply(ne_someip_ipc_reg_unreg_event_handler_t* req,
    ne_someip_endpoint_unix_addr_t* client_unix_path, bool result);

// subscribe eventgroup_id
void ne_someip_comm_reuse_mgr_subscribe_info_notify(uint8_t* data, uint32_t size);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_COMMON_REUSE_NE_SOMEIP_COMMON_REUSE_MANAGER_H
/* EOF */
