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
#ifndef SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_DEFINE_H
#define SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_DEFINE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>
#include "ne_someip_internal_define.h"
#include "ne_someip_config_define.h"
#include "ne_someip_transmit.h"
#include "ne_someip_object.h"
#include "ne_someip_sync_obj.h"
#include "ne_someip_list.h"
#include "ne_someip_map.h"
#include "ne_someip_tp_define.h"

typedef enum ne_someip_endpoint_type
{
    ne_someip_endpoint_type_tcp = 0x0,
    ne_someip_endpoint_type_udp = 0x1,
    ne_someip_endpoint_type_unix = 0x2,
    ne_someip_endpoint_type_sd_udp = 0x3,
}ne_someip_endpoint_type_t;

typedef enum ne_someip_endpoint_role_type
{
    ne_someip_endpoint_role_type_client = 0x0,
    ne_someip_endpoint_role_type_server = 0x1,
    ne_someip_endpoint_role_type_undef = 0x2,
}ne_someip_endpoint_role_type_t;

typedef enum ne_someip_endpoint_state
{
    ne_someip_endpoint_state_not_created = 0x0,
    ne_someip_endpoint_state_created = 0x1,
    ne_someip_endpoint_state_destroy = 0x2,
}ne_someip_endpoint_state_t;

typedef enum ne_someip_endpoint_send_policy_type
{
    ne_someip_endpoint_send_policy_type_normal_send = 0x0,
    ne_someip_endpoint_send_policy_type_circular_send = 0x1,
    ne_someip_endpoint_send_policy_type_delay_send = 0x2,
    ne_someip_endpoint_send_policy_type_udp_collection_send = 0x3,
}ne_someip_endpoint_send_policy_type_t;

typedef enum ne_someip_endpoint_receive_policy_type
{
    ne_someip_endpoint_receive_policy_type_udp_decode = 0x0,
    ne_someip_endpoint_receive_policy_type_tcp_unix_decode = 0x1,
}ne_someip_endpoint_receive_policy_type_t;

typedef enum ne_someip_endpoint_interface_ret_code
{
    ne_someip_endpoint_interface_ret_code_success = 0x0,
    ne_someip_endpoint_interface_ret_code_fail = 0x1,
}ne_someip_endpoint_interface_ret_code_t;

typedef enum ne_someip_endpoint_transmit_state
{
    ne_someip_endpoint_transmit_state_not_created = 0x0,  // default state
    ne_someip_endpoint_transmit_state_prepared = 0x1,  // transmit was prepare state, can't communication
    ne_someip_endpoint_transmit_state_startd = 0x2,  // transmit can communication now
    ne_someip_endpoint_transmit_state_stopped = 0x3,  // transmit was stop state，can't communication
    ne_someip_endpoint_transmit_state_error = 0x4,  // transmit have error
}ne_someip_endpoint_transmit_state_t;

typedef enum ne_someip_endpoint_transmit_link_state
{
    ne_someip_endpoint_transmit_link_state_connecting = 0x0,
    ne_someip_endpoint_transmit_link_state_connected = 0x1,
    ne_someip_endpoint_transmit_link_state_disconnected = 0x2,
    ne_someip_endpoint_transmit_link_state_error = 0x3,  // transmit link created error
    ne_someip_endpoint_transmit_link_state_invalid = 0x4,
    ne_someip_endpoint_transmit_link_state_unknow
}ne_someip_endpoint_transmit_link_state_t;

typedef enum ne_someip_endpoint_link_role
{
    ne_someip_endpoint_link_role_client = 0x0,  // unix domain client or tcp socket client
    ne_someip_endpoint_link_role_server = 0x1,  // unix domain client or tcp socket server
    ne_someip_enepoint_link_role_idle
}ne_someip_endpoint_link_role_t;

typedef enum ne_someip_endpoint_add_multicast_addr_state
{
    ne_someip_endpoint_add_multicast_addr_state_adding = 0x0,  // add multicast addr interface called
    ne_someip_endpoint_add_multicast_addr_state_success = 0x1,  // add multicast addr successed
    ne_someip_endpoint_add_multicast_addr_state_fail = 0x2,  // add multicast addr failed
    ne_someip_endpoint_add_multicast_addr_state_remove = 0x3,  // remove multicast addr
}ne_someip_endpoint_add_multicast_addr_state_t;

typedef enum ne_someip_endpoint_check_recv_data_state
{
    ne_someip_endpoint_check_data_state_error = 0x0,
    ne_someip_endpoint_check_data_state_recv_finish = 0x1,
    ne_someip_endpoint_check_data_state_recv_continue = 0x2,
    ne_someip_endpoint_check_recv_data_state_finish_no_payload = 0x3,
    ne_someip_endpoint_check_recv_data_state_idle = 0x4,
}ne_someip_endpoint_check_recv_data_state_t;

typedef enum ne_someip_endpoint_dtls_status
{
    ne_someip_endpoint_dtls_status_unknow = 0x0,
    ne_someip_endpoint_dtls_status_verifying = 0x1,
    ne_someip_endpoint_dtls_status_verified = 0x2,
    ne_someip_endpoint_dtls_status_failed = 0x3,
}ne_someip_endpoint_dtls_status_t;

typedef struct ne_someip_endpoint_client_instance_spec
{
    ne_someip_client_id_t client_id;
    ne_someip_service_instance_spec_t inst_spec;
}ne_someip_endpoint_client_instance_spec_t;

/*************************************endpoint saved info*****************************************/
typedef struct ne_someip_endpoint_transmit_link_info
{
    ne_someip_endpoint_link_role_t link_role;
    ne_someip_transmit_link_t* transmit_link;
    ne_someip_endpoint_transmit_link_state_t transmit_link_state;
}ne_someip_endpoint_transmit_link_info_t;

typedef struct ne_someip_endpoint_transmit_info
{
    ne_someip_transmit_t* transmit;
    ne_someip_endpoint_transmit_state_t transmit_state;
}ne_someip_endpoint_transmit_info_t;

typedef struct ne_someip_endpoint_add_multicast_addr_info
{
    ne_someip_enpoint_multicast_addr_t* addr;
    ne_someip_endpoint_add_multicast_addr_state_t state;
}ne_someip_endpoint_add_multicast_addr_info_t;

typedef struct ne_someip_endpoint_instance_info
{
    ne_someip_endpoint_instance_type_t instance_type;
    const void* service_instance;  // ne_someip_provided_instance_t or ne_someip_required_service_instance_t
}ne_someip_endpoint_instance_info_t;
/*************************************endpoint saved info*****************************************/

/*************************************endpoint send/recv data*************************************/
typedef struct ne_someip_trans_buffer_struct
{
    ne_someip_endpoint_buffer_t* ipc_data;
    ne_someip_endpoint_buffer_t* someip_header;
    ne_someip_payload_t* payload;
}ne_someip_trans_buffer_struct_t;

typedef struct ne_someip_endpoint_send_policy
{
    ne_someip_endpoint_send_policy_type_t send_policy_type;
    uint32_t delay_time;
    uint32_t collection_timeout;
    uint32_t collection_buffer_size;
    ne_someip_udp_collection_trigger_t trigger_mode;
    uint32_t segment_length;
    uint32_t sepration_time;
}ne_someip_endpoint_send_policy_t;

// typedef struct ne_someip_endpoint_send_handler
// {
//     ne_someip_list_t* buffer_list; //<ne_someip_endpoint_buffer_t>
//     void* peer_addr;
//     ne_someip_endpoint_send_policy_t policy;
//     ne_someip_transmit_type_t transmit_type;
//     ne_someip_transmit_t* transmit;
//     ne_someip_transmit_link_t* transmit_link;
//     void* endpoint;
//     uint32_t seq_id;
// }ne_someip_endpoint_send_handler_t;

// typedef struct ne_someip_endpoint_receive_handler
// {
//     ne_someip_list_t* buffer_list;  // <ne_someip_endpoint_buffer_t>
//     uint32_t buffer_num;
//     ne_someip_endpoint_receive_policy_type_t policy_type;
//     const void* policy_handler;
// }ne_someip_endpoint_receive_handler_t;

typedef struct ne_someip_endpoint_buffer_info
{
    ne_someip_endpoint_buffer_t* buffer;
    uint32_t seq_id;
}ne_someip_endpoint_buffer_info_t;

typedef struct ne_someip_endpoint_send_cache
{
    ne_someip_trans_buffer_struct_t* buffer;
    const void* seq_data;  // user_data, used to notify
}ne_someip_endpoint_send_cache_t;

typedef struct ne_someip_endpoint_udp_iov_cache
{
    ne_someip_transmit_iov_buffer_t* buffer;
    ne_someip_endpoint_net_addr_t* peer_addr;
    ne_someip_list_t* tp_buffer_list;  // <ne_someip_transmit_iov_buffer_t*>
    ne_someip_endpoint_send_cache_t* orig_data;  // TP分包前的原始数据，ne_someip_trans_buffer_struct_t*
    bool is_tp_data;
    bool is_first_tp_data;
    uint32_t sepration_time;
}ne_someip_endpoint_udp_iov_cache_t;

typedef struct ne_someip_endpoint_tp_cache
{
    bool is_tp_data_send;
    ne_someip_list_t* tp_list;  // <ne_someip_endpoint_udp_iov_cache_t*>
}ne_someip_endpoint_tp_cache_t;

typedef struct ne_someip_endpoint_core_send_info
{
    void* peer_addr;  // 当前发送的对端地址，该值用于release的时候删除数据
    ne_someip_list_t* udp_send_list;  // <ne_someip_endpoint_udp_iov_cache_t*>
    ne_someip_map_t* tp_cache_map;  // <ne_someip_endpoint_net_addr_t*, ne_someip_endpoint_tp_cache_t*>
    ne_someip_map_t* udp_tls_cache_map;  // <ne_someip_endpoint_net_addr_t*, ne_someip_list_t<ne_someip_endpoint_udp_iov_cache_t*>>
    ne_someip_map_t* tcp_send_map;  // <ne_someip_endpoint_net_addr_t*, ne_someip_list_t<ne_someip_transmit_iov_buffer_t*>>
    ne_someip_map_t* unix_send_map;  // <ne_someip_endpoint_unix_addr_t*, ne_someip_list_t<ne_someip_transmit_iov_buffer_t*>>
    ne_someip_map_t* udp_collection_send_buffer_map;  // <ne_someip_endpoint_net_addr_t*, ne_someip_list_t<ne_someip_endpoint_send_cache_t*>>
}ne_someip_endpoint_core_send_info_t;

typedef struct ne_someip_endpoint_core_recv_info
{
    ne_someip_map_t* unix_receive_buffer_map;  // <ne_someip_endpoint_unix_addr_t*, ne_someip_list_t<ne_someip_transmit_normal_buffer_t*>>
    ne_someip_map_t* unix_ipc_type_len_map;  // <ne_someip_endpoint_unix_addr_t*, ne_someip_transmit_normal_buffer_t*
    ne_someip_map_t* tcp_receive_buffer_map;  // <ne_someip_endpoint_net_addr_t*, ne_someip_list_t<ne_someip_transmit_normal_buffer_t*>>
                                     // remote addr as the key
}ne_someip_endpoint_core_recv_info_t;

typedef struct ne_someip_endpoint_core
{
    ne_someip_looper_t* looper;
    ne_someip_endpoint_type_t ep_type;
    ne_someip_endpoint_core_send_info_t* send_info;
    ne_someip_endpoint_core_recv_info_t* recv_info;
    ne_someip_transmit_iov_buffer_t* udp_iov_cache_buffer;
    NEOBJECT_MEMBER
}ne_someip_endpoint_core_t;
/*************************************endpoint send/recv data*************************************/

/***************************************endpoint callback*****************************************/
typedef enum ne_someip_endpoint_addr_type
{
    ne_someip_endpoint_addr_type_net = 0,
    ne_someip_endpoint_addr_type_unix = 1,
    ne_someip_endpoint_addr_type_unknow = 2,
}ne_someip_endpoint_addr_type_t;

typedef struct ne_someip_endpoint_addr_pair_base
{
    ne_someip_endpoint_addr_type_t type;
}ne_someip_endpoint_addr_pair_base_t;

typedef struct ne_someip_endpoint_net_addr_pair
{
    ne_someip_endpoint_addr_pair_base_t base;
    ne_someip_endpoint_net_addr_t* local_addr;
    ne_someip_endpoint_net_addr_t* remote_addr;
    ne_someip_endpoint_type_t type;
    bool is_multicast;
}ne_someip_endpoint_net_addr_pair_t;

typedef struct ne_someip_endpoint_unix_addr_pair
{
    ne_someip_endpoint_addr_pair_base_t base;
    ne_someip_endpoint_unix_addr_t* local_addr;
    ne_someip_endpoint_unix_addr_t* remote_addr;
    ne_someip_endpoint_type_t type;
}ne_someip_endpoint_unix_addr_pair_t;

typedef struct ne_someip_endpoint_link_state
{
    void* addr_pair;
    ne_someip_endpoint_transmit_link_state_t state;
}ne_someip_endpoint_link_state_t;

typedef struct ne_someip_endpoint_callback
{
    void (*socket_send_status)(void* seq_data, ne_someip_error_code_t result, void* user_data);
    void (*seq_data_free)(void* seq_data);
    void (*link_state_notify)(ne_someip_endpoint_link_state_t* state, void* user_data);
    void (*receiver_on_message)(ne_someip_trans_buffer_struct_t* trans_buffer, void* addr_pair, void* user_data);
    void (*free)(void* user_data);
    void* user_data;
}ne_someip_endpoint_callback_t;

/***************************************endpoint callback*****************************************/

/*************************************endpoint*************************************/
typedef struct ne_someip_endpoint_base
{
    ne_someip_endpoint_type_t endpoint_type;
}ne_someip_endpoint_base_t;

typedef struct ne_someip_endpoint_unix
{
    ne_someip_endpoint_base_t base;
    ne_someip_endpoint_role_type_t role_type;
    bool is_need_switch_thread;
    ne_someip_endpoint_unix_addr_t* local_addr;
    ne_someip_endpoint_state_t endpoint_state;
    ne_someip_endpoint_core_t* ep_core;
    ne_someip_endpoint_transmit_info_t* unix_transmit;
    ne_someip_map_t* unix_transmit_link_map;// <ne_someip_endpoint_unix_addr_t, ne_someip_endpoint_transmit_link_info_t*>
    ne_someip_looper_t* endpoint_io_looper;
    ne_someip_looper_t* work_looper;
    ne_someip_map_t* service_instance_map; // <ne_someip_service_instance_spec_t, ne_someip_endpoint_instance_info_t*>
    ne_someip_map_t* client_instance_map;  // <ne_someip_endpoint_client_instance_spec_t, ne_someip_list_t*<ne_someip_endpoint_instance_info_t*>>
    ne_someip_map_t* common_instance_map;  // <ne_someip_endpoint_client_instance_spec_t, ne_someip_endpoint_instance_info_t*>
    ne_someip_sync_obj_t* instance_map_sync;
    ne_someip_endpoint_callback_t* callback;
    ne_someip_map_t* instance_id_map;  // <ne_someip_service_id_t(major_version?), ne_someip_instance_id_t>
    ne_someip_map_t* subscribe_event_id_map;  // <ne_someip_subscribe_event_key_t, ne_someip_list_t<ne_someip_endpoint_instance_info_t*>>
    ne_someip_sync_obj_t* seq_id_sync;
    int32_t sync_seq_id;
    ne_someip_sync_obj_t* res_code_map_sync;
    ne_someip_map_t* sync_seq_res_code_map;  // <sync_seq_id, ne_someip_error_code_t>
    bool is_transmit_stop;
    NEOBJECT_MEMBER
}ne_someip_endpoint_unix_t;

typedef struct ne_someip_endpoint_udp_data
{
    ne_someip_endpoint_base_t base;
    bool is_need_switch_thread;
    ne_someip_endpoint_net_addr_t* local_addr;
    ne_someip_endpoint_state_t endpoint_state;
    ne_someip_endpoint_core_t* ep_core;
    bool is_tls_used;
    ne_someip_endpoint_transmit_info_t* udp_transmit;
    ne_someip_ssl_key_info_t* key_info;
    ne_someip_map_t* tls_status_map;  // <ne_someip_endpoint_net_addr_t, ne_someip_endpoint_dtls_status_t>
    ne_someip_looper_t* endpoint_io_looper;
    ne_someip_looper_t* work_looper;
    ne_someip_endpoint_callback_t* callback;
    ne_someip_list_t* req_instance_list;  // ne_someip_required_service_instance_t
    ne_someip_list_t* group_addr_list;  // ne_someip_endpoint_add_multicast_addr_info_t
    ne_someip_sync_obj_t* seq_id_sync;
    int32_t sync_seq_id;
    ne_someip_sync_obj_t* res_code_map_sync;
    ne_someip_map_t* sync_seq_res_code_map;  // <sync_seq_id, ne_someip_error_code_t>
    ne_someip_tp_ctx_t* tp_ctx;
    bool is_transmit_stop;
    bool is_multicast_endpoint;
    NEOBJECT_MEMBER
}ne_someip_endpoint_udp_data_t;

typedef struct ne_someip_endpoint_tcp_data
{
    ne_someip_endpoint_base_t base;
    ne_someip_endpoint_role_type_t role_type;
    bool is_need_switch_thread;
    ne_someip_endpoint_net_addr_t* local_addr;
    ne_someip_endpoint_state_t endpoint_state;
    ne_someip_endpoint_core_t* ep_core;
    bool is_tls_used;
    ne_someip_endpoint_transmit_info_t* tcp_transmit;
    ne_someip_ssl_key_info_t* key_info;
    ne_someip_map_t* tcp_transmit_link_map;  // <ne_someip_endpoint_net_addr_t, ne_someip_endpoint_transmit_link_info_t*>
    ne_someip_looper_t* endpoint_io_looper;
    ne_someip_looper_t* work_looper;
    ne_someip_endpoint_callback_t* callback;
    ne_someip_list_t* req_instance_list;  // ne_someip_required_service_instance_t
    ne_someip_sync_obj_t* seq_id_sync;
    int32_t sync_seq_id;
    ne_someip_sync_obj_t* res_code_map_sync;
    ne_someip_map_t* sync_seq_res_code_map;  // <sync_seq_id, ne_someip_error_code_t>
    bool is_transmit_stop;
    NEOBJECT_MEMBER
}ne_someip_endpoint_tcp_data_t;

typedef struct ne_someip_endpoint_udp_sd
{
    ne_someip_endpoint_base_t base;
    ne_someip_endpoint_net_addr_t* local_addr;
    ne_someip_endpoint_state_t endpoint_state;
    ne_someip_endpoint_core_t* ep_core;
    ne_someip_endpoint_transmit_info_t* udp_transmit;
    ne_someip_looper_t* endpoint_io_looper;
    ne_someip_endpoint_callback_t* callback;
    ne_someip_endpoint_add_multicast_addr_info_t* group_addr_info;
    bool is_multicast_endpoint;
    ne_someip_tp_ctx_t* tp_ctx;
    NEOBJECT_MEMBER
}ne_someip_endpoint_udp_sd_t;
/*************************************endpoint*************************************/

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_DEFINE_H
/* EOF */
