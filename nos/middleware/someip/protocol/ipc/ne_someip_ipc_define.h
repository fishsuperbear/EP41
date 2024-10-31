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
#ifndef MANAGER_NE_SOMEIP_IPC_DEFINE_H
#define MANAGER_NE_SOMEIP_IPC_DEFINE_H

#include <stdbool.h>
#include "ne_someip_define.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_endpoint_define.h"
#include "ne_someip_sd_define.h"
#include "ne_someip_server_define.h"
#include "ne_someip_client_define.h"

// define all forward message type
typedef enum ne_someip_ipc_msg_type
{
    ne_someip_ipc_msg_type_unknown = 0x00,
    ne_someip_ipc_msg_type_send_request = 0x01,
    ne_someip_ipc_msg_type_send_request_reply = 0x02,
    ne_someip_ipc_msg_type_send_response = 0x03,
    ne_someip_ipc_msg_type_send_response_reply = 0x04,
    ne_someip_ipc_msg_type_send_event = 0x05,
    ne_someip_ipc_msg_type_send_event_reply = 0x06,
    ne_someip_ipc_msg_type_client_tcp_connect = 0x07,
    ne_someip_ipc_msg_type_client_tcp_connect_reply = 0x08,
    ne_someip_ipc_msg_type_client_create_tcp_socket = 0x09,
    ne_someip_ipc_msg_type_client_create_tcp_socket_reply = 0x0a,
    ne_someip_ipc_msg_type_server_create_tcp_socket = 0x0b,
    ne_someip_ipc_msg_type_server_create_tcp_socket_reply = 0x0c,
    ne_someip_ipc_msg_type_client_create_udp_socket = 0x0d,
    ne_someip_ipc_msg_type_client_create_udp_socket_reply = 0x0e,
    ne_someip_ipc_msg_type_server_create_udp_socket = 0x0f,
    ne_someip_ipc_msg_type_server_create_udp_socket_reply = 0x10,
    ne_someip_ipc_msg_type_register_request_handler = 0x11,
    ne_someip_ipc_msg_type_register_request_handler_reply = 0x12,
    ne_someip_ipc_msg_type_register_response_handler = 0x13,
    ne_someip_ipc_msg_type_register_response_handler_reply = 0x14,
    ne_someip_ipc_msg_type_unregister_request_handler = 0x15,
    ne_someip_ipc_msg_type_unregister_request_handler_reply = 0x16,
    ne_someip_ipc_msg_type_unregister_response_handler = 0x17,
    ne_someip_ipc_msg_type_unregister_response_handler_reply = 0x18,
    ne_someip_ipc_msg_type_register_event_handler = 0x19,
    ne_someip_ipc_msg_type_register_event_handler_reply = 0x1a,
    ne_someip_ipc_msg_type_unregister_event_handler = 0x1b,
    ne_someip_ipc_msg_type_unregister_event_handler_reply = 0x1c,
    ne_someip_ipc_msg_type_client_tcp_status_changed = 0x1d,
    ne_someip_ipc_msg_type_server_tcp_status_changed = 0x1e,
    ne_someip_ipc_msg_type_client_udp_status_changed = 0x1f,
    ne_someip_ipc_msg_type_server_udp_status_changed = 0x20,
    ne_someip_ipc_msg_type_receive_remote_event = 0x21,
    ne_someip_ipc_msg_type_receive_remote_request = 0x22,
    ne_someip_ipc_msg_type_receive_remote_response = 0x23,
    ne_someip_ipc_msg_type_join_group = 0x24,
    ne_someip_ipc_msg_type_leave_group = 0x25,
    ne_someip_ipc_msg_type_join_group_reply = 0x26,
    ne_someip_ipc_msg_type_leave_group_reply = 0x27,
    ne_someip_ipc_msg_type_send_offer = 0x30,
    ne_someip_ipc_msg_type_send_stop_offer = 0x31,
    ne_someip_ipc_msg_type_send_find = 0x32,
    ne_someip_ipc_msg_type_send_stop_find = 0x33,
    ne_someip_ipc_msg_type_send_subscribe = 0x34,
    ne_someip_ipc_msg_type_send_stop_subscribe = 0x35,
    ne_someip_ipc_msg_type_send_subscribe_ack = 0x36,
    ne_someip_ipc_msg_type_recv_offer = 0x37,
    ne_someip_ipc_msg_type_recv_stop_offer = 0x38,
    ne_someip_ipc_msg_type_recv_subscribe = 0x39,
    ne_someip_ipc_msg_type_recv_stop_subscribe = 0x3a,
    ne_someip_ipc_msg_type_recv_subscribe_ack = 0x3b,
    ne_someip_ipc_msg_type_recv_subscribe_nack = 0x3c,
    ne_someip_ipc_msg_type_subscribe_success = 0x3d,
    ne_someip_ipc_msg_type_remote_reboot = 0x3e,
    ne_someip_ipc_msg_type_register_service_status_handler = 0x3f,
    ne_someip_ipc_msg_type_register_service_status_handler_reply = 0x40,
    ne_someip_ipc_msg_type_unregister_service_status_handler = 0x41,
    ne_someip_ipc_msg_type_unregister_service_status_handler_reply = 0x42,
    ne_someip_ipc_msg_type_notify_network_status = 0x43,
    ne_someip_ipc_msg_type_recv_local_offer_status = 0x44,
    ne_someip_ipc_msg_type_recv_local_find_status = 0x45,
    ne_someip_ipc_msg_type_recv_local_subscribe_status = 0x46,
    ne_someip_ipc_msg_type_get_client_id = 0x47,
    ne_someip_ipc_msg_type_get_client_id_reply = 0x48,
    ne_someip_ipc_msg_type_find_remote_svs = 0x49,
    ne_someip_ipc_msg_type_find_remote_svs_reply = 0x4a,
    ne_someip_ipc_msg_type_client_tcp_disconnect = 0x4b,
    ne_someip_ipc_msg_type_client_tcp_disconnect_reply = 0x4c,
    ne_someip_ipc_msg_type_client_destory_tcp_socket = 0x4d,
    ne_someip_ipc_msg_type_client_destory_tcp_socket_reply = 0x4e,
    ne_someip_ipc_msg_type_server_destory_tcp_socket = 0x4f,
    ne_someip_ipc_msg_type_server_destory_tcp_socket_reply = 0x50,
    ne_someip_ipc_msg_type_client_destory_udp_socket = 0x51,
    ne_someip_ipc_msg_type_client_destory_udp_socket_reply = 0x52,
    ne_someip_ipc_msg_type_server_destory_udp_socket = 0x53,
    ne_someip_ipc_msg_type_server_destory_udp_socket_reply = 0x54,
} ne_someip_ipc_msg_type_t;

typedef enum ne_someip_ipc_local_offer_status
{
    ne_someip_ipc_local_offer_status_unkown = 0x00,
    ne_someip_ipc_local_offer_status_send_success = 0x01,
    ne_someip_ipc_local_offer_status_send_failed = 0x02,
    ne_someip_ipc_local_offer_status_send_initial_success = 0x03,
    ne_someip_ipc_local_offer_status_send_initial_failed = 0x04,
    ne_someip_ipc_local_offer_status_send_repetition_success = 0x05,
    ne_someip_ipc_local_offer_status_send_repetition_failed = 0x06,
    ne_someip_ipc_local_offer_status_send_main_success = 0x07,
    ne_someip_ipc_local_offer_status_send_main_failed = 0x08,
    ne_someip_ipc_local_offer_status_send_ne_someip_ttl_time_out = 0x09,
} ne_someip_ipc_local_offer_status_t;

typedef enum ne_someip_ipc_local_find_status
{
    ne_someip_ipc_local_find_status_unkown = 0x00,
    ne_someip_ipc_local_find_status_send_success = 0x01,
    ne_someip_ipc_local_find_status_send_failed = 0x02,
    ne_someip_ipc_local_find_status_send_initial_success = 0x03,
    ne_someip_ipc_local_find_status_send_initial_failed = 0x04,
    ne_someip_ipc_local_find_status_send_repetition_success = 0x05,
    ne_someip_ipc_local_find_status_send_repetition_failed = 0x06,
    ne_someip_ipc_local_find_status_send_ne_someip_ttl_time_out = 0x09,
} ne_someip_ipc_local_find_status_t;

typedef enum ne_someip_ipc_local_subscribe_status
{
    ne_someip_ipc_local_subscribe_status_unkown = 0x00,
    ne_someip_ipc_local_subscribe_status_send_success = 0x01,
    ne_someip_ipc_local_subscribe_status_send_failed = 0x02,
    ne_someip_ipc_local_subscribe_status_send_cycle_success = 0x03,
    ne_someip_ipc_local_subscribe_status_send_cycle_failed = 0x04,
    ne_someip_ipc_local_subscribe_status_send_ne_someip_ttl_time_out = 0x09,
} ne_someip_ipc_local_subscribe_status_t;

typedef struct ne_someip_sd_subscribe_timer
{
    uint16_t request_response_delay_max;
    uint16_t request_response_delay_min;
    uint16_t retry_delay;
    uint16_t retry_max;
    uint32_t ttl;
} ne_someip_sd_subscribe_timer_t;

typedef struct ne_someip_ipc_subscribe_eg
{
    ne_someip_eventgroup_id_t eventgroup_id;
    uint16_t request_response_delay_max;
    uint16_t request_response_delay_min;
    uint16_t retry_delay;
    uint16_t retry_max;
    uint32_t ttl;
    uint8_t initial_data_flag;
    bool reliable_flag;
    bool unreliable_flag;
} ne_someip_ipc_subscribe_eg_t;

typedef struct ne_someip_ipc_rpc_msg_header
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;                                //完整数据包长度,包括转发包头、someip包头和payload
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_method_id_t method_id;
    ne_someip_session_id_t session_id;
    // 如果是event或者response，则是0，如果是request，则是对应的clientid
    ne_someip_client_id_t client_id;
    ne_someip_l4_protocol_t protocol;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_port;
    uint32_t destination_address;
    uint16_t destination_port;
    ne_someip_endpoint_send_policy_t send_policy;
} ne_someip_ipc_rpc_msg_header_t;

typedef struct ne_someip_ipc_send_rpc_msg_reply
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_method_id_t method_id;
    ne_someip_session_id_t session_id;
    // 如果是event或者response，则是0，如果是request，则是对应的clientid
    ne_someip_client_id_t client_id;
    ne_someip_l4_protocol_t protocol;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_port;
    uint32_t destination_address;
    uint16_t destination_port;
    ne_someip_error_code_t result;
} ne_someip_ipc_send_rpc_msg_reply_t;

/*endpoint 会保存socket的状态，比如正在创建中；
如果有多个instance都要请求同一个创建，instance会先在endpoint查找对应的socket的状态，如果是创建中，则不会再次发送创建请求，要把对应的instance指针注册到endpoint;*/
typedef struct ne_someip_ipc_create_socket
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    //ne_someip_service_id_t service_id;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_port;
    bool is_tls_used;
    ne_someip_ssl_key_info_t key_info;
} ne_someip_ipc_create_socket_t;

/*endpoint收到创建好的信息后，会将socket状态保存，遍历instance，如果instance中有等待创建socket的状态，则将该信息callback给instance*/
typedef struct ne_someip_ipc_create_socket_reply
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    //ne_someip_service_id_t service_id;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_port;
    bool result;
} ne_someip_ipc_create_socket_reply_t;

typedef struct ne_someip_ipc_join_group
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_address_type_t address_type;
    uint32_t multi_address;
    uint16_t multi_port;
    uint32_t if_address;
    uint16_t if_port;
}ne_someip_ipc_join_group_t;

typedef struct ne_someip_ipc_join_group_reply
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_address_type_t address_type;
    uint32_t multi_address;
    uint16_t multi_port;
    uint32_t if_address;
    uint16_t if_port;
    bool result;
}ne_someip_ipc_join_group_reply_t;

typedef struct ne_someip_ipc_tcp_connect {
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    //ne_someip_service_id_t service_id;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_port;
    uint32_t destination_address;
    uint16_t destination_port;
} ne_someip_ipc_tcp_connect_t;

typedef struct ne_someip_ipc_tcp_connect_reply
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    //ne_someip_service_id_t service_id;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_port;
    uint32_t destination_address;
    uint16_t destination_port;
    bool result;
} ne_someip_ipc_tcp_connect_reply_t;

/*注册handler到router，router保存的是pid信息*/
//如果之前有其他的instance注册过，是不是就不用再次注册了？
typedef struct ne_someip_ipc_reg_unreg_method_handler
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_tcp_port;
    uint16_t source_udp_port;
    ne_someip_list_t* method_id_list;  // <ne_someip_method_info_t*>
} ne_someip_ipc_reg_unreg_method_handler_t;

/*endpoint收到回复后，将遍历instance，如果instance里面有等待注册回复的状态，则call给这个instance*/
typedef struct ne_someip_ipc_reg_unreg_method_handler_reply
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_tcp_port;
    uint16_t source_udp_port;
    bool result;
} ne_someip_ipc_reg_unreg_method_handler_reply_t;

/*endpoint 收到socket的变化通知，会遍历instance，如果该instance关心socket状态，则将这个状态callback给instance*/
typedef struct ne_someip_ipc_reg_unreg_event_handler
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_udp_port;
    uint16_t source_tcp_port;
    ne_someip_list_t* event_eg_info_list;  // <ne_someip_event_info_t*>
}ne_someip_ipc_reg_unreg_event_handler_t;

typedef struct ne_someip_ipc_reg_unreg_event_handler_reply
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_udp_port;
    uint16_t source_tcp_port;
    bool result;
}ne_someip_ipc_reg_unreg_event_handler_reply_t;

/* endpoint 收到socket的变化通知，会遍历instance，如果该instance关心socket状态，则将这个状态callback给instance */
typedef struct ne_someip_ipc_tcp_udp_status_changed {
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_recv_tcp_udp_message_info_t socket_info;
    ne_someip_endpoint_transmit_link_state_t status;
} ne_someip_ipc_tcp_udp_status_changed_t;

/*endpoint 收到rpc消息后，遍历instance，如果该instance关心这个method，则将这个rpc callback给instance*/
typedef struct ne_someip_ipc_recv_remote_rpc_msg
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_recv_tcp_udp_message_info_t socket_info;
} ne_someip_ipc_recv_remote_rpc_msg_t;


/*注册handler到router，router保存的是pid信息*/
//如果之前有其他的instance注册过，是不是就不用再次注册了？
typedef struct ne_someip_ipc_reg_unreg_service_handler
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
} ne_someip_ipc_reg_unreg_service_handler_t;

typedef struct ne_someip_ipc_reg_unreg_service_handler_reply
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    bool result;
} ne_someip_ipc_reg_unreg_service_handler_reply_t;

/*************************************************sd ipc message*********************************************/
typedef struct ne_someip_ipc_notify_network_status
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    uint32_t ip_addr;
    ne_someip_network_states_t status;
} ne_someip_ipc_notify_network_status_t;

typedef struct ne_someip_ipc_send_offer
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_minor_version_t minor_version;
    bool reliable_flag;
    bool unreliable_flag;
    uint16_t tcp_port;
    uint16_t udp_port;
    ne_someip_address_type_t addr_type;
    uint32_t service_addr;
    ne_someip_server_offer_time_config_t timer;
    uint32_t remote_addr;
    uint16_t remote_port;
    ne_someip_endpoint_unix_addr_t proxy_unix_addr; //for router
    pthread_t tid;
} ne_someip_ipc_send_offer_t;

typedef struct ne_someip_ipc_send_find
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_minor_version_t minor_version;
    ne_someip_address_type_t addr_type;
    uint32_t local_addr;
    ne_someip_client_find_time_config_t timer;
    uint32_t remote_addr;
    uint16_t remote_port;
    ne_someip_endpoint_unix_addr_t proxy_unix_addr; // for router
    pthread_t tid;
} ne_someip_ipc_send_find_t;

//change to list or the first subscribe for multicast
typedef struct ne_someip_ipc_send_subscribe
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    bool is_delay;
    ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_list_t* eventgroup_list;//ne_someip_ipc_subscribe_eg_t
    uint8_t counter;
    uint16_t tcp_port;
    uint16_t udp_port;
    ne_someip_address_type_t addr_type;
    uint32_t local_addr;
    uint32_t remote_addr;
    uint16_t remote_port;
    ne_someip_endpoint_unix_addr_t proxy_unix_addr; // for router
    pthread_t tid;
} ne_someip_ipc_send_subscribe_t;

typedef struct ne_someip_ipc_send_subscribe_ack
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_list_t* sub_acks; //ne_someip_sd_recv_subscribe_ack_t
    uint32_t local_addr;
    uint16_t local_port;
    uint32_t remote_addr;
    uint16_t remote_port;
} ne_someip_ipc_send_subscribe_ack_t;

typedef struct ne_someip_ipc_recv_local_offer_status
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_minor_version_t minor_version;
    ne_someip_ser_offer_status_t status;
    pthread_t tid;
} ne_someip_ipc_recv_local_offer_status_t;

typedef struct ne_someip_ipc_recv_local_find_status
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_minor_version_t minor_version;
    ne_someip_find_service_states_t status;
    pthread_t tid;
} ne_someip_ipc_recv_local_find_status_t;

//change to list or the first subscribe for multicast
typedef struct ne_someip_ipc_recv_local_subscribe_status
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    // bool is_delay;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_list_t* eventgroup_list;//ne_someip_eventgroup_id_t*
    ne_someip_eventgroup_subscribe_states_t status;
    pthread_t tid;
} ne_someip_ipc_recv_local_subscribe_status_t;

/*endpoint收到offer后，遍历client的instance，如果该instance关心该offer，就将这个offer信息callback给instance*/
typedef struct ne_someip_ipc_recv_offer
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    // ne_someip_list_t* offer_list; //ne_someip_sd_recv_offer_t
    ne_someip_sd_recv_offer_t service;
} ne_someip_ipc_recv_offer_t;

typedef struct ne_someip_ipc_recv_subscribe
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_list_t* subscribe_list; // ne_someip_sd_recv_subscribe_t
    uint32_t remote_sd_addr;
    uint16_t remote_sd_port;
    uint32_t local_sd_addr;
    uint16_t local_sd_port;
} ne_someip_ipc_recv_subscribe_t;

typedef struct ne_someip_ipc_recv_subscribe_ack
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_list_t* sub_ack_list; //ne_someip_sd_recv_subscribe_ack_t
} ne_someip_ipc_recv_subscribe_ack_t;

typedef struct ne_someip_ipc_remote_reboot
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    uint32_t ip_addr;
} ne_someip_ipc_remote_reboot_t;

typedef struct  ne_someip_ipc_get_client_id
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_sequence_id_t seq_id;
    pthread_t tid;
} ne_someip_ipc_get_client_id_t;

typedef struct  ne_someip_ipc_get_client_id_reply
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_sequence_id_t seq_id;
    pthread_t tid;
    ne_someip_client_id_t client_id;
    ne_someip_client_id_t client_id_min;
    ne_someip_client_id_t client_id_max;
} ne_someip_ipc_get_client_id_reply_t;

typedef struct ne_someip_ipc_call_back_user_data
{
    ne_someip_ipc_msg_type_t type;
    void* instance;
} ne_someip_ipc_call_back_user_data_t;

typedef struct ne_someip_ipc_find_remote_svs
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_client_id_t client_id;
    ne_someip_sequence_id_t seq_id;
    pthread_t tid;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
} ne_someip_ipc_find_remote_svs_t;

typedef struct ne_someip_ipc_find_remote_svs_reply
{
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_client_id_t client_id;
    ne_someip_sequence_id_t seq_id;
    pthread_t tid;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_list_t* service_list; //ne_someip_sd_recv_offer_t
} ne_someip_ipc_find_remote_svs_reply_t;

typedef struct ne_someip_ipc_unix_addr_info {
    ne_someip_endpoint_unix_addr_t unix_addr;
    uint32_t count;
}ne_someip_ipc_unix_addr_info_t;

typedef struct ne_someip_ipc_endpoint_info {
    // tcp: ne_someip_endpoint_tcp_data_t*
    // udp: ne_someip_endpoint_udp_data_t*
    void* ep_data;
    ne_someip_list_t* unix_addr_list;    // ne_someip_ipc_unix_addr_info_t*
}ne_someip_ipc_endpoint_info_t;

typedef struct ne_someip_ipc_header_client_id {
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
}ne_someip_ipc_header_client_id_t;

typedef struct ne_someip_ipc_header_no_client_id {
    ne_someip_ipc_msg_type_t type;
    uint32_t length;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
}ne_someip_ipc_header_no_client_id_t;

#endif // MANAGER_NE_SOMEIP_IPC_DEFINE_H
/* EOF */
