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
#ifndef SRC_PROTOCOL_RPC_INCLUDE_NE_SOMEIP_INTERNAL_DEFINE_H
#define SRC_PROTOCOL_RPC_INCLUDE_NE_SOMEIP_INTERNAL_DEFINE_H

#ifndef INCLUDE_NE_SOMEIP_DEFINE_H
#    include "ne_someip_define.h"
#endif
#include "ne_someip_list.h"
#include <stdbool.h>

#define NE_SOMEIP_HEADER_LENGTH 16
#define NE_SOMEIP_HEADER_LEN_IN_MSG_LEN 8
#define NE_SOMEIP_TP_HEADER_LEN 4
// #define NE_SOMEIP_MAX_UDP_PAYLOAD_SIZE 65511//2^16 - 1 - 8(udp) - 16(someip)
#define NE_SOMEIP_MAX_UDP_PAYLOAD_SIZE 1400

#define UINT_32_T_MAX 2147483647
#define UINT_16_T_MAX 65535

#define NESOMEIP_EVENT_H_VALUE 0x8000

#define NE_SOMEIP_SYNC_TIMER_VALUE 2000
#define NE_SOMEIP_MULTI_THREAD_SYNC_OBJ_DESTROY_VALUE 1000

#define NE_SOMEIP_SSL_PATH_NAME_LEN 256
#define NE_SOMEIP_TLS_PATH_LEN 120

static const char ne_someip_server_unix_path[NESOMEIP_FILE_NAME_LENGTH] = "@/tmp/isomeipd_unix_server";
static const char ne_someip_server_rpc_unix_path[NESOMEIP_FILE_NAME_LENGTH] = "@/tmp/isomeipd_unix_server_rpc";
static const char ne_someip_client_unix_path[NESOMEIP_FILE_NAME_LENGTH] = "@/tmp/isomeipd_unix_server";
static const char ne_someip_tls_path[NE_SOMEIP_TLS_PATH_LEN] = "/system/etc/ara/";
static const char ne_someip_tls_ca_name[NE_SOMEIP_TLS_PATH_LEN] = "ca.crt";

static const uint8_t NESOMEIP_TP_MESSAGE_TYPE = 0x20;  // tp flag bit is 1

//define the ANY id
static const ne_someip_client_id_t NE_SOMEIP_ANY_CLIENT = 0xFFFF;
static const ne_someip_service_id_t NE_SOMEIP_ANY_SERVICE = 0xFFFF;
static const ne_someip_instance_id_t NE_SOMEIP_ANY_INSTANCE = 0xFFFF;
static const ne_someip_method_id_t NE_SOMEIP_ANY_METHOD = 0xFFFF;
static const ne_someip_event_id_t NE_SOMEIP_ANY_EVENT = 0xFFFF;
static const ne_someip_major_version_t NE_SOMEIP_ANY_MAJOR = 0xFF;
static const ne_someip_minor_version_t NE_SOMEIP_ANY_MINOR = 0xFFFFFFFF;

static const ne_someip_major_version_t NE_SOMEIP_DEFAULT_MAJOR = 0x00;
static const ne_someip_minor_version_t NE_SOMEIP_DEFAULT_MINOR = 0x000000;
static const ne_someip_interface_version_t NE_SOMEIP_DEFAULT_INTERFACE_VERSION = 0x01;
static const ne_someip_protocol_version_t NE_SOMEIP_DEFAULT_PROTOCOL_VERSION = 0x01;


typedef enum ne_someip_internal_error_code
{
    ne_someip_error_code_malloc_error = 0x01,  // malloc error
    ne_someip_error_code_link_monitor_error = 0x02,  // network link monitor error
    ne_someip_error_code_create_subscription_error = 0x03,  // create subscription failed when subcribe eventgroup
    ne_someip_error_code_server_endpoint_not_find = 0x04,  // server send response to client, but not find server endpoint
    ne_someip_error_code_tcp_client_endpoint_not_find = 0x05,  // tcp client send message to server, but not find tcp client endpoint
    ne_someip_error_code_tcp_disconnect = 0x06,  // tcp server disconnect by tcp client
    ne_someip_error_code_serialize_error = 0x07,  // message serializing failed when message was sending
    ne_someip_error_code_send_tcp_message_error = 0x08,  // send message using Tcp failed
    ne_someip_error_code_send_udp_message_error = 0x09,  // send message using Udp failed
    ne_someip_error_code_send_message_error = 0x0a,  // send message failed (UDP or TCP)
    ne_someip_error_code_null_pointer_error = 0x0b,  // NULL pointer
    ne_someip_error_code_create_notification_error = 0x0c,  // create Notification failed
    ne_someip_error_code_param_error = 0x0d,  // parameters error
    ne_someip_error_code_path_name_too_long = 0x0e,  // path name too long
    ne_someip_error_code_map_error = 0x0f,
    ne_someip_error_code_list_error = 0x10,
    ne_someip_error_code_client_context_not_create = 0x11,
    ne_someip_error_code_sync_create_error = 0x12,  // ne_someip_sync_obj_create failed
    ne_someip_error_code_post_looper_error = 0x13,
    ne_someip_error_code_ref_count_error = 0x14,  // reference count error
    ne_someip_error_code_req_ser_ins_config_error = 0x15,  // required_service_instance config have wrong info
    ne_someip_error_code_minor_version_error = 0x16,
    ne_someip_error_code_event_id_error = 0x17,
    ne_someip_error_code_method_id_error = 0x18,
    ne_someip_error_code_eventgroup_id_error = 0x19,
    ne_someip_error_code_dispatch_looper_error = 0x2A,
    ne_someip_error_code_snprintf_error = 0x2B,
    ne_someip_error_code_endpoint_type_error = 0x2C,
    ne_someip_error_code_udp_client_endpoint_not_find = 0x2D,
    ne_someip_error_code_create_unix_error = 0x2F, //create unix endpoint failed.
    ne_someip_error_code_socket_buffer_full = 0x30,  // send socket buffer full, wait to send data later
    ne_someip_error_code_comm_not_prepare_ok = 0x31,
    ne_someip_error_code_tp_segment_error = 0x32,
    ne_someip_error_code_tp_reassembe_error = 0x33,
    ne_someip_error_code_tp_send_not_finish = 0x34,
    ne_someip_error_code_send_not_finish = 0x35,
    ne_someip_error_code_internal_data = 0x36,
    ne_someip_internal_error_code_unknown  = 0xFF,
}ne_someip_internal_error_code_t;

/* send message states (state machine enum) for client and server */
typedef enum ne_someip_send_message_states
{
    ne_someip_send_message_states_not_triggered = 0x0,
    ne_someip_send_message_states_wait_link_ack = 0x1,
    ne_someip_send_message_states_sending = 0x2,
    ne_someip_send_message_states_send_success = 0x3,
    ne_someip_send_message_states_send_fail = 0x4,
}ne_someip_send_message_states_t;

/* network link states (state machine enum) for monitor */
typedef enum ne_someip_link_states
{
    ne_someip_link_states_up = 0x0,
    ne_someip_link_states_down = 0x1,
}ne_someip_link_states_t;

/* ip address states (state machine enum) for monitor */
typedef enum ne_someip_ip_address_states
{
    ne_someip_ip_address_states_unkown = 0x0,
    ne_someip_ip_address_states_known = 0x1,
}ne_someip_ip_address_states_t;

/* network states (state machine enum) for client and server */
typedef enum ne_someip_network_states
{
    ne_someip_network_states_unknown = 0x0,
    ne_someip_network_states_down = 0x1,
    ne_someip_network_states_up = 0x2,
}ne_someip_network_states_t;

typedef enum ne_someip_register_unregister_status {
    ne_someip_register_unregister_status_initial = 0,
    ne_someip_register_unregister_status_register = 1,
    ne_someip_register_unregister_status_unregister = 2,
    ne_someip_register_unregister_status_recv_register_reply = 3,
    ne_someip_register_unregister_status_recv_unregister_reply = 4,
} ne_someip_register_unregister_status_t;

typedef enum ne_someip_send_message_status {
    ne_someip_send_message_status_initial = 0,
    ne_someip_send_message_status_send = 1,
    ne_someip_send_message_status_recv_response = 2,
} ne_someip_send_message_status_t;

typedef enum ne_someip_endpoint_instance_type
{
    ne_someip_endpoint_instance_type_client = 0x0,
    ne_someip_endpoint_instance_type_server = 0x1,
    ne_someip_endpoint_instance_type_common = 0x2,
    ne_someip_endpoint_instance_type_no_use = 0x3,  // used in daemo
    ne_someip_endpoint_instance_type_unkown = 0x4,
} ne_someip_endpoint_instance_type_t;

typedef enum ne_someip_socket_create_states {
    ne_someip_socket_create_states_not_create = 0x0,
    ne_someip_socket_create_states_creating = 0x1,
    ne_someip_socket_create_states_created = 0x2,
}ne_someip_socket_create_states_t;

typedef struct ne_someip_endpoint_net_addr {
    uint32_t ip_addr;
    uint16_t port;
    ne_someip_address_type_t type;
}ne_someip_endpoint_net_addr_t;

typedef struct ne_someip_endpoint_unix_addr
{
    char unix_path[NESOMEIP_FILE_NAME_LENGTH];
}ne_someip_endpoint_unix_addr_t;

typedef struct ne_someip_enpoint_multicast_addr
{
    uint32_t multicast_ip;
}ne_someip_enpoint_multicast_addr_t;

typedef struct ne_someip_endpoint_buffer
{
    char* iov_buffer;
    uint32_t size;
}ne_someip_endpoint_buffer_t;

typedef struct ne_someip_if_name_info
{
    char if_name[NESOMEIP_IF_NAME_LENGTH];
    uint32_t ip_addr;
    bool is_enable;
}ne_someip_if_name_info_t;

typedef struct ne_someip_recv_tcp_udp_message_info {
    ne_someip_address_type_t addr_type;
    uint32_t local_addr;
    uint16_t local_port;
    uint32_t remote_addr;
    uint16_t remote_port;
    ne_someip_l4_protocol_t protocol;
    bool is_multicast;
} ne_someip_recv_tcp_udp_message_info_t;

typedef struct ne_someip_service_instance_base
{
    ne_someip_endpoint_instance_type_t type;
} ne_someip_service_instance_base_t;

typedef struct ne_someip_network_info
{
    char ifname[NESOMEIP_IF_NAME_LENGTH];
    uint32_t ip_addr;
    bool is_enabled;
	void* user_data;
} ne_someip_network_info_t;

typedef struct ne_someip_ssl_key_info
{
    char ca_crt_path[NE_SOMEIP_SSL_PATH_NAME_LEN];
    char crt_path[NE_SOMEIP_SSL_PATH_NAME_LEN];
    char key_path[NE_SOMEIP_SSL_PATH_NAME_LEN];
}ne_someip_ssl_key_info_t;

typedef struct ne_someip_event_eg_info
{
    ne_someip_event_id_t event_id;
    ne_someip_list_t* eg_id_list;  // <ne_someip_eventgroup_id_t*>
    ne_someip_l4_protocol_t protocol;
}ne_someip_event_eg_info_t;

typedef struct ne_someip_event_info
{
    ne_someip_event_id_t event_id;
    ne_someip_eventgroup_id_t eg_id;
    ne_someip_l4_protocol_t protocol;
}ne_someip_event_info_t;

typedef struct ne_someip_method_info
{
    ne_someip_method_id_t method_id;
    ne_someip_l4_protocol_t protocol;
}ne_someip_method_info_t;

#endif // SRC_PROTOCOL_RPC_INCLUDE_NE_SOMEIP_INTERNAL_DEFINE_H
/* EOF */
