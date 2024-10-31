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

#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_CLIENT_DEFINE_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_CLIENT_DEFINE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "ne_someip_looper.h"
#include "ne_someip_thread.h"
#include "ne_someip_map.h"
#include "ne_someip_list.h"
#include "ne_someip_config_define.h"
#include "ne_someip_handler.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_endpoint_define.h"
#include "ne_someip_sd_define.h"
#include "ne_someip_sync_obj.h"
#include "ne_someip_object.h"
#include "ne_someip_network_monitor.h"

/* remote service connect states (state machine enum) */
typedef enum ne_someip_remote_service_states
{
    ne_someip_remote_service_states_unknown = 0x0,
    ne_someip_remote_service_states_connect = 0x1,
    ne_someip_remote_service_states_disconnect = 0x2,
}ne_someip_remote_service_status_t;

typedef enum ne_someip_tcp_connect_states {
    ne_someip_tcp_connect_states_disconnect = 0x0,
    ne_someip_tcp_connect_states_connecting = 0x1,
    ne_someip_tcp_connect_states_connected = 0x2,
}ne_someip_tcp_connect_states_t;

/* find service states (state machine enum) for client */
typedef enum ne_someip_find_service_states
{
    ne_someip_find_service_states_not_triggered = 0x0,
    ne_someip_find_service_states_wait_network_up = 0x1,
    ne_someip_find_service_states_start_find = 0x2,
    ne_someip_find_service_states_finding = 0x3,
    ne_someip_find_service_states_stop_find = 0x4,
    ne_someip_find_service_states_timeout = 0x5,
}ne_someip_find_service_states_t;

/* find service states (state machine enum) for client */
typedef enum ne_someip_eventgroup_subscribe_states
{
    ne_someip_eventgroup_subscribe_states_not_triggered = 0x0,
    ne_someip_eventgroup_subscribe_states_wait_network_up = 0x1,
    ne_someip_eventgroup_subscribe_states_wait_available = 0x2,
    ne_someip_eventgroup_subscribe_states_start_subscribe = 0x3,
    ne_someip_eventgroup_subscribe_states_subscribing = 0x4,
    ne_someip_eventgroup_subscribe_states_stop_subscribe = 0x5,
    ne_someip_eventgroup_subscribe_states_subscribe_ack = 0x6,
    ne_someip_eventgroup_subscribe_states_subscribe_nack = 0x7,
    ne_someip_eventgroup_subscribe_states_timeout = 0x8,
}ne_someip_eventgroup_subscribe_states_t;

typedef enum ne_someip_required_com_type
{
    ne_someip_required_com_type_unknow = 0x0,
    ne_someip_required_com_type_tcp = 0x1,
    ne_someip_required_com_type_udp = 0x2,
    ne_someip_required_com_type_tcp_udp = 0x5,
}ne_someip_required_com_type_t;

typedef enum ne_someip_required_handler_type
{
    ne_someip_required_handler_type_find = 0x1,
    ne_someip_required_handler_type_avail = 0x2,
    ne_someip_required_handler_type_sub = 0x3,
    ne_someip_required_handler_type_event = 0x4,
    ne_someip_required_handler_type_resp = 0x5,
    ne_someip_required_handler_type_send = 0x6,
}ne_someip_required_handler_type_t;

typedef struct ne_someip_subscribe_event_key
{
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_event_id_t event_id;
}ne_someip_subscribe_event_key_t;

typedef struct ne_someip_required_event_behaviour
{
    ne_someip_event_config_t* config;
}ne_someip_required_event_behaviour_t;

typedef struct ne_someip_required_eventgroup_behaviour
{
    ne_someip_required_eventgroup_config_t* config;
    ne_someip_eventgroup_subscribe_states_t sub_state;
    ne_someip_subscribe_status_t pre_upper_sub_status;
}ne_someip_required_eventgroup_behaviour_t;

typedef struct ne_someip_required_method_behaviour
{
    ne_someip_method_config_t* config;
    uint32_t udp_collection_buffer_timeout;
    ne_someip_udp_collection_trigger_t udp_collection_trigger;
}ne_someip_required_method_behaviour_t;

typedef struct ne_someip_required_network_connect_behaviour
{
    const ne_someip_network_config_t* config;
    ne_someip_network_states_t network_status;
    uint32_t ip_addr;
    ne_someip_map_t* find_behav_map;  // <ne_someip_find_offer_service_spec_t*, ne_someip_required_find_service_behaviour_t*>
                                        // down 的时候，记录下当前网卡的find_behav_up 的时候，当前网卡发送完成后remove_all（只清除map，不清除data）
}ne_someip_required_network_connect_behaviour_t;

typedef struct ne_someip_required_find_ser_behaviour
{
    ne_someip_client_id_t client_id;
    ne_someip_find_offer_service_spec_t service_spec;
    ne_someip_find_service_states_t find_service_state;
    ne_someip_find_status_t prev_upper_find_status;
    const ne_someip_required_service_instance_config_t* config;
    const ne_someip_network_config_t* net_config;
    const ne_someip_client_find_time_config_t* find_timer_config;
    ne_someip_looper_t* io_looper;
    void* instance;
}ne_someip_required_find_service_behaviour_t;

typedef struct ne_someip_required_service_connect_behaviour
{
    ne_someip_service_status_t avail_status;
    ne_someip_remote_service_status_t status;  // remote service availability status
    ne_someip_socket_create_states_t remote_tcp_socket_states;  // (forward used)
    ne_someip_socket_create_states_t remote_udp_socket_states;  // (forward used)
    ne_someip_tcp_connect_states_t tcp_connect_states;  // (forward used)
    bool reliable_flag;
    bool unreliable_flag;
    ne_someip_endpoint_net_addr_t* peer_udp_addr;
    ne_someip_endpoint_net_addr_t* peer_tcp_addr;
    // ne_someip_map_t* sub_behav_map;  // <ne_someip_eventgroup_id_t*, ne_someip_required_eventgroup_behaviour_t*>
    //                                    // unavailable 的时候，记录下sub_behav；availble 的时候，发送完成后remove_all
    NEOBJECT_MEMBER
}ne_someip_required_service_connect_behaviour_t;

typedef struct ne_someip_saved_find_handler
{
    ne_someip_find_status_handler handler;
    const void* user_data;
}ne_someip_saved_find_handler_t;

typedef struct ne_someip_saved_available_handler
{
    ne_someip_service_available_handler handler;
    const void* user_data;
}ne_someip_saved_available_handler_t;

typedef struct ne_someip_saved_subscribe_status_handler
{
    ne_someip_subscribe_status_handler handler;
    const void* user_data;
}ne_someip_saved_subscribe_status_handler_t;

typedef struct ne_someip_saved_recv_event_handler
{
    ne_someip_recv_event_handler handler;
    const void* user_data;
}ne_someip_saved_recv_event_handler_t;

typedef struct ne_someip_saved_recv_response_handler
{
    ne_someip_recv_response_handler handler;
    const void* user_data;
}ne_someip_saved_recv_response_handler_t;

typedef struct ne_someip_saved_send_status_handler
{
    ne_someip_send_req_status_handler handler;
    const void* user_data;
}ne_someip_saved_send_status_handler_t;

typedef struct ne_someip_client_send_seq_data
{
    ne_someip_method_id_t method_id;
    const void* seq_data;
}ne_someip_client_send_seq_data_t;

typedef struct ne_someip_client_find_local_services
{
    ne_someip_find_local_offer_services_t* offer_services;
    ne_someip_error_code_t res;
}ne_someip_client_find_local_services_t;

typedef struct ne_someip_client_daemon_client_id_info
{
    ne_someip_client_id_t client_id;
    ne_someip_error_code_t res;
}ne_someip_client_daemon_client_id_info_t;

typedef struct ne_someip_client_inter_config
{
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_minor_version_t minor_version;
    uint16_t tcp_port;
    uint16_t udp_port;
    bool tcp_reuse;
    bool udp_reuse;
    uint32_t udp_collection;
    bool tcp_tls_flag;
    bool udp_tls_flag;
    ne_someip_address_type_t addr_type;
    uint16_t sd_multicast_port;
}ne_someip_client_inter_config_t;

typedef struct ne_someip_saved_req_seq_info
{
    ne_someip_method_id_t method_id;
    ne_someip_session_id_t session_id;
}ne_someip_saved_req_seq_info_t;

typedef struct ne_someip_common_service_instance {
    ne_someip_service_instance_base_t type;
    ne_someip_client_id_t client_id;
    ne_someip_endpoint_unix_t* unix_endpoint;
    ne_someip_looper_t* work_looper;
    ne_someip_looper_t* io_looper;
    ne_someip_map_t* find_behav_map;  // <ne_someip_find_offer_service_spec_t*, ne_someip_required_find_service_behaviour_t*>
                                        // 这个map用来管理find_service的状态等信息，key的值可以是any
    ne_someip_map_t* ser_con_behav_map;  // <ne_someip_find_offer_service_spec_t*, ne_someip_required_service_connect_behaviour_t*>，保存通知service status的信息
                                         // 这个结构体用来管理接收offer service的信息，key的值必须是确定的值
    ne_someip_sync_obj_t* ser_con_behav_sync;
    ne_someip_map_t* ser_con_behav_any_map;  // <ne_someip_service_instance_spec_t*, ne_someip_required_service_connect_behaviour_t*>
                                             // 保存收到offer之前创建req_instance时创建的ser_connect_behaviour，此时不知道minor_version的信息，
                                             // 收到offer后，移到ser_con_behav_map中
    ne_someip_sync_obj_t* ser_con_behav_any_sync;
    ne_someip_map_t* reg_requir_service_map;  // <ne_someip_service_instance_spec_t*, ne_someip_list_t*<ne_someip_required_service_instance_t*>>
                                              // required_service_instance 注册到 common instance 中，用于链接require service和service status
                                              // 根据此表回调到 required_service_instance 中
    ne_someip_sync_obj_t* reg_requir_service_map_sync;
    ne_someip_map_t* find_handler_map;  // <ne_someip_find_offer_service_spec_t*, ne_someip_list_t*<ne_someip_saved_find_handler_t*>>
                                        // 这个map用来管理find_service的状态等信息，key的值可以是any
    ne_someip_sync_obj_t* find_handler_sync;
    ne_someip_map_t* serv_avail_handler_map;  // <ne_someip_find_offer_service_spec_t*, ne_someip_list_t*<ne_someip_saved_available_handler_t*>>
                                             // 这个map用来保存available_handler，key的值可以是any
    ne_someip_sync_obj_t* serv_avail_handler_sync;
    ne_someip_map_t* find_offer_serv_map;  // <seq_id, ne_someip_client_find_local_services_t*>
                                           // 用于query_offer_service 同步接口返回查询结果
    ne_someip_sync_obj_t* find_offer_serv_sync;
    ne_someip_map_t* net_connect_behav_map;  // 网卡监听的信息, <ne_someip_network_config_t*, ne_someip_required_network_connect_behaviour_t*>
}ne_someip_common_service_instance_t;

struct ne_someip_client_context
{
    void* app_context;
    ne_someip_client_id_t client_id;
    ne_someip_sequence_id_t seq_id;
    ne_someip_sync_obj_t* seq_id_sync;
    ne_someip_looper_t* work_looper;
    ne_someip_thread_t* work_thread;
    ne_someip_looper_t* io_looper;
    ne_someip_thread_t* io_thread;
    bool is_shared_thread;
    ne_someip_common_service_instance_t* common_instance;
    ne_someip_endpoint_unix_t* unix_endpoint;
    NEOBJECT_MEMBER
};

struct ne_someip_required_service_instance
{
    ne_someip_service_instance_base_t type;
    ne_someip_client_id_t client_id;
    ne_someip_session_id_t session_id;
    ne_someip_client_inter_config_t inter_config;
    ne_someip_looper_t* work_looper;
    ne_someip_looper_t* io_looper;
    const ne_someip_required_service_instance_config_t* config;
    const ne_someip_ssl_key_info_t* tcp_key_info;
    const ne_someip_ssl_key_info_t* udp_key_info;
    ne_someip_required_service_connect_behaviour_t* service_connect_behaviour;
    ne_someip_required_network_connect_behaviour_t* network_connect_behaviour;
    ne_someip_map_t* event_behaviour_map;  // <event_id, ne_someip_required_event_behaviour_t*>
    ne_someip_map_t* eventgroup_behaviour_map;  // <eventgroup_id, ne_someip_required_eventgroup_behaviour_t*>
    bool is_sub_event_wait_send;
    ne_someip_map_t* method_behaviour_map;  // <method_id, ne_someip_required_method_behaviour_t*>
    ne_someip_list_t* sub_handler_list;  // <ne_someip_saved_subscribe_status_handler_t*>
    ne_someip_sync_obj_t* sub_handler_sync;
    ne_someip_list_t* event_handler_list; // <ne_someip_saved_recv_event_handler_t*>
    ne_someip_sync_obj_t* event_handler_sync;
    ne_someip_list_t* resp_handler_list;  // <ne_someip_saved_recv_response_handler_t*>
    ne_someip_sync_obj_t* resp_handler_sync;
    ne_someip_list_t* send_status_handler_list;  // <ne_someip_saved_send_status_handler_t*>
    ne_someip_sync_obj_t* send_status_handler_sync;
    ne_someip_endpoint_unix_t* unix_endpoint;
    ne_someip_endpoint_udp_data_t* udp_endpoint;
    ne_someip_map_t* udp_multi_ep_map;  // <ne_someip_endpoint_net_addr_t*, ne_someip_endpoint_udp_data_t*>
    ne_someip_endpoint_tcp_data_t* tcp_endpoint;
    int tcp_retry_num;
    ne_someip_map_t* save_req_seq_map;  // <ne_someip_saved_req_seq_info_t*, void*>
    NEOBJECT_MEMBER
};

typedef struct ne_someip_net_status_notify_data
{
    char if_name[NE_SOMEIP_NETWORK_IFNAME_LEN];
    uint32_t ip_addr;
    bool is_enabled;
    void* user_data;
}ne_someip_net_status_notify_data_t;

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_CLIENT_DEFINE_H
/* EOF */
