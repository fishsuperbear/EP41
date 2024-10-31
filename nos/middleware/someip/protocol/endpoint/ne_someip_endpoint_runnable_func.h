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
#ifndef SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_RUNNABLE_FUNC_H
#define SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_RUNNABLE_FUNC_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "ne_someip_endpoint_define.h"
#include "ne_someip_looper.h"
#include "ne_someip_thread.h"

// typedef void (*ne_someip_ep_runna_common)(void*);

/***********************for sync call interface************************/
typedef struct ne_someip_ep_runnable_reg_instan_client
{
    ne_someip_thread_t* thread;
    void* endpoint;
    ne_someip_client_id_t client_id;
    const void* service_instance;
    int32_t sync_seq_id;
    ne_someip_endpoint_instance_type_t type;
}ne_someip_ep_runnable_reg_instan_client_t;

typedef struct ne_someip_ep_runnable_unreg_instan_client
{
    ne_someip_thread_t* thread;
    void* endpoint;
    ne_someip_client_id_t client_id;
    int32_t sync_seq_id;
    ne_someip_endpoint_instance_type_t type;
}ne_someip_ep_runnable_unreg_instan_client_t;

typedef struct ne_someip_ep_runnable_reg_instan_server
{
    ne_someip_thread_t* thread;
    void* endpoint;
    ne_someip_service_instance_spec_t service_instance_key;
    const void* service_instance;
    int32_t sync_seq_id;
}ne_someip_ep_runnable_reg_instan_server_t;

typedef struct ne_someip_ep_runnable_unreg_instan_server
{
    ne_someip_thread_t* thread;
    void* endpoint;
    ne_someip_service_instance_spec_t service_instance_key;
    int32_t sync_seq_id;
}ne_someip_ep_runnable_unreg_instan_server_t;

typedef struct ne_someip_ep_runnable_reg_callback
{
    ne_someip_thread_t* thread;
    void* endpoint;
    ne_someip_endpoint_callback_t* callback;
}ne_someip_ep_runnable_reg_callback_t;

typedef struct ne_someip_ep_runnable_unreg_callback
{
    ne_someip_thread_t* thread;
    void* endpoint;
    ne_someip_endpoint_callback_t* callback;
}ne_someip_ep_runnable_unreg_callback_t;

typedef struct ne_someip_ep_runnable_create_link
{
    ne_someip_thread_t* thread;
    void* endpoint;
    void* peer_addr;
    ne_someip_endpoint_link_role_t role;
    int32_t sync_seq_id;
}ne_someip_ep_runnable_create_link_t;

typedef struct ne_someip_ep_runnable_destroy_link
{
    ne_someip_thread_t* thread;
    void* endpoint;
    void* peer_addr;
    ne_someip_endpoint_link_role_t role;
    int32_t sync_seq_id;
}ne_someip_ep_runnable_destroy_link_t;

typedef struct ne_someip_ep_runnable_join_group
{
    ne_someip_thread_t* thread;
    void* endpoint;
    ne_someip_endpoint_net_addr_t* interface_addr;
    int32_t sync_seq_id;
}ne_someip_ep_runnable_join_group_t;

typedef struct ne_someip_ep_runnable_leave_group
{
    ne_someip_thread_t* thread;
    void* endpoint;
    ne_someip_endpoint_net_addr_t* interface_addr;
    int32_t sync_seq_id;
}ne_someip_ep_runnable_leave_group_t;

typedef struct ne_someip_ep_runnable_core_stop
{
    ne_someip_thread_t* thread;
    void* endpoint;
    int32_t sync_seq_id;
}ne_someip_ep_runnable_core_stop_t;
/***********************for sync call interface************************/

typedef struct ne_someip_ep_runnable_link_state_notify
{
    void* endpoint;
    ne_someip_endpoint_transmit_link_state_t state;
    void* pair_addr;
}ne_someip_ep_runnable_link_state_notify_t;

typedef struct ne_someip_ep_runnable_recv_msg
{
    void* endpoint;
    ne_someip_trans_buffer_struct_t* trans_buffer;
    void* pair_addr;
}ne_someip_ep_runnable_recv_msg_t;

typedef struct ne_someip_ep_runnable_async_reply
{
    void* endpoint;
    const void* seq_data;
    ne_someip_error_code_t result;
}ne_someip_ep_runnable_async_reply_t;

typedef struct ne_someip_ep_runnable_send_msg
{
    void* endpoint;
    ne_someip_trans_buffer_struct_t* trans_buffer;
    void* peer_addr;
    ne_someip_endpoint_send_policy_t send_policy;
    const void* seq_data;
}ne_someip_ep_runnable_send_msg_t;

typedef struct ne_someip_ep_runnable_send_on_timer
{
    void* endpoint;
    ne_someip_trans_buffer_struct_t* trans_buffer;
    void* peer_addr;
    const void* seq_data;
}ne_someip_ep_runnable_send_on_timer_t;

typedef struct ne_someip_ep_runnable_send_tp_data
{
    void* endpoint;
    ne_someip_trans_buffer_struct_t* trans_buffer;  // 原始数据
    ne_someip_list_t* tp_data_list;  // 拆分后的TP包
    ne_someip_endpoint_net_addr_t* peer_addr;
    ne_someip_endpoint_send_policy_t send_policy;
    const void* seq_data;
}ne_someip_ep_runnable_send_tp_data_t;

typedef struct ne_someip_ep_runnable_send_tp_data_on_timer
{
    void* endpoint;
    ne_someip_endpoint_udp_iov_cache_t* tp_data;
}ne_someip_ep_runnable_send_tp_data_on_timer_t;

void ne_someip_ep_reg_instan_client_run(ne_someip_ep_runnable_reg_instan_client_t* reg_instance_client);
void ne_someip_ep_reg_instan_client_free(ne_someip_ep_runnable_reg_instan_client_t* reg_instance_client);
void ne_someip_ep_unreg_instan_client_run(ne_someip_ep_runnable_unreg_instan_client_t* unreg_instance_client);
void ne_someip_ep_unreg_instan_client_free(ne_someip_ep_runnable_unreg_instan_client_t* unreg_instance_client);
void ne_someip_ep_reg_instan_server_run(ne_someip_ep_runnable_reg_instan_server_t* reg_instance_server);
void ne_someip_ep_reg_instan_server_free(ne_someip_ep_runnable_reg_instan_server_t* reg_instance_server);
void ne_someip_ep_unreg_instan_server_run(ne_someip_ep_runnable_unreg_instan_server_t* unreg_instance_server);
void ne_someip_ep_unreg_instan_server_free(ne_someip_ep_runnable_unreg_instan_server_t* unreg_instance_server);
void ne_someip_ep_reg_callback_run(ne_someip_ep_runnable_reg_callback_t* reg_callback);
void ne_someip_ep_reg_callback_free(ne_someip_ep_runnable_reg_callback_t* reg_callback);
void ne_someip_ep_unreg_callback_run(ne_someip_ep_runnable_unreg_callback_t* unreg_callback);
void ne_someip_ep_unreg_callback_free(ne_someip_ep_runnable_unreg_callback_t* unreg_callback);
void ne_someip_ep_create_link_run(ne_someip_ep_runnable_create_link_t* create_link);
void ne_someip_ep_create_link_free(ne_someip_ep_runnable_create_link_t* create_link);
void ne_someip_ep_destroy_link_run(ne_someip_ep_runnable_destroy_link_t* destroy_link);
void ne_someip_ep_destroy_link_free(ne_someip_ep_runnable_destroy_link_t* destroy_link);
void ne_someip_ep_join_group_run(ne_someip_ep_runnable_join_group_t* join_group);
void ne_someip_ep_join_group_free(ne_someip_ep_runnable_join_group_t* join_group);
void ne_someip_ep_leave_group_run(ne_someip_ep_runnable_leave_group_t* leave_group);
void ne_someip_ep_leave_group_free(ne_someip_ep_runnable_leave_group_t* leave_group);
void ne_someip_ep_link_state_notify_run(ne_someip_ep_runnable_link_state_notify_t* link_state_notify);
void ne_someip_ep_link_state_notify_free(ne_someip_ep_runnable_link_state_notify_t* link_state_notify);
void ne_someip_ep_recv_msg_run(ne_someip_ep_runnable_recv_msg_t* recv_msg);
void ne_someip_ep_recv_msg_free(ne_someip_ep_runnable_recv_msg_t* recv_msg);
void ne_someip_ep_async_reply_run(ne_someip_ep_runnable_async_reply_t* reply);
void ne_someip_ep_async_reply_free(ne_someip_ep_runnable_async_reply_t* reply);
void ne_someip_ep_send_msg_run(ne_someip_ep_runnable_send_msg_t* send_msg);
void ne_someip_ep_send_msg_free(ne_someip_ep_runnable_send_msg_t* send_msg);
void ne_someip_ep_send_on_timer_run(ne_someip_ep_runnable_send_on_timer_t* send_on_timer);
void ne_someip_ep_send_on_timer_free(ne_someip_ep_runnable_send_on_timer_t* send_on_timer);
void ne_someip_ep_send_tp_data_run(ne_someip_ep_runnable_send_tp_data_t* send_tp_data);
void ne_someip_ep_send_tp_data_free(ne_someip_ep_runnable_send_tp_data_t* send_tp_data);
void ne_someip_ep_send_tp_on_timer_run(ne_someip_ep_runnable_send_tp_data_on_timer_t* send_on_timer);
void ne_someip_ep_send_tp_on_timer_free(ne_someip_ep_runnable_send_tp_data_on_timer_t* send_on_timer);
void ne_someip_ep_core_stop_run(ne_someip_ep_runnable_core_stop_t* core_stop_struct);
void ne_someip_ep_core_stop_free(ne_someip_ep_runnable_core_stop_t* core_stop_struct);

ne_someip_looper_runnable_t* ne_someip_ep_create_reg_instan_client_runnable(ne_someip_thread_t* thread, void* endpoint,
    ne_someip_client_id_t client_id, const void* service_instance, int32_t sync_seq_id, ne_someip_endpoint_instance_type_t type);
ne_someip_looper_runnable_t* ne_someip_ep_create_unreg_instan_client_runnable(ne_someip_thread_t* thread, void* endpoint,
    ne_someip_client_id_t client_id, int32_t sync_seq_id, ne_someip_endpoint_instance_type_t type);
ne_someip_looper_runnable_t* ne_someip_ep_create_reg_instan_server_runnable(ne_someip_thread_t* thread, void* endpoint, 
    ne_someip_service_instance_spec_t service_instance_key, const void* service_instance, int32_t sync_seq_id);
ne_someip_looper_runnable_t* ne_someip_ep_create_unreg_instan_server_runnable(ne_someip_thread_t* thread, void* endpoint, 
    ne_someip_service_instance_spec_t service_instance_key, int32_t sync_seq_id);
ne_someip_looper_runnable_t* ne_someip_ep_create_reg_callback_runnable(ne_someip_thread_t* thread, void* endpoint, ne_someip_endpoint_callback_t* callback);
ne_someip_looper_runnable_t* ne_someip_ep_create_unreg_callback_runnable(ne_someip_thread_t* thread, void* endpoint,
    ne_someip_endpoint_callback_t* callback);
ne_someip_looper_runnable_t* ne_someip_ep_create_create_link_runnable(ne_someip_thread_t* thread, void* endpoint, void* peer_addr,
    ne_someip_endpoint_link_role_t role, int32_t sync_seq_id);
ne_someip_looper_runnable_t* ne_someip_ep_create_destroy_link_runnable(ne_someip_thread_t* thread, void* endpoint, void* peer_addr,
    ne_someip_endpoint_link_role_t role, int32_t sync_seq_id);
ne_someip_looper_runnable_t* ne_someip_ep_create_join_group_runnable(ne_someip_thread_t* thread, void* endpoint,
    ne_someip_endpoint_net_addr_t* interface_addr, int32_t sync_seq_id);
ne_someip_looper_runnable_t* ne_someip_ep_create_leave_group_runnable(ne_someip_thread_t* thread, void* endpoint,
    ne_someip_endpoint_net_addr_t* interface_addr, int32_t sync_seq_id);
ne_someip_looper_runnable_t* ne_someip_ep_create_link_state_notify_runnable(void* endpoint, ne_someip_endpoint_transmit_link_state_t state,
    void* pair_addr);
ne_someip_looper_runnable_t* ne_someip_ep_create_recv_msg_runnable(void* endpoint, ne_someip_trans_buffer_struct_t* trans_buffer,
    void* pair_addr);
ne_someip_looper_runnable_t* ne_someip_ep_create_async_reply_runnable(void* endpoint, const void* seq_data, ne_someip_error_code_t result);
ne_someip_looper_runnable_t* ne_someip_ep_create_send_msg_runnable(void* endpoint, ne_someip_trans_buffer_struct_t* trans_buffer,
    void* peer_addr, ne_someip_endpoint_send_policy_t* send_policy, const void* seq_data);
ne_someip_looper_runnable_t* ne_someip_ep_create_send_tp_data_runnable(void* endpoint, ne_someip_trans_buffer_struct_t* trans_buffer,
    ne_someip_list_t* tp_data_list, void* peer_addr, ne_someip_endpoint_send_policy_t* send_policy, const void* seq_data);
ne_someip_looper_runnable_t* ne_someip_ep_create_core_stop_runnable(void* endpoint, ne_someip_thread_t* thread, int32_t sync_seq_id);

ne_someip_looper_timer_runnable_t* ne_someip_ep_create_send_on_timer_runnable(void* endpoint, ne_someip_trans_buffer_struct_t* trans_buffer,
    void* peer_addr, const void* seq_data);
ne_someip_looper_timer_runnable_t* ne_someip_ep_create_send_tp_on_timer_runnable(void* endpoint, ne_someip_endpoint_udp_iov_cache_t* tp_data);

ne_someip_looper_runnable_t* ne_someip_ep_create_runnable(void* run, void* free, void* user_data);
void ne_someip_ep_destroy_runnable(ne_someip_looper_runnable_t* runnable);
ne_someip_looper_timer_runnable_t*
    ne_someip_ep_create_timer_runnable(void* run, void* free, void* user_data);
void ne_someip_ep_destroy_timer_runnable(ne_someip_looper_timer_runnable_t* runnable);

void* ne_someip_ep_create_addr_mem(void* endpoint, void* peer_addr);
void ne_someip_ep_destroy_addr_mem(void* endpoint, void* peer_addr);
void* ne_someip_ep_create_pair_addr_mem(void* pair_addr);
void ne_someip_ep_destroy_pair_addr_mem(void* pair_addr);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_RUNNABLE_FUNC_H
/* EOF */
