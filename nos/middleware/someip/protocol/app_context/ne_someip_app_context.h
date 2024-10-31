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
#ifndef SRC_PROTOCOL_APP_CONTEXT_NE_SOMEIP_APP_CONTEXT_H
#define SRC_PROTOCOL_APP_CONTEXT_NE_SOMEIP_APP_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_endpoint_define.h"
#include "ne_someip_client_define.h"
#include "ne_someip_sync_obj.h"
#include "ne_someip_thread.h"
#include "ne_someip_ipc_define.h"
#include "ne_someip_sync_wait_obj.h"

#if SOMEIP_QNX_PLATFORM
#include <process.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

#include "ne_someip_thread.h"

typedef struct ne_someip_app_context
{
    ne_someip_sequence_id_t seq_id;
    ne_someip_sync_obj_t* seq_id_sync;
    ne_someip_map_t* client_context_map;  // <client_id, ne_someip_client_context_t*>
    ne_someip_sync_obj_t* client_context_map_sync;
    ne_someip_map_t* tcp_endpoint_map;  // <ne_someip_endpoint_net_addr_t*, ne_someip_endpoint_tcp_data_t*>
    ne_someip_sync_obj_t* tcp_endpoint_map_sync;
    ne_someip_map_t* udp_ednpoint_map;  // <ne_someip_endpoint_net_addr_t*, ne_someip_endpoint_udp_data_t*>
    ne_someip_sync_obj_t* udp_ednpoint_map_sync;
    ne_someip_map_t* unix_endpoint_map;  // <ne_someip_endpoint_unix_addr_t*, ne_someip_endpoint_unix_t*>
    ne_someip_sync_obj_t* unix_endpoint_map_sync;
    ne_someip_endpoint_unix_t* shared_unix_endpoint;
    ne_someip_looper_t* io_looper;
    ne_someip_looper_t* work_looper;
    ne_someip_thread_t* io_thread;
    ne_someip_thread_t* work_thread;
    ne_someip_sync_obj_t* io_mutex;
    ne_someip_sync_obj_t* work_mutex;
    char unix_path[NESOMEIP_FILE_NAME_LENGTH];
    char server_unix_path[NESOMEIP_FILE_NAME_LENGTH];
    char forward_unix_path[NESOMEIP_FILE_NAME_LENGTH];
    ne_someip_client_id_t daemon_client_max;
    ne_someip_client_id_t daemon_client_min;
    bool is_daemon_client_id_get;
    ne_someip_network_config_t* default_net_config;
    ne_someip_map_t* reg_service_client_map;  // <ne_someip_find_offer_service_spec_t*, ne_someip_list_t*<ne_someip_client_id_t*>>
                                              // common_service_instance 注册所需要的 service status 到app_context中
                                              // 根据此表查找client_id，然后回调到 common_service_instance 中
    ne_someip_sync_obj_t* reg_service_client_map_sync;
    ne_someip_map_t* wait_obj_map;  // <pthread_t, ne_someip_sync_wait_obj_t*>
                                    // 用于实现假同步调用
    ne_someip_sync_obj_t* wait_obj_sync;
    ne_someip_map_t* client_id_map;  // <seq_id, ne_someip_client_daemon_client_id_info_t*>
                                     // 用于从daemon获取client_id，同步接口返回查询结果
    ne_someip_sync_obj_t* client_id_sync;

}ne_someip_app_context_t;

uint32_t ne_someip_app_context_get_payload_len(const ne_someip_payload_t* payload);
bool ne_someip_app_context_trans_payload(ne_someip_endpoint_buffer_t* buffer, ne_someip_payload_t* payload);

void ne_someip_app_context_unref(ne_someip_app_context_t* context);

ne_someip_app_context_t* ne_someip_app_context_create();

void ne_someip_app_context_t_free(ne_someip_app_context_t* context);

// interface for thread
void ne_someip_app_context_dispatch_looper(bool is_shared, ne_someip_looper_t** work_looper,
    ne_someip_thread_t** work_thread, ne_someip_looper_t** io_looper,
    ne_someip_thread_t** io_thread);

// interface for client_context
ne_someip_client_context_t* ne_someip_app_context_find_client_context(ne_someip_client_id_t client_id);  // ne_someip_client_context_t*
bool ne_someip_app_context_save_client_context(ne_someip_client_id_t client_id, ne_someip_client_context_t* client_context);  // ne_someip_client_context_t*
bool ne_someip_app_context_delete_client_context(ne_someip_client_id_t client_id);

// interface for endpoint
ne_someip_endpoint_tcp_data_t* ne_someip_app_context_find_tcp_endpoint(ne_someip_endpoint_net_addr_t* addr);
bool ne_someip_app_context_save_tcp_endpoint(ne_someip_endpoint_net_addr_t* addr, ne_someip_endpoint_tcp_data_t* endpoint);
bool ne_someip_app_context_delete_tcp_endpoint(ne_someip_endpoint_net_addr_t* addr);
ne_someip_endpoint_udp_data_t* ne_someip_app_context_find_udp_endpoint(ne_someip_endpoint_net_addr_t* addr);
bool ne_someip_app_context_save_udp_endpoint(ne_someip_endpoint_net_addr_t* addr, ne_someip_endpoint_udp_data_t* endpoint);
bool ne_someip_app_context_delete_udp_endpoint(ne_someip_endpoint_net_addr_t* addr);

ne_someip_endpoint_unix_t*
ne_someip_app_context_create_unix_endpoint(const ne_someip_thread_t* work_thread, const ne_someip_looper_t* work_looper,
	const ne_someip_looper_t* io_looper);
ne_someip_endpoint_unix_t* ne_someip_app_context_find_unix_endpoint(ne_someip_endpoint_unix_addr_t* addr);
ne_someip_endpoint_unix_t* ne_someip_app_context_find_common_unix_endpoint();
bool ne_someip_app_context_save_unix_endpoint(ne_someip_endpoint_unix_addr_t* addr, ne_someip_endpoint_unix_t* endpoint);
bool ne_someip_app_context_delete_unix_endpoint(ne_someip_endpoint_unix_addr_t* addr);

pid_t ne_someip_app_context_get_pid_id();
pthread_t ne_someip_app_context_get_tid_id();
bool ne_someip_app_context_get_saved_unix_path(ne_someip_endpoint_unix_addr_t* unix_addr);
bool ne_someip_app_context_get_unix_path(ne_someip_endpoint_unix_addr_t* unix_addr, pthread_t id);
char* ne_someip_app_context_get_server_unix_path();

bool ne_someip_app_context_get_ipc_unix_path(ne_someip_endpoint_unix_addr_t* unix_addr, pthread_t id);
char* ne_someip_app_context_get_ipc_server_unix_path();

ne_someip_sync_wait_obj_t* ne_someip_app_context_create_save_wait_obj(pthread_t id);
ne_someip_sync_wait_obj_t* ne_someip_app_context_find_wait_obj(pthread_t id);
bool ne_someip_app_context_destroy_remove_wait_obj(pthread_t id);

// interface for seq_id
int64_t ne_someip_app_context_get_seq_id();

// interface for client_id
bool ne_someip_app_conetxt_is_daemon_client_id_get();
bool ne_someip_app_context_is_client_id_right(ne_someip_client_id_t client_id);
void ne_someip_app_context_get_daemon_client_id(ne_someip_sequence_id_t seq_id, pthread_t tid);
// save current sharedpoint, used for server exit
bool ne_someip_app_context_save_shared_endpoint(ne_someip_endpoint_unix_t* endpoint);

bool ne_someip_app_conetxt_create_client_id_info(ne_someip_sequence_id_t seq_id);
void ne_someip_app_conetxt_add_client_id_info(ne_someip_sequence_id_t seq_id, ne_someip_client_id_t client_id,
    ne_someip_error_code_t find_res);
ne_someip_client_daemon_client_id_info_t* ne_someip_app_conetxt_find_client_id_info(ne_someip_sequence_id_t seq_id);
bool ne_someip_app_conetxt_delete_client_id_info(ne_someip_sequence_id_t seq_id);

// interface for get default network info from someip daemon
ne_someip_network_config_t* ne_someip_app_context_get_default_net_info();

// interface for common_service to register/unregister service available handler
void ne_someip_app_context_query_client_id_for_avail_status(const ne_someip_find_offer_service_spec_t* spec,
    ne_someip_list_t* client_id_List);
bool ne_someip_app_context_reg_client_id_for_avail_status(const ne_someip_find_offer_service_spec_t* spec,
    ne_someip_client_id_t client_id, void* instance);
bool ne_someip_app_context_unreg_client_id_for_avail_status(const ne_someip_find_offer_service_spec_t* spec,
    ne_someip_client_id_t client_id, void* instance);
bool ne_someip_app_context_unreg_all_client_id_for_service_status(ne_someip_client_id_t client_id);

// interface for common service to update serv_connect_behav status
void ne_someip_app_context_update_ser_connect_behav_by_net_down(const ne_someip_find_offer_service_spec_t* spec);

/*********************************callback**********************************/
// recv client id from daemon
void ne_someip_app_context_recv_daemon_client_id(const ne_someip_ipc_get_client_id_reply_t* client_id_info);
// recv find reply info (from sd)
void ne_someip_app_context_recv_find_reply(const ne_someip_ipc_recv_local_find_status_t* find_status);
// recv offer service info (sd)
void ne_someip_app_context_recv_offer_service_handler(const ne_someip_ipc_recv_offer_t* offer_msg);
/*********************************callback**********************************/

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_APP_CONTEXT_NE_SOMEIP_APP_CONTEXT_H
/* EOF */
