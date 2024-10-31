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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_CLIENT_RUNNABLE_FUNC_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_CLIENT_RUNNABLE_FUNC_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_client_define.h"
#include "ne_someip_client_context.h"
#include "ne_someip_app_context.h"
#include "ne_someip_list.h"

typedef enum ne_someip_client_runna_timer_type
{
    ne_someip_client_runna_timer_type_get_client_id = 0x0,
    ne_someip_client_runna_timer_type_get_local_services = 0x1,
    ne_someip_client_runna_timer_type_stop_find = 0x2,
    ne_someip_client_runna_timer_type_stop_subscribe = 0x3,
}ne_someip_client_runna_timer_type_t;

typedef struct ne_someip_client_runna_get_daemon_client_id
{
    ne_someip_sequence_id_t seq_id;
    pthread_t tid;
}ne_someip_client_runna_get_daemon_client_id_t;

typedef struct ne_someip_client_runna_find_local_service
{
    ne_someip_client_context_t* context;
    ne_someip_find_offer_service_spec_t spec;
    ne_someip_sequence_id_t seq_id;
    pthread_t tid;
}ne_someip_client_runna_find_local_service_t;

typedef struct ne_someip_client_runna_find_service  // start_find_service and stop_find_service
{
    ne_someip_client_context_t* context;
    ne_someip_find_offer_service_spec_t spec;
    const ne_someip_required_service_instance_config_t* inst_config;
    bool operate_type;  // true: start find service; false: stop find service
    pthread_t tid;
}ne_someip_client_runna_find_service_t;

typedef struct ne_someip_client_runna_reg_serv_status
{
    ne_someip_client_context_t* context;
    ne_someip_find_offer_service_spec_t spec;
    bool operate_type;  // true: register; false: unregister
    ne_someip_saved_available_handler_t* handler;  // used to notify service status when register
}ne_someip_client_runna_reg_serv_status_t;

typedef struct ne_someip_client_runna_reg_to_daemon {
    ne_someip_required_service_instance_t* instance;
    bool operate_type;  // true register; false: unregister
}ne_someip_client_runna_reg_to_daemon_t;

// start_subscribe_eventgroup and stop_subscribe_eventgroup
typedef struct ne_someip_client_runna_subscribe_eventgroup
{
    ne_someip_required_service_instance_t* instance;
    ne_someip_eventgroup_id_t eventgroup_id;
    bool operate_type;  // true: start subscribe eventgroup; false: stop subscribe eventgroup
    pthread_t tid;
}ne_someip_client_runna_subscribe_eventgroup_t;

typedef struct ne_someip_client_runna_send_request
{
    ne_someip_required_service_instance_t* instance;
    ne_someip_header_t* header;
    ne_someip_payload_t* payload;
    const void* user_seq;
}ne_someip_client_runna_send_request_t;

typedef struct ne_someip_client_runna_unref_context
{
    ne_someip_client_context_t* context;
    ne_someip_app_context_t* app_context;
    pthread_t tid;
}ne_someip_client_runna_unref_context_t;

typedef struct ne_someip_client_runna_destroy_instance
{
    ne_someip_client_context_t* context;
    ne_someip_required_service_instance_t* req_instance;
    pthread_t tid;
}ne_someip_client_runna_destroy_instance_t;

typedef struct ne_someip_client_runna_sync_wait_timer
{
    ne_someip_client_context_t* context;
    ne_someip_sequence_id_t seq_id;
    pthread_t tid;
    ne_someip_client_runna_timer_type_t type;
}ne_someip_client_runna_sync_wait_timer_t;

void ne_someip_client_runna_get_daemon_client_id_run(ne_someip_client_runna_get_daemon_client_id_t* client_id_info);
void ne_someip_client_runna_get_daemon_client_id_free(ne_someip_client_runna_get_daemon_client_id_t* client_id_info);
void ne_someip_client_runna_find_local_service_run(ne_someip_client_runna_find_local_service_t* find_local_service);
void ne_someip_client_runna_find_local_service_free(ne_someip_client_runna_find_local_service_t* find_local_service);
void ne_someip_client_runna_find_service_run(ne_someip_client_runna_find_service_t* find_service);
void ne_someip_client_runna_find_service_free(ne_someip_client_runna_find_service_t* find_service);
void ne_someip_client_runna_subscribe_eventgroup_run(ne_someip_client_runna_subscribe_eventgroup_t* subscribe_eventgroup);
void ne_someip_client_runna_subscribe_eventgroup_free(ne_someip_client_runna_subscribe_eventgroup_t* subscribe_eventgroup);
void ne_someip_client_runna_send_request_run(ne_someip_client_runna_send_request_t* send_request);
void ne_someip_client_runna_send_request_free(ne_someip_client_runna_send_request_t* send_request);
void ne_someip_client_runna_register_status_to_daemon_run(ne_someip_client_runna_reg_serv_status_t* reg_status);
void ne_someip_client_runna_register_status_to_daemon_free(ne_someip_client_runna_reg_serv_status_t* reg_status);
void ne_someip_client_runna_reg_event_to_daemon_run(ne_someip_client_runna_reg_to_daemon_t* reg_data);
void ne_someip_client_runna_reg_event_to_daemon_free(ne_someip_client_runna_reg_to_daemon_t* reg_data);
void ne_someip_client_runna_reg_resp_to_daemon_run(ne_someip_client_runna_reg_to_daemon_t* reg_data);
void ne_someip_client_runna_reg_resp_to_daemon_free(ne_someip_client_runna_reg_to_daemon_t* reg_data);
void ne_someip_client_runna_destroy_instance_run(ne_someip_client_runna_destroy_instance_t* destroy_instance);
void ne_someip_client_runna_destroy_instance_free(ne_someip_client_runna_destroy_instance_t* destroy_instance);

void ne_someip_client_runna_start_timer_run(ne_someip_looper_timer_t* timer);
void ne_someip_client_runna_start_timer_free(ne_someip_looper_timer_t* timer);
void ne_someip_client_runna_stop_timer_run(ne_someip_looper_timer_t* timer);
void ne_someip_client_runna_stop_timer_free(ne_someip_looper_timer_t* timer);
void ne_someip_client_runna_sync_wait_timeout_run(ne_someip_client_runna_sync_wait_timer_t* timer_info);
void ne_someip_client_runna_sync_wait_timeout_free(ne_someip_client_runna_sync_wait_timer_t* timer_info);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_RUNNABLE_FUNC_H
/* EOF */
