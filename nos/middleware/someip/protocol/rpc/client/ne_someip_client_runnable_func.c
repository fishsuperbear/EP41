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
#include "ne_someip_client_runnable_func.h"
#include "ne_someip_required_service_instance.h"
#include "ne_someip_log.h"
#include "ne_someip_app_context.h"
#include "ne_someip_common_service_instance.h"
#include "ne_someip_required_event_behaviour.h"
#include "ne_someip_required_method_behaviour.h"
#include "ne_someip_required_network_connect_behaviour.h"
// static ne_someip_list_t* ne_someip_client_runnable_find_service_instance(ne_someip_map_t* required_service_instance_map,
//     ne_someip_service_instance_spec_t service_instance);

void ne_someip_client_runna_get_daemon_client_id_run(ne_someip_client_runna_get_daemon_client_id_t* client_id_info)
{
    if (NULL == client_id_info) {
        ne_someip_log_error("impossible case!");
        return;
    }

    ne_someip_app_context_get_daemon_client_id(client_id_info->seq_id, client_id_info->tid);
}

void ne_someip_client_runna_get_daemon_client_id_free(ne_someip_client_runna_get_daemon_client_id_t* client_id_info)
{
    if (NULL != client_id_info) {
        free(client_id_info);
        client_id_info = NULL;
    }
}

void ne_someip_client_runna_find_local_service_run(ne_someip_client_runna_find_local_service_t* find_local_service)
{
    if (NULL == find_local_service || NULL == find_local_service->context) {
        ne_someip_log_error("impossible case!");
        return;
    }

    ne_someip_comm_serv_inst_query_offered_instances(find_local_service->context->common_instance,
        &(find_local_service->spec), find_local_service->seq_id, find_local_service->tid);
}

void ne_someip_client_runna_find_local_service_free(ne_someip_client_runna_find_local_service_t* find_local_service)
{
    if (NULL != find_local_service) {
        // ne_someip_client_context_unref(find_local_service->context);
        free(find_local_service);
        find_local_service = NULL;
    }
}

void ne_someip_client_runna_find_service_run(ne_someip_client_runna_find_service_t* find_service)
{
    if (NULL == find_service || NULL == find_service->context || NULL == find_service->context->common_instance) {
        ne_someip_log_error("impossible case!");
        return;
    }

    ne_someip_log_debug("call common instances interface find_service");
    // call required instances interface
    ne_someip_error_code_t ret = ne_someip_error_code_unknown;
    if (find_service->operate_type) {  // start find service
        ret = ne_someip_comm_serv_inst_start_find_service(find_service->context->common_instance, &(find_service->spec),
            find_service->context->io_looper, find_service->inst_config);
    }
    else {  // stop find service
        bool notify_obj = true;
        ret = ne_someip_comm_serv_inst_stop_find_service(find_service->context->common_instance, &(find_service->spec),
            find_service->context->io_looper, &notify_obj, find_service->tid);
        if (notify_obj) {
            ne_someip_sync_wait_obj_t* wait_obj = ne_someip_app_context_find_wait_obj(find_service->tid);
            if (NULL != wait_obj) {
                // 唤醒同步等待
                ne_someip_sync_wait_obj_notify(wait_obj);
            }
        }
    }

    if (ne_someip_error_code_ok != ret) {
        ne_someip_comm_serv_inst_notify_find_status(find_service->context->common_instance, &(find_service->spec),
            ne_someip_find_status_stopped, ret);
    }
}

void ne_someip_client_runna_find_service_free(ne_someip_client_runna_find_service_t* find_service)
{
    if (NULL != find_service) {
        // ne_someip_client_context_unref(find_service->context);
        find_service->inst_config = NULL;
        free(find_service);
        find_service = NULL;
    }
}

void ne_someip_client_runna_subscribe_eventgroup_run(ne_someip_client_runna_subscribe_eventgroup_t* subscribe_eventgroup)
{
    if (NULL == subscribe_eventgroup || NULL == subscribe_eventgroup->instance) {
        ne_someip_log_error("impossible case!");
        return;
    }

    ne_someip_log_debug("call required instances interface subscribe_eventgroup");
    // call required instances interface
    ne_someip_error_code_t ret = ne_someip_error_code_unknown;
    if (subscribe_eventgroup->operate_type) {  // start subscribe eventgroup
        ret = ne_someip_req_serv_inst_start_subscribe_eventgroup(subscribe_eventgroup->instance,
            subscribe_eventgroup->eventgroup_id);
    }
    else {  // stop subscribe eventgroup
        bool notify_obj = true;
        ret = ne_someip_req_serv_inst_stop_subscribe_eventgroup(subscribe_eventgroup->instance,
            subscribe_eventgroup->eventgroup_id, &notify_obj, subscribe_eventgroup->tid);
        if (notify_obj) {
            ne_someip_sync_wait_obj_t* wait_obj = ne_someip_app_context_find_wait_obj(subscribe_eventgroup->tid);
            if (NULL != wait_obj) {
                // 唤醒同步等待
                ne_someip_sync_wait_obj_notify(wait_obj);
            }
        }
    }

    if (ne_someip_error_code_ok != ret) {
        ne_someip_req_serv_inst_notify_sub_status(subscribe_eventgroup->instance, subscribe_eventgroup->eventgroup_id,
            ne_someip_subscribe_status_failed, ret);
    }
}

void ne_someip_client_runna_subscribe_eventgroup_free(ne_someip_client_runna_subscribe_eventgroup_t* subscribe_eventgroup)
{
    if (NULL != subscribe_eventgroup) {
        ne_someip_required_service_instance_unref(subscribe_eventgroup->instance);
        free(subscribe_eventgroup);
        subscribe_eventgroup = NULL;
    }
}

// 不实现同步返回，send_request只能是异步调用
void ne_someip_client_runna_send_request_run(ne_someip_client_runna_send_request_t* send_request)
{
    if (NULL == send_request || NULL == send_request->instance) {
        ne_someip_log_error("impossible case!");
        return;
    }

    ne_someip_log_debug("call required instances interface send_request");
    // call required instances interface
    ne_someip_error_code_t ret = ne_someip_error_code_unknown;
    ret = ne_someip_req_serv_inst_send_request(send_request->instance, send_request->user_seq,
        send_request->header, send_request->payload);

    if (ne_someip_error_code_ok != ret) {
        ne_someip_method_id_t method_id = send_request->header->method_id;
        ne_someip_req_serv_inst_notify_send_status(send_request->instance, send_request->user_seq, method_id,
            ret);
    }
}

void ne_someip_client_runna_send_request_free(ne_someip_client_runna_send_request_t* send_request)
{
    if (NULL != send_request) {
        ne_someip_required_service_instance_unref(send_request->instance);
        if (NULL != send_request->header) {
            free(send_request->header);
            send_request->header = NULL;
        }
        ne_someip_payload_unref(send_request->payload);
        send_request->user_seq = NULL;
        free(send_request);
        send_request = NULL;
    }
}

void ne_someip_client_runna_register_status_to_daemon_run(ne_someip_client_runna_reg_serv_status_t* reg_status)
{
    if (NULL == reg_status || NULL == reg_status->context || NULL == reg_status->context->common_instance) {
        ne_someip_log_error("reg_status or reg_status->context or reg_status->context->common_instance is NULL.");
        return;
    }

    ne_someip_log_debug("call register/unregister to daemon interface");
    // call app_context interface
    if (reg_status->operate_type) {  // register
        // notify service status to upper when service_status was konwm
        ne_someip_comm_serv_inst_notify_local_avail_status(reg_status->context->common_instance, &(reg_status->spec),
            reg_status->handler);

        bool is_reg_ok = ne_someip_app_context_reg_client_id_for_avail_status(&(reg_status->spec),
            reg_status->context->common_instance->client_id, (void*)(reg_status->context->common_instance));
        if (!is_reg_ok) {
            ne_someip_log_error("register common_service to app_context failed.");
            return;
        }
    }
    else {  // unregister
        ne_someip_app_context_unreg_client_id_for_avail_status(&(reg_status->spec),
            reg_status->context->common_instance->client_id, (void*)(reg_status->context->common_instance));
    }
}

void ne_someip_client_runna_register_status_to_daemon_free(ne_someip_client_runna_reg_serv_status_t* reg_status)
{
    if (NULL != reg_status) {
        // ne_someip_client_context_unref(reg_status->context);
        if (NULL != reg_status->handler) {
            free(reg_status->handler);
            reg_status->handler = NULL;
        }
        free(reg_status);
        reg_status = NULL;
    }
}

void ne_someip_client_runna_reg_event_to_daemon_run(ne_someip_client_runna_reg_to_daemon_t* reg_data) {
    if (NULL == reg_data || NULL == reg_data->instance) {
        ne_someip_log_error("reg_data is NULL.");
        return;
    }

    if (ne_someip_network_states_up !=
        ne_someip_req_network_connect_behav_net_status_get(reg_data->instance->network_connect_behaviour)) {
        ne_someip_log_info("wait network up.");
        return;
    }

    if (reg_data->operate_type) {  // register
        ne_someip_req_event_behaviour_reg_event_handler_to_daemon(reg_data->instance);
    } else {  // unregister
        ne_someip_req_event_behaviour_unreg_event_handler_to_daemon(reg_data->instance);
    }
}

void ne_someip_client_runna_reg_event_to_daemon_free(ne_someip_client_runna_reg_to_daemon_t* reg_data) {
    if (NULL != reg_data) {
        ne_someip_required_service_instance_unref(reg_data->instance);
        free(reg_data);
    }
}

void ne_someip_client_runna_reg_resp_to_daemon_run(ne_someip_client_runna_reg_to_daemon_t* reg_data)
{
    if (NULL == reg_data || NULL == reg_data->instance) {
        ne_someip_log_error("reg_data is NULL.");
        return;
    }

    if (ne_someip_network_states_up !=
        ne_someip_req_network_connect_behav_net_status_get(reg_data->instance->network_connect_behaviour)) {
        ne_someip_log_info("wait network up.");
        return;
    }

    if (reg_data->operate_type) {  // register
        ne_someip_req_method_behaviour_reg_response_handler_to_daemon(reg_data->instance);
    } else {  // unregister
        ne_someip_req_method_behaviour_unreg_response_handler_to_daemon(reg_data->instance);
    }
}

void ne_someip_client_runna_reg_resp_to_daemon_free(ne_someip_client_runna_reg_to_daemon_t* reg_data)
{
    if (NULL != reg_data) {
        ne_someip_required_service_instance_unref(reg_data->instance);
        free(reg_data);
    }
}

void ne_someip_client_runna_destroy_instance_run(ne_someip_client_runna_destroy_instance_t* destroy_instance)
{
    if (NULL == destroy_instance) {
        ne_someip_log_error("destroy_instance is NULL.");
        return;
    }

    if (NULL != destroy_instance->context) {
        ne_someip_service_instance_spec_t spec;
        spec.service_id = ne_someip_req_serv_inst_get_service_id(destroy_instance->req_instance);
        spec.instance_id = ne_someip_req_serv_inst_get_instance_id(destroy_instance->req_instance);
        spec.major_version = ne_someip_req_serv_inst_get_major_version(destroy_instance->req_instance);
        ne_someip_comm_serv_inst_unre_requir_service(destroy_instance->context->common_instance, &spec,
            destroy_instance->req_instance);
    }

    ne_someip_required_service_instance_unref(destroy_instance->req_instance);

    ne_someip_sync_wait_obj_t* wait_obj = ne_someip_app_context_find_wait_obj(destroy_instance->tid);
    if (NULL != wait_obj) {
        // 唤醒同步等待
        ne_someip_sync_wait_obj_notify(wait_obj);
    }
}

void ne_someip_client_runna_destroy_instance_free(ne_someip_client_runna_destroy_instance_t* destroy_instance)
{
    if (NULL != destroy_instance) {
        free(destroy_instance);
    }
}

void ne_someip_client_runna_start_timer_run(ne_someip_looper_timer_t* timer)
{
    if (NULL != timer) {
        ne_someip_looper_timer_start(timer, NE_LOOPER_TIMER_TYPE_INTERVAL_ONE_SHOT, NE_SOMEIP_SYNC_TIMER_VALUE);
    }
}

void ne_someip_client_runna_start_timer_free(ne_someip_looper_timer_t* timer)
{
    if (NULL != timer) {
        ne_someip_looper_timer_unref(timer);
    }
}

void ne_someip_client_runna_stop_timer_run(ne_someip_looper_timer_t* timer)
{
    if (NULL != timer) {
        ne_someip_looper_timer_stop(timer);
    }
}

void ne_someip_client_runna_stop_timer_free(ne_someip_looper_timer_t* timer)
{
    if (NULL != timer) {
        ne_someip_looper_timer_unref(timer);
    }
}

void ne_someip_client_runna_sync_wait_timeout_run(ne_someip_client_runna_sync_wait_timer_t* timer_info)
{
    if (NULL == timer_info) {
        ne_someip_log_error("timer_info is NULL.");
        return;
    }
    ne_someip_log_debug("timeout, seq_id is %d", timer_info->seq_id);

    switch (timer_info->type) {
        case ne_someip_client_runna_timer_type_get_client_id:
            {
                ne_someip_app_conetxt_add_client_id_info(timer_info->seq_id, 0, ne_someip_error_code_sync_call_timeout);
                break;
            }
        case ne_someip_client_runna_timer_type_get_local_services:
            {
                if (NULL == timer_info->context) {
                    ne_someip_log_error("timer_info->context is NULL.");
                    break;
                }
                ne_someip_comm_serv_inst_add_find_offer_serv(timer_info->context->common_instance, timer_info->seq_id, NULL,
                    ne_someip_error_code_sync_call_timeout);
                break;
            }
        case ne_someip_client_runna_timer_type_stop_find:
        case ne_someip_client_runna_timer_type_stop_subscribe:
        default :
            break;
    }

    ne_someip_sync_wait_obj_t* wait_obj = ne_someip_app_context_find_wait_obj(timer_info->tid);
    if (NULL != wait_obj) {
        // timerout，唤醒同步等待
        ne_someip_sync_wait_obj_notify(wait_obj);
    }
}

void ne_someip_client_runna_sync_wait_timeout_free(ne_someip_client_runna_sync_wait_timer_t* timer_info)
{
    if (NULL == timer_info) {
        ne_someip_log_error("timer_info is NULL.");
        return;
    }

    if (NULL != timer_info->context) {
        // ne_someip_client_context_unref(timer_info->context);
        timer_info->context = NULL;
    }
    free(timer_info);
    timer_info = NULL;
}

// static ne_someip_list_t* ne_someip_client_runnable_find_service_instance(ne_someip_map_t* required_service_instance_map,
//     ne_someip_service_instance_spec_t service_instance)
// {
//     ne_someip_list_t* temp_ser_instance = NULL;

//     if (NULL == required_service_instance_map) {
//         ne_someip_log_error("required_service_instance_map is NULL.");
//         return temp_ser_instance;
//     }

//     if (NE_SOMEIP_ANY_SERVICE != service_instance.service_id &&
//         NE_SOMEIP_ANY_INSTANCE != service_instance.instance_id &&
//         NE_SOMEIP_ANY_MAJOR != service_instance.major_version) {
//         uint32_t hash_return = 0;
//         ne_someip_required_service_instance_t* req_ser_ins =
//             (ne_someip_required_service_instance_t*)ne_someip_map_find(required_service_instance_map,
//             &service_instance, &hash_return);
//         if (NULL == req_ser_ins) {
//             ne_someip_log_error("required_service_instance_map find fail.");
//             ne_someip_log_error("service_id [0x%x], instance_id [0x%x], major_version [0x%x]", service_instance.service_id,
//                 service_instance.instance_id, service_instance.major_version);
//             return temp_ser_instance;
//         }

//         temp_ser_instance = ne_someip_list_append(temp_ser_instance, req_ser_ins);
//     }
//     else {
//         ne_someip_map_iter_t* it = ne_someip_map_iter_new(required_service_instance_map);
//         ne_someip_service_instance_spec_t* instance_key = NULL;
//         ne_someip_required_service_instance_t* instance_value = NULL;
//         while (ne_someip_map_iter_next(it, (void**)(&instance_key), (void**)(&instance_value))) {
//             if (NULL == instance_key || NULL == instance_value) {
//                 ne_someip_log_error("impossible case.");
//                 continue;
//             }

//             if ((instance_key->service_id == service_instance.service_id || NE_SOMEIP_ANY_SERVICE == service_instance.service_id) &&
//                 (instance_key->instance_id == service_instance.instance_id || NE_SOMEIP_ANY_INSTANCE == service_instance.instance_id) &&
//                 (instance_key->major_version == service_instance.major_version || NE_SOMEIP_ANY_MAJOR == service_instance.major_version)) {
//                 temp_ser_instance = ne_someip_list_append(temp_ser_instance, instance_value);
//             }
//         }
//         ne_someip_map_iter_destroy(it);
//     }

//     return temp_ser_instance;
// }

// 检查eventgroup id是否正确
// {
//     ne_someip_required_service_instance_t* tmp_ins = (ne_someip_required_service_instance_t*)(iter->data);
//             if (NULL == tmp_ins->config || NULL == tmp_ins->config->eventgroups_config_array ||
//                 NULL == tmp_ins->config->service_config) {
//                 ne_someip_log_error("Config info that was saved in required_service_instance_map is wrong.");
//                 continue;
//             }

//             ne_someip_log_debug("required instance: service_id [0x%x], instance_id [0x%x], major_version [0x%x]",
//                 tmp_ins->config->service_config->service_id,
//                 tmp_ins->config->instance_id, tmp_ins->config->service_config->major_version);

//             // check if the eventgroup_id appears in the required instances config
//             bool is_eventgroup_id_in_config = false;
//             for (int i = 0; i < tmp_ins->config->eventgroups_config_array->eventgroup_array_num; ++i) {
//                 ne_someip_required_eventgroup_config_t* req_eventgroup =
//                     tmp_ins->config->eventgroups_config_array->eventgroup_config_array[i];
//                 if (NULL == req_eventgroup || NULL == req_eventgroup->eventgroup_config) {
//                     continue;
//                 }

//                 if (subscribe_eventgroup->eventgroup_id == req_eventgroup->eventgroup_config->eventgroup_id) {
//                     ne_someip_log_debug("subscribe_eventgroup_id [0x%x] is in required instances config",
//                         subscribe_eventgroup->eventgroup_id);
//                     is_eventgroup_id_in_config = true;
//                     break;
//                 }
//             }

//             if (!is_eventgroup_id_in_config) {
//                 continue;
//             }
// }

// 检查method id 是否正确
// {
//     ne_someip_required_service_instance_t* tmp_ins = (ne_someip_required_service_instance_t*)(iter->data);
//             if (NULL == tmp_ins->config || NULL == tmp_ins->config->method_config_array ||
//                 NULL == tmp_ins->config->service_config) {
//                 ne_someip_log_error("Config info that was saved in required_service_instance_map is wrong.");
//                 continue;
//             }

//             ne_someip_log_debug("required instance: service_id [0x%x], instance_id [0x%x], major_version [0x%x]",
//                 tmp_ins->config->service_config->service_id,
//                 tmp_ins->config->instance_id, tmp_ins->config->service_config->major_version);

//             // check if the method id appears in the required instances config
//             bool is_method_id_in_config = false;
//             for (int i = 0; i < tmp_ins->config->method_config_array->method_array_num; ++i) {
//                 ne_someip_required_provided_method_config_t* req_method =
//                     tmp_ins->config->method_config_array->method_config_array[i];
//                 if (NULL == req_method || NULL == req_method->method_config) {
//                     continue;
//                 }

//                 if (send_request->message->method_id == req_method->method_config->method_id) {
//                     ne_someip_log_debug("method_id [0x%x] is in required instances config", send_request->message->method_id);
//                     is_method_id_in_config = true;
//                     break;
//                 }
//             }

//             if (!is_method_id_in_config) {
//                 continue;
//             }

// }