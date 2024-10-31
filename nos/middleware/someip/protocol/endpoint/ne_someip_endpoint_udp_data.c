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
#include "ne_someip_endpoint_udp_data.h"
#include "ne_someip_app_context.h"
#include "ne_someip_endpoint_runnable_func.h"
#include "ne_someip_endpoint_core.h"
#include "ne_someip_endpoint_tool.h"
#include "ne_someip_tp.h"
#include "ne_someip_log.h"
#include <unistd.h>
#include <assert.h>
/********************internal func**********************/
static void ne_someip_endpoint_udp_data_t_free(ne_someip_endpoint_udp_data_t* endpoint);
/********************internal func**********************/

NEOBJECT_FUNCTION(ne_someip_endpoint_udp_data_t);

ne_someip_endpoint_udp_data_t* ne_someip_endpoint_udp_data_create(
    ne_someip_endpoint_type_t type, ne_someip_endpoint_net_addr_t* local_addr, ne_someip_looper_t* looper,
    bool is_need_switch_thread, const ne_someip_ssl_key_info_t* key_info)
{
    if ((NULL == looper) || (NULL == local_addr)) {
        ne_someip_log_error("io looper or local_addr is NULL");
        return NULL;
    }

    ne_someip_endpoint_udp_data_t* endpoint = (ne_someip_endpoint_udp_data_t*)malloc(sizeof(ne_someip_endpoint_udp_data_t));
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint malloc error.");
        return NULL;
    }
    memset(endpoint, 0, sizeof(*endpoint));

    endpoint->base.endpoint_type = type;
    // endpoint->is_need_switch_thread = is_need_switch_thread;
    if (looper != ne_someip_looper_self()) {
        endpoint->is_need_switch_thread = true;
    } else {
        endpoint->is_need_switch_thread = false;
    }

    ne_someip_endpoint_net_addr_t* tmp_local_addr =
        (ne_someip_endpoint_net_addr_t*)malloc(sizeof(ne_someip_endpoint_net_addr_t));
    if (NULL == tmp_local_addr) {
        ne_someip_log_error("malloc ne_someip_endpoint_net_addr_t error.");
        ne_someip_endpoint_udp_data_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    memcpy(tmp_local_addr, local_addr, sizeof(*local_addr));
    ne_someip_log_debug("tmp_local_addr->ip_addr is [%d],  tmp_local_addr->port is [%d]", tmp_local_addr->ip_addr,
        tmp_local_addr->port);
    endpoint->local_addr = tmp_local_addr;
    endpoint->endpoint_state = ne_someip_endpoint_state_created;

    ne_someip_looper_t* looper_ref = ne_someip_looper_ref(looper);
    if (NULL == looper_ref) {
        ne_someip_log_error("ne_someip_looper_ref return NULL.");
        ne_someip_endpoint_udp_data_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    endpoint->endpoint_io_looper = looper_ref;

    ne_someip_endpoint_core_t* core = ne_someip_ep_core_new(looper_ref, type);
    if (NULL == core) {
        ne_someip_log_error("ne_someip_ep_core_new return NULL.");
        ne_someip_endpoint_udp_data_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    endpoint->ep_core = core;

    ne_someip_endpoint_transmit_info_t* transmit_info = (ne_someip_endpoint_transmit_info_t*)malloc(sizeof(ne_someip_endpoint_transmit_info_t));
    if (NULL == transmit_info) {
        ne_someip_log_error("transmit_info malloc error.");
        ne_someip_endpoint_udp_data_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    memset(transmit_info, 0, sizeof(*transmit_info));
    endpoint->udp_transmit = transmit_info;
    ne_someip_transmit_t* transmit = NULL;
    if (NULL == key_info) {
        endpoint->is_tls_used = false;
        transmit = ne_someip_transmit_new(NE_SOMEIP_TRANSMIT_TYPE_UDP, local_addr, false);
    } else {
        endpoint->is_tls_used = true;
        transmit = ne_someip_transmit_new(NE_SOMEIP_TRANSMIT_TYPE_DTLS, local_addr, false);
    }
    if (NULL == transmit) {
        ne_someip_log_error("ne_someip_transmit_new return NULL.");
        ne_someip_endpoint_udp_data_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    endpoint->udp_transmit->transmit = transmit;
    endpoint->udp_transmit->transmit_state = ne_someip_endpoint_transmit_state_stopped;
    endpoint->key_info = key_info;

    endpoint->tls_status_map = ne_someip_map_new(ne_someip_ep_tool_net_key_hash_func, ne_someip_ep_tool_net_addr_compare,
        ne_someip_ep_tool_net_addr_free, free);
    if (NULL == endpoint->tls_status_map) {
        ne_someip_log_error("ne_someip_map_new return NULL.");
        ne_someip_endpoint_udp_data_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }

    endpoint->work_looper = ne_someip_looper_ref(ne_someip_looper_self());
    endpoint->callback = NULL;
    endpoint->req_instance_list = ne_someip_list_create();
    if (NULL == endpoint->req_instance_list) {
        ne_someip_log_error("ne_someip_list_create return NULL.");
        ne_someip_endpoint_udp_data_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    endpoint->group_addr_list = NULL;
    endpoint->sync_seq_id = 0;
    endpoint->sync_seq_res_code_map =  ne_someip_map_new(ne_someip_map_int32_hash_func, ne_someip_ep_tool_int32_compare,
        ne_someip_ep_tool_int32_free, ne_someip_ep_tool_error_code_free);
    if (NULL == endpoint->sync_seq_res_code_map) {
        ne_someip_log_error("ne_someip_map_new return NULL.");
        ne_someip_endpoint_udp_data_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    endpoint->seq_id_sync = ne_someip_sync_obj_create();
    endpoint->res_code_map_sync = ne_someip_sync_obj_create();

    endpoint->tp_ctx = ne_someip_tp_init();
    if (NULL == endpoint->tp_ctx) {
        ne_someip_log_error("ne_someip_tp_init return NULL.");
        ne_someip_endpoint_udp_data_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }

    endpoint->is_transmit_stop = false;
    endpoint->is_multicast_endpoint = false;

    ne_someip_endpoint_udp_data_t_ref_count_init(endpoint);

    ne_someip_log_info("ne_someip_endpoint_udp_data_create create object finish.");

    ne_someip_error_code_t res_code = ne_someip_ep_core_start(endpoint, key_info);
    if (ne_someip_error_code_ok != res_code) {
        ne_someip_log_error("ne_someip_endpoint_start fail.");
        ne_someip_endpoint_udp_data_unref(endpoint);
        return NULL;
    }

    return endpoint;
}

ne_someip_endpoint_udp_data_t* ne_someip_endpoint_udp_data_ref(ne_someip_endpoint_udp_data_t* endpoint)
{
    ne_someip_endpoint_udp_data_t* ref = NULL;
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return ref;
    }

    ref = ne_someip_endpoint_udp_data_t_ref(endpoint);
    return ref;
}

void ne_someip_endpoint_udp_data_unref(ne_someip_endpoint_udp_data_t* endpoint)
{
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return;
    }

    ne_someip_endpoint_udp_data_t_unref(endpoint);
}

ne_someip_error_code_t ne_someip_endpoint_udp_data_stop(ne_someip_endpoint_udp_data_t* endpoint)
{

    ne_someip_error_code_t res_code = ne_someip_error_code_failed;
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL;");
        return res_code;
    }

    if (endpoint->is_transmit_stop) {
        res_code = ne_someip_error_code_ok;
        return res_code;
    }

    if (endpoint->endpoint_io_looper != ne_someip_looper_self()) {  // TODO: add lock?
        ne_someip_log_debug("need post to io thread.");
        ne_someip_thread_t* thread = ne_someip_thread_self();
        if (NULL == thread) {
            ne_someip_log_error("get self thread fail.");
            return res_code;
        }
        int32_t sync_seq_id = ne_someip_endpoint_get_sync_seq_id(endpoint);
        ne_someip_looper_runnable_t* runnable = ne_someip_ep_create_core_stop_runnable(endpoint, thread, sync_seq_id);
        if (NULL == runnable) {
            ne_someip_log_error("ne_someip_ep_create_core_stop_runnable return error.");
            return res_code;
        }

        ne_someip_endpoint_set_sync_res_code(endpoint, sync_seq_id, res_code);

        int res = ne_someip_looper_post(endpoint->endpoint_io_looper, runnable);
        if (0 != res) {
            ne_someip_endpoint_get_sync_res_code(endpoint, sync_seq_id);  // delete the saved data of this seq_id
            ne_someip_log_error("ne_someip_looper_post fail.");
            ne_someip_ep_destroy_runnable(runnable);
            return res_code;
        }

        ne_someip_thread_wait(thread);  // 线程同步wait，等呆notify
        res_code = ne_someip_endpoint_get_sync_res_code(endpoint, sync_seq_id);
        ne_someip_log_debug("seq_id [%d], core_stop res is [%d]", sync_seq_id, res_code);
    }
    else {
        res_code = ne_someip_ep_core_stop(endpoint);
        if (ne_someip_error_code_failed == res_code) {
            ne_someip_log_error("ne_someip_endpoint_stop return fail.");
        }
    }

    endpoint->is_transmit_stop = true;
    return res_code;
}

static void ne_someip_endpoint_udp_data_t_free(ne_someip_endpoint_udp_data_t* endpoint)
{
    ne_someip_error_code_t res_code = ne_someip_error_code_failed;
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return;
    }

    res_code = ne_someip_endpoint_udp_data_stop(endpoint);
    if (ne_someip_error_code_ok != res_code) {
        ne_someip_log_error("ne_someip_endpoint_udp_data_stop return fail.");
    }

    bool del_ret = ne_someip_app_context_delete_udp_endpoint(endpoint->local_addr);
    if (!del_ret) {
        ne_someip_log_error("delete udp endpoint in app_context error.");
    }

    ne_someip_endpoint_udp_data_t_ref_count_deinit(endpoint);

    if (NULL != endpoint->local_addr) {
        free(endpoint->local_addr);
        endpoint->local_addr = NULL;
    }
    if (NULL != endpoint->ep_core) {
        ne_someip_ep_core_unref(endpoint->ep_core);
        endpoint->ep_core = NULL;
    }
    if (NULL != endpoint->udp_transmit) {
        if (NULL != endpoint->udp_transmit->transmit) {
            ne_someip_transmit_unref(endpoint->udp_transmit->transmit);
            endpoint->udp_transmit->transmit = NULL;
        }
        free(endpoint->udp_transmit);
        endpoint->udp_transmit = NULL;
    }
    endpoint->key_info = NULL;
    if (NULL != endpoint->tls_status_map) {
        ne_someip_map_unref(endpoint->tls_status_map);
        endpoint->tls_status_map = NULL;
    }
    if (NULL != endpoint->endpoint_io_looper) {
        ne_someip_looper_unref(endpoint->endpoint_io_looper);
    }
    if (NULL != endpoint->work_looper) {
        ne_someip_looper_unref(endpoint->work_looper);
    }
    if (NULL != endpoint->callback) {
        if (NULL != endpoint->callback->user_data && NULL != endpoint->callback->free) {
            endpoint->callback->free(endpoint->callback->user_data);
            endpoint->callback->user_data = NULL;
        }
        free(endpoint->callback);
        endpoint->callback = NULL;
    }
    if (NULL != endpoint->req_instance_list) {
        ne_someip_list_destroy(endpoint->req_instance_list, NULL);
        endpoint->req_instance_list = NULL;
    }
    if (NULL != endpoint->group_addr_list) {
        ne_someip_list_destroy(endpoint->group_addr_list, ne_someip_ep_tool_ep_group_addr_free);
        endpoint->group_addr_list = NULL;
    }
    if (NULL != endpoint->sync_seq_res_code_map) {
        ne_someip_map_unref(endpoint->sync_seq_res_code_map);
        endpoint->sync_seq_res_code_map = NULL;
    }
    if (NULL != endpoint->seq_id_sync) {
        ne_someip_sync_obj_destroy(endpoint->seq_id_sync);
        endpoint->seq_id_sync = NULL;
    }
    if (NULL != endpoint->res_code_map_sync) {
        ne_someip_sync_obj_destroy(endpoint->res_code_map_sync);
        endpoint->res_code_map_sync = NULL;
    }
    if (NULL != endpoint->tp_ctx) {
        ne_someip_tp_deinit(endpoint->tp_ctx);
        endpoint->tp_ctx = NULL;
    }

    free(endpoint);
    endpoint = NULL;

    ne_someip_log_info("ne_someip_endpoint_udp_data_t_free finish.");
}

ne_someip_error_code_t ne_someip_endpoint_udp_data_join_group(
    ne_someip_endpoint_udp_data_t* endpoint, ne_someip_endpoint_net_addr_t* interface_addr)
{
    ne_someip_error_code_t res_code = ne_someip_error_code_failed;

    if (NULL == endpoint || NULL == interface_addr) {
        ne_someip_log_error("endpoint or interface_addr is NULL.");
        return res_code;
    }

    ne_someip_log_info("join group, interface_addr->ip_addr is [%d],  interface_addr->port is [%d]", interface_addr->ip_addr,
        interface_addr->port);

    endpoint->is_multicast_endpoint = true;

    if (endpoint->is_need_switch_thread) {  // TODO: add lock?
        ne_someip_log_debug("need post to io thread.");
        ne_someip_thread_t* thread = ne_someip_thread_self();
        if (NULL == thread) {
            ne_someip_log_error("get self thread fail.");
            return res_code;
        }
        int32_t sync_seq_id = ne_someip_endpoint_get_sync_seq_id(endpoint);
        ne_someip_looper_runnable_t* runnable = ne_someip_ep_create_join_group_runnable(thread, endpoint, interface_addr, sync_seq_id);
        if (NULL == runnable) {
            ne_someip_log_error("ne_someip_ep_create_create_link_runnable return error.");
            return res_code;
        }

        ne_someip_endpoint_set_sync_res_code(endpoint, sync_seq_id, res_code);

        int res = ne_someip_looper_post(endpoint->endpoint_io_looper, runnable);
        if (0 != res) {
            ne_someip_endpoint_get_sync_res_code(endpoint, sync_seq_id);  // delete the saved data of this seq_id
            ne_someip_log_error("ne_someip_looper_post fail.");
            ne_someip_ep_destroy_runnable(runnable);
            return res_code;
        }

        ne_someip_thread_wait(thread);  // 线程同步wait，等呆notify
        res_code = ne_someip_endpoint_get_sync_res_code(endpoint, sync_seq_id);
        ne_someip_log_debug("seq_id [%d], udp join group sync res is [%d]", sync_seq_id, res_code);
    }
    else {
        res_code = ne_someip_endpoint_join_group(endpoint, interface_addr);
        ne_someip_log_debug("udp join group res is [%d]", res_code);
    }

    return res_code;
}

ne_someip_error_code_t ne_someip_endpoint_udp_data_leave_group(
    ne_someip_endpoint_udp_data_t* endpoint, ne_someip_endpoint_net_addr_t* interface_addr)
{
    ne_someip_error_code_t res_code = ne_someip_error_code_failed;

    if (NULL == endpoint || NULL == interface_addr) {
        ne_someip_log_error("endpoint or interface_addr is NULL.");
        return res_code;
    }

    ne_someip_log_info("join group, interface_addr->ip_addr is [%d],  interface_addr->port is [%d]", interface_addr->ip_addr,
        interface_addr->port);

    if (endpoint->is_need_switch_thread) {  // TODO: add lock?
        ne_someip_log_debug("need post to io thread.");
        ne_someip_thread_t* thread = ne_someip_thread_self();
        if (NULL == thread) {
            ne_someip_log_error("get self thread fail.");
            return res_code;
        }
        int32_t sync_seq_id = ne_someip_endpoint_get_sync_seq_id(endpoint);
        ne_someip_looper_runnable_t* runnable = ne_someip_ep_create_leave_group_runnable(thread, endpoint, interface_addr, sync_seq_id);
        if (NULL == runnable) {
            ne_someip_log_error("ne_someip_ep_create_create_link_runnable return error.");
            return res_code;
        }

        ne_someip_endpoint_set_sync_res_code(endpoint, sync_seq_id, res_code);

        int res = ne_someip_looper_post(endpoint->endpoint_io_looper, runnable);
        if (0 != res) {
            ne_someip_endpoint_get_sync_res_code(endpoint, sync_seq_id);  // delete the saved data of this seq_id
            ne_someip_log_error("ne_someip_looper_post fail.");
            ne_someip_ep_destroy_runnable(runnable);
            return res_code;
        }

        ne_someip_thread_wait(thread);  // 线程同步wait，等呆notify
        res_code = ne_someip_endpoint_get_sync_res_code(endpoint, sync_seq_id);
        ne_someip_log_debug("seq_id [%d], udp leave group sync res is [%d]", sync_seq_id, res_code);
    }
    else {
        res_code = ne_someip_endpoint_leave_group(endpoint, interface_addr);
        ne_someip_log_debug("udp leave group res is [%d]", res_code);
    }

    return res_code;
}

ne_someip_error_code_t ne_someip_endpoint_udp_data_send_async(ne_someip_endpoint_udp_data_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer, ne_someip_endpoint_net_addr_t* peer_addr,
    ne_someip_endpoint_send_policy_t* send_policy, const void* seq_data)
{
    ne_someip_error_code_t res_code = ne_someip_error_code_failed;
    if (NULL == endpoint || NULL == endpoint->endpoint_io_looper || NULL == trans_buffer ||
        NULL == peer_addr || NULL == send_policy) {  // TODO: is endpoint_io_looper need add lock?
        ne_someip_log_error("endpoint or endpoint_io_looper or trans_buffer or peer_addr or send_policy is NULL.");
        return res_code;
    }

    ne_someip_log_info("send data use udp endpoint, peer_addr ip [%d], port [%d]", peer_addr->ip_addr,
        peer_addr->port);

    if (0 == send_policy->segment_length || NE_SOMEIP_MAX_UDP_PAYLOAD_SIZE < send_policy->segment_length) {
        ne_someip_log_warn("send_policy->segment_length is %d.", send_policy->segment_length);
        send_policy->segment_length = NE_SOMEIP_MAX_UDP_PAYLOAD_SIZE;
    }
    if (ne_someip_tp_is_segment_needed(trans_buffer, send_policy->segment_length)) {
        // TP data
        ne_someip_list_t* tp_data_list = ne_someip_tp_segment_send_msg(trans_buffer, send_policy->segment_length);
        if (NULL == tp_data_list) {
            ne_someip_log_error("ne_someip_tp_segment_send_msg failed.");
            return ne_someip_error_code_tp_segment_error;
        }

        if (endpoint->is_need_switch_thread) {
            ne_someip_looper_runnable_t* runnable = ne_someip_ep_create_send_tp_data_runnable(endpoint, trans_buffer,
                tp_data_list, peer_addr, send_policy, seq_data);
            if (NULL == runnable) {
                ne_someip_log_error("ne_someip_ep_create_trans_state_runnable return error.");
                ne_someip_ep_tool_tp_data_list_all_free(tp_data_list);
                return res_code;
            }
            int res = ne_someip_looper_post(endpoint->endpoint_io_looper, runnable);
            if (0 == res) {
                res_code = ne_someip_error_code_ok;
            }
            else {
                ne_someip_log_error("ne_someip_looper_post fail.");
                ne_someip_ep_destroy_runnable(runnable);
            }
        } else {
            res_code = ne_someip_ep_core_send_tp_data(endpoint, trans_buffer, tp_data_list, peer_addr, send_policy, seq_data);
            // if (ne_someip_error_code_ok != res_code) {
                ne_someip_ep_tool_tp_data_list_free(tp_data_list);
            // }
        }
    } else {
        // not TP data
        if (endpoint->is_need_switch_thread) {  // TODO: add lock?
            ne_someip_looper_runnable_t* runnable = ne_someip_ep_create_send_msg_runnable(endpoint, trans_buffer,
                peer_addr, send_policy, seq_data);
            if (NULL == runnable) {
                ne_someip_log_error("ne_someip_ep_create_trans_state_runnable return error.");
                return res_code;
            }
            int res = ne_someip_looper_post(endpoint->endpoint_io_looper, runnable);
            if (0 == res) {
                res_code = ne_someip_error_code_ok;
            }
            else {
                ne_someip_log_error("ne_someip_looper_post fail.");
                ne_someip_ep_destroy_runnable(runnable);
            }
        } else {
            res_code = ne_someip_ep_core_send(endpoint, trans_buffer, peer_addr, send_policy, seq_data);
        }
    }

    return res_code;
}

bool ne_someip_endpoint_udp_data_is_dtls_used(const ne_someip_endpoint_udp_data_t* endpoint)
{
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return false;
    }

    return endpoint->is_tls_used;
}

bool ne_someip_endpoint_udp_data_set_dtls_status(ne_someip_endpoint_udp_data_t* endpoint,
    ne_someip_endpoint_net_addr_t* peer_addr, ne_someip_endpoint_dtls_status_t status)
{
    if (NULL == endpoint || NULL == endpoint->tls_status_map || NULL == peer_addr) {
        ne_someip_log_error("endpoint or endpoint->tls_status_map or peer_addr is NULL.");
        return false;
    }

    bool res = false;
    uint32_t hash_return = 0;
    ne_someip_endpoint_dtls_status_t* tls_status = ne_someip_map_find(endpoint->tls_status_map, (void*)peer_addr, &hash_return);
    if (NULL != tls_status) {
        *tls_status = status;
        res = true;
    } else {
        ne_someip_endpoint_net_addr_t* saved_key =
            (ne_someip_endpoint_net_addr_t*)malloc(sizeof(ne_someip_endpoint_net_addr_t));
        if (NULL == saved_key) {
            ne_someip_log_error("malloc ne_someip_endpoint_net_addr_t error.");
            return res;
        }
        memcpy(saved_key, peer_addr, sizeof(ne_someip_endpoint_net_addr_t));
        ne_someip_endpoint_dtls_status_t* saved_value =
            (ne_someip_endpoint_dtls_status_t*)malloc(sizeof(ne_someip_endpoint_dtls_status_t));
        if (NULL == saved_value) {
            ne_someip_log_error("malloc ne_someip_endpoint_dtls_status_t error.");
            free(saved_key);
            return res;
        }
        *saved_value = status;
        res = ne_someip_map_insert(endpoint->tls_status_map, saved_key, saved_value);
    }

    return res;
}

ne_someip_endpoint_dtls_status_t ne_someip_endpoint_udp_data_get_dtls_status(ne_someip_endpoint_udp_data_t* endpoint,
    ne_someip_endpoint_net_addr_t* peer_addr)
{
    ne_someip_endpoint_dtls_status_t res = ne_someip_endpoint_dtls_status_unknow;
    if (NULL == endpoint || NULL == endpoint->tls_status_map || NULL == peer_addr) {
        ne_someip_log_error("endpoint or endpoint->tls_status_map or peer_addr is NULL.");
        return res;
    }

    uint32_t hash_return = 0;
    ne_someip_endpoint_dtls_status_t* tls_status = ne_someip_map_find(endpoint->tls_status_map, (void*)peer_addr, &hash_return);
    if (NULL != tls_status) {
        res = *tls_status;
    }

    return res;
}

static void ne_someip_endpoint_udp_data_recreate_transmit(ne_someip_endpoint_udp_data_t* endpoint)
{
    if (NULL == endpoint || NULL == endpoint->udp_transmit) {
        return;
    }

    if (!endpoint->is_tls_used) {
        endpoint->udp_transmit->transmit = ne_someip_transmit_new(NE_SOMEIP_TRANSMIT_TYPE_UDP, endpoint->local_addr, false);
    } else {
        endpoint->udp_transmit->transmit = ne_someip_transmit_new(NE_SOMEIP_TRANSMIT_TYPE_DTLS, endpoint->local_addr, false);
    }
    if (NULL == endpoint->udp_transmit->transmit) {
        ne_someip_log_error("ne_someip_transmit_new return NULL.");
        return;
    }

    int i = 0;
    while(i < 10) {
        ne_someip_error_code_t res = ne_someip_ep_core_start(endpoint, endpoint->key_info);
        if (ne_someip_error_code_ok == res) {
            break;
        }
        ++i;
        usleep(200000);
    }
}

/***********************callback**************************/
void ne_someip_endpoint_udp_data_transmit_state_change(ne_someip_endpoint_udp_data_t* endpoint, ne_someip_endpoint_transmit_state_t state)
{
    ne_someip_log_info("start. udp transmit state change to [%d]", state);
    if (NULL == endpoint || NULL == endpoint->udp_transmit) {
        ne_someip_log_error("endpoint or endpoint->udp_transmit is NULL.");
        return;
    }

    if (endpoint->udp_transmit->transmit_state == state) {
        ne_someip_log_info("transmit_state is %d, not changed.", state);
        return;
    }

    endpoint->udp_transmit->transmit_state = state;

    switch(state) {
        case ne_someip_endpoint_transmit_state_stopped:
            ne_someip_log_debug("transmit stop, nothing need to do.");
            break;
        case ne_someip_endpoint_transmit_state_error:
        {
            // 清除出错的transmit相关的数据，该状态后续需要通知上层，此种情况需销毁endpoint并创建新的endpoint
            ne_someip_ep_core_stop(endpoint);

            if (NULL != endpoint->udp_transmit) {
                if (NULL != endpoint->udp_transmit->transmit) {
                    ne_someip_transmit_unref(endpoint->udp_transmit->transmit);
                    endpoint->udp_transmit->transmit = NULL;
                }
            }

            if (endpoint->is_need_switch_thread) {
                ne_someip_endpoint_udp_data_ref(endpoint);
                int32_t res = ne_someip_looper_runnable_task_create_and_post(endpoint->work_looper,
                    ne_someip_endpoint_udp_data_recreate_transmit, ne_someip_endpoint_udp_data_unref, endpoint);
                if (0 != res) {
                    ne_someip_log_error("ne_someip_looper_runnable_task_create_and_post fail.");
                    ne_someip_endpoint_udp_data_unref(endpoint);
                    return;
                }
            } else {
                ne_someip_endpoint_udp_data_recreate_transmit(endpoint);
            }
            break;
        }
        case ne_someip_endpoint_transmit_state_not_created:
            ne_someip_log_error("impossibe case.");
            break;
        case ne_someip_endpoint_transmit_state_prepared:
        case ne_someip_endpoint_transmit_state_startd:
        default:
            break;
    }
}

// void ne_someip_endpoint_udp_data_add_multicast_addr_state_change(ne_someip_endpoint_udp_data_t* endpoint,
//     ne_someip_enpoint_multicast_addr_t group_addr, ne_someip_endpoint_add_multicast_addr_state_t state)
// {
//     // TODO
// }

void ne_someip_endpoint_udp_data_async_send_reply(ne_someip_endpoint_udp_data_t* endpoint,
    const void* seq_data, ne_someip_error_code_t result)
{
    ne_someip_log_debug("async send reply, seq_data [%p], result [%d].", seq_data, result);
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return;
    }

    // notify to upper instance, switch thread
    if (endpoint->is_need_switch_thread) {
        ne_someip_looper_t* looper = endpoint->work_looper;
        if (NULL == looper) {
            ne_someip_log_error("work_looper is NULL.");
            return;
        }

        ne_someip_looper_runnable_t* runnable = ne_someip_ep_create_async_reply_runnable(endpoint, seq_data, result);
        if (NULL == runnable) {
            ne_someip_log_error("ne_someip_ep_create_async_reply_runnable return NULL.");
            return;
        }
        int post_res = ne_someip_looper_post(looper, runnable);
        if (0 != post_res) {
            ne_someip_log_error("ne_someip_looper_post fail.");
            ne_someip_ep_destroy_runnable(runnable);
        }
    }
    else {
        ne_someip_endpoint_async_send_reply(endpoint, seq_data, result);
    }
}

ne_someip_error_code_t ne_someip_endpoint_udp_data_on_receive(ne_someip_endpoint_udp_data_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer, uint32_t size, ne_someip_endpoint_net_addr_t* peer_addr)
{
    if (NULL == endpoint || NULL == trans_buffer || NULL == peer_addr) {
        ne_someip_log_error("endpoint or trans_buffer or peer_addr is NULL.");
        return ne_someip_error_code_failed;
    }

    ne_someip_log_info("receive data from udp transmit, data len is [%d], peer_addr ip is [%d], port is [%d].",
        size, peer_addr->ip_addr, peer_addr->port);

    ne_someip_endpoint_net_addr_pair_t addr_pair;
    addr_pair.base.type = ne_someip_endpoint_addr_type_net;
    addr_pair.local_addr = endpoint->local_addr;
    addr_pair.remote_addr = peer_addr;
    addr_pair.type = ne_someip_endpoint_type_udp;
    addr_pair.is_multicast = endpoint->is_multicast_endpoint;

    // notify to upper instance, switch thread
    ne_someip_error_code_t res = ne_someip_error_code_failed;
    if (endpoint->is_need_switch_thread) {
        ne_someip_looper_t* looper = endpoint->work_looper;
        if (NULL == looper) {
            ne_someip_log_error("work_looper is NULL.");
            return res;
        }

        ne_someip_looper_runnable_t* runnable = ne_someip_ep_create_recv_msg_runnable(endpoint, trans_buffer, &addr_pair);
        if (NULL == runnable) {
            ne_someip_log_error("ne_someip_ep_create_recv_msg_runnable return NULL.");
            return res;
        }
        int post_res = ne_someip_looper_post(looper, runnable);
        if (0 != post_res) {
            ne_someip_log_error("ne_someip_looper_post fail.");
            ne_someip_ep_destroy_runnable(runnable);
        }
        else {
            res = ne_someip_error_code_ok;
        }
    }
    else {
        ne_someip_endpoint_on_receive(endpoint, trans_buffer, &addr_pair);
        ne_someip_ep_tool_trans_buffer_free(trans_buffer);
        res = ne_someip_error_code_ok;
    }
    return res;
}

// return: ne_someip_list_t* dispatcher_instance(/*ne_someip_endpoint_instance_info_t*/)
ne_someip_list_t* ne_someip_endpoint_udp_data_dispatcher_instance(ne_someip_endpoint_udp_data_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer)
{
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return NULL;
    }


    if (NULL == endpoint->req_instance_list || 0 == ne_someip_list_length(endpoint->req_instance_list)) {
        return NULL;
    }

    return endpoint->req_instance_list;
}
