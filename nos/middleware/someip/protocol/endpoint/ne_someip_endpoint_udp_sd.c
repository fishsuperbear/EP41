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
#include "ne_someip_endpoint_udp_sd.h"

#include <unistd.h>
#include <arpa/inet.h>
#include "ne_someip_app_context.h"
#include "ne_someip_endpoint_core.h"
#include "ne_someip_endpoint_tool.h"
#include "ne_someip_tp.h"
#include "ne_someip_log.h"


/********************internal func**********************/
static void ne_someip_endpoint_udp_sd_t_free(ne_someip_endpoint_udp_sd_t* endpoint);
/********************internal func**********************/

NEOBJECT_FUNCTION(ne_someip_endpoint_udp_sd_t);

ne_someip_endpoint_udp_sd_t* ne_someip_endpoint_udp_sd_create(
    ne_someip_endpoint_type_t type, ne_someip_endpoint_net_addr_t* addr, ne_someip_looper_t* looper)
{
    if ((NULL == looper) || (NULL == addr)) {
        ne_someip_log_error("io looper or addr is NULL");
        return NULL;
    }

    ne_someip_endpoint_udp_sd_t* endpoint = (ne_someip_endpoint_udp_sd_t*)malloc(sizeof(ne_someip_endpoint_udp_sd_t));
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint malloc error.");
        return NULL;
    }
    memset(endpoint, 0, sizeof(*endpoint));

    endpoint->base.endpoint_type = type;

    ne_someip_endpoint_net_addr_t* tmp_local_addr =
        (ne_someip_endpoint_net_addr_t*)malloc(sizeof(ne_someip_endpoint_net_addr_t));
    if (NULL == tmp_local_addr) {
        ne_someip_log_error("malloc ne_someip_endpoint_net_addr_t error.");
        ne_someip_endpoint_udp_sd_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    memcpy(tmp_local_addr, addr, sizeof(*addr));
    ne_someip_log_debug("tmp_local_addr->ip_addr is [%d],  tmp_local_addr->port is [%d]", tmp_local_addr->ip_addr,
        tmp_local_addr->port);
    endpoint->local_addr = tmp_local_addr;

    endpoint->endpoint_state = ne_someip_endpoint_state_created;

    ne_someip_looper_t* looper_ref = ne_someip_looper_ref(looper);
    if (NULL == looper_ref) {
        ne_someip_log_error("ne_someip_looper_ref return NULL.");
        ne_someip_endpoint_udp_sd_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    endpoint->endpoint_io_looper = looper_ref;

    ne_someip_endpoint_core_t* core = ne_someip_ep_core_new(looper_ref, type);
    if (NULL == core) {
        ne_someip_log_error("ne_someip_ep_core_new return NULL.");
        ne_someip_endpoint_udp_sd_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    endpoint->ep_core = core;

    ne_someip_endpoint_transmit_info_t* transmit_info = (ne_someip_endpoint_transmit_info_t*)malloc(sizeof(ne_someip_endpoint_transmit_info_t));
    if (NULL == transmit_info) {
        ne_someip_log_error("transmit_info malloc error.");
        ne_someip_endpoint_udp_sd_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    memset(transmit_info, 0, sizeof(*transmit_info));
    endpoint->udp_transmit = transmit_info;

    ne_someip_transmit_t* transmit = ne_someip_transmit_new(NE_SOMEIP_TRANSMIT_TYPE_UDP, addr, false);
    if (NULL == transmit) {
        ne_someip_log_error("ne_someip_transmit_new return NULL.");
        ne_someip_endpoint_udp_sd_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }
    endpoint->udp_transmit->transmit = transmit;
    endpoint->udp_transmit->transmit_state = ne_someip_endpoint_transmit_state_stopped;

    endpoint->callback = NULL;
    endpoint->group_addr_info = NULL;  // TODO: 考虑是否还需要这项，或者是否还需要新增
    endpoint->is_multicast_endpoint = false;

    endpoint->tp_ctx = ne_someip_tp_init();
    if (NULL == endpoint->tp_ctx) {
        ne_someip_log_error("ne_someip_tp_init return NULL.");
        ne_someip_endpoint_udp_sd_t_free(endpoint);
        endpoint = NULL;
        return NULL;
    }

    ne_someip_endpoint_udp_sd_t_ref_count_init(endpoint);

    ne_someip_log_info("ne_someip_endpoint_udp_sd_create create object finish.");

    ne_someip_error_code_t res_code = ne_someip_ep_core_start(endpoint, NULL);
    if (ne_someip_error_code_ok != res_code) {
        ne_someip_log_error("ne_someip_endpoint_start fail.");
        ne_someip_endpoint_udp_sd_unref(endpoint);
        return NULL;
    }

    return endpoint;
}


ne_someip_endpoint_udp_sd_t* ne_someip_endpoint_udp_sd_ref(ne_someip_endpoint_udp_sd_t* endpoint)
{
    ne_someip_endpoint_udp_sd_t* ref = NULL;
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return ref;
    }

    ref = ne_someip_endpoint_udp_sd_t_ref(endpoint);
    return ref;
}

void ne_someip_endpoint_udp_sd_unref(ne_someip_endpoint_udp_sd_t* endpoint)
{
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return;
    }

    ne_someip_endpoint_udp_sd_t_unref(endpoint);
}

static void ne_someip_endpoint_udp_sd_t_free(ne_someip_endpoint_udp_sd_t* endpoint)
{
    ne_someip_error_code_t res_code = ne_someip_error_code_failed;
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return;
    }

    res_code = ne_someip_ep_core_stop(endpoint);
    if (ne_someip_error_code_failed == res_code) {
        ne_someip_log_error("ne_someip_endpoint_stop return fail.");
    }

    ne_someip_endpoint_udp_sd_t_ref_count_deinit(endpoint);

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
    if (NULL != endpoint->endpoint_io_looper) {
        ne_someip_looper_unref(endpoint->endpoint_io_looper);
        endpoint->endpoint_io_looper = NULL;
    }
    if (NULL != endpoint->callback) {
        if (NULL != endpoint->callback->user_data && NULL != endpoint->callback->free) {
            endpoint->callback->free(endpoint->callback->user_data);
            endpoint->callback->user_data = NULL;
        }
        free(endpoint->callback);
        endpoint->callback = NULL;
    }
    if (NULL != endpoint->group_addr_info) {  // TODO: 考虑是否还需要这项，或者是否还需要新增
        if (NULL != endpoint->group_addr_info->addr) {
            free(endpoint->group_addr_info->addr);
            endpoint->group_addr_info->addr = NULL;
        }
        free(endpoint->group_addr_info);
        endpoint->group_addr_info = NULL;
    }
    if (NULL != endpoint->tp_ctx) {
        ne_someip_tp_deinit(endpoint->tp_ctx);
        endpoint->tp_ctx = NULL;
    }

    free(endpoint);
    endpoint = NULL;

    ne_someip_log_info("ne_someip_endpoint_udp_sd_t_free finish.");
}

ne_someip_error_code_t ne_someip_endpoint_udp_sd_join_group(
    ne_someip_endpoint_udp_sd_t* endpoint, ne_someip_endpoint_net_addr_t* interface_addr)
{
    ne_someip_error_code_t res_code = ne_someip_error_code_failed;

    if (NULL == endpoint || NULL == interface_addr) {
        ne_someip_log_error("endpoint or interface_addr is NULL.");
        return res_code;
    }

    ne_someip_log_info("join group, interface_addr->ip_addr is [%d],  interface_addr->port is [%d]", interface_addr->ip_addr,
        interface_addr->port);

    res_code = ne_someip_endpoint_join_group(endpoint, interface_addr);
    ne_someip_log_debug("udp join group res is [%d]", res_code);

    endpoint->is_multicast_endpoint = true;
    return res_code;
}

ne_someip_error_code_t ne_someip_endpoint_udp_sd_leave_group(
    ne_someip_endpoint_udp_sd_t* endpoint, ne_someip_endpoint_net_addr_t* interface_addr)
{
    ne_someip_error_code_t res_code = ne_someip_error_code_failed;

    if (NULL == endpoint || NULL == interface_addr) {
        ne_someip_log_error("endpoint or interface_addr is NULL.");
        return res_code;
    }

    ne_someip_log_info("join group, interface_addr->ip_addr is [%d],  interface_addr->port is [%d]", interface_addr->ip_addr,
        interface_addr->port);

    res_code = ne_someip_endpoint_leave_group(endpoint, interface_addr);
    ne_someip_log_debug("udp leave group res is [%d]", res_code);
    return res_code;
}

ne_someip_error_code_t ne_someip_endpoint_udp_sd_send_async(ne_someip_endpoint_udp_sd_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer, ne_someip_endpoint_net_addr_t* peer_addr,
    ne_someip_endpoint_send_policy_t* send_policy, const void* seq_data)
{
    ne_someip_error_code_t res_code = ne_someip_error_code_failed;
    if (NULL == endpoint || NULL == trans_buffer || NULL == peer_addr || NULL == send_policy) {
        ne_someip_log_error("endpoint or trans_buffer or peer_addr or send_policy is NULL.");
        return res_code;
    }
    if (NULL == endpoint->endpoint_io_looper) {  // TODO: add lock?
        ne_someip_log_error("io_looper is NULL.");
        return res_code;
    }

    if (0 == send_policy->segment_length || NE_SOMEIP_MAX_UDP_PAYLOAD_SIZE < send_policy->segment_length) {
        ne_someip_log_debug("send_policy->segment_length is %d.", send_policy->segment_length);
        send_policy->segment_length = NE_SOMEIP_MAX_UDP_PAYLOAD_SIZE;
    }

    if (ne_someip_tp_is_segment_needed(trans_buffer, send_policy->segment_length)) {
        // TP data
        ne_someip_list_t* tp_data_list = ne_someip_tp_segment_send_msg(trans_buffer, send_policy->segment_length);
        if (NULL == tp_data_list) {
            ne_someip_log_error("ne_someip_tp_segment_send_msg failed.");
            return ne_someip_error_code_tp_segment_error;
        }
        res_code = ne_someip_ep_core_send_tp_data(endpoint, trans_buffer, tp_data_list, peer_addr, send_policy, seq_data);
        // if (ne_someip_error_code_ok != res_code) {
            ne_someip_ep_tool_tp_data_list_free(tp_data_list);
        // }
    } else {
        // not TP data
        res_code = ne_someip_ep_core_send(endpoint, trans_buffer, peer_addr, send_policy, seq_data);
    }

    struct in_addr ip_addr_str;
    ip_addr_str.s_addr = peer_addr->ip_addr;
    ne_someip_log_info("send data use udp sd endpoint, peer_addr ip [%s], port [%d], res [%d]", inet_ntoa(ip_addr_str),
        peer_addr->port, res_code);
    // if（ne_someip_error_code_ok != res） {  // trigger失败，删除数据并通知上层user；trigger成功，考虑异步和缓存情况，等数据发送动作后再释放及通知
    //     ne_someip_ep_tool_trans_buffer_free(trans_buffer);
    // }

    return res_code;
}

static void ne_someip_endpoint_udp_sd_recreate_transmit(ne_someip_endpoint_udp_sd_t* endpoint)
{
    if (NULL == endpoint || NULL == endpoint->udp_transmit) {
        return;
    }

    endpoint->udp_transmit->transmit = ne_someip_transmit_new(NE_SOMEIP_TRANSMIT_TYPE_UDP, endpoint->local_addr, false);
    if (NULL == endpoint->udp_transmit->transmit) {
        ne_someip_log_error("ne_someip_transmit_new return NULL.");
        return;
    }

    int i = 0;
    while(i < 10) {
        ne_someip_error_code_t res = ne_someip_ep_core_start(endpoint, NULL);
        if (ne_someip_error_code_ok == res) {
            break;
        }
        ++i;
        usleep(200000);
    }
}

/***********************callback**************************/

void ne_someip_endpoint_udp_sd_transmit_state_change(ne_someip_endpoint_udp_sd_t* endpoint, ne_someip_endpoint_transmit_state_t state)
{
    ne_someip_log_info("start. udp transmit state change to [%d]", state);
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return;
    }

    if (NULL == endpoint->udp_transmit) {
        ne_someip_log_error("endpoint->udp_transmit is NULL.");
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

            ne_someip_endpoint_udp_sd_recreate_transmit(endpoint);
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

// void ne_someip_endpoint_udp_sd_add_multicast_addr_state_change(ne_someip_endpoint_udp_sd_t* endpoint,
//     ne_someip_enpoint_multicast_addr_t* group_addr, ne_someip_endpoint_add_multicast_addr_state_t state)
// {
//     // TODO
// }

void ne_someip_endpoint_udp_sd_async_send_reply(ne_someip_endpoint_udp_sd_t* endpoint,
    const void* seq_data, ne_someip_error_code_t result)
{
    if (NULL == endpoint) {
        ne_someip_log_error("endpoint is NULL.");
        return;
    }

    ne_someip_log_debug("async send reply, seq_data [%p], result [%d].", seq_data, result);
    ne_someip_endpoint_async_send_reply(endpoint, seq_data, result);
}

ne_someip_error_code_t ne_someip_endpoint_udp_sd_on_receive(ne_someip_endpoint_udp_sd_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer, uint32_t size, ne_someip_endpoint_net_addr_t* peer_addr)
{
    if (NULL == endpoint || NULL == trans_buffer || NULL == peer_addr) {
        ne_someip_log_error("endpoint or trans_buffer or peer_addr is NULL.");
        return ne_someip_error_code_failed;
    }

    struct in_addr ip_addr_str;
    ip_addr_str.s_addr = peer_addr->ip_addr;
    ne_someip_log_info("receive data from udp transmit, data len is [%d], peer_addr ip is [%s], port is [%d].",
        size, inet_ntoa(ip_addr_str), peer_addr->port);

    ne_someip_endpoint_net_addr_pair_t addr_pair;
    addr_pair.base.type = ne_someip_endpoint_addr_type_net;
    addr_pair.local_addr = endpoint->local_addr;
    addr_pair.remote_addr = peer_addr;
    addr_pair.type = ne_someip_endpoint_type_udp;
    addr_pair.is_multicast = endpoint->is_multicast_endpoint;
    ne_someip_endpoint_on_receive(endpoint, trans_buffer, &addr_pair);
    ne_someip_ep_tool_trans_buffer_free(trans_buffer);
    return ne_someip_error_code_ok;
}

ne_someip_list_t* ne_someip_endpoint_udp_sd_dispatcher_instance(ne_someip_endpoint_udp_sd_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer) // return: ne_someip_list_t* dispatcher_instance(/*ne_someip_endpoint_instance_info_t*/)
{
    return NULL;
}
