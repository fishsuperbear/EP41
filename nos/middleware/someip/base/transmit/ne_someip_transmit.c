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
#include <stdlib.h>
#include <string.h>
#include "ne_someip_log.h"
#include "ne_someip_object.h"
#include "ne_someip_transmit.h"
#include "ne_someip_transmitimpltcp.h"
#include "ne_someip_transmitimpludp.h"
#include "ne_someip_transmitimpltls.h"
#include "ne_someip_transmitimpldtls.h"
#include "ne_someip_transmitimplunixDomain.h"
#include "ne_someip_transmitcore.h"

struct ne_someip_transmit {
    ne_someip_transmit_type_t transmit_type;   // transmit类型
    ne_someip_transmit_impl_t* impl;           // impl对象. TCP UDP UnixDomain
    ne_someip_transmit_core_t* core;           // core对象
    NEOBJECT_MEMBER
};

static void ne_someip_transmit_t_free(ne_someip_transmit_t *obj);
static ne_someip_transmit_t* ne_someip_transmit_t_new();
NEOBJECT_FUNCTION(ne_someip_transmit_t);

ne_someip_transmit_t*
ne_someip_transmit_t_new()
{
    ne_someip_transmit_t* transmit = malloc(sizeof(ne_someip_transmit_t));
    if (NULL == transmit) {
        ne_someip_log_error("[Transmit] malloc error");
        return NULL;
    }

    memset(transmit, 0, sizeof(ne_someip_transmit_t));
    ne_someip_transmit_t_ref_count_init(transmit);

    ne_someip_log_debug("[Transmit] create transmit %p", transmit);
    return transmit;
}

void ne_someip_transmit_t_free(ne_someip_transmit_t *obj) {
    if (NULL == obj) {
        ne_someip_log_error("[Transmit] error: obj is null");
        return;
    }
    ne_someip_log_debug("[Transmit] free transmit %p", obj);
    // impl由core释放

    // 释放core
    if (NULL != obj->core) {
        ne_someip_transmit_core_unref(obj->core);
        obj->core = NULL;
    }

    ne_someip_transmit_t_ref_count_deinit(obj);

    free(obj);
    obj = NULL;
    return;
}

ne_someip_transmit_t*
ne_someip_transmit_new(ne_someip_transmit_type_t type, const void* local_address, bool is_listen_mode)
{
    // Create transmit
    ne_someip_transmit_t* transmit = ne_someip_transmit_t_new();
    if (NULL == transmit) {
        ne_someip_log_error("[Transmit] transmit create error");
        return NULL;
    }
    transmit->transmit_type = type;
    transmit->impl = NULL;

    // create transmit impl by type(TCP/UDP/UnixDomain)
    switch (type) {
        case NE_SOMEIP_TRANSMIT_TYPE_TCP:
        {
            ne_someip_transmit_impl_tcp_t* impl_tcp = ne_someip_transmit_impl_tcp_new();
            transmit->impl = (ne_someip_transmit_impl_t*)impl_tcp;
            break;
        }
        case NE_SOMEIP_TRANSMIT_TYPE_UDP:
        {
            ne_someip_transmit_impl_udp_t* impl_udp = ne_someip_transmit_impl_udp_new();
            transmit->impl = (ne_someip_transmit_impl_t*)impl_udp;
            break;
        }
        case NE_SOMEIP_TRANSMIT_TYPE_TLS:
        {
            ne_someip_transmit_impl_tls_t* impl_tls = ne_someip_transmit_impl_tls_new();
            transmit->impl = (ne_someip_transmit_impl_t*)impl_tls;
            break;
        }
        // case NE_SOMEIP_TRANSMIT_TYPE_DTLS:
        // {
        //     ne_someip_transmit_impl_dtls_t* impl_dtls = ne_someip_transmit_impl_dtls_new();
        //     transmit->impl = (ne_someip_transmit_impl_t*)impl_dtls;
        //     break;
        // }
        case NE_SOMEIP_TRANSMIT_TYPE_UNIX_DOMAIN:
        {
            ne_someip_transmit_impl_unix_domain_t* impl_unixDomain = ne_someip_transmit_impl_unix_domain_new();
            transmit->impl = (ne_someip_transmit_impl_t*)impl_unixDomain;
            break;
        }
        default:
        {
            ne_someip_log_error("[Transmit] transmit_type error");
            break;
        }
    }
    if (NULL == transmit->impl) {
        ne_someip_log_error("[Transmit] ne_someip_transmit_impl_tcp_new:impl create error");
        ne_someip_transmit_unref(transmit);
        transmit = NULL;
        return NULL;
    }

    // create core by transmit impl
    transmit->core = ne_someip_transmit_core_new(transmit->impl, type, local_address, is_listen_mode);
    if (NULL == transmit->core) {
        ne_someip_log_error("[Transmit] ne_someip_transmit_core_new error");
        ne_someip_transmit_unref(transmit);
        return NULL;
    }

    ne_someip_log_debug("[Transmit] create transmit:%p, is_listen_mode:%d", transmit, is_listen_mode);
    if (NE_SOMEIP_TRANSMIT_TYPE_TCP == type || NE_SOMEIP_TRANSMIT_TYPE_UDP == type || \
        NE_SOMEIP_TRANSMIT_TYPE_TLS == type || NE_SOMEIP_TRANSMIT_TYPE_DTLS == type) {
        ne_someip_log_debug("[Transmit] local addr is :%d:%d", ((ne_someip_endpoint_net_addr_t*)local_address)->ip_addr, ((ne_someip_endpoint_net_addr_t*)local_address)->port);
    }
    else {
        ne_someip_log_debug("[Transmit] local addr is %s", ((ne_someip_endpoint_unix_addr_t*)local_address)->unix_path);
    }

    return transmit;
}

ne_someip_transmit_t* ne_someip_transmit_ref(ne_someip_transmit_t* handle)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return NULL;
    }
    return ne_someip_transmit_t_ref(handle);
}

int ne_someip_transmit_unref(ne_someip_transmit_t* handle)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }

    ne_someip_transmit_t_unref(handle);
    return 0;
}

int ne_someip_transmit_set_looper(ne_someip_transmit_t* handle, ne_someip_looper_t* looper)
{
    if (NULL == handle || NULL == looper) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_set_looper(handle->core, looper);
}

ne_someip_looper_t* ne_someip_transmit_get_looper(ne_someip_transmit_t* handle)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return NULL;
    }
    return ne_someip_transmit_core_get_looper(handle->core);
}

int ne_someip_transmit_set_config(ne_someip_transmit_t* handle, ne_someip_transmit_config_type_t type, const void* config)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_set_config(handle->core, type, config);
}

int ne_someip_transmit_get_config(ne_someip_transmit_t* handle, ne_someip_transmit_config_type_t type, void* config)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_get_config(handle->core, type, config);
}

int ne_someip_transmit_set_callback(ne_someip_transmit_t* handle, ne_someip_transmit_callback_t* callback, void* user_data)
{
    if (NULL == handle || NULL == callback) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_set_callback(handle->core, callback, user_data);
}

int ne_someip_transmit_set_source(ne_someip_transmit_t* handle, ne_someip_transmit_source_t* source, void* user_data)
{
    if (NULL == handle || NULL == source) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_set_source(handle->core, source, user_data);
}

int ne_someip_transmit_set_sink(ne_someip_transmit_t* handle, ne_someip_transmit_sink_t* sink, void* user_data)
{
    if (NULL == handle || NULL == sink) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_set_sink(handle->core, sink, user_data);
}

int ne_someip_transmit_prepare(ne_someip_transmit_t* handle)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_prepare(handle->core);
}

int ne_someip_transmit_start(ne_someip_transmit_t* handle)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_start(handle->core);
}

int ne_someip_transmit_stop(ne_someip_transmit_t* handle)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }

    int ret = 0;
    if (NULL != handle->core) {
        ret = ne_someip_transmit_core_stop(handle->core);
    }

    return ret;
}

ne_someip_transmit_state_t ne_someip_transmit_get_state(ne_someip_transmit_t* handle)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return NE_SOMEIP_TRANSMIT_STATE_MAX;
    }
    return ne_someip_transmit_core_get_state(handle->core);
}

int ne_someip_transmit_query(ne_someip_transmit_t* handle, ne_someip_transmit_query_type_t type, void* data)
{
    if (NULL == handle || NULL == data) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_query(handle->core, type, data);
}

int ne_someip_transmit_join_group(ne_someip_transmit_t* handle, ne_someip_transmit_group_type_t group_type,
    ne_someip_endpoint_net_addr_t* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address)
{
    ne_someip_log_debug("[Transmit] start");
    if (NULL == handle || NULL == unicast_addr || NULL == group_address) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }

    ne_someip_log_info("[Transmit] unicast ip = %d, mul_ip = %d",
        unicast_addr->ip_addr, group_address->multicast_ip);
    return ne_someip_transmit_core_join_group(handle->core, group_type, unicast_addr, group_address);
}

int ne_someip_transmit_leave_group(ne_someip_transmit_t* handle, ne_someip_transmit_group_type_t group_type,
    ne_someip_endpoint_net_addr_t* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address)
{
    if (NULL == handle || NULL == unicast_addr || NULL == group_address) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_leave_group(handle->core, group_type, unicast_addr, group_address);
}

int ne_someip_transmit_link_trigger_available(ne_someip_transmit_t* handle, ne_someip_transmit_link_type_t type, ne_someip_transmit_link_t* link)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_link_trigger_available(handle->core, type, link);
}

int ne_someip_transmit_link_setup(ne_someip_transmit_t* handle, ne_someip_transmit_link_type_t link_type,
    void* peer_address, void* user_data)
{
    if (NULL == handle) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_link_setup(handle->core, link_type, peer_address, user_data);
}

int ne_someip_transmit_link_teardown(ne_someip_transmit_t* handle, ne_someip_transmit_link_t* link)
{
    if (NULL == handle || NULL == link) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_link_teardown(handle->core, link);
}

ne_someip_transmit_link_state_t ne_someip_transmit_link_get_state(ne_someip_transmit_link_t* link)
{
    if (NULL == link) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return NE_SOMEIP_TRANSMIT_LINK_STATE_MAX;
    }
    return ne_someip_transmit_core_link_get_state(link);
}

ne_someip_transmit_link_type_t ne_someip_transmit_link_get_type(ne_someip_transmit_link_t* link)
{
    if (NULL == link) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return NE_SOMEIP_TRANSMIT_LINK_TYPE_MAX;
    }
    return ne_someip_transmit_core_link_get_type(link);
}

int ne_someip_transmit_link_get_peer_address(ne_someip_transmit_link_t* link, void* peer_address)
{
    if (NULL == link) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return -1;
    }
    return ne_someip_transmit_core_link_get_peer_address(link, peer_address);
}

void* ne_someip_transmit_link_get_userdata(ne_someip_transmit_link_t* link)
{
    if (NULL == link) {
        ne_someip_log_error("[Transmit] param is NULL, error");
        return NULL;
    }
    return ne_someip_transmit_core_link_get_userdata(link);
}

uint16_t ne_someip_transmit_get_port(const ne_someip_transmit_t* transmit)
{
    if (NULL == transmit || NULL == transmit->core) {
        return 0;
    }
    ne_someip_endpoint_net_addr_t* net_addr = (ne_someip_endpoint_net_addr_t*)transmit->core->local_address;
    if (NULL == net_addr) {
        ne_someip_log_error("net_addr is null");
        return 0;
    }

    return net_addr->port;
}
/* EOF */
