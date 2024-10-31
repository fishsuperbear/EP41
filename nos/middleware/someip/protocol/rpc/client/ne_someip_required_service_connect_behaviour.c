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
#include "ne_someip_required_service_connect_behaviour.h"
#include "ne_someip_ipc_behaviour.h"
#include "ne_someip_log.h"
#include "ne_someip_sd_tool.h"

static void ne_someip_required_service_connect_behaviour_t_free(ne_someip_required_service_connect_behaviour_t* behaviour);

NEOBJECT_FUNCTION(ne_someip_required_service_connect_behaviour_t);

ne_someip_required_service_connect_behaviour_t* ne_someip_req_service_connect_behaviour_new()
{
    ne_someip_required_service_connect_behaviour_t* ser_con_behav =
        (ne_someip_required_service_connect_behaviour_t*)malloc(sizeof(ne_someip_required_service_connect_behaviour_t));
    if (NULL == ser_con_behav) {
        ne_someip_log_error("malloc ne_someip_required_service_connect_behaviour_t error.");
        return ser_con_behav;
    }
    memset(ser_con_behav, 0, sizeof(*ser_con_behav));
    ser_con_behav->avail_status = ne_someip_service_status_unavailable;
    ser_con_behav->status = ne_someip_remote_service_states_unknown;
    ser_con_behav->remote_tcp_socket_states = ne_someip_socket_create_states_not_create;
    ser_con_behav->remote_udp_socket_states = ne_someip_socket_create_states_not_create;
    ser_con_behav->tcp_connect_states = ne_someip_tcp_connect_states_disconnect;
    ser_con_behav->reliable_flag = false;
    ser_con_behav->unreliable_flag = false;
    ser_con_behav->peer_udp_addr = NULL;
    ser_con_behav->peer_tcp_addr = NULL;

    ne_someip_required_service_connect_behaviour_t_ref_count_init(ser_con_behav);

    return ser_con_behav;
}

ne_someip_required_service_connect_behaviour_t*
    ne_someip_req_service_connect_behaviour_ref(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return NULL;
    }

    return ne_someip_required_service_connect_behaviour_t_ref(behaviour);
}

void ne_someip_req_service_connect_behaviour_unref(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }

    return ne_someip_required_service_connect_behaviour_t_unref(behaviour);
}

void ne_someip_req_serv_connect_behav_avail_status_set(ne_someip_required_service_connect_behaviour_t* behaviour,
    ne_someip_service_status_t status)
{
    if (NULL == behaviour) {
        return;
    }

    behaviour->avail_status = status;
}

ne_someip_service_status_t
    ne_someip_req_serv_connect_behav_avail_status_get(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        return ne_someip_service_status_unavailable;
    }

    return behaviour->avail_status;
}

//  get service_connect_behaviour state
ne_someip_remote_service_status_t
ne_someip_req_serv_connect_behav_serv_status_get(ne_someip_required_service_connect_behaviour_t* behaviour)
{
	if (NULL == behaviour) {
        return ne_someip_remote_service_states_unknown;
	}
    ne_someip_log_debug("get remote service status %d", behaviour->status);
	return behaviour->status;
}

// set service connect behaviour state
void ne_someip_req_serv_connect_behav_serv_status_set(ne_someip_required_service_connect_behaviour_t* behaviour,
    ne_someip_find_offer_service_spec_t offer_service, ne_someip_remote_service_status_t serv_status)
{
	if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }
    ne_someip_log_debug("set remote service status %d", serv_status);
    behaviour->status = serv_status;
}

void ne_someip_req_serv_connect_behav_save_peer_info(ne_someip_required_service_connect_behaviour_t* behaviour,
    const ne_someip_sd_recv_offer_t* offer_msg)
{
    if (NULL == behaviour || NULL == offer_msg) {
        ne_someip_log_error("behaviour or offer_msg is NULL.");
        return;
    }

    if (NULL == behaviour->peer_tcp_addr) {
        behaviour->peer_tcp_addr = (ne_someip_endpoint_net_addr_t*)malloc(sizeof(ne_someip_endpoint_net_addr_t));
        if (NULL == behaviour->peer_tcp_addr) {
            ne_someip_log_error("malloc ne_someip_endpoint_net_addr_t error.");
            return;
        }
    }

    if (NULL == behaviour->peer_udp_addr) {
        behaviour->peer_udp_addr = (ne_someip_endpoint_net_addr_t*)malloc(sizeof(ne_someip_endpoint_net_addr_t));
        if (NULL == behaviour->peer_udp_addr) {
            ne_someip_log_error("malloc ne_someip_endpoint_net_addr_t error.");
            return;
        }
    }

    ne_someip_sd_convert_uint32_to_ip("ip :", offer_msg->service_addr, __FILE__, __LINE__, __FUNCTION__);
    ne_someip_log_info("tcp_port [%d], udp_port [%d], addr_type [%d]", offer_msg->tcp_port,
        offer_msg->udp_port, offer_msg->addr_type);

    behaviour->peer_tcp_addr->ip_addr = offer_msg->service_addr;
    behaviour->peer_tcp_addr->port = offer_msg->tcp_port;
    behaviour->peer_tcp_addr->type = offer_msg->addr_type;
    
    behaviour->peer_udp_addr->ip_addr = offer_msg->service_addr;
    behaviour->peer_udp_addr->port = offer_msg->udp_port;
    behaviour->peer_udp_addr->type = offer_msg->addr_type;

    behaviour->reliable_flag = offer_msg->reliable_flag;
    behaviour->unreliable_flag = offer_msg->unreliable_flag;
    ne_someip_log_debug("cmj-----udp_addr %p, udp port %d", behaviour->peer_udp_addr, behaviour->peer_udp_addr->port);
}

void ne_someip_req_serv_connect_behav_clear_peer_info(ne_someip_required_service_connect_behaviour_t* behaviour,
    bool is_tcp)
{
    ne_someip_log_info("cmj-----clear peer_info");
    behaviour->avail_status = ne_someip_service_status_unavailable;
    behaviour->status = ne_someip_remote_service_states_unknown;
    if (is_tcp) {
        behaviour->reliable_flag = false;
        if (NULL != behaviour->peer_tcp_addr) {
            ne_someip_log_error("free tcp_addr is %p", behaviour->peer_tcp_addr);
            free(behaviour->peer_tcp_addr);
            behaviour->peer_tcp_addr = NULL;
        }
    } else {
        behaviour->unreliable_flag = false;
        if (NULL != behaviour->peer_udp_addr) {
            free(behaviour->peer_udp_addr);
            behaviour->peer_udp_addr = NULL;
        }
    }
}

void ne_someip_req_serv_connect_behav_tcp_connect_status_set(ne_someip_required_service_connect_behaviour_t* behaviour,
    ne_someip_tcp_connect_states_t status)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }

    behaviour->tcp_connect_states = status;
}

ne_someip_tcp_connect_states_t
ne_someip_req_serv_connect_behav_tcp_connect_status_get(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return false;
    }

    return behaviour->tcp_connect_states;
}

void ne_someip_req_serv_connect_behav_tcp_socket_create_status_set(
    ne_someip_required_service_connect_behaviour_t* behaviour, ne_someip_socket_create_states_t status)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }

    behaviour->remote_tcp_socket_states = status;
}

ne_someip_socket_create_states_t ne_someip_req_serv_connect_behav_tcp_socket_create_status_get(
    ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return false;
    }

    return behaviour->remote_tcp_socket_states;
}

void ne_someip_req_serv_connect_behav_udp_socket_create_status_set(
    ne_someip_required_service_connect_behaviour_t* behaviour, ne_someip_socket_create_states_t status)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }

    behaviour->remote_udp_socket_states = status;
}

ne_someip_socket_create_states_t ne_someip_req_serv_connect_behav_udp_socket_create_status_get(
    ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return false;
    }

    return behaviour->remote_udp_socket_states;
}

ne_someip_endpoint_net_addr_t*
ne_someip_req_serv_connect_behav_udp_peer_addr_get(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return NULL;
    }

    return behaviour->peer_udp_addr;
}

ne_someip_endpoint_net_addr_t*
ne_someip_req_serv_connect_behav_tcp_peer_addr_get(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return NULL;
    }

    return behaviour->peer_tcp_addr;
}

uint32_t ne_someip_req_serv_connect_behav_peer_ip_get(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return 0;
    }

    ne_someip_endpoint_net_addr_t* addr = ne_someip_req_serv_connect_behav_udp_peer_addr_get(behaviour);
    if (NULL == addr) {
        addr = ne_someip_req_serv_connect_behav_tcp_peer_addr_get(behaviour);   
    }

    if (NULL == addr) {
        return 0;
    }

    return addr->ip_addr;
}

bool ne_someip_req_serv_connect_behav_is_tcp_used(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return false;
    }

    return behaviour->reliable_flag;
}

bool ne_someip_req_serv_connect_behav_is_udp_used(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return false;
    }

    return behaviour->unreliable_flag;
}

static void ne_someip_required_service_connect_behaviour_t_free(ne_someip_required_service_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_info("behaviour is NULL.");
        return;
    }

    ne_someip_required_service_connect_behaviour_t_ref_count_deinit(behaviour);

    if (NULL != behaviour->peer_udp_addr) {
        free(behaviour->peer_udp_addr);
        behaviour->peer_udp_addr = NULL;
    }

    if (NULL != behaviour->peer_tcp_addr) {
        free(behaviour->peer_tcp_addr);
        behaviour->peer_tcp_addr = NULL;
    }

    free(behaviour);
    behaviour = NULL;
}