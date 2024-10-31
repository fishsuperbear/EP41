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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_SERVICE_CONNECT_BEHAVIOUR_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_SERVICE_CONNECT_BEHAVIOUR_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "ne_someip_client_define.h"
#include "ne_someip_endpoint_udp_data.h"

ne_someip_required_service_connect_behaviour_t* ne_someip_req_service_connect_behaviour_new();
ne_someip_required_service_connect_behaviour_t*
    ne_someip_req_service_connect_behaviour_ref(ne_someip_required_service_connect_behaviour_t* behaviour);
void ne_someip_req_service_connect_behaviour_unref(ne_someip_required_service_connect_behaviour_t* behaviour);

// void ne_someip_req_service_connect_behaviour_free(ne_someip_required_service_connect_behaviour_t* behaviour);

void ne_someip_req_serv_connect_behav_avail_status_set(ne_someip_required_service_connect_behaviour_t* behaviour,
    ne_someip_service_status_t status);
ne_someip_service_status_t
    ne_someip_req_serv_connect_behav_avail_status_get(ne_someip_required_service_connect_behaviour_t* behaviour);

//  get service_connect_behaviour state
ne_someip_remote_service_status_t
ne_someip_req_serv_connect_behav_serv_status_get(ne_someip_required_service_connect_behaviour_t* behaviour);
void ne_someip_req_serv_connect_behav_serv_status_set(ne_someip_required_service_connect_behaviour_t* behaviour,
    ne_someip_find_offer_service_spec_t offer_service, ne_someip_remote_service_status_t serv_status);

// save status
void ne_someip_req_serv_connect_behav_save_peer_info(ne_someip_required_service_connect_behaviour_t* behaviour,
    const ne_someip_sd_recv_offer_t* offer_msg);

// clear status
void ne_someip_req_serv_connect_behav_clear_peer_info(ne_someip_required_service_connect_behaviour_t* behaviour, bool is_tcp);

void ne_someip_req_serv_connect_behav_tcp_connect_status_set(ne_someip_required_service_connect_behaviour_t* behaviour,
    ne_someip_tcp_connect_states_t status);

ne_someip_tcp_connect_states_t
ne_someip_req_serv_connect_behav_tcp_connect_status_get(ne_someip_required_service_connect_behaviour_t* behaviour);

void ne_someip_req_serv_connect_behav_tcp_socket_create_status_set(
    ne_someip_required_service_connect_behaviour_t* behaviour, ne_someip_socket_create_states_t status);

ne_someip_socket_create_states_t ne_someip_req_serv_connect_behav_tcp_socket_create_status_get(
    ne_someip_required_service_connect_behaviour_t* behaviour);

void ne_someip_req_serv_connect_behav_udp_socket_create_status_set(
    ne_someip_required_service_connect_behaviour_t* behaviour, ne_someip_socket_create_states_t status);

ne_someip_socket_create_states_t ne_someip_req_serv_connect_behav_udp_socket_create_status_get(
    ne_someip_required_service_connect_behaviour_t* behaviour);

ne_someip_endpoint_net_addr_t*
ne_someip_req_serv_connect_behav_udp_peer_addr_get(ne_someip_required_service_connect_behaviour_t* behaviour);

ne_someip_endpoint_net_addr_t*
ne_someip_req_serv_connect_behav_tcp_peer_addr_get(ne_someip_required_service_connect_behaviour_t* behaviour);

uint32_t ne_someip_req_serv_connect_behav_peer_ip_get(ne_someip_required_service_connect_behaviour_t* behaviour);

bool ne_someip_req_serv_connect_behav_is_tcp_used(ne_someip_required_service_connect_behaviour_t* behaviour);
bool ne_someip_req_serv_connect_behav_is_udp_used(ne_someip_required_service_connect_behaviour_t* behaviour);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_SERVICE_CONNECT_BEHAVIOUR_H
/* EOF */
