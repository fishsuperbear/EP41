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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_EVENTGROUP_BEHAVIOUR_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_EVENTGROUP_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_client_define.h"
#include "ne_someip_endpoint_udp_data.h"

ne_someip_required_eventgroup_behaviour_t*
ne_someip_req_eventgroup_behaviour_new(const ne_someip_required_eventgroup_config_t* config);

void ne_someip_req_eventgroup_behaviour_free(ne_someip_required_eventgroup_behaviour_t* behaviour);

//  send to endpoint
ne_someip_error_code_t ne_someip_req_eg_behav_start_subscribe_eventgroup(ne_someip_required_eventgroup_behaviour_t* behaviour,
    ne_someip_eventgroup_id_t eventgroup_id, void* required_instance, ne_someip_network_states_t state,
    ne_someip_service_status_t serv_status, bool is_comm_ok);

ne_someip_error_code_t ne_someip_req_eg_behav_stop_subscribe_eventgroup(ne_someip_required_eventgroup_behaviour_t* behaviour,
    ne_someip_eventgroup_id_t eventgroup_id, void* required_instance, bool* notify_obj, pthread_t tid);

//  receive from endpoint
void ne_someip_req_eg_behav_recv_subscribe_ack(ne_someip_required_eventgroup_behaviour_t* behaviour,
    const ne_someip_sd_recv_subscribe_ack_t* sub_ack);

bool ne_someip_req_eg_behav_is_sub_eventgroup_wait_net_up(ne_someip_required_eventgroup_behaviour_t* behaviour);

bool ne_someip_req_eg_behav_is_sub_eventgroup_wait_avaliable(ne_someip_required_eventgroup_behaviour_t* behaviour);

void ne_someip_req_eg_behav_set_sub_state(ne_someip_required_eventgroup_behaviour_t* behaviour,
    ne_someip_eventgroup_subscribe_states_t state);

bool ne_someip_req_eg_behav_is_sub_ack(ne_someip_required_eventgroup_behaviour_t* behaviour);

ne_someip_subscribe_status_t ne_someip_req_eg_behav_get_upper_sub_state(
    ne_someip_required_eventgroup_behaviour_t* behaviour, bool* is_need_notify);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_EVENTGROUP_BEHAVIOUR_H
/* EOF */
