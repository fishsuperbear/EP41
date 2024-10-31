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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_NETWORK_CONNECT_BEHAVIOUR_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_NETWORK_CONNECT_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_client_define.h"

ne_someip_required_network_connect_behaviour_t*
    ne_someip_req_network_connect_behaviour_new(const ne_someip_network_config_t* net_config);

void ne_someip_req_network_connect_behaviour_free(ne_someip_required_network_connect_behaviour_t* behaviour);

ne_someip_network_config_t*
    ne_someip_req_network_connect_behav_net_config_get(ne_someip_required_network_connect_behaviour_t* behaviour);

// network status changing
void ne_someip_req_network_connect_behav_net_status_set(ne_someip_required_network_connect_behaviour_t* behaviour,
    ne_someip_network_states_t status);
//  get network_status
ne_someip_network_states_t
    ne_someip_req_network_connect_behav_net_status_get(ne_someip_required_network_connect_behaviour_t* behaviour);

void ne_someip_req_network_connect_behav_ip_set(ne_someip_required_network_connect_behaviour_t* behaviour,
    uint32_t ip_addr);
uint32_t ne_someip_req_network_connect_behav_ip_get(ne_someip_required_network_connect_behaviour_t* behaviour);

ne_someip_error_code_t
    ne_someip_req_network_connect_behav_save_find_behav(ne_someip_required_network_connect_behaviour_t* behaviour,
    const ne_someip_find_offer_service_spec_t* service_spec, ne_someip_required_find_service_behaviour_t* find_behav);
void ne_someip_req_network_connect_behav_remove_specific_find_behav(ne_someip_required_network_connect_behaviour_t* behaviour,
    const ne_someip_find_offer_service_spec_t* service_spec);
void ne_someip_req_network_connect_behav_remove_all_find_behav(ne_someip_required_network_connect_behaviour_t* behaviour);


#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_NETWORK_CONNECT_BEHAVIOUR_H
/* EOF */