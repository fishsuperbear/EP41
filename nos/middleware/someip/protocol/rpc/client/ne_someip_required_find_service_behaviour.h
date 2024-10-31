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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_FIND_SERVICE_BEHAVIOUR_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_FIND_SERVICE_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_client_define.h"

ne_someip_required_find_service_behaviour_t* ne_someip_req_find_service_behaviour_new(ne_someip_client_id_t client_id,
	const ne_someip_find_offer_service_spec_t* service_spec, const ne_someip_required_service_instance_config_t* config,
	ne_someip_looper_t* io_looper, ne_someip_common_service_instance_t* instance);

void ne_someip_req_find_service_behaviour_free(ne_someip_required_find_service_behaviour_t* behaviour);

//  send to endpoint
ne_someip_error_code_t ne_someip_req_find_serv_behav_start_find_service(ne_someip_required_find_service_behaviour_t* behaviour,
	ne_someip_network_states_t state, uint32_t ip);

ne_someip_error_code_t ne_someip_req_find_serv_behav_stop_find_service(ne_someip_required_find_service_behaviour_t* behaviour,
	uint32_t ip, bool* notify_obj, pthread_t tid);

ne_someip_required_service_instance_config_t*
    ne_someip_req_find_serv_behav_get_config(ne_someip_required_find_service_behaviour_t* behaviour);

// network up/down notify
ne_someip_error_code_t ne_someip_req_find_serv_behav_net_status_notify(ne_someip_required_find_service_behaviour_t* behaviour,
	ne_someip_network_states_t state, uint32_t ip);

void ne_someip_req_find_serv_behav_find_reply(ne_someip_required_find_service_behaviour_t* behaviour,
	ne_someip_find_service_states_t res);

ne_someip_find_service_states_t
ne_someip_req_find_serv_behav_find_status_get(ne_someip_required_find_service_behaviour_t* behaviour);

ne_someip_find_status_t ne_someip_req_find_serv_behav_get_upper_find_state(ne_someip_required_find_service_behaviour_t* behaviour,
	bool* is_need_notify);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_FIND_SERVICE_BEHAVIOUR_H
/* EOF */
