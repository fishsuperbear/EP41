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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_METHOD_BEHAVIOUR_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_METHOD_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_client_define.h"

ne_someip_required_method_behaviour_t* ne_someip_req_method_behaviour_new(ne_someip_method_config_t* config);

void ne_someip_req_method_behaviour_free(ne_someip_required_method_behaviour_t* behaviour);

//  send to endpoint
ne_someip_error_code_t ne_someip_req_method_behav_send_request(void* endpoint, ne_someip_endpoint_net_addr_t* peer_addr,
    ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_endpoint_send_policy_t* policy, const void* seq_data);

ne_someip_error_code_t ne_someip_req_method_behav_send_request_to_daemon(ne_someip_header_t* header,
    ne_someip_payload_t* payload, void* required_instance, ne_someip_endpoint_send_policy_t* send_plicy);

//  when port reuse, send to daemon
ne_someip_error_code_t ne_someip_req_method_behaviour_reg_response_handler_to_daemon(void* required_instance);

ne_someip_error_code_t ne_someip_req_method_behaviour_unreg_response_handler_to_daemon(void* required_instance);

//  receive from endpoint
void ne_someip_req_method_behav_recv_response(ne_someip_required_service_instance_t* instance, ne_someip_header_t* header,
	ne_someip_payload_t* payload);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_METHOD_BEHAVIOUR_H
/* EOF */