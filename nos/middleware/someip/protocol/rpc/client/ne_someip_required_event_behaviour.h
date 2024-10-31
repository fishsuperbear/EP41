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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_EVENT_BEHAVIOUR_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_EVENT_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_client_define.h"

ne_someip_required_event_behaviour_t* ne_someip_req_event_behaviour_new(ne_someip_event_config_t* config);

void ne_someip_req_event_behaviour_free(ne_someip_required_event_behaviour_t* behaviour);

//  when port reuse, send to daemon
ne_someip_error_code_t ne_someip_req_event_behaviour_reg_event_handler_to_daemon(void* required_instance);

ne_someip_error_code_t ne_someip_req_event_behaviour_unreg_event_handler_to_daemon(void* required_instance);

//  receive from endpoint
void ne_someip_req_event_behaviour_recv_event(ne_someip_required_service_instance_t* instance, ne_someip_header_t* header,
	ne_someip_payload_t* payload);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_REQUIRED_EVENT_BEHAVIOUR_H
/* EOF */