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

#ifndef INCLUDE_NE_SOMEIP_HANDLER_H
#define INCLUDE_NE_SOMEIP_HANDLER_H

#include <stdbool.h>
#include "ne_someip_define.h"

// ne_someip_find_offer_service_spec_t 中的值可以为FF
typedef void (*ne_someip_find_status_handler)(const ne_someip_find_offer_service_spec_t*, ne_someip_find_status_t, ne_someip_error_code_t,
	void*);
// ne_someip_find_offer_service_spec_t 中的值必须是精确值
typedef void (*ne_someip_service_available_handler)(const ne_someip_find_offer_service_spec_t*, ne_someip_service_status_t, void*);
typedef void (*ne_someip_offer_status_handler)(ne_someip_provided_instance_t*, ne_someip_offer_status_t,
	ne_someip_error_code_t, void*);
typedef void (*ne_someip_subscribe_status_handler)(ne_someip_required_service_instance_t*, ne_someip_eventgroup_id_t,
	ne_someip_subscribe_status_t, ne_someip_error_code_t, void*);

//the second parameter is seq_id, the last is user_data
typedef void (*ne_someip_send_event_status_handler)(ne_someip_provided_instance_t*, void*, ne_someip_event_id_t,
	ne_someip_error_code_t, void*);
typedef void (*ne_someip_send_req_status_handler)(ne_someip_required_service_instance_t*, void*, ne_someip_method_id_t,
	ne_someip_error_code_t, void*);
typedef void (*ne_someip_send_resp_status_handler)(ne_someip_provided_instance_t*, void*, ne_someip_method_id_t,
	ne_someip_error_code_t, void*);

typedef void (*ne_someip_recv_request_handler)(ne_someip_provided_instance_t*, ne_someip_header_t*, ne_someip_payload_t*,
	ne_someip_remote_client_info_t*, void*);
typedef void (*ne_someip_recv_response_handler)(ne_someip_required_service_instance_t*, ne_someip_header_t*, ne_someip_payload_t*, void*);
typedef void (*ne_someip_recv_event_handler)(ne_someip_required_service_instance_t*, ne_someip_header_t*, ne_someip_payload_t*, void*);
typedef void (*ne_someip_recv_subscribe_handler)(ne_someip_provided_instance_t*, const ne_someip_eventgroup_id_list_t*,
	ne_someip_remote_client_info_t*, void*);

#endif // INCLUDE_NE_SOMEIP_HANDLER_H
/* EOF */
