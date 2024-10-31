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
#ifndef SERVICE_DISCOVERY_NE_SOMEIP_SD_H
#define SERVICE_DISCOVERY_NE_SOMEIP_SD_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_define.h"
#include "ne_someip_ipc_define.h"

//init
bool ne_someip_sd_init(ne_someip_looper_t* looper);

//deinit
void ne_someip_sd_deinit();

//define interface to offer
ne_someip_error_code_t ne_someip_sd_offer(ne_someip_ipc_send_offer_t* msg);

ne_someip_error_code_t ne_someip_sd_stop_offer(ne_someip_ipc_send_offer_t* msg);

//define interface to find
ne_someip_error_code_t ne_someip_sd_find(ne_someip_ipc_send_find_t* msg);

ne_someip_error_code_t ne_someip_sd_stop_find(ne_someip_ipc_send_find_t* msg);

//define interface to subscribe
ne_someip_error_code_t ne_someip_sd_subscribe(ne_someip_ipc_send_subscribe_t* msg);

ne_someip_error_code_t ne_someip_sd_stop_subscribe(ne_someip_ipc_send_subscribe_t* msg);

//define interface to send subscribe ack, ne_someip_ipc_send_subscribe_ack_t
ne_someip_error_code_t ne_someip_sd_subscribe_ack(ne_someip_ipc_send_subscribe_ack_t* msg);

void ne_someip_sd_notify_network_status(ne_someip_ipc_notify_network_status_t* msg);

bool ne_someip_sd_add_service_handler(ne_someip_ipc_reg_unreg_service_handler_t* ser_handler,
	ne_someip_endpoint_unix_addr_t* unix_path);

bool ne_someip_sd_remove_service_handler(ne_someip_ipc_reg_unreg_service_handler_t* ser_handler);

void ne_someip_sd_find_remote_services(ne_someip_ipc_find_remote_svs_t* msg, ne_someip_list_t* service_list);

void ne_someip_sd_unix_link_disconnect(ne_someip_endpoint_unix_addr_t* remote_unix_addr);

void ne_someip_sd_reset_session();

#ifdef __cplusplus
}
#endif
#endif //SERVICE_DISCOVERY_NE_SOMEIP_SD_H
/* EOF */