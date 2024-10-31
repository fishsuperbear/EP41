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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_CLIENT_TOOL_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_CLIENT_TOOL_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "stdint.h"

/************************hash function****************************/
uint32_t ne_someip_client_network_config_hash_func(const void* key);
uint32_t ne_someip_client_uint16_hash_func(const void* key);
uint32_t ne_someip_client_pthread_t_hash_func(const void* key);
uint32_t ne_someip_client_saved_req_seq_hasn_func(const void* key);

/************************compare function****************************/
int ne_someip_client_find_offer_spec_fuzzy_compare(const void* k1, const void* k2);
int ne_someip_client_find_offer_spec_exact_compare(const void* k1, const void* k2);
int ne_someip_client_serv_inst_spec_exact_compare(const void* k1, const void* k2);
int ne_someip_client_uint32_compare(const void* k1, const void* k2);
int ne_someip_client_network_config_compare(const void* k1, const void* k2);
int ne_someip_client_pthread_t_compare(const void* k1, const void* k2);
int ne_someip_client_saved_req_seq_compare(const void* k1, const void* k2);

int ne_someip_client_network_info_compare(const void* net_config, const void* net_notify);

/************************free function****************************/
void ne_someip_client_client_id_info_free(void* data);
void ne_someip_client_find_offer_spec_free(void* data);
void ne_someip_client_find_offer_spec_list_free(void* data);
void ne_someip_client_find_offer_services_free(void* data);
void ne_someip_client_serv_inst_spec_free(void* data);
void ne_someip_client_client_id_list_free(void* data);
void ne_someip_client_uint32_free(void* data);
void ne_someip_client_saved_find_handler_free(void* data);
void ne_someip_client_saved_avail_handler_free(void* data);
void ne_someip_client_saved_sub_status_handler_free(void* data);
void ne_someip_client_saved_recv_event_handler_free(void* data);
void ne_someip_client_saved_recv_resp_handler_free(void* data);
void ne_someip_client_saved_send_handler_free(void* data);
void ne_someip_client_context_find_local_service_free(void* data);
void ne_someip_client_pthread_t_free(void* data);
void ne_someip_client_sync_wait_obj_free(void* data);

/*******************************************log print*****************************************/
void ne_someip_client_printf_instance_info(const void* instance,
	const char* file, uint16_t line, const char* fun);
void ne_someip_client_printf_instance_debug(const void* instance,
	const char* file, uint16_t line, const char* fun);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_CLIENT_TOOL_H
/* EOF */
