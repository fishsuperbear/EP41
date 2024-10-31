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

#ifndef SERVICE_DISCOVERY_NE_SOMEIP_SD_TOOL_H
#define SERVICE_DISCOVERY_NE_SOMEIP_SD_TOOL_H

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

/*****************************************free*******************************/
void ne_someip_sd_free(void* data);
void ne_someip_sd_service_free(void* data);
void ne_someip_sd_timer_key_free(void* data);
void ne_someip_sd_timer_value_free(void* data);
void ne_someip_sd_subscribe_value_free(void* data);
void ne_someip_sd_endpoint_free(void* data);
void ne_someip_sd_remote_service_free(void* data);
void ne_someip_sd_remote_service_list_free(void* data);
void ne_someip_sd_ins_spec_free(void* data);
void ne_someip_sd_ins_sepc_ip_free(void* data);
void ne_someip_sd_ip_list_free(void* data);
void ne_someip_sd_subscribe_list_free(void* data);

/****************************************cmp*********************************/
int ne_someip_sd_offer_key_cmp(void* key1, void* key2);
int ne_someip_sd_offer_r_key_cmp(void* key1, void* key2);
int ne_someip_sd_find_key_cmp(void* key1, void* key2);
int ne_someip_sd_find_r_key_cmp(void* key1, void* key2);
int ne_someip_sd_subscribe_key_cmp(void* key1, void* key2);
int ne_someip_sd_session_key_cmp(void* key1, void* key2);
int ne_someip_sd_endpoint_key_cmp(void* key1, void* key2);
int ne_someip_sd_finished_subscribe_key_cmp(void* key1, void* key2);
int ne_someip_sd_service_key_cmp(void* key1, void* key2);
int ne_someip_sd_uint32_key_cmp(void* key1, void* key2);
int ne_someip_sd_ins_spec_cmp(void* key1, void* key2);
int ne_someip_sd_ins_sepc_ip_cmp(void* key1, void* key2);
int ne_someip_sd_uint16_cmp(void* key1, void* key2);

/****************************************hash fun**********************************/
uint32_t ne_someip_sd_offer_key_hash_fun(const void* key);
uint32_t ne_someip_sd_offer_r_key_hash_fun(const void* key);
uint32_t ne_someip_sd_find_key_hash_fun(const void* key);
uint32_t ne_someip_sd_find_r_key_hash_fun(const void* key);
uint32_t ne_someip_sd_subsribe_key_hash_fun(const void* key);
uint32_t ne_someip_sd_ins_spec_key_hash_fun(const void* key);
uint32_t ne_someip_sd_session_key_hash_fun(const void* key);
uint32_t ne_someip_sd_net_addr_key_hash_fun(const void* key);
uint32_t ne_someip_sd_service_handler_key_hash_fun(const void* key);
uint32_t ne_someip_sd_uint32_key_hash_fun(const void* key);
uint32_t ne_someip_sd_ins_sepc_ip_hash_fun(const void* key);
uint32_t ne_someip_sd_uint16_hash_fun(const void* key);

/************************************uint32 to char***************************/
void ne_someip_uint32_convert_to_char(uint32_t data1, char* data2);

void ne_someip_sd_convert_uint32_to_ip(const char* description, uint32_t ip, const char* file, uint16_t line,
	const char* fun);

void ne_someip_sd_convert_uint32_to_ip_t(const char* description, uint32_t src_ip, uint32_t dst_ip,
	const char* file, uint16_t line, const char* fun);

#ifdef __cplusplus
}
#endif
#endif  // SERVICE_DISCOVERY_NE_SOMEIP_SD_TOOL_H
/* EOF */