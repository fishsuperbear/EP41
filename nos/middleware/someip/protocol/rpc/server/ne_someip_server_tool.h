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
#ifndef MANAGER_SERVER_NE_SOMEIP_SERVER_TOOL_H
#define MANAGER_SERVER_NE_SOMEIP_SERVER_TOOL_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_define.h"
#include "ne_someip_map.h"
#include "ne_someip_server_context.h"
#include "ne_someip_server_define.h"

/********************************************free fun****************************************/
void ne_someip_server_network_info_free(void* data);
void ne_someip_server_subscriber_free(void* data);
void ne_someip_server_remote_info_free(void* data);

/*********************************************cmp fun****************************************/
int ne_someip_uint16_cmp(void* key1, void* key2);
int ne_someip_ins_spec_cmp(void* key1, void* key2);
int ne_someip_subscriber_cmp(ne_someip_endpoint_net_addr_t* first_subsriber, ne_someip_endpoint_net_addr_t* second_subsriber);
uint32_t ne_someip_server_subscriber_addr_cmp(void* key1, void* key2);
int ne_someip_saved_event_resp_seq_cmp(void* key1, void* key2);
int ne_someip_forward_link_cmp(void* key1, void* key2);

/*********************************************hash func**************************************/
uint32_t ne_someip_p_ins_uint16_key_hash_fun(const void* key);
uint32_t ne_someip_server_subscriber_addr_hash_fun(const void* key);
uint32_t ne_someip_p_ins_saved_event_resp_seq_hasn_func(const void* key);
uint32_t ne_someip_p_ins_forward_link_hasn_func(const void* key);

/*******************************************handle map*****************************************/
void ne_someip_remove_all_map(ne_someip_map_t* map);

/*******************************************log print*****************************************/
uint32_t ne_someip_server_get_payload_len(const ne_someip_payload_t* payload);

#ifdef __cplusplus
}
#endif
#endif // MANAGER_SERVER_NE_SOMEIP_SERVER_TOOL_H
/* EOF */