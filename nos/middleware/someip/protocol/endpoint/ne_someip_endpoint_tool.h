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
#ifndef SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_TOOL_H
#define SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_TOOL_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "ne_someip_endpoint_define.h"

/***********************************hashMap compare function*************************************/
uint32_t ne_someip_ep_tool_unix_key_hash_func(const void* key);
uint32_t ne_someip_ep_tool_net_key_hash_func(const void* key);
uint32_t ne_someip_ep_tool_inst_spec_key_hash_func(const void* key);
uint32_t ne_someip_ep_tool_client_inst_spec_key_hash_func(const void* key);
uint32_t ne_someip_ep_tool_find_offer_spec_key_hash_func(const void* key);
uint32_t ne_someip_ep_tool_uint16_hash_func(const void* key);
int ne_someip_ep_tool_net_addr_compare(const void* k1, const void* k2);
int ne_someip_ep_tool_unix_addr_compare(const void* k1, const void* k2);
int ne_someip_ep_tool_service_instance_spec_compare(const void* k1, const void* k2);
int ne_someip_ep_tool_client_instance_spec_compare(const void* k1, const void* k2);
int ne_someip_ep_tool_int32_compare(const void* k1, const void* k2);
int ne_someip_ep_tool_uint16_compare(const void* k1, const void* k2);

/***********************************free function*************************************/
void ne_someip_ep_tool_net_addr_free(void* data);
void ne_someip_ep_tool_unix_addr_free(void* data);
void ne_someip_ep_tool_udp_receive_buff_list_free(void* data);
void ne_someip_ep_tool_service_instance_info_free(void* data);
void ne_someip_ep_tool_transmit_link_info_free(void* data);
void ne_someip_ep_tool_int32_free(void* data);
void ne_someip_ep_tool_uint16_free(void* data);
void ne_someip_ep_tool_error_code_free(void* data);
void ne_someip_ep_tool_service_instance_spec_free(void* data);
void ne_someip_ep_tool_client_instance_spec_free(void* data);

/***********************************free function for list*************************************/
void ne_someip_ep_tool_service_instance_info_list_free(void* data);
void ne_someip_ep_tool_send_cache_list_free(void* data);
void ne_someip_ep_tool_ep_send_cache_free(void* data);
void ne_someip_ep_tool_ep_send_cache_tp_free(void* data);
void ne_someip_ep_tool_tp_data_list_free(void* data);
void ne_someip_ep_tool_tp_data_list_all_free(void* data);
void ne_someip_ep_tool_tp_iov_data_list_free(void* data);
void ne_someip_ep_tool_trans_buffer_free(ne_someip_trans_buffer_struct_t* trans_buffer);  // delete buffer data
void ne_someip_ep_tool_trans_buffer_no_payload_free(ne_someip_trans_buffer_struct_t* trans_buffer);  // for instance
void ne_someip_ep_tool_trans_normal_buf_free(void* data);
void ne_someip_ep_tool_trans_normal_buf_complete_free(void* data);
void ne_someip_ep_tool_trans_normal_buf_list_free(void* data);  // just delete list and normal_buf struct, not delete buffer
void ne_someip_ep_tool_udp_iov_cache_free(void* data);
void ne_someip_ep_tool_udp_iov_cache_fail_free(void* data);
void ne_someip_ep_tool_udp_iov_cache_list_free(void* data);
void ne_someip_ep_tool_trans_iov_buf_free(void* data);
void ne_someip_ep_tool_trans_iov_buf_tp_free(void* data);
void ne_someip_ep_tool_trans_iov_buf_trig_fail_free(void* data);
void ne_someip_ep_tool_trans_iov_buf_complete_free(void* data);
void ne_someip_ep_tool_trans_iov_buf_list_free(void* data);
void ne_someip_ep_tool_ep_buf_list_free(void* data);
void ne_someip_ep_tool_ep_buf_list_self_free(void* data);
void ne_someip_ep_tool_ep_group_addr_free(void* data);
void ne_someip_ep_tool_ep_tp_cache_free(void* data);

uint32_t ne_someip_ep_tool_get_total_len(ne_someip_transmit_iov_buffer_t* buffer);
uint32_t ne_someip_ep_tool_get_trans_total_len(ne_someip_trans_buffer_struct_t* ep_trans_buffer);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_TOOL_H
/* EOF */
