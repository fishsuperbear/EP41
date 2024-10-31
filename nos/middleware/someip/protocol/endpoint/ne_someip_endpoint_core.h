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
#ifndef SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_CORE_H
#define SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_looper.h"
#include "ne_someip_endpoint_define.h"

/*
 * @brief create ne_someip_endpoint_core_t object，the original reference counting of ne_someip_endpoint_core_t
 *  object is 1. (The function was called in work thread.)
 *
 * @param [in] looper : The looper of the IO thread.
 *
 * @return 返回ne_someip_unix_data_endpoint_t对象指针.
 *
 * @attention Synchronous I/F.
 */

ne_someip_endpoint_core_t* ne_someip_ep_core_new(ne_someip_looper_t* looper, ne_someip_endpoint_type_t ep_type);
ne_someip_endpoint_core_t* ne_someip_ep_core_ref(ne_someip_endpoint_core_t* core);
void ne_someip_ep_core_unref(ne_someip_endpoint_core_t* core);
// The function will called when ne_someip_xxx_endpoint_create() is called.
ne_someip_error_code_t ne_someip_ep_core_start(void* ep, const ne_someip_ssl_key_info_t* key_info);
// The function will called when ne_someip_xxx_endpoint_free() is called.
ne_someip_error_code_t ne_someip_ep_core_stop(void* ep);

ne_someip_error_code_t ne_someip_ep_core_send(void* endpoint, ne_someip_trans_buffer_struct_t* trans_buffer, void* peer_addr,
    ne_someip_endpoint_send_policy_t* policy, const void* seq_data);
ne_someip_error_code_t ne_someip_ep_core_send_tp_data(void* ep, ne_someip_trans_buffer_struct_t* trans_buffer, ne_someip_list_t* tp_data,
	ne_someip_endpoint_net_addr_t* peer_addr, ne_someip_endpoint_send_policy_t* policy, const void* seq_data);

ne_someip_error_code_t ne_someip_ep_core_delay_send_ontimer(void* endpoint, ne_someip_trans_buffer_struct_t* trans_buffer,
	void* peer_addr, const void* seq_data);
ne_someip_error_code_t ne_someip_ep_core_udp_collect_send_ontimer(void* endpoint, ne_someip_trans_buffer_struct_t* trans_buffer,
	void* peer_addr, const void* seq_data);
ne_someip_error_code_t ne_someip_ep_core_send_tp_data_ontimer(void* ep, ne_someip_endpoint_udp_iov_cache_t* tp_data);
void ne_someip_ep_core_async_send_reply(void* endpoint, const void* seq_data, ne_someip_error_code_t result);

void ne_someip_ep_core_free_buffer_cache_for_link(void* endpoint, void* peer_addr);
bool ne_someip_ep_core_is_cache_saved(void* endpoint, void* peer_addr);
void ne_someip_ep_core_trigger_send_cache_data(void* endpoint, ne_someip_transmit_t* transmit,
    ne_someip_transmit_link_t *link, void* peer_addr);

// ne_someip_transmit_buffer_t* ne_someip_ep_core_alloc_memory(void* endpoint, const void* peer_address);
// ne_someip_error_code_t ne_someip_ep_core_receive(void* endpoint, const void* peer_address);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_CORE_H
/* EOF */