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
#ifndef EXTEND_TP_NE_SOMEIP_TP_H_
#define EXTEND_TP_NE_SOMEIP_TP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_list.h"
#include "ne_someip_map.h"
#include "ne_someip_sync_obj.h"
#include "ne_someip_tp_define.h"
#include "ne_someip_endpoint_define.h"

// Send
bool ne_someip_tp_is_segment_needed(const ne_someip_trans_buffer_struct_t* buffer, uint32_t segment_length);
// ne_someip_list_t: <ne_someip_trans_buffer_struct_t*>
ne_someip_list_t* ne_someip_tp_segment_send_msg(const ne_someip_trans_buffer_struct_t* buffer, uint32_t segment_length);
void ne_someip_tp_free_buffer_struct(ne_someip_trans_buffer_struct_t* buffer);
void ne_someip_tp_free_buffer_struct_no_header_data(ne_someip_trans_buffer_struct_t* buffer);
void ne_someip_tp_free_buffer_struct_header_data(ne_someip_trans_buffer_struct_t* buffer);

// Receive
ne_someip_tp_ctx_t* ne_someip_tp_init();
void ne_someip_tp_deinit(ne_someip_tp_ctx_t* tp_ctx);
bool ne_someip_tp_set_config_info(ne_someip_tp_ctx_t* tp_ctx, ne_someip_tp_config_info_t config_info);
ne_someip_tp_config_info_t ne_someip_tp_get_config_info(ne_someip_tp_ctx_t* tp_ctx);
ne_someip_tp_reassembe_result_t ne_someip_tp_can_data_notify_to_upper(ne_someip_tp_ctx_t* tp_ctx,
    ne_someip_endpoint_net_addr_pair_t* pair_addr, ne_someip_trans_buffer_struct_t* in_trans_buffer,
    ne_someip_trans_buffer_struct_t** out_trans_buffer);

#ifdef __cplusplus
}
#endif
#endif  // EXTEND_TP_NE_SOMEIP_TP_H_
/* EOF */
