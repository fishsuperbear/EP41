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
#ifndef EXTEND_TP_NE_SOMEIP_TP_DEFINE_H_
#define EXTEND_TP_NE_SOMEIP_TP_DEFINE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "ne_someip_define.h"

// default value of SOME/IP TP config info
static const uint32_t NESOMEIP_TP_BUFFER_LEN = 14000;
static const uint32_t NESOMEIP_TP_TIMER_INTERVAL = 2000;  // 2s
static const uint32_t NESOMEIP_TP_MAX_PARALLER_BUFFER_NUM = 5;

static const uint32_t NESOMEIP_HEADER_SIZE = 16;
static const uint32_t NESOMEIP_IN_LEN_HEADER_SIZE = 8;
static const uint32_t NESOMEIP_FINAL_TP_HEADER_SIZE = 20;
static const uint32_t NESOMEIP_TP_HEADER_SIZE = 4;
static const uint32_t NESOMEIP_TP_IN_LEN_HEADER_SIZE = 12;
static const uint32_t NESOMEIP_TP_MAX_SEGMENT_SIZE = 1392;
static const uint32_t NESOMEIP_TP_OFFSET = 0xFFFFFFF0;
static const uint32_t NESOMEIP_TP_MORE_SEG_FLAG = 1;
static const uint32_t NESOMEIP_TP_MAX_UINT32_LEN = 0xFFFFFFFF;

#define NESOMEIP_TP_TIMER_THREAD_NAME "someip_tp_timer"

// define the result of TP reassembe result
typedef enum ne_someip_tp_reassembe_result {
    ne_someip_tp_reassembe_result_success = 0x00,
    ne_someip_tp_reassembe_result_failed = 0x01,
    ne_someip_tp_reassembe_result_wait = 0x02,
    ne_someip_tp_reassembe_result_no_need = 0x03,
} ne_someip_tp_reassembe_result_t;

// define the order of received segment message
typedef enum ne_someip_tp_reassembe_order {
    ne_someip_tp_reassembe_order_unknown = 0x00,
    ne_someip_tp_reassembe_order_ascend = 0x01,
    ne_someip_tp_reassembe_order_descend = 0x02,
} ne_someip_tp_reassembe_order_t;

// TP config info
typedef struct ne_someip_tp_config_info {
    uint32_t buffer_len;
    uint32_t timer_interval;
    uint32_t max_paraller_buffer_num;
} ne_someip_tp_config_info_t;

// header info for TP segment received
typedef struct ne_someip_tp_segment_header_info {
    ne_someip_address_type_t addr_type;
    uint32_t local_addr;
    uint16_t local_port;
    uint32_t remote_addr;
    uint16_t remote_port;
    uint32_t message_id;
    uint32_t request_id;
    uint8_t protocol_version;
    uint8_t interface_version;
    uint8_t message_type;
    uint8_t return_code;
} ne_someip_tp_segment_header_info_t;

// information of buffer that used to reassembe recevied data(include TP header info)
typedef struct ne_someip_tp_buffer_data {
    // char* buffer;
    ne_someip_endpoint_buffer_t* someip_header;
    ne_someip_list_t* payload_list;  // <ne_someip_payload_t*>
    ne_someip_tp_reassembe_order_t reassembe_order;
    uint32_t offset;
    uint32_t valid_payload_buffers;
    uint8_t more_segment_flag;
} ne_someip_tp_buffer_data_t;

typedef struct ne_someip_tp_ctx
{
    uint32_t reassembe_buffer_id;
    ne_someip_tp_config_info_t config_info;
    // ne_someip_sync_obj_t* segment_info_sync;
    ne_someip_map_t* segment_info;         // <uint32_t buffer_id, ne_someip_tp_segment_header_info_t>
    // ne_someip_sync_obj_t* reassembe_buffer_sync;
    ne_someip_map_t* reassembe_buffer;     // <uint32_t buffer_id, ne_someip_tp_buffer_data_t>
    // ne_someip_sync_obj_t* buffer_timer_sync;
    ne_someip_map_t* buffer_timer;         // <uint32_t buffer_id, ne_someip_looper_timer_t*>
} ne_someip_tp_ctx_t;

#ifdef __cplusplus
}
#endif
#endif  // EXTEND_TP_NE_SOMEIP_TP_DEFINE_H_
/* EOF */
