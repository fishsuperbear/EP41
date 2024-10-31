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
#ifndef SRC_PROTOCOL_RPC_COMMON_REUSE_NE_SOMEIP_COMMON_REUSE_DEFINE_H
#define SRC_PROTOCOL_RPC_COMMON_REUSE_NE_SOMEIP_COMMON_REUSE_DEFINE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "ne_someip_map.h"
#include "ne_someip_list.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_endpoint_define.h"
#include "ne_someip_ipc_define.h"

typedef struct ne_someip_reg_method_key
{
    ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    // ne_someip_major_version_t major_version;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_port;
    ne_someip_l4_protocol_t protocol;
    ne_someip_method_id_t method_id;
}ne_someip_reg_method_key_t;

typedef struct ne_someip_reg_method_info
{
    ne_someip_endpoint_unix_addr_t proxy_unix_addr;
    bool is_resp_wait;
}ne_someip_reg_method_info_t;

typedef struct ne_someip_reg_event_key
{
    ne_someip_service_id_t service_id;
    // ne_someip_major_version_t major_version;
    ne_someip_address_type_t address_type;
    uint32_t source_address;
    uint16_t source_port;
    ne_someip_l4_protocol_t protocol;
    ne_someip_event_id_t event_id;
}ne_someip_reg_event_key_t;

typedef struct ne_someip_reg_sub_eg_key
{
    ne_someip_client_id_t client_id;
    ne_someip_instance_id_t instance_id;
}ne_someip_reg_sub_eg_key_t;

typedef struct ne_someip_reg_event_info
{
    ne_someip_map_t* sub_eg_map;  // <ne_someip_eventgroup_id_t*,
                                  // ne_someip_map_t*<ne_someip_reg_sub_eg_key_t*, ne_someip_list_t*<ne_someip_endpoint_unix_addr_t*>>>   
    ne_someip_map_t* reg_event_unix_map;  // <ne_someip_endpoint_unix_addr_t*, bool* is_event_sub>
}ne_someip_reg_event_info_t;

typedef struct ne_someip_send_rpc_reply_info
{
    ne_someip_ipc_rpc_msg_header_t reply;
    ne_someip_endpoint_unix_addr_t proxy_unix_addr;
} ne_someip_send_rpc_reply_info_t;

typedef struct ne_someip_reuse_subscribe_info
{
    uint8_t* data;
    uint32_t size;
}ne_someip_reuse_subscribe_info_t;


#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_COMMON_REUSE_NE_SOMEIP_COMMON_REUSE_DEFINE_H
/* EOF */
