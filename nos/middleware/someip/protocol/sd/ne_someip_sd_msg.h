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
#ifndef SERVICE_DISCOVERY_NE_SOMEIP_SD_MSG_H
#define SERVICE_DISCOVERY_NE_SOMEIP_SD_MSG_H

#include "ne_someip_list.h"
#include "ne_someip_define.h"
#include "ne_someip_sd_define.h"
#include "ne_someip_ipc_define.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

ne_someip_sd_msg_t* ne_someip_sd_create_msg();
void ne_someip_sd_destroy_msg(ne_someip_sd_msg_t* msg);

ne_someip_error_code_t
ne_someip_sd_create_offer_msg(ne_someip_sd_msg_t* msg, const ne_someip_list_t* data, uint32_t src_addr,
    uint32_t dst_addr, bool is_stop, bool reboot_flag, bool unicast_flag,
    ne_someip_session_id_t session_id);

ne_someip_error_code_t
ne_someip_sd_create_find_msg(ne_someip_sd_msg_t* msg, const ne_someip_list_t* data, uint32_t src_addr,
    uint32_t dst_addr, bool is_stop, bool reboot_flag, bool unicast_flag,
    ne_someip_session_id_t session_id);

ne_someip_error_code_t
ne_someip_sd_create_subscribe_msg(ne_someip_sd_msg_t* msg, const ne_someip_ipc_send_subscribe_t* subscribe,
    uint32_t src_addr, uint32_t dst_addr, bool is_stop, bool reboot_flag, bool unicast_flag,
    ne_someip_session_id_t session_id);

ne_someip_error_code_t
ne_someip_sd_create_subscribe_ack_msg(ne_someip_sd_msg_t* msg, const ne_someip_list_t* data, uint32_t src_addr,
    uint32_t dst_addr, bool is_ack, bool reboot_flag, bool unicast_flag,
    ne_someip_session_id_t session_id);

ne_someip_error_code_t ne_someip_sd_create_base_msg(ne_someip_sd_msg_t* msg);

ne_someip_error_code_t
ne_someip_sd_create_service_entry(ne_someip_sd_entry_type_t type, ne_someip_service_id_t service_id,
	ne_someip_instance_id_t instance_id, ne_someip_major_version_t major_version, ne_someip_minor_version_t minor_version,
    uint32_t ttl, ne_someip_sd_offer_find_entry_t* entry);

ne_someip_error_code_t 
ne_someip_sd_create_subscribe_entry(ne_someip_sd_entry_type_t type, ne_someip_service_id_t service_id,
	ne_someip_instance_id_t instance_id, ne_someip_major_version_t major_version, uint32_t ttl, uint8_t counter,
    ne_someip_eventgroup_id_t eventgroup_id, ne_someip_sd_subscribe_entry_t* entry);

ne_someip_error_code_t
ne_someip_sd_create_ip_option(ne_someip_sd_option_type_t type, uint32_t ip_addr,
	ne_someip_l4_protocol_t protocol, uint32_t port, ne_someip_sd_ip_option_t* option);

void* ne_someip_sd_find_entry(const ne_someip_sd_msg_t* msg, ne_someip_sd_entry_type_t entry_type,
	ne_someip_service_id_t service_id, ne_someip_instance_id_t instance_id, ne_someip_major_version_t major_version,
	ne_someip_minor_version_t minor_version, uint32_t ttl, uint16_t counter, ne_someip_eventgroup_id_t eventgroup_id);

void* ne_someip_sd_find_option(const ne_someip_sd_msg_t* msg, ne_someip_sd_option_type_t option_type,
	ne_someip_l4_protocol_t protocol, uint32_t ip_addr, uint16_t port);

void ne_someip_sd_add_option_for_entry(ne_someip_sd_entry_t* entry, uint16_t index);

uint16_t ne_someip_sd_get_option_index(const ne_someip_sd_msg_t* msg, ne_someip_sd_ip_option_t* option);

bool ne_someip_sd_chk_msg(const ne_someip_sd_msg_t* msg, const ne_someip_endpoint_net_addr_pair_t* net_addr);

void ne_someip_sd_msg_handle_sub_error(const ne_someip_sd_msg_t* msg, ne_someip_list_t* nack_list);

ne_someip_message_length_t ne_someip_sd_msg_get_length(const ne_someip_sd_msg_t* msg);

ne_someip_header_t* ne_someip_sd_msg_get_header(const ne_someip_sd_msg_t* msg);

bool ne_someip_sd_find_option_content(const ne_someip_list_t* options, ne_someip_sd_option_type_t type,
	ne_someip_l4_protocol_t protocol, uint32_t* addr, uint16_t* port);

#ifdef __cplusplus
}
#endif
#endif  // SERVICE_DISCOVERY_NE_SOMEIP_SD_TOOL_H
/* EOF */