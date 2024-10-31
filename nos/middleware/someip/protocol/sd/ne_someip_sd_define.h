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

#ifndef SERVICE_DISCOVERY_NE_SOMEIP_SD_DEFINE_H
#define SERVICE_DISCOVERY_NE_SOMEIP_SD_DEFINE_H

#include "ne_someip_config_define.h"
#include "ne_someip_list.h"
#include "ne_someip_looper.h"

#define NE_SOMEIP_SD_ENTRY_LEAST_LENGTH 16
#define NESOMEIP_SD_BASE_OPTION_LENGTH 4
#define NE_SOMEIP_SD_IP_OPTION_LENGTH 12
#define NE_SOMEIP_SD_IP_NUM 4
#define NE_SOMEIP_SD_IP_OPTION_DEFAULT_LENGTH 9

#define NE_SOMEIP_SD_MSG_SERVICE_ID 0xFFFF
#define NE_SOMEIP_SD_MSG_METHOD_ID 0x8100
#define NE_SOMEIP_SD_MSG_CLIENT_ID 0x0000
#define NE_SOMEIP_SD_MSG_PROTOCOL_VER 0x01
#define NE_SOMEIP_SD_MSG_INTERFACE_VER 0x01
#define NE_SOMEIP_SD_MSG_TYPE 0x02
#define NE_SOMEIP_SD_MSG_RET 0x00

#define PACK_DELAY_TIME 100//ms

#define NE_SOMEIP_SD_LEAST_LENGTH 28

#define NE_SOMEIP_SD_COMMON_ENTRY_LENGTH 12
#define NE_SOMEIP_SD_SERVICE_ENTRY_LENGTH 16
#define NE_SOMEIP_SD_EVENTGROUP_ENTRY_LENGTH 16
#define NE_SOMEIP_SD_IP_OPTION_LENGTH 12
#define NE_SOMEIP_SD_IPV4_OPTION_LENGTH 0x0009
#define NE_SOMEIP_SD_IPV6_OPTION_LENGTH 0x0015
#define NE_SOMEIP_SD_IP_PROTOCOL_UDP 0x11
#define NE_SOMEIP_SD_IP_PROTOCOL_TCP 0x6
#define NE_SOMEIP_SD_LOAD_BALANCING_LEN 8
#define NE_SOMEIP_SD_IPV6_LEN 136


#define NE_SOMEIP_SD_SESSION_ID_MAX 0xFFFF

#define NE_SOMEIP_SD_HEADER_MIN_LEN 8

#define NE_SOMEIP_IP_NUM 4
#define NE_SOMEIP_IP_MAX 255

#define NE_SOMEIP_SD_INITIAL_SEQ_ID 1

typedef struct ne_someip_sd_msg
{
	ne_someip_header_t header;
    uint8_t reboot_flag;
    uint8_t unicast_flag;
    uint32_t reserved;
    uint32_t entry_length;
    ne_someip_list_t* entry_list; //ne_someip_sd_entry_t
    uint32_t option_length;
    ne_someip_list_t* option_list;//ne_someip_sd_option_t
} ne_someip_sd_msg_t;

typedef enum ne_someip_sd_entry_type
{
	ne_someip_sd_entry_type_unknown = 0xFF,
	ne_someip_sd_entry_type_find = 0x00,
	ne_someip_sd_entry_type_offer = 0x01,
	ne_someip_sd_entry_type_subscribe = 0x06,
	ne_someip_sd_entry_type_subscribe_ack = 0x07,
} ne_someip_sd_entry_type_t;

typedef enum ne_someip_sd_option_type
{
	ne_someip_sd_option_type_unknown = 0x00,
	ne_someip_sd_option_type_configuration = 0x01,
	ne_someip_sd_option_type_load_balancing = 0x02,
	ne_someip_sd_option_type_ipv4_endpoint = 0x04,
	ne_someip_sd_option_type_ipv6_endpoint = 0x06,
	ne_someip_sd_option_type_ipv4_multicast = 0x14,
	ne_someip_sd_option_type_ipv6_multicast = 0x16,
	ne_someip_sd_option_type_ipv4_sd_endpoint = 0x24,
	ne_someip_sd_option_type_ipv6_sd_endpoint = 0x26,
} ne_someip_sd_option_type_t;

typedef enum ne_someip_sd_find_offer_phase_status
{
	ne_someip_sd_find_ffer_phase_status_down = 0x00,
	ne_someip_sd_find_offer_phase_status_initial = 0x01,
	ne_someip_sd_find_offer_phase_status_repetition = 0x02,
	ne_someip_sd_find_offer_phase_status_main = 0x03,
	ne_someip_sd_find_offer_phase_status_ne_someip_ttl_timeout = 0x04,
} ne_someip_sd_find_offer_phase_status_t;

typedef enum ne_someip_subscribe_phase_status
{
	ne_someip_subscribe_phase_status_down = 0x00,
	ne_someip_subscribe_phase_status_cycle = 0x00,
	ne_someip_subscribe_phase_status_ne_someip_ttl_timeout = 0x01,
} ne_someip_subscribe_phase_status_t;

//define the message type
typedef enum ne_someip_sd_msg_type
{
	ne_someip_sd_msg_type_unknown = 0x00,
	ne_someip_sd_msg_type_find_initial = 0x01,
	ne_someip_sd_msg_type_find_repetition = 0x02,
	ne_someip_sd_msg_type_find_main = 0x03,
	ne_someip_sd_msg_type_offer_initial = 0x04,
	ne_someip_sd_msg_type_offer_repetition = 0x05,
	ne_someip_sd_msg_type_offer_main = 0x06,
	ne_someip_sd_msg_type_subscribe = 0x07,
} ne_someip_sd_msg_type_t;

typedef enum ne_someip_sd_timer_type
{
	ne_someip_sd_timer_type_unknown = 0x00,
	ne_someip_sd_timer_type_find_initial = 0x01,
	ne_someip_sd_timer_type_find_repetition = 0x02,
	ne_someip_sd_timer_type_find_main = 0x03,
	ne_someip_sd_timer_type_offer_initial = 0x04,
	ne_someip_sd_timer_type_offer_repetition = 0x05,
	ne_someip_sd_timer_type_offer_main = 0x06,
	ne_someip_sd_timer_type_subscribe = 0x07,
	ne_someip_sd_timer_type_offer_ttl = 0x08,
	ne_someip_sd_timer_type_find_ttl = 0x09,
	ne_someip_sd_timer_type_subscribe_ttl = 0x0a,
	ne_someip_sd_timer_type_delay_offer = 0x0b,
	ne_someip_sd_timer_type_delay_subscribe = 0x0c,
	ne_someip_sd_timer_type_remote_service = 0x0d,
} ne_someip_sd_timer_type_t;

//define base entry
typedef struct ne_someip_sd_base_entry
{
	ne_someip_sd_entry_type_t type;
} ne_someip_sd_base_entry_t;

typedef struct ne_someip_sd_entry
{
	ne_someip_sd_entry_type_t type;
	uint8_t index1;
	uint8_t index2;
	uint8_t option_number1;
	uint8_t option_number2;
	ne_someip_service_id_t service_id;
	ne_someip_instance_id_t instance_id;
	ne_someip_major_version_t major_version;
	uint32_t ttl;
} ne_someip_sd_entry_t;

typedef struct ne_someip_sd_offer_find_entry
{
	ne_someip_sd_entry_type_t type;
	uint8_t index1;
	uint8_t index2;
	uint8_t option_number1;
	uint8_t option_number2;
	ne_someip_service_id_t service_id;
	ne_someip_instance_id_t instance_id;
	ne_someip_major_version_t major_version;
	ne_someip_minor_version_t minor_version;
	uint32_t ttl;
} ne_someip_sd_offer_find_entry_t;

typedef struct ne_someip_sd_subscribe_entry
{
	ne_someip_sd_entry_type_t type;
	uint8_t index1;
	uint8_t index2;
	uint8_t option_number1;
	uint8_t option_number2;
	ne_someip_service_id_t service_id;
	ne_someip_instance_id_t instance_id;
	ne_someip_major_version_t major_version;
	uint32_t ttl;
	uint16_t counter;
	ne_someip_eventgroup_id_t eventgroup_id;
} ne_someip_sd_subscribe_entry_t;

// define base option
typedef struct ne_someip_sd_base_option
{
	ne_someip_sd_option_type_t type;
} ne_someip_sd_base_option_t;

typedef struct ne_someip_sd_configuration_option
{
	ne_someip_sd_option_type_t type;
} ne_someip_sd_configuration_option_t;

typedef struct ne_someip_sd_load_balancing_option
{
	ne_someip_sd_option_type_t type;
	uint16_t index;
	uint16_t length;
	uint8_t reserved;
	uint16_t priority;
	uint16_t weight;
} ne_someip_sd_load_balancing_option_t;

typedef struct ne_someip_sd_ip_option
{
	ne_someip_sd_option_type_t type;
	uint16_t length;
	uint32_t ip_addr;
	uint8_t reserved;
	ne_someip_l4_protocol_t protocol;
	uint16_t port;
} ne_someip_sd_ip_option_t;

typedef struct ne_someip_sd_recv_find
{
	ne_someip_sd_option_type_t type;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_minor_version_t minor_version;
    uint32_t ttl;
} ne_someip_sd_recv_find_t;

typedef struct ne_someip_sd_recv_offer
{
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_minor_version_t minor_version;
    bool reliable_flag;
    bool unreliable_flag;
    uint16_t tcp_port;
    uint16_t udp_port;
    ne_someip_address_type_t addr_type;
    uint32_t service_addr;
    uint32_t ttl;
    uint32_t remote_sd_addr;
    uint16_t remote_sd_port;
} ne_someip_sd_recv_offer_t;

typedef struct ne_someip_sd_recv_sub_ack
{
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_eventgroup_id_t eventgroup_id;
    uint8_t counter;
    uint32_t ttl;
    bool multicast_flag;
    uint32_t multicast_addr;
    uint16_t multicast_port;
    bool is_ack;
} ne_someip_sd_recv_subscribe_ack_t;

typedef struct ne_someip_sd_recv_subscribe
{
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_eventgroup_id_t eventgroup_id;
    uint8_t counter;
    bool reliable_flag;
    bool unreliable_flag;
    uint16_t tcp_port;
    uint16_t udp_port;
    ne_someip_address_type_t addr_type;
    uint32_t client_addr;
    uint32_t ttl;
    // char remote_sd_addr[NESOMEIP_IP_ADDR_LENGTH];
    // uint16_t remote_sd_port;
    // char local_sd_addr[NESOMEIP_IP_ADDR_LENGTH];
    // uint16_t local_sd_port;
} ne_someip_sd_recv_subscribe_t;

// //define the user data for receive callback
// typedef struct ne_someip_sd_user_data
// {
// 	void* data;  //ne_someip_sd_offer_key_t, ne_someip_sd_offer_key_r_t, ne_someip_sd_find_key_t, ne_someip_sd_find_key_r_t, ne_someip_sd_subscribe_key_t
// } ne_someip_sd_user_data_t;

typedef struct ne_someip_sd_base_key
{
	ne_someip_sd_msg_type_t type;
} ne_someip_sd_base_key_t;

//define the offer key
typedef struct ne_someip_sd_offer_key
{
	ne_someip_sd_base_key_t base;
	ne_someip_server_offer_time_config_t offer_timer;
	uint32_t src_addr;
	uint16_t src_port;
	uint32_t dst_addr;
	uint16_t dst_port;
	ne_someip_looper_timer_t* timer;
} ne_someip_sd_offer_key_t;

typedef struct ne_someip_sd_offer_key_r
{
	ne_someip_sd_base_key_t base;
	uint32_t sequence_id;
	uint16_t counter;
	ne_someip_server_offer_time_config_t offer_timer;
	uint32_t src_addr;
	uint16_t src_port;
	uint32_t dst_addr;
	uint16_t dst_port;
	ne_someip_looper_timer_t* timer;
} ne_someip_sd_offer_key_r_t;

//define the find key
typedef struct ne_someip_sd_find_key
{
	ne_someip_sd_base_key_t base;
	ne_someip_client_find_time_config_t find_timer;
	uint32_t src_addr;
	uint16_t src_port;
	uint32_t dst_addr;
	uint16_t dst_port;
} ne_someip_sd_find_key_t;

typedef struct ne_someip_sd_find_key_r
{
	ne_someip_sd_base_key_t base;
	uint32_t sequence_id;
	uint32_t counter;
	ne_someip_client_find_time_config_t find_timer;
	uint32_t src_addr;
	uint16_t src_port;
	uint32_t dst_addr;
	uint16_t dst_port;
} ne_someip_sd_find_key_r_t;

//define the subscribe key
typedef struct ne_someip_sd_subscribe_key
{
	ne_someip_sd_base_key_t base;
	ne_someip_client_id_t client_id;
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
    ne_someip_list_t* eventgroup_list;//ne_someip_ipc_subscribe_eg_t
    uint32_t local_addr;
    uint16_t local_port;
	uint32_t counter;
} ne_someip_sd_subscribe_key_t;

typedef struct ne_someip_sd_send_session_map
{
	ne_someip_session_id_t session_id;
	bool is_reboot;
} ne_someip_sd_send_session_t;

typedef struct ne_someip_sd_session_key
{
	uint32_t local_addr;
	uint32_t remote_addr;
} ne_someip_sd_session_key_t;

typedef struct ne_someip_sd_network_status
{
	bool is_reboot;
	bool network_available;
} ne_someip_sd_network_status_t;

typedef struct ne_someip_sd_delay_send_offer_info
{
	uint32_t dst_ip;
	bool is_unicast;
	void* service;
}ne_someip_sd_delay_send_offer_info_t;

typedef struct ne_someip_sd_delay_send_sub_info
{
	uint32_t dst_ip;
	void* service;
}ne_someip_sd_delay_send_sub_info_t;

typedef struct ne_someip_sd_ins_ip_spec
{
	uint32_t ip_addr;
	ne_someip_service_instance_spec_t ins_spec;
}ne_someip_sd_ins_ip_spec_t;

#endif // SERVICE_DISCOVERY_NE_SOMEIP_SD_DEFINE_H
/* EOF */
