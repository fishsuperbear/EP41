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

#ifndef INCLUDE_NE_SOMEIP_CONFIG_DEFINE_H
#define INCLUDE_NE_SOMEIP_CONFIG_DEFINE_H

#include <stdbool.h>
#include "ne_someip_define.h"

#define NE_SOMEIP_DEFAULT_SEGMENT_LENGTH                1408
#define NE_SOMEIP_DEFAULT_SEPARATION_TIME               0
#define NE_SOMEIP_NETWORK_DEFAULT_TTL                   0xFF
#define NE_SOMEIP_SD_DEFAULT_TTL                        0xFFFFFF
#define NE_SOMEIP_SD_DEFAULT_INITIAL_DELAY_MIN          10
#define NE_SOMEIP_SD_DEFAULT_INITIAL_DELAY_MAX          50
#define NE_SOMEIP_SD_DEFAULT_REPETITIONS_BASE_DELAY     100
#define NE_SOMEIP_SD_DEFAULT_REPETITIONS_MAX            5
#define NE_SOMEIP_SD_DEFAULT_CYCLIC_OFFER_DELAY         3000
#define NE_SOMEIP_SD_DEFAULT_REQUEST_RESPONSE_DELAY_MIN 10
#define NE_SOMEIP_SD_DEFAULT_REQUEST_RESPONSE_DELAY_MAX 50
#define NE_SOMEIP_SD_DEFAULT_SUBSCRIBE_RETRY_TIMES      0
#define NE_SOMEIP_SD_DEFAULT_SUBSCRIBE_RETRY_DELAY      0
#define NE_SOMEIP_DEFAULT_SUBNET_MASK                   "255.255.255.0"
#define NE_SOMEIP_SHORT_NAME_LENGTH                     256
#define NE_SOMEIP_DEFAULT_MINOR_VERSION                 1
#define NE_SOMEIP_DEFAULT_LOAD_BALANCING_PRIORITY       0
#define NE_SOMEIP_DEFAULT_LOAD_BALANCING_WEIGHT         0
#define NE_SOMEIP_DEFAULT_OFFER_TIME_REFERENCE          "SomeipDefaultServerOfferTimeConfig"
#define NE_SOMEIP_DEFAULT_FIND_TIME_REFERENCE           "SomeipDefaultClientFindTimeConfig"
#define NE_SOMEIP_DEFAULT_SERVER_SUBSCRIBE_REFERENCE    "SomeipDefaultServerSubscribeConfig"
#define NE_SOMEIP_DEFAULT_CLIENT_SUBSCRIBE_REFERENCE    "SomeipDefaultClientSubscribeConfig"
#define NE_SOMEIP_TLS_NAME_LEN                     120

typedef enum ne_someip_serializer_type
{
    ne_someip_serializer_type_someip = 0x0,
    ne_someip_serializer_type_signal_base = 0x1,
}ne_someip_serializer_type_t;

typedef enum ne_someip_udp_collection_trigger
{
    ne_someip_udp_collection_trigger_unknown = 0xFF,
    ne_someip_udp_collection_trigger_always = 0x0,
    ne_someip_udp_collection_trigger_never = 0x1,
}ne_someip_udp_collection_trigger_t;

typedef struct ne_someip_event_config
{
    char short_name[NE_SOMEIP_SHORT_NAME_LENGTH];
    ne_someip_event_id_t event_id;
    ne_someip_serializer_type_t serializer_type;
    uint32_t max_segment_length;
    uint32_t separation_time;
    ne_someip_l4_protocol_t comm_type;
    bool is_field;
}ne_someip_event_config_t;

typedef struct ne_someip_method_config
{
    char short_name[NE_SOMEIP_SHORT_NAME_LENGTH];
    ne_someip_method_id_t method_id;
    uint32_t segment_length_request;
    uint32_t segment_length_response;
    uint32_t sepration_time_request;
    uint32_t sepration_time_response;
    ne_someip_l4_protocol_t comm_type;
    bool fire_and_forget;
}ne_someip_method_config_t;

typedef struct ne_someip_field_config
{
    char short_name[NE_SOMEIP_SHORT_NAME_LENGTH];
    ne_someip_event_config_t* notifier;
    ne_someip_method_config_t* getter;
    ne_someip_method_config_t* setter;
}ne_someip_field_config_t;

typedef struct ne_someip_event_config_array
{
    ne_someip_event_config_t* event_config_array;
    uint32_t event_array_num;
}ne_someip_event_config_array_t;

typedef struct ne_someip_event_ref_config_array
{
    ne_someip_event_config_t** event_config_array;
    uint32_t event_array_num;
}ne_someip_event_ref_config_array_t;

typedef struct ne_someip_method_config_array
{
    ne_someip_method_config_t* method_config_array;
    uint32_t method_array_num;
}ne_someip_method_config_array_t;

typedef struct ne_someip_field_config_array
{
    ne_someip_field_config_t* field_config_array;
    uint32_t field_array_num;
}ne_someip_field_config_array_t;

typedef struct ne_someip_eventgroup_config
{
    char short_name[NE_SOMEIP_SHORT_NAME_LENGTH];
    ne_someip_eventgroup_id_t eventgroup_id;
    ne_someip_event_ref_config_array_t events_array;
    ne_someip_l4_protocol_t comm_type;
}ne_someip_eventgroup_config_t;

typedef struct ne_someip_eventgroup_config_array
{
    ne_someip_eventgroup_config_t* eventgroup_config_array;
    uint32_t eventgroup_array_num;
}ne_someip_eventgroup_config_array_t;

typedef struct ne_someip_service_config
{
    char short_name[NE_SOMEIP_SHORT_NAME_LENGTH];
    ne_someip_service_id_t service_id;
    ne_someip_major_version_t major_version;
    ne_someip_minor_version_t minor_version;
    ne_someip_event_config_array_t events_config;
    ne_someip_method_config_array_t methods_config;
    ne_someip_field_config_array_t fields_config;
    ne_someip_eventgroup_config_array_t eventgroups_config;
    bool reliable_flag;
    bool unreliable_flag;
}ne_someip_service_config_t;

typedef struct ne_someip_service_config_array
{
    ne_someip_service_config_t* service_config;
    uint32_t num;
}ne_someip_service_config_array_t;

typedef struct ne_someip_network_config
{
    char network_id[NE_SOMEIP_SHORT_NAME_LENGTH];
    char if_name[NESOMEIP_IF_NAME_LENGTH];
    uint32_t ip_addr;
    ne_someip_address_type_t addr_type;
    uint32_t subnet_mask;
    uint16_t priority;
    uint8_t ttl;
    uint32_t multicast_ip;
    uint16_t multicast_port;
}ne_someip_network_config_t;

typedef struct ne_someip_network_config_array
{
    ne_someip_network_config_t* network_config;
    uint32_t num;
}ne_someip_network_config_array_t;

typedef struct ne_someip_server_offer_time_config
{
    char time_id[NE_SOMEIP_SHORT_NAME_LENGTH];
    uint16_t initial_delay_min;
    uint16_t initial_delay_max;
    uint16_t repetition_base_delay;
    uint16_t repetition_max;
    uint32_t ttl;
    uint32_t cyclic_offer_delay;
    uint16_t request_response_delay_max;
    uint16_t request_response_delay_min;
}ne_someip_server_offer_time_config_t;

typedef struct ne_someip_server_offer_time_config_array
{
    ne_someip_server_offer_time_config_t* offer_time_config;
    uint32_t num;
}ne_someip_server_offer_time_config_array_t;

typedef struct ne_someip_required_provided_method_config
{
    ne_someip_method_config_t* method_config;
    uint32_t udp_collection_buffer_timeout;
    ne_someip_udp_collection_trigger_t udp_collection_trigger;
}ne_someip_required_provided_method_config_t;

typedef struct ne_someip_required_provided_event_config
{
    ne_someip_event_config_t* event_config;
    uint32_t udp_collection_buffer_timeout;
    ne_someip_udp_collection_trigger_t udp_collection_trigger;
}ne_someip_required_provided_event_config_t;

typedef struct ne_someip_client_subscribe_time_config
{
    char time_id[NE_SOMEIP_SHORT_NAME_LENGTH];
    uint16_t request_response_delay_min;
    uint16_t request_response_delay_max;
    uint32_t retry_times;
    uint32_t retry_delay;
    uint32_t ttl;
}ne_someip_client_subscribe_time_config_t;

typedef struct ne_someip_client_subscribe_time_config_array
{
    ne_someip_client_subscribe_time_config_t* client_subscribe_time_config;
    uint32_t num;
}ne_someip_client_subscribe_time_config_array_t;

typedef struct ne_someip_server_subscribe_time_config
{
    char time_id[NE_SOMEIP_SHORT_NAME_LENGTH];
    uint16_t request_response_delay_min;
    uint16_t request_response_delay_max;
}ne_someip_server_subscribe_time_config_t;

typedef struct ne_someip_server_subscribe_time_config_array
{
    ne_someip_server_subscribe_time_config_t* server_subscribe_time_config;
    uint32_t num;
}ne_someip_server_subscribe_time_config_array_t;

typedef struct ne_someip_provided_eventgroup_config
{
    ne_someip_eventgroup_config_t* eventgroup_config;
    uint32_t multicast_ip_addr;
    uint16_t multicast_port;
    uint16_t threshold;
    ne_someip_server_subscribe_time_config_t* time;
}ne_someip_provided_eventgroup_config_t;

typedef struct ne_someip_required_provided_method_config_array {
    ne_someip_required_provided_method_config_t* method_config_array;
    uint32_t method_array_num;
}ne_someip_required_provided_method_config_array_t;

typedef struct ne_someip_required_provided_event_config_array {
    ne_someip_required_provided_event_config_t* event_config_array;
    uint32_t event_array_num;
}ne_someip_required_provided_event_config_array_t;

typedef struct ne_someip_provided_eventgroup_config_array {
    ne_someip_provided_eventgroup_config_t* eventgroup_config_array;
    uint32_t eventgroup_array_num;
}ne_someip_provided_eventgroup_config_array_t;

typedef struct ne_someip_tls_handshake_key {
    char algorithm_family[NE_SOMEIP_TLS_NAME_LEN];
    char algorithm_mode[NE_SOMEIP_TLS_NAME_LEN];
    char algorithm_second_family[NE_SOMEIP_TLS_NAME_LEN];
}ne_someip_tls_handshake_key_t;

typedef struct ne_someip_tls_handshake_key_array {
    ne_someip_tls_handshake_key_t* handshake_key;
    uint32_t handshake_key_num;
}ne_someip_tls_handshake_key_array_t;

typedef struct ne_someip_provided_service_instance_config
{
    char short_name[NE_SOMEIP_SHORT_NAME_LENGTH];
    ne_someip_instance_id_t instance_id;
    uint16_t load_balancing_priority;
    uint16_t load_balancing_weight;
    bool reliable_flag;
    bool unreliable_flag;
    uint16_t tcp_port;
    uint16_t udp_port;
    bool tcp_reuse;
    bool udp_reuse;
    uint32_t udp_collection;
    bool tcp_tls_flag;
    uint64_t tp_separation_time_usec;
    ne_someip_tls_handshake_key_array_t tcp_handshake_key_array;
    char tcp_private_key_name[NE_SOMEIP_TLS_NAME_LEN];
    char tcp_certification_name[NE_SOMEIP_TLS_NAME_LEN];
    bool udp_tls_flag;
    ne_someip_tls_handshake_key_array_t udp_handshake_key_array;
    char udp_private_key_name[NE_SOMEIP_TLS_NAME_LEN];
    char udp_certification_name[NE_SOMEIP_TLS_NAME_LEN];
    ne_someip_required_provided_method_config_array_t method_config_array;
    ne_someip_required_provided_event_config_array_t event_config_array;
    ne_someip_provided_eventgroup_config_array_t eventgroups_config_array;
    ne_someip_server_offer_time_config_t* offer_time;
    ne_someip_network_config_t* network_config;
    ne_someip_service_config_t* service_config;
}ne_someip_provided_service_instance_config_t;

typedef struct ne_someip_provided_service_instance_config_array
{
    ne_someip_provided_service_instance_config_t* provided_instance_config;
    uint32_t num;
}ne_someip_provided_service_instance_config_array_t;

typedef struct ne_someip_client_find_time_config
{
    char time_id[NE_SOMEIP_SHORT_NAME_LENGTH];
    uint16_t initial_delay_min;
    uint16_t initial_delay_max;
    uint16_t repetition_base_delay;
    uint16_t repetition_max;
    uint32_t ttl;
}ne_someip_client_find_time_config_t;

typedef struct ne_someip_client_find_time_config_array
{
    ne_someip_client_find_time_config_t* find_time_config;
    uint32_t num;
}ne_someip_client_find_time_config_array_t;

typedef struct ne_someip_required_eventgroup_config
{
    ne_someip_eventgroup_config_t* eventgroup_config;
    ne_someip_client_subscribe_time_config_t* time;
}ne_someip_required_eventgroup_config_t;

typedef struct ne_someip_required_eventgroup_config_array
{
    ne_someip_required_eventgroup_config_t* eventgroup_config_array;
    uint32_t eventgroup_array_num;
}ne_someip_required_eventgroup_config_array_t;

typedef struct ne_someip_required_service_instance_config
{
    char short_name[NE_SOMEIP_SHORT_NAME_LENGTH];
    ne_someip_instance_id_t instance_id;
    bool reliable_flag;
    bool unreliable_flag;
    uint16_t tcp_port;
    uint16_t udp_port;
    bool tcp_reuse;
    bool udp_reuse;
    uint64_t tp_separation_time_usec;
    uint32_t udp_collection;
    bool tcp_tls_flag;
    ne_someip_tls_handshake_key_array_t tcp_handshake_key_array;
    char tcp_private_key_name[NE_SOMEIP_TLS_NAME_LEN];
    char tcp_certification_name[NE_SOMEIP_TLS_NAME_LEN];
    bool udp_tls_flag;
    ne_someip_tls_handshake_key_array_t udp_handshake_key_array;
    char udp_private_key_name[NE_SOMEIP_TLS_NAME_LEN];
    char udp_certification_name[NE_SOMEIP_TLS_NAME_LEN];
    ne_someip_required_provided_method_config_array_t method_config_array;
    ne_someip_required_eventgroup_config_array_t eventgroups_config_array;
    ne_someip_client_find_time_config_t* find_time;
    ne_someip_network_config_t* network_config;
    ne_someip_service_config_t* service_config;
}ne_someip_required_service_instance_config_t;

typedef struct ne_someip_required_service_instance_config_array
{
    ne_someip_required_service_instance_config_t* required_instance_config;
    uint32_t num;
}ne_someip_required_service_instance_config_array_t;

typedef struct ne_someip_config
{
    ne_someip_service_config_array_t service_array;
    ne_someip_network_config_array_t network_array;
    ne_someip_server_offer_time_config_array_t offer_time_array;
    ne_someip_client_find_time_config_array_t find_time_array;
    ne_someip_server_subscribe_time_config_array_t server_sub_time_array;
    ne_someip_client_subscribe_time_config_array_t client_sub_time_array;
    ne_someip_provided_service_instance_config_array_t provided_instance_array;
    ne_someip_required_service_instance_config_array_t required_instance_array;
}ne_someip_config_t;

#endif // INCLUDE_NE_SOMEIP_CONFIG_DEFINE_H
/* EOF */
