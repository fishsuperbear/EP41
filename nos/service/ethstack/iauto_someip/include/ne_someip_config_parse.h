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

#ifndef INCLUDE_NE_SOMEIP_CONFIG_PARSE_H
#define INCLUDE_NE_SOMEIP_CONFIG_PARSE_H

#ifdef  __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "ne_someip_config_define.h"

typedef struct cJSON cJSON;

// parse
ne_someip_config_t* ne_someip_config_parse_someip_config(const cJSON* object);
ne_someip_config_t* ne_someip_config_parse_someip_config_by_content(const char* content);
ne_someip_error_code_t ne_someip_config_parse_service_array_config(const cJSON* services_obj,
    ne_someip_service_config_array_t* service_array);
ne_someip_error_code_t ne_someip_config_parse_service_config(const cJSON* service_obj,
    ne_someip_service_config_t* service_config);
ne_someip_error_code_t ne_someip_config_parse_server_offer_time_array_config(const cJSON* times_obj,
    ne_someip_server_offer_time_config_array_t* time_array);
ne_someip_error_code_t ne_someip_config_parse_server_offer_time_config(const cJSON* time_obj,
    ne_someip_server_offer_time_config_t* time_config);
ne_someip_error_code_t ne_someip_config_parse_client_find_time_array_config(const cJSON* times_obj,
    ne_someip_client_find_time_config_array_t* time_array);
ne_someip_error_code_t ne_someip_config_parse_client_find_time_config(const cJSON* time_obj,
    ne_someip_client_find_time_config_t* time_config);
ne_someip_error_code_t ne_someip_config_parse_server_subscribe_time_array_config(const cJSON* times_obj,
    ne_someip_server_subscribe_time_config_array_t* time_array);
ne_someip_error_code_t ne_someip_config_parse_server_subscribe_time_config(const cJSON* time_obj,
    ne_someip_server_subscribe_time_config_t* time_config);
ne_someip_error_code_t ne_someip_config_parse_client_subscribe_time_array_config(const cJSON* times_obj,
    ne_someip_client_subscribe_time_config_array_t* time_array);
ne_someip_error_code_t ne_someip_config_parse_client_subscribe_time_config(const cJSON* time_obj,
    ne_someip_client_subscribe_time_config_t* time_config);
ne_someip_error_code_t ne_someip_config_parse_network_array_config(const cJSON* networks_obj,
    ne_someip_network_config_array_t* network_array);
ne_someip_error_code_t ne_someip_config_parse_network_config(const cJSON* network_obj,
    ne_someip_network_config_t* network_config);
ne_someip_error_code_t ne_someip_config_parse_tls_handshake_key_array_config(const cJSON* handshake_keys_obj,
    ne_someip_tls_handshake_key_array_t* handshake_key_array);
ne_someip_error_code_t ne_someip_config_parse_tls_handshake_key_config(const cJSON* handshake_key_obj,
    ne_someip_tls_handshake_key_t* handshake_key_config);
ne_someip_error_code_t ne_someip_config_parse_provided_instance_array_config(const cJSON* object,
    ne_someip_provided_service_instance_config_array_t* pro_instance_array,
    const ne_someip_service_config_array_t* service_array,
    const ne_someip_server_offer_time_config_array_t* offer_time_array,
    const ne_someip_network_config_array_t* network_array,
    const ne_someip_server_subscribe_time_config_array_t* server_sub_time_array);
ne_someip_error_code_t ne_someip_config_parse_provided_instance_config(const cJSON* object,
    ne_someip_provided_service_instance_config_t* pro_instance_config,
    const ne_someip_service_config_array_t* service_array,
    const ne_someip_server_offer_time_config_array_t* offer_time_array,
    const ne_someip_network_config_array_t* network_array,
    const ne_someip_server_subscribe_time_config_array_t* server_sub_time_array);
ne_someip_error_code_t ne_someip_config_parse_required_instances_array_config(const cJSON* object,
    ne_someip_required_service_instance_config_array_t* req_instance_array,
    const ne_someip_service_config_array_t* service_array,
    const ne_someip_client_find_time_config_array_t* find_time_array,
    const ne_someip_network_config_array_t* network_array,
    const ne_someip_client_subscribe_time_config_array_t* client_sub_time_array);
ne_someip_error_code_t ne_someip_config_parse_required_instances_config(const cJSON* object,
    ne_someip_required_service_instance_config_t* req_instance_config,
    const ne_someip_service_config_array_t* service_array,
    const ne_someip_client_find_time_config_array_t* find_time_array,
    const ne_someip_network_config_array_t* network_array,
    const ne_someip_client_subscribe_time_config_array_t* client_sub_time_array);

// release
void ne_someip_config_release_someip_config(ne_someip_config_t** someip_config);
void ne_someip_config_release_service_array_config(ne_someip_service_config_array_t* service_array);
void ne_someip_config_release_service_config(ne_someip_service_config_t* service_config);
void ne_someip_config_release_server_offer_time_array_config(ne_someip_server_offer_time_config_array_t* time_array);
void ne_someip_config_release_client_find_time_array_config(ne_someip_client_find_time_config_array_t* time_array);
void ne_someip_config_release_server_subscribe_time_array_config(ne_someip_server_subscribe_time_config_array_t* time_array);
void ne_someip_config_release_client_subscribe_time_array_config(ne_someip_client_subscribe_time_config_array_t* time_array);
void ne_someip_config_release_network_array_config(ne_someip_network_config_array_t* network_array);
void ne_someip_config_release_provided_instance_array_config(ne_someip_provided_service_instance_config_array_t* pro_instance_array);
void ne_someip_config_release_provided_instance_config(ne_someip_provided_service_instance_config_t* pro_instance_config);
void ne_someip_config_release_required_instances_array_config(ne_someip_required_service_instance_config_array_t* req_instance_array);
void ne_someip_config_release_required_instances_config(ne_someip_required_service_instance_config_t* req_instance_config);

// find
ne_someip_event_config_t* ne_someip_config_find_event(const ne_someip_service_config_t* service_config, ne_someip_event_id_t event_id);
ne_someip_method_config_t* ne_someip_config_find_method(const ne_someip_service_config_t* service_config, ne_someip_method_id_t method_id);
ne_someip_eventgroup_config_t* ne_someip_config_find_eventgroup(const ne_someip_service_config_t* service_config,
    ne_someip_eventgroup_id_t eventgroup_id);
ne_someip_service_config_t* ne_someip_config_find_service(const ne_someip_service_config_array_t* service_array,
    ne_someip_service_id_t service_id, ne_someip_major_version_t major_version);
ne_someip_server_offer_time_config_t* ne_someip_config_find_server_offer_time(char* offer_time_ref,
    const ne_someip_server_offer_time_config_array_t* offer_time_array);
ne_someip_client_find_time_config_t* ne_someip_config_find_client_find_time(char* find_time_ref,
    const ne_someip_client_find_time_config_array_t* find_time_array);
ne_someip_network_config_t* ne_someip_config_find_network(char* ethernet_ref,
    const ne_someip_network_config_array_t* network_array);
ne_someip_server_subscribe_time_config_t* ne_someip_config_find_server_subscribe_time(char* subscribe_time_ref,
    const ne_someip_server_subscribe_time_config_array_t* server_sub_time_array);
ne_someip_client_subscribe_time_config_t* ne_someip_config_find_client_subscribe_time(char* subscribe_time_ref,
    const ne_someip_client_subscribe_time_config_array_t* client_sub_time_array);

// printf
void ne_someip_config_printf_someip_config(const ne_someip_config_t* someip_config);
void ne_someip_config_printf_service_array_config(const ne_someip_service_config_array_t* service_array);
void ne_someip_config_printf_service_config(const ne_someip_service_config_t* service_config);
void ne_someip_config_printf_server_offer_time_array_config(const ne_someip_server_offer_time_config_array_t* time_array);
void ne_someip_config_printf_client_find_time_array_config(const ne_someip_client_find_time_config_array_t* time_array);
void ne_someip_config_printf_server_subscribe_time_array_config(const ne_someip_server_subscribe_time_config_array_t* time_array);
void ne_someip_config_printf_client_subscribe_time_array_config(const ne_someip_client_subscribe_time_config_array_t* time_array);
void ne_someip_config_printf_network_array_config(const ne_someip_network_config_array_t* network_array);
void ne_someip_config_printf_provided_instance_array_config(const ne_someip_provided_service_instance_config_array_t* pro_instance_array);
void ne_someip_config_printf_provided_instance_config(const ne_someip_provided_service_instance_config_t* config);
void ne_someip_config_printf_required_instances_array_config(const ne_someip_required_service_instance_config_array_t* req_instance_array);
void ne_someip_config_printf_required_instances_config(const ne_someip_required_service_instance_config_t* config);

#ifdef __cplusplus
}
#endif
#endif // INCLUDE_NE_SOMEIP_CONFIG_PARSE_H
/* EOF */
