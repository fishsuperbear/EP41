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
#include "ne_someip_client_tool.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_client_define.h"
#include "ne_someip_endpoint_tool.h"
#include "ne_someip_config_define.h"
#include "ne_someip_sync_wait_obj.h"
#include "ne_someip_required_service_instance.h"
#include "ne_someip_log.h"

/************************hash function****************************/
uint32_t ne_someip_client_network_config_hash_func(const void* key)
{
    if (NULL == key) {
        return -1;
    }

    ne_someip_network_config_t* config = (ne_someip_network_config_t*)key;
    return ne_someip_map_string_hash_func((const void*)(config->network_id));
}

uint32_t ne_someip_client_uint16_hash_func(const void* key)
{
    if (!key) {
        return 0;
    }
    return (uint32_t)(*((uint16_t*)key) & 0x0000FFFF);
}

uint32_t ne_someip_client_pthread_t_hash_func(const void* key)
{
    if (!key) {
        return 0;
    }
    return (uint32_t)(*(pthread_t*)key);
}

uint32_t ne_someip_client_saved_req_seq_hasn_func(const void* key) {
    if (!key) {
        return 0;
    }
    ne_someip_saved_req_seq_info_t* tmp_key = key;
    return (uint32_t)(tmp_key->method_id + tmp_key->session_id);
}

/************************compare function****************************/
int ne_someip_client_find_offer_spec_fuzzy_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    ne_someip_find_offer_service_spec_t* k1_tmp = (ne_someip_find_offer_service_spec_t*)k1;
    ne_someip_find_offer_service_spec_t* k2_tmp = (ne_someip_find_offer_service_spec_t*)k2;

    if ((k1_tmp->ins_spec.service_id == k2_tmp->ins_spec.service_id || NE_SOMEIP_ANY_SERVICE == k1_tmp->ins_spec.service_id ||
    	NE_SOMEIP_ANY_SERVICE == k2_tmp->ins_spec.service_id) && (k1_tmp->ins_spec.instance_id == k2_tmp->ins_spec.instance_id ||
    	NE_SOMEIP_ANY_INSTANCE == k1_tmp->ins_spec.instance_id || NE_SOMEIP_ANY_INSTANCE == k2_tmp->ins_spec.instance_id) &&
        (k1_tmp->ins_spec.major_version == k2_tmp->ins_spec.major_version || NE_SOMEIP_ANY_MAJOR == k1_tmp->ins_spec.major_version ||
        NE_SOMEIP_ANY_MAJOR == k2_tmp->ins_spec.major_version) && (k1_tmp->minor_version == k2_tmp->minor_version ||
        NE_SOMEIP_ANY_MINOR == k1_tmp->minor_version || NE_SOMEIP_ANY_MINOR == k2_tmp->minor_version)) {
    	return 0;
    }

    return -1;
}

int ne_someip_client_find_offer_spec_exact_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    ne_someip_find_offer_service_spec_t* k1_tmp = (ne_someip_find_offer_service_spec_t*)k1;
    ne_someip_find_offer_service_spec_t* k2_tmp = (ne_someip_find_offer_service_spec_t*)k2;

    if ((k1_tmp->ins_spec.service_id == k2_tmp->ins_spec.service_id) && (k1_tmp->ins_spec.instance_id == k2_tmp->ins_spec.instance_id) &&
        (k1_tmp->ins_spec.major_version == k2_tmp->ins_spec.major_version) && (k1_tmp->minor_version == k2_tmp->minor_version)) {
    	return 0;
    }

    return -1;
}

int ne_someip_client_serv_inst_spec_exact_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    ne_someip_service_instance_spec_t* k1_tmp = (ne_someip_service_instance_spec_t*)k1;
    ne_someip_service_instance_spec_t* k2_tmp = (ne_someip_service_instance_spec_t*)k2;

    if ((k1_tmp->service_id == k2_tmp->service_id) && (k1_tmp->instance_id == k2_tmp->instance_id) &&
        (k1_tmp->major_version == k2_tmp->major_version)) {
        return 0;
    }
    return -1;
}

int ne_someip_client_uint32_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    uint32_t* k1_tmp = (uint32_t*)k1;
    uint32_t* k2_tmp = (uint32_t*)k2;

    if (*k1_tmp == *k2_tmp) {
        return 0;
    }
    return -1;
}

int ne_someip_client_network_config_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    ne_someip_network_config_t* k1_tmp = (ne_someip_network_config_t*)k1;
    ne_someip_network_config_t* k2_tmp = (ne_someip_network_config_t*)k2;
    int num = memcmp(k1_tmp->network_id, k2_tmp->network_id, NE_SOMEIP_SHORT_NAME_LENGTH);
    if (0 != num) {
        return -1;
    }
    return 0;
}

int ne_someip_client_pthread_t_compare(const void* k1, const void* k2)
{
    if (NULL == k1 && NULL == k2) {
        return -1;
    }
    if ((NULL == k1 && NULL != k2) || (NULL != k1 && NULL == k2)) {
        return -1;
    }

    pthread_t* k1_tmp = (pthread_t*)k1;
    pthread_t* k2_tmp = (pthread_t*)k2;

    if (*k1_tmp == *k2_tmp) {
        return 0;
    }

    return -1;
}

int ne_someip_client_saved_req_seq_compare(const void* k1, const void* k2) {
    if (NULL == k1 || NULL == k2) {
        return -1;
    }

    ne_someip_saved_req_seq_info_t* k1_tmp = k1;
    ne_someip_saved_req_seq_info_t* k2_tmp = k2;
    if (k1_tmp->method_id == k2_tmp->method_id &&
        k1_tmp->session_id == k2_tmp->session_id) {
        return 0;
    }

    return -1;
}

int ne_someip_client_network_info_compare(const void* net_config, const void* net_notify)
{
    if (NULL == net_config && NULL == net_notify) {
        return -1;
    }
    if ((NULL == net_config && NULL != net_notify) || (NULL != net_config && NULL == net_notify)) {
        return -1;
    }

    ne_someip_network_config_t* tmp_net_config = (ne_someip_network_config_t*)net_config;
    ne_someip_net_status_notify_data_t* tmp_net_notify = (ne_someip_net_status_notify_data_t*)net_notify;
    if (0 != tmp_net_config->ip_addr) {
        if (tmp_net_config->ip_addr != tmp_net_notify->ip_addr) {
            return -1;
        } else {
            return 0;
        }
    } else {
        if (0 == strcmp(tmp_net_config->if_name, tmp_net_notify->if_name)) {
            return 0;
        } else {
            return -1;
        }
    }

}

/************************free function****************************/
void ne_someip_client_client_id_info_free(void* data)
{
    if (NULL != data) {
        free((ne_someip_client_daemon_client_id_info_t*)data);
    }
}

void ne_someip_client_find_offer_spec_free(void* data)
{
   if (NULL != data) {
    	free((ne_someip_find_offer_service_spec_t*)data);
    }
}

void ne_someip_client_find_offer_spec_list_free(void* data)
{
    if (NULL != data) {
        ne_someip_list_destroy((ne_someip_list_t*)data, ne_someip_client_find_offer_spec_free);
    }
}

void ne_someip_client_find_offer_services_free(void* data)
{
    if (NULL != data) {
        free((ne_someip_client_find_local_services_t*)data);
    }
}

void ne_someip_client_serv_inst_spec_free(void* data)
{
    if (NULL != data) {
        free((ne_someip_service_instance_spec_t*)data);
    }
}

void ne_someip_client_client_id_list_free(void* data)
{
    if (NULL != data) {
    	ne_someip_list_destroy((ne_someip_list_t*)data, ne_someip_ep_tool_uint16_free);
    }
}

void ne_someip_client_uint32_free(void* data)
{
    if (NULL != data) {
        free((uint32_t*)data);
    }
}

void ne_someip_client_saved_find_handler_free(void* data)
{
    if (NULL != data) {
        free((ne_someip_saved_find_handler_t*)data);
    }
}

void ne_someip_client_saved_avail_handler_free(void* data)
{
    if (NULL != data) {
        free((ne_someip_saved_available_handler_t*)data);
    }
}

void ne_someip_client_saved_sub_status_handler_free(void* data)
{
    if (NULL != data) {
        free((ne_someip_saved_subscribe_status_handler_t*)data);
    }
}

void ne_someip_client_saved_recv_event_handler_free(void* data)
{
    if (NULL != data) {
        free((ne_someip_saved_recv_event_handler_t*)data);
    }
}

void ne_someip_client_saved_recv_resp_handler_free(void* data)
{
    if (NULL != data) {
        free((ne_someip_saved_recv_response_handler_t*)data);
    }
}

void ne_someip_client_saved_send_handler_free(void* data)
{
    if (NULL != data) {
        free((ne_someip_saved_send_status_handler_t*)data);
    }
}

void ne_someip_client_context_find_local_service_free(void* data)
{
    if (NULL == data) {
        return;
    }

    free((ne_someip_find_offer_service_spec_t*)data);
    data = NULL;
}

void ne_someip_client_pthread_t_free(void* data)
{
    if (NULL == data) {
        return;
    }

    free((pthread_t*)data);
    data = NULL;
}

void ne_someip_client_sync_wait_obj_free(void* data)
{
    if (NULL == data) {
        return;
    }

    ne_someip_sync_wait_obj_destroy((ne_someip_sync_wait_obj_t*)data);
    data = NULL;
}

/*******************************************log print*****************************************/
void ne_someip_client_printf_instance_info(const void* instance,
    const char* file, uint16_t line, const char* fun)
{
    ne_someip_client_inter_config_t* config =
        ne_someip_req_serv_inst_get_inter_config((ne_someip_required_service_instance_t*)instance);
    if (NULL == config) {
        return;
    }
    ne_someip_log_info(
        "start, instance: service id [0x%x], instance_id [0x%x], major_version [0x%x], minor_version [0x%x]",
        config->service_id, config->instance_id,
        config->major_version, config->minor_version);
}

void ne_someip_client_printf_instance_debug(const void* instance,
    const char* file, uint16_t line, const char* fun)
{
    ne_someip_client_inter_config_t* config =
        ne_someip_req_serv_inst_get_inter_config((ne_someip_required_service_instance_t*)instance);
    if (NULL == config) {
        return;
    }
    ne_someip_log_info(
        "start, instance: service id [0x%x], instance_id [0x%x], major_version [0x%x], minor_version [0x%x]",
        config->service_id, config->instance_id,
        config->major_version, config->minor_version);
}
