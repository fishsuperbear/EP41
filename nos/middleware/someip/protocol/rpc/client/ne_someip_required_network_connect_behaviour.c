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
#include "ne_someip_required_network_connect_behaviour.h"
#include "ne_someip_ipc_behaviour.h"
#include "ne_someip_endpoint_tool.h"
#include "ne_someip_client_tool.h"
#include "ne_someip_log.h"

ne_someip_required_network_connect_behaviour_t*
    ne_someip_req_network_connect_behaviour_new(const ne_someip_network_config_t* net_config)
{
    if (NULL == net_config) {
        ne_someip_log_error("net_config is NULL.");
        return NULL;
    }

	ne_someip_required_network_connect_behaviour_t* network_conn_behav =
        (ne_someip_required_network_connect_behaviour_t*)malloc(sizeof(ne_someip_required_network_connect_behaviour_t));
    if (NULL == network_conn_behav) {
        ne_someip_log_error("malloc ne_someip_required_network_connect_behaviour_t error.");
        return NULL;
    }
    memset(network_conn_behav, 0, sizeof(*network_conn_behav));
    network_conn_behav->config = net_config;
    network_conn_behav->network_status = ne_someip_network_states_unknown;
    network_conn_behav->ip_addr = net_config->ip_addr;
    network_conn_behav->find_behav_map = ne_someip_map_new(ne_someip_ep_tool_find_offer_spec_key_hash_func,
        ne_someip_client_find_offer_spec_exact_compare, ne_someip_client_find_offer_spec_free, NULL);
    if (NULL == network_conn_behav->find_behav_map) {
        ne_someip_log_error("ne_someip_map_new retun NULL.");
        ne_someip_req_network_connect_behaviour_free(network_conn_behav);
        return NULL;
    }

    return network_conn_behav;
}

void ne_someip_req_network_connect_behaviour_free(ne_someip_required_network_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }

    behaviour->config = NULL;
    if (NULL != behaviour->find_behav_map) {
        ne_someip_map_unref(behaviour->find_behav_map);
        behaviour->find_behav_map = NULL;
    }

    free(behaviour);
    behaviour = NULL;
}

ne_someip_network_config_t*
    ne_someip_req_network_connect_behav_net_config_get(ne_someip_required_network_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return NULL;
    }

    return behaviour->config;
}

void ne_someip_req_network_connect_behav_net_status_set(ne_someip_required_network_connect_behaviour_t* behaviour,
    ne_someip_network_states_t status)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }

	behaviour->network_status = status;
}

ne_someip_network_states_t ne_someip_req_network_connect_behav_net_status_get(ne_someip_required_network_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return ne_someip_network_states_down;
    }

    return behaviour->network_status;
}

void ne_someip_req_network_connect_behav_ip_set(ne_someip_required_network_connect_behaviour_t* behaviour,
    uint32_t ip_addr)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }

    behaviour->ip_addr = ip_addr;
}

uint32_t ne_someip_req_network_connect_behav_ip_get(ne_someip_required_network_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return 0;
    }

    return  behaviour->ip_addr;
}

ne_someip_error_code_t
    ne_someip_req_network_connect_behav_save_find_behav(ne_someip_required_network_connect_behaviour_t* behaviour,
    const ne_someip_find_offer_service_spec_t* service_spec, ne_someip_required_find_service_behaviour_t* find_behav)
{
    if (NULL == behaviour || NULL == service_spec || NULL == find_behav || NULL == behaviour->find_behav_map) {
        ne_someip_log_error("behaviour or service_spec or find_behav or behaviour->find_behav_map is NULL.");
        return ne_someip_error_code_failed;
    }

    ne_someip_find_offer_service_spec_t* key_spec =
        (ne_someip_find_offer_service_spec_t*)malloc(sizeof(ne_someip_find_offer_service_spec_t));
    memset(key_spec, 0, sizeof(ne_someip_find_offer_service_spec_t));
    memcpy(key_spec, service_spec, sizeof(ne_someip_find_offer_service_spec_t));
    bool map_ret = ne_someip_map_insert(behaviour->find_behav_map, (void*)key_spec, (void*)find_behav);
    if (map_ret) {
        return ne_someip_error_code_ok;
    } else {
        ne_someip_log_error("ne_someip_map_insert failed.");
        free(key_spec);
        return ne_someip_error_code_failed;
    }
}

void ne_someip_req_network_connect_behav_remove_specific_find_behav(ne_someip_required_network_connect_behaviour_t* behaviour,
    const ne_someip_find_offer_service_spec_t* service_spec)
{
    if (NULL == behaviour || NULL == service_spec || NULL == behaviour->find_behav_map) {
        ne_someip_log_error("behaviour or service_spec or behaviour->find_behav_map is NULL.");
        return;
    }

    ne_someip_map_remove(behaviour->find_behav_map, (void*)service_spec, true);
}

void ne_someip_req_network_connect_behav_remove_all_find_behav(ne_someip_required_network_connect_behaviour_t* behaviour)
{
    if (NULL == behaviour || NULL == behaviour->find_behav_map) {
        ne_someip_log_error("behaviour or behaviour->find_behav_map is NULL.");
        return;
    }

    ne_someip_map_remove_all(behaviour->find_behav_map);
}