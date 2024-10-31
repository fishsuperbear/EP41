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
#include "ne_someip_required_find_service_behaviour.h"
#include "ne_someip_ipc_define.h"
#include "ne_someip_ipc_behaviour.h"
#include "ne_someip_log.h"
#include "ne_someip_app_context.h"
#include "ne_someip_sd_tool.h"

static void ne_someip_req_find_serv_behav_create_ipc_find_info(ne_someip_required_find_service_behaviour_t* behaviour,
    ne_someip_ipc_send_find_t* find_info, bool is_find, uint32_t ip, pthread_t tid);

ne_someip_required_find_service_behaviour_t* ne_someip_req_find_service_behaviour_new(ne_someip_client_id_t client_id,
	const ne_someip_find_offer_service_spec_t* service_spec, const ne_someip_required_service_instance_config_t* config,
    ne_someip_looper_t* io_looper, ne_someip_common_service_instance_t* instance)
{
	if (NULL == service_spec || NULL == config) {
        ne_someip_log_error("service_spec or config is NULL.");
        return NULL;
	}

    if (NULL == config->network_config || NULL == config->find_time) {
        ne_someip_log_error("config->network_config or config->find_time is NULL.");
        return NULL;
    }

    ne_someip_required_find_service_behaviour_t* find_behav = malloc(sizeof(ne_someip_required_find_service_behaviour_t));
    if (NULL == find_behav) {
    	ne_someip_log_error("malloc find_behav error.");
        return NULL;
    }

    find_behav->client_id =client_id;
    find_behav->service_spec.ins_spec.service_id = service_spec->ins_spec.service_id;
    find_behav->service_spec.ins_spec.instance_id = service_spec->ins_spec.instance_id;
    find_behav->service_spec.ins_spec.major_version = service_spec->ins_spec.major_version;
    find_behav->service_spec.minor_version = service_spec->minor_version;
    find_behav->find_service_state = ne_someip_find_service_states_not_triggered;
    find_behav->prev_upper_find_status = ne_someip_find_status_stopped;
    find_behav->config = config;
    find_behav->net_config = (ne_someip_network_config_t*)malloc(sizeof(ne_someip_network_config_t));
    if (NULL == find_behav->net_config) {
        ne_someip_log_error("malloc find_behav->net_config error.");
        ne_someip_req_find_service_behaviour_free(find_behav);
        return NULL;
    }
    memcpy(find_behav->net_config, config->network_config, sizeof(ne_someip_network_config_t));
    find_behav->find_timer_config = (ne_someip_client_find_time_config_t*)malloc(sizeof(ne_someip_client_find_time_config_t));
    if (NULL == find_behav->find_timer_config) {
        ne_someip_log_error("malloc find_behav->find_timer_config error.");
        ne_someip_req_find_service_behaviour_free(find_behav);
        return NULL;
    }
    memcpy(find_behav->find_timer_config, config->find_time, sizeof(ne_someip_client_find_time_config_t));
    find_behav->io_looper = io_looper;
    find_behav->instance = instance;
    return find_behav;
}

void ne_someip_req_find_service_behaviour_free(ne_someip_required_find_service_behaviour_t* behaviour)
{
    if (NULL != behaviour) {
        if (NULL != behaviour->net_config) {
            free(behaviour->net_config);
            behaviour->net_config = NULL;
        }
        if (NULL != behaviour->find_timer_config) {
            free(behaviour->find_timer_config);
            behaviour->find_timer_config = NULL;
        }
        behaviour->config = NULL;
        behaviour->io_looper = NULL;
        behaviour->instance = NULL;
        free(behaviour);
    }
}

//  send to endpoint
ne_someip_error_code_t ne_someip_req_find_serv_behav_start_find_service(ne_someip_required_find_service_behaviour_t* behaviour,
    ne_someip_network_states_t state, uint32_t ip)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return ne_someip_error_code_unknown;
    }

    if (ne_someip_network_states_up != state) {
        behaviour->find_service_state = ne_someip_find_service_states_wait_network_up;
        return ne_someip_error_code_network_down;
    }

    ne_someip_error_code_t find_res = ne_someip_error_code_unknown;
    switch (behaviour->find_service_state) {
        case ne_someip_find_service_states_not_triggered:
        case ne_someip_find_service_states_stop_find:
        case ne_someip_find_service_states_timeout:
            {
                ne_someip_ipc_send_find_t find_info;
                ne_someip_req_find_serv_behav_create_ipc_find_info(behaviour, &find_info, true, ip, 0);
                find_res = ne_someip_ipc_behaviour_find((ne_someip_common_service_instance_t*)(behaviour->instance), &find_info);
                behaviour->find_service_state = ne_someip_find_service_states_start_find;
                break;
            }
        case ne_someip_find_service_states_wait_network_up:
            {
                find_res = ne_someip_error_code_network_down;
                break;
            }
        case ne_someip_find_service_states_start_find:
        case ne_someip_find_service_states_finding:
            {
                ne_someip_log_debug("find service called before.");
                find_res = ne_someip_error_code_ok;
                break;
            }
        default:
            {
                ne_someip_log_error("impossible! behaviour->find_service_state error.");
                break;
            }
    }

    return find_res;
}

ne_someip_error_code_t ne_someip_req_find_serv_behav_stop_find_service(ne_someip_required_find_service_behaviour_t* behaviour,
    uint32_t ip, bool* notify_obj, pthread_t tid)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return ne_someip_error_code_unknown;
    }

    ne_someip_error_code_t find_res = ne_someip_error_code_unknown;
    switch (behaviour->find_service_state) {
        case ne_someip_find_service_states_not_triggered:
        case ne_someip_find_service_states_stop_find:
        case ne_someip_find_service_states_wait_network_up:
            {
                behaviour->find_service_state = ne_someip_find_service_states_stop_find;
                find_res = ne_someip_error_code_ok;
                break;
            }
        case ne_someip_find_service_states_timeout:
        case ne_someip_find_service_states_start_find:
        case ne_someip_find_service_states_finding:
            {
                ne_someip_ipc_send_find_t find_info;
                ne_someip_req_find_serv_behav_create_ipc_find_info(behaviour, &find_info, false, ip, tid);
                find_res = ne_someip_ipc_behaviour_stop_find((ne_someip_common_service_instance_t*)(behaviour->instance), &find_info);
                if (ne_someip_error_code_ok == find_res && NULL != notify_obj) {
                    *notify_obj = false;
                }
                behaviour->find_service_state = ne_someip_find_service_states_stop_find;
                break;
            }
        default:
            {
                ne_someip_log_error("impossible! behaviour->find_service_state error.");
                break;
            }
    }

    return find_res;
}

ne_someip_required_service_instance_config_t*
    ne_someip_req_find_serv_behav_get_config(ne_someip_required_find_service_behaviour_t* behaviour)
{
	if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return NULL;
    }

	return behaviour->config;
}

ne_someip_error_code_t ne_someip_req_find_serv_behav_net_status_notify(ne_someip_required_find_service_behaviour_t* behaviour,
	ne_someip_network_states_t state, uint32_t ip)
{
    ne_someip_log_info("network states %d", state);
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return ne_someip_error_code_unknown;
    }

    ne_someip_error_code_t find_res = ne_someip_error_code_unknown;
    switch (behaviour->find_service_state) {
        case ne_someip_find_service_states_not_triggered:
        case ne_someip_find_service_states_stop_find:
        case ne_someip_find_service_states_timeout:
            {
                find_res = ne_someip_error_code_ok;
            }
        case ne_someip_find_service_states_wait_network_up:
            {
                if (ne_someip_network_states_up == state) {
                    ne_someip_ipc_send_find_t find_info;
                    ne_someip_req_find_serv_behav_create_ipc_find_info(behaviour, &find_info, true, ip, 0);
                    find_res = ne_someip_ipc_behaviour_find((ne_someip_common_service_instance_t*)(behaviour->instance), &find_info);
                    behaviour->find_service_state = ne_someip_find_service_states_start_find;
                }
                else if (ne_someip_network_states_down == state) {
                    find_res = ne_someip_error_code_ok;
                }
                break;
            }

        case ne_someip_find_service_states_start_find:
        case ne_someip_find_service_states_finding:
            {
                if (ne_someip_network_states_up == state) {
                    find_res = ne_someip_error_code_ok;
                }
                else if (ne_someip_network_states_down == state) {
                    behaviour->find_service_state = ne_someip_find_service_states_wait_network_up;
                }
                break;
            }
        default:
            {
                ne_someip_log_error("impossible! behaviour->find_service_state error.");
                break;
            }
    }

    return find_res;
}

void ne_someip_req_find_serv_behav_find_reply(ne_someip_required_find_service_behaviour_t* behaviour,
    ne_someip_find_service_states_t res)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }

    behaviour->find_service_state = res;
}

ne_someip_find_service_states_t
ne_someip_req_find_serv_behav_find_status_get(ne_someip_required_find_service_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return ne_someip_find_service_states_not_triggered;
    }

    return behaviour->find_service_state;
}

ne_someip_find_status_t ne_someip_req_find_serv_behav_get_upper_find_state(ne_someip_required_find_service_behaviour_t* behaviour,
	bool* is_need_notify)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return ne_someip_find_status_stopped;
    }

    ne_someip_find_status_t upper_find_status = ne_someip_find_status_stopped;
    switch (behaviour->find_service_state) {
        case ne_someip_find_service_states_not_triggered:
        case ne_someip_find_service_states_stop_find:
        case ne_someip_find_service_states_timeout:
            {
                upper_find_status = ne_someip_find_status_stopped;
                break;
            }
        case ne_someip_find_service_states_wait_network_up:
        case ne_someip_find_service_states_start_find:
            {
                upper_find_status = ne_someip_find_status_pending;
                break;
            }
        case ne_someip_find_service_states_finding:
            {
                upper_find_status = ne_someip_find_status_running;
                break;
            }
        default:
            {
                ne_someip_log_error("impossible! behaviour->find_service_state error.");
                break;
            }
    }

    ne_someip_log_info("upper_find_status %d, find_service_state %d, prev_upper_find_status %d", upper_find_status,
        behaviour->find_service_state, behaviour->prev_upper_find_status);
    if (upper_find_status != behaviour->prev_upper_find_status) {
        behaviour->prev_upper_find_status = upper_find_status;
        *is_need_notify = true;
    }

    return upper_find_status;
}

static void ne_someip_req_find_serv_behav_create_ipc_find_info(ne_someip_required_find_service_behaviour_t* behaviour,
    ne_someip_ipc_send_find_t* find_info, bool is_find, uint32_t ip, pthread_t tid)
{
    if (NULL == behaviour || NULL == find_info || NULL == behaviour->config) {
        ne_someip_log_error("instance or find_info or behaviour->config is NULL.");
        return;
    }

    ne_someip_sd_convert_uint32_to_ip("ip :", ip, __FILE__, __LINE__, __FUNCTION__);

    if (NULL == behaviour->net_config || NULL ==behaviour->find_timer_config) {
        ne_someip_log_error("behaviour->net_config or behaviour->find_timer_config is NULL.");
        return;
    }
    ne_someip_network_config_t* net_config = behaviour->net_config;
    ne_someip_client_find_time_config_t* find_timer = behaviour->find_timer_config;

    find_info->type = is_find ? ne_someip_ipc_msg_type_send_find : ne_someip_ipc_msg_type_send_stop_find;
    find_info->length = sizeof(ne_someip_ipc_send_find_t);
    find_info->client_id = behaviour->client_id;
    find_info->service_id = behaviour->service_spec.ins_spec.service_id;
    find_info->instance_id = behaviour->service_spec.ins_spec.instance_id;
    find_info->major_version = behaviour->service_spec.ins_spec.major_version;
    find_info->minor_version = behaviour->service_spec.minor_version;
    find_info->addr_type = net_config->addr_type;
    find_info->local_addr = ip;
    memcpy(&(find_info->timer), find_timer, sizeof(ne_someip_client_find_time_config_t));
    find_info->remote_addr = net_config->multicast_ip;
    find_info->remote_port = net_config->multicast_port;
    ne_someip_app_context_get_saved_unix_path(&find_info->proxy_unix_addr);
    find_info->tid = tid;
}
