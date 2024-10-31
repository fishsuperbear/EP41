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
#include "ne_someip_required_eventgroup_behaviour.h"
#include "ne_someip_ipc_behaviour.h"
#include "ne_someip_log.h"

static void ne_someip_req_eventgroup_behaviour_uint16_free(void* data)
{
    if (NULL == data) {
        return;
    }

    free((uint16_t*)data);
    data = NULL;
}

ne_someip_required_eventgroup_behaviour_t*
ne_someip_req_eventgroup_behaviour_new(const ne_someip_required_eventgroup_config_t* config)
{
    if (NULL == config) {
        ne_someip_log_error("config is NULL.");
        return NULL;
    }
	ne_someip_required_eventgroup_behaviour_t* eventgroup_behav =
        (ne_someip_required_eventgroup_behaviour_t*)malloc(sizeof(ne_someip_required_eventgroup_behaviour_t));
    if (NULL == eventgroup_behav) {
        ne_someip_log_error("malloc ne_someip_required_eventgroup_behaviour_t error.");
        return eventgroup_behav;
    }
    memset(eventgroup_behav, 0, sizeof(*eventgroup_behav));
    eventgroup_behav->config =
        (ne_someip_required_eventgroup_config_t*)malloc(sizeof(ne_someip_required_eventgroup_config_t));
    if (NULL == eventgroup_behav->config) {
        ne_someip_log_error("malloc eventgroup_behav->config error.");
        ne_someip_req_eventgroup_behaviour_free(eventgroup_behav);
        return NULL;
    }
    eventgroup_behav->config->eventgroup_config =
        (ne_someip_eventgroup_config_t*)malloc(sizeof(ne_someip_eventgroup_config_t));
    if (NULL == eventgroup_behav->config->eventgroup_config) {
        ne_someip_log_error("malloc eventgroup_behav->config->eventgroup_config error.");
        ne_someip_req_eventgroup_behaviour_free(eventgroup_behav);
        return NULL;
    }
    memcpy(eventgroup_behav->config->eventgroup_config, config->eventgroup_config,
        sizeof(ne_someip_eventgroup_config_t));
    eventgroup_behav->config->time =
        (ne_someip_client_subscribe_time_config_t*)malloc(sizeof(ne_someip_client_subscribe_time_config_t));
    if (NULL == eventgroup_behav->config->time) {
        ne_someip_log_error("malloc eventgroup_behav->config->time error.");
        ne_someip_req_eventgroup_behaviour_free(eventgroup_behav);
        return NULL;
    }
    memcpy(eventgroup_behav->config->time, config->time, sizeof(ne_someip_client_subscribe_time_config_t));
    eventgroup_behav->sub_state = ne_someip_eventgroup_subscribe_states_not_triggered;
    eventgroup_behav->pre_upper_sub_status = ne_someip_subscribe_status_unsubscribed;

    return eventgroup_behav;
}

void ne_someip_req_eventgroup_behaviour_free(ne_someip_required_eventgroup_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_info("behaviour is NULL.");
        return;
    }

    if (NULL != behaviour->config) {
        if (NULL != behaviour->config->eventgroup_config) {
            free(behaviour->config->eventgroup_config);
            behaviour->config->eventgroup_config = NULL;
        }
        if (NULL != behaviour->config->time) {
            free(behaviour->config->time);
            behaviour->config->time = NULL;
        }

        free(behaviour->config);
        behaviour->config = NULL;
    }

    free(behaviour);
    behaviour = NULL;
}

//  send to endpoint
ne_someip_error_code_t ne_someip_req_eg_behav_start_subscribe_eventgroup(ne_someip_required_eventgroup_behaviour_t* behaviour,
    ne_someip_eventgroup_id_t eventgroup_id, void* required_instance, ne_someip_network_states_t state,
    ne_someip_service_status_t serv_status, bool is_comm_ok)
{
	if (NULL == behaviour || NULL == required_instance) {
        ne_someip_log_error("behaviour or required_instance is NULL.");
        return ne_someip_error_code_failed;
    }

    if (ne_someip_network_states_up != state) {
        behaviour->sub_state = ne_someip_eventgroup_subscribe_states_wait_network_up;
        return ne_someip_error_code_network_down;
    }

    if (ne_someip_service_status_available != serv_status || !is_comm_ok) {
        behaviour->sub_state = ne_someip_eventgroup_subscribe_states_wait_available;
        return ne_someip_error_code_comm_not_prepare_ok;
    }

    ne_someip_error_code_t ret = ne_someip_error_code_unknown;
    ne_someip_list_t* eventgroup_id_list = ne_someip_list_create();
    if (NULL == eventgroup_id_list) {
        ne_someip_log_error("ne_someip_list_create eventgroup_id_list error.");
        return ne_someip_error_code_failed;
    }
    ne_someip_eventgroup_id_t* save_eventgroup_id = malloc(sizeof(ne_someip_eventgroup_id_t));
    if (NULL == save_eventgroup_id) {
        ne_someip_log_error("malloc save_eventgroup_id error.");
        return ne_someip_error_code_failed;
    }
    *save_eventgroup_id = eventgroup_id;
    eventgroup_id_list = ne_someip_list_append(eventgroup_id_list, (void*)save_eventgroup_id);
    ne_someip_required_service_instance_t* ser_inst = (ne_someip_required_service_instance_t*)required_instance;
    switch (behaviour->sub_state) {
        case ne_someip_eventgroup_subscribe_states_not_triggered:
        case ne_someip_eventgroup_subscribe_states_stop_subscribe:
        case ne_someip_eventgroup_subscribe_states_subscribe_nack:
        case ne_someip_eventgroup_subscribe_states_timeout:
            {
                ret = ne_someip_ipc_behaviour_subscribe(ser_inst, eventgroup_id_list, false);
                behaviour->sub_state = ne_someip_eventgroup_subscribe_states_subscribing;
                break;
            }
        case ne_someip_eventgroup_subscribe_states_wait_network_up:
        case ne_someip_eventgroup_subscribe_states_wait_available:
            {
                ret = ne_someip_ipc_behaviour_subscribe(ser_inst, eventgroup_id_list, true);
                behaviour->sub_state = ne_someip_eventgroup_subscribe_states_subscribing;
                break;
            }
        case ne_someip_eventgroup_subscribe_states_start_subscribe:
        case ne_someip_eventgroup_subscribe_states_subscribing:
        case ne_someip_eventgroup_subscribe_states_subscribe_ack:
            {
                ne_someip_log_debug("subscribe eventgroup called before.");
                ret = ne_someip_error_code_ok;
                break;
            }
        default:
            {
                ne_someip_log_error("impossible! behaviour->sub_state error.");
                ret = ne_someip_error_code_failed;
                break;
            }
    }

    if (NULL != eventgroup_id_list) {
        ne_someip_list_destroy(eventgroup_id_list, ne_someip_req_eventgroup_behaviour_uint16_free);
    }

    return ret;
}

ne_someip_error_code_t ne_someip_req_eg_behav_stop_subscribe_eventgroup(ne_someip_required_eventgroup_behaviour_t* behaviour,
    ne_someip_eventgroup_id_t eventgroup_id, void* required_instance, bool* notify_obj, pthread_t tid)
{
	if (NULL == behaviour || NULL == required_instance) {
        ne_someip_log_error("behaviour or required_instance is NULL.");
        return ne_someip_error_code_failed;
    }

    ne_someip_error_code_t ret = ne_someip_error_code_unknown;
    ne_someip_required_service_instance_t* ser_inst = (ne_someip_required_service_instance_t*)required_instance;
    switch (behaviour->sub_state) {
        case ne_someip_eventgroup_subscribe_states_not_triggered:
            {
                ne_someip_log_debug("start subscribe eventgroup not called before;");
                behaviour->sub_state = ne_someip_eventgroup_subscribe_states_stop_subscribe;
                ret = ne_someip_error_code_ok;
                break;
            }
        case ne_someip_eventgroup_subscribe_states_stop_subscribe:
            {
                ne_someip_log_debug("stop subscribe eventgroup called before;");
                ret = ne_someip_error_code_ok;
                break;
            }
        case ne_someip_eventgroup_subscribe_states_wait_network_up:
        case ne_someip_eventgroup_subscribe_states_wait_available:
            {
                behaviour->sub_state = ne_someip_eventgroup_subscribe_states_stop_subscribe;
                ret = ne_someip_error_code_ok;
                break;
            }
        case ne_someip_eventgroup_subscribe_states_timeout:
        case ne_someip_eventgroup_subscribe_states_start_subscribe:
        case ne_someip_eventgroup_subscribe_states_subscribing:
        case ne_someip_eventgroup_subscribe_states_subscribe_ack:
        case ne_someip_eventgroup_subscribe_states_subscribe_nack:
            {
                ne_someip_log_info("tid is %d", tid);
                ret = ne_someip_ipc_behaviour_stop_subscribe(ser_inst, eventgroup_id, tid);
                if (ne_someip_error_code_ok == ret && NULL != notify_obj) {
                    *notify_obj = false;
                }
                behaviour->sub_state = ne_someip_eventgroup_subscribe_states_stop_subscribe;
                break;
            }
        default:
            {
                ne_someip_log_error("impossible! behaviour->sub_state error.");
                break;
            }
    }

    return ret;
}

//  receive from endpoint
void ne_someip_req_eg_behav_recv_subscribe_ack(ne_someip_required_eventgroup_behaviour_t* behaviour,
    const ne_someip_sd_recv_subscribe_ack_t* sub_ack)
{
	if (NULL == behaviour || NULL == sub_ack) {
        ne_someip_log_error("behaviour or sub_ack is NULL.");
        return;
    }

    if (sub_ack->is_ack) {  // subscribe ack
        behaviour->sub_state = ne_someip_eventgroup_subscribe_states_subscribe_ack;
    }
    else {  // subscribe nack
        behaviour->sub_state = ne_someip_eventgroup_subscribe_states_subscribe_nack;
    }
}

bool ne_someip_req_eg_behav_is_sub_eventgroup_wait_net_up(ne_someip_required_eventgroup_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return false;
    }

    if (ne_someip_eventgroup_subscribe_states_wait_network_up == behaviour->sub_state) {
        return true;
    }
    return false;
}

bool ne_someip_req_eg_behav_is_sub_eventgroup_wait_avaliable(ne_someip_required_eventgroup_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return false;
    }

    ne_someip_log_debug("check_sub_state [%d]", behaviour->sub_state);
    if (ne_someip_eventgroup_subscribe_states_wait_available == behaviour->sub_state ||
        ne_someip_eventgroup_subscribe_states_subscribe_nack == behaviour->sub_state) {
        return true;
    }
    return false;
}

void ne_someip_req_eg_behav_set_sub_state(ne_someip_required_eventgroup_behaviour_t* behaviour,
    ne_someip_eventgroup_subscribe_states_t state)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return;
    }

    ne_someip_log_debug("set_sub_state [%d]", state);
    behaviour->sub_state = state;
}

bool ne_someip_req_eg_behav_is_sub_ack(ne_someip_required_eventgroup_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_error("behaviour is NULL.");
        return false;
    }

    if (ne_someip_eventgroup_subscribe_states_subscribe_ack == behaviour->sub_state) {
        return true;
    }
    return false;
}

ne_someip_subscribe_status_t ne_someip_req_eg_behav_get_upper_sub_state(
    ne_someip_required_eventgroup_behaviour_t* behaviour, bool* is_need_notify)
{
    if (NULL == behaviour || NULL == is_need_notify) {
        ne_someip_log_error("behaviour or is_need_notify is NULL.");
        return ne_someip_subscribe_status_failed;
    }

    ne_someip_subscribe_status_t upper_sub_status = ne_someip_subscribe_status_failed;
    switch (behaviour->sub_state) {
        case ne_someip_eventgroup_subscribe_states_not_triggered:
        case ne_someip_eventgroup_subscribe_states_stop_subscribe:
        case ne_someip_eventgroup_subscribe_states_timeout:
            {
                upper_sub_status = ne_someip_subscribe_status_unsubscribed;
                break;
            }
        case ne_someip_eventgroup_subscribe_states_subscribe_nack:
            {
                upper_sub_status = ne_someip_subscribe_status_failed;
                break;
            }
        case ne_someip_eventgroup_subscribe_states_wait_network_up:
        case ne_someip_eventgroup_subscribe_states_wait_available:
        case ne_someip_eventgroup_subscribe_states_start_subscribe:
            {
                upper_sub_status = ne_someip_subscribe_status_pending;
                break;
            }
        case ne_someip_eventgroup_subscribe_states_subscribing:
            {
                upper_sub_status = ne_someip_subscribe_status_pending;
                break;
            }
        case ne_someip_eventgroup_subscribe_states_subscribe_ack:
            {
                upper_sub_status = ne_someip_subscribe_status_subscribed;
                break;
            }
        default:
            {
                ne_someip_log_error("impossible! behaviour->sub_state error.");
                break;
            }
    }

    if (upper_sub_status != behaviour->pre_upper_sub_status) {
        behaviour->pre_upper_sub_status = upper_sub_status;
        *is_need_notify = true;
    }

    return upper_sub_status;
}
