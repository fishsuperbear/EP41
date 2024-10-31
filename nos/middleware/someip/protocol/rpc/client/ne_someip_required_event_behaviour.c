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
#include "ne_someip_required_event_behaviour.h"
#include "ne_someip_ipc_behaviour.h"
#include "ne_someip_log.h"

ne_someip_required_event_behaviour_t* ne_someip_req_event_behaviour_new(ne_someip_event_config_t* config)
{
    if (NULL == config) {
        ne_someip_log_error("config is NULL.");
        return NULL;
    }

	ne_someip_required_event_behaviour_t* event_behav =
        (ne_someip_required_event_behaviour_t*)malloc(sizeof(ne_someip_required_event_behaviour_t));
    if (NULL == event_behav) {
        ne_someip_log_error("malloc ne_someip_required_event_behaviour_t error.");
        return event_behav;
    }
    memset(event_behav, 0, sizeof(*event_behav));
    event_behav->config = config;

    return event_behav;
}

void ne_someip_req_event_behaviour_free(ne_someip_required_event_behaviour_t* behaviour)
{
    if (NULL == behaviour) {
        ne_someip_log_info("behaviour is NULL.");
        return;
    }

    behaviour->config = NULL;
    free(behaviour);
    behaviour = NULL;
}

ne_someip_error_code_t ne_someip_req_event_behaviour_reg_event_handler_to_daemon(void* required_instance)
{
    if (NULL == required_instance) {
        ne_someip_log_error("required_instance is NULL.");
        return ne_someip_error_code_failed;
    }

    ne_someip_error_code_t ret =
        ne_someip_ipc_behaviour_reg_event_handler((ne_someip_required_service_instance_t*)required_instance);

    return ret;
}

ne_someip_error_code_t ne_someip_req_event_behaviour_unreg_event_handler_to_daemon(void* required_instance)
{
    if (NULL == required_instance) {
        ne_someip_log_error("required_instance is NULL.");
        return ne_someip_error_code_failed;
    }

    ne_someip_error_code_t ret =
        ne_someip_ipc_behaviour_unreg_event_handler((ne_someip_required_service_instance_t*)required_instance);

    return ret;
}

void ne_someip_req_event_behaviour_recv_event(ne_someip_required_service_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload)
{
	if (NULL == instance || NULL == header) {
        ne_someip_log_error("behaviour or header is NULL.");
        return;
    }

    ne_someip_sync_obj_sync_start(instance->event_handler_sync);
    ne_someip_list_iterator_t* iter = ne_someip_list_iterator_create(instance->event_handler_list);
    while(ne_someip_list_iterator_valid(iter)) {
        if (NULL != ne_someip_list_iterator_data(iter)) {
            ((ne_someip_saved_recv_event_handler_t*)(ne_someip_list_iterator_data(iter)))->handler(instance,
                header, payload,
                ((ne_someip_saved_recv_event_handler_t*)(ne_someip_list_iterator_data(iter)))->user_data);
        }
        ne_someip_list_iterator_next(iter);
    }
    ne_someip_list_iterator_destroy(iter);
    ne_someip_sync_obj_sync_end(instance->event_handler_sync);
}