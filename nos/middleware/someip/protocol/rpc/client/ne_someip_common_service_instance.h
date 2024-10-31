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
#ifndef SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_COMMON_SERVICE_INSTANCE_H
#define SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_COMMON_SERVICE_INSTANCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_define.h"
#include "ne_someip_client_define.h"
#include "ne_someip_ipc_define.h"

typedef struct ne_someip_comm_serv_inst_destroy
{
    ne_someip_common_service_instance_t* instance;
    pthread_t tid;
}ne_someip_comm_serv_inst_destroy_t;

// user thread
ne_someip_common_service_instance_t* ne_someip_common_service_instance_new(ne_someip_client_id_t client_id,
    ne_someip_looper_t* work_looper, ne_someip_looper_t* io_looper, ne_someip_endpoint_unix_t* unix_endpoint);

void ne_someip_common_service_instance_destroy(ne_someip_common_service_instance_t* instance);

ne_someip_required_service_connect_behaviour_t*
    ne_someip_comm_serv_inst_create_connect_behav_by_create_req_inst(ne_someip_common_service_instance_t* instance,
    const ne_someip_service_instance_spec_t* spec);
ne_someip_required_service_connect_behaviour_t*
    ne_someip_comm_serv_inst_create_connect_behav_by_recv_offer(ne_someip_common_service_instance_t* instance,
    const ne_someip_find_offer_service_spec_t* spec);

void ne_someip_common_serv_inst_notify_available_by_net_down(ne_someip_common_service_instance_t* instance,
    const ne_someip_find_offer_service_spec_t* ser_spec);

uint32_t ne_someip_common_serv_inst_get_ip_by_net_config(ne_someip_common_service_instance_t* instance,
    const ne_someip_network_config_t* net_config);

// interface for required_service to register/unregister service available handler
bool ne_someip_comm_serv_inst_reg_requir_service(ne_someip_common_service_instance_t* com_instance,
	const ne_someip_service_instance_spec_t* spec, ne_someip_required_service_instance_t* req_instance);
bool ne_someip_comm_serv_inst_unre_requir_service(ne_someip_common_service_instance_t* com_instance,
	const ne_someip_service_instance_spec_t* spec, const ne_someip_required_service_instance_t* req_instance);

ne_someip_error_code_t ne_someip_comm_serv_inst_reg_find_service_handler(ne_someip_common_service_instance_t* instance,
    const ne_someip_find_offer_service_spec_t* service_spec, ne_someip_find_status_handler handler, const void* user_data);
ne_someip_error_code_t ne_someip_comm_serv_inst_unreg_find_service_handler(ne_someip_common_service_instance_t* instance,
    const ne_someip_find_offer_service_spec_t* service_spec, ne_someip_find_status_handler handler);

ne_someip_error_code_t ne_someip_comm_serv_inst_reg_service_status_handler(
    ne_someip_common_service_instance_t* instance, const ne_someip_find_offer_service_spec_t* service_spec,
    ne_someip_service_available_handler handler, const void* user_data);
ne_someip_error_code_t ne_someip_comm_serv_inst_unreg_service_status_handler(
    ne_someip_common_service_instance_t* instance, const ne_someip_find_offer_service_spec_t* service_spec,
    ne_someip_service_available_handler handler);

bool ne_someip_comm_serv_inst_create_find_offer_serv(ne_someip_common_service_instance_t* instance,
	ne_someip_sequence_id_t seq_id, ne_someip_find_local_offer_services_t* offer_services);
void ne_someip_comm_serv_inst_add_find_offer_serv(ne_someip_common_service_instance_t* instance,
	ne_someip_sequence_id_t seq_id, ne_someip_list_t* offer_services_list, ne_someip_error_code_t find_res);
ne_someip_client_find_local_services_t*
    ne_someip_comm_serv_inst_find_saved_find_offer_serv(ne_someip_common_service_instance_t* instance,
    ne_someip_sequence_id_t seq_id);
bool ne_someip_comm_serv_inst_delete_find_offer_serv(ne_someip_common_service_instance_t* instance,
	ne_someip_sequence_id_t seq_id);

// work thread
ne_someip_error_code_t ne_someip_comm_serv_inst_start_find_service(ne_someip_common_service_instance_t* instance,
	const ne_someip_find_offer_service_spec_t* service_spec, ne_someip_looper_t* io_looper,
    const ne_someip_required_service_instance_config_t* inst_config);

ne_someip_error_code_t ne_someip_comm_serv_inst_stop_find_service(ne_someip_common_service_instance_t* instance,
	const ne_someip_find_offer_service_spec_t* service_spec, ne_someip_looper_t* io_looper, bool* notify_obj, pthread_t tid);

ne_someip_required_service_instance_config_t* ne_someip_comm_serv_inst_get_config(ne_someip_common_service_instance_t* instance,
	const ne_someip_find_offer_service_spec_t* service_spec);

ne_someip_service_status_t ne_someip_comm_serv_inst_get_available_status(ne_someip_common_service_instance_t* instance,
	const ne_someip_service_instance_spec_t* spec);

// interface for query offered instances, running in work thread
void ne_someip_comm_serv_inst_query_offered_instances(ne_someip_common_service_instance_t* instance,
    ne_someip_find_offer_service_spec_t* spec, ne_someip_sequence_id_t seq_id, pthread_t tid);

// notify the find status to upper app
void ne_someip_comm_serv_inst_notify_find_status(ne_someip_common_service_instance_t* instance,
	const ne_someip_find_offer_service_spec_t* service_spec, ne_someip_find_status_t status, ne_someip_error_code_t code);

// notify the service available status to upper app
void ne_someip_comm_serv_inst_notify_avail_status(ne_someip_common_service_instance_t* instance,
	const ne_someip_find_offer_service_spec_t* service_spec, ne_someip_service_status_t status);

// notify the service available status to upper app when register_service_status_handler
void ne_someip_comm_serv_inst_notify_local_avail_status(ne_someip_common_service_instance_t* instance,
    const ne_someip_find_offer_service_spec_t* service_spec, ne_someip_saved_available_handler_t* handler_info);
/*********************************callback*************************************/
// recv find local services(received offer services) (from sd)
void ne_someip_comm_serv_inst_recv_query_offer_services(ne_someip_common_service_instance_t* instance,
    const ne_someip_ipc_find_remote_svs_reply_t* remote_service);
// recv start/stop find service reply result (from app context)
void ne_someip_comm_serv_inst_recv_find_reply(ne_someip_common_service_instance_t* instance,
	const ne_someip_find_offer_service_spec_t* spec, ne_someip_find_service_states_t status);
//  recv service status available/unavailable status (fromn app_context)
void ne_someip_comm_serv_inst_recv_service_avail_handler(ne_someip_common_service_instance_t* instance,
	const ne_someip_sd_recv_offer_t* offer_msg);
// recv remote reboot notify (sd)
void ne_someip_comm_serv_inst_remote_reboot_handler(ne_someip_common_service_instance_t* instance, uint32_t ip_addr);
// recv unix disconnect
void ne_someip_comm_serv_inst_unix_link_change(ne_someip_common_service_instance_t* instance, ne_someip_endpoint_transmit_link_state_t state);
/*********************************callback*************************************/

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_CLIENT_NE_SOMEIP_COMMON_SERVICE_INSTANCE_H
/* EOF */
