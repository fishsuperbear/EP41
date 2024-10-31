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
#ifndef MANAGER_NE_SOMEIP_IPC_BEHAVIOUR_H
#define MANAGER_NE_SOMEIP_IPC_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_define.h"
#include "ne_someip_ipc_define.h"
#include "ne_someip_server_define.h"
#include "ne_someip_client_define.h"
#include "ne_someip_required_service_instance.h"
#include "ne_someip_provided_service_instance.h"

// TCP: create/destory/connect/disconnect
ne_someip_error_code_t
ne_someip_ipc_behaviour_create_client_tcp_socket(const ne_someip_required_service_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_destory_client_tcp_socket(const ne_someip_required_service_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_create_server_tcp_socket(const ne_someip_provided_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_destory_server_tcp_socket(const ne_someip_provided_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_tcp_client_connect(const ne_someip_required_service_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_tcp_client_disconnect(const ne_someip_required_service_instance_t* instance);

// UDP: create/destory/joinMulticast/leaveMulticast
ne_someip_error_code_t
ne_someip_ipc_behaviour_create_client_udp_socket(const ne_someip_required_service_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_destory_client_udp_socket(const ne_someip_required_service_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_create_destroy_multicast_socket(const ne_someip_required_service_instance_t* instance,
    const ne_someip_endpoint_net_addr_t* multi_addr, bool is_create);
ne_someip_error_code_t
ne_someip_ipc_behaviour_create_server_udp_socket(const ne_someip_provided_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_destory_server_udp_socket(const ne_someip_provided_instance_t* instance);
ne_someip_return_code_t
ne_someip_ipc_behaviour_client_join_group(const ne_someip_required_service_instance_t* instance,
    const ne_someip_endpoint_net_addr_t* unicast_addr, const ne_someip_endpoint_net_addr_t* group_address);
ne_someip_return_code_t
ne_someip_ipc_behaviour_client_leave_group(const ne_someip_required_service_instance_t* instance,
    const ne_someip_endpoint_net_addr_t* unicast_addr, const ne_someip_endpoint_net_addr_t* group_address);

// register/unregister (request/response/event) handler
ne_someip_error_code_t
ne_someip_ipc_behaviour_reg_req_handler(const ne_someip_provided_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_unreg_req_handler(const ne_someip_provided_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_reg_resp_handler(const ne_someip_required_service_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_unreg_resp_handler(const ne_someip_required_service_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_reg_event_handler(const ne_someip_required_service_instance_t* instance);
ne_someip_error_code_t
ne_someip_ipc_behaviour_unreg_event_handler(const ne_someip_required_service_instance_t* instance);

// send rpc message (request/response/event)
ne_someip_error_code_t
ne_someip_ipc_behaviour_send_req_msg(const ne_someip_header_t* header, const ne_someip_payload_t* payload,
    const ne_someip_required_service_instance_t* instance, const ne_someip_endpoint_send_policy_t* send_policy);
ne_someip_error_code_t
ne_someip_ipc_behaviour_send_event_msg(ne_someip_header_t* header, const ne_someip_payload_t* payload,
    const ne_someip_provided_instance_t* instance, const ne_someip_endpoint_send_policy_t* send_policy,
    const ne_someip_provided_event_t* event_behaviour, ne_someip_send_spec_t* send_msg);
ne_someip_error_code_t
ne_someip_ipc_behaviour_send_resp_msg(const ne_someip_header_t* header, const ne_someip_payload_t* payload,
    const ne_someip_remote_client_info_t* remote_info, const ne_someip_provided_instance_t* instance,
    const ne_someip_endpoint_send_policy_t* send_policy);

/**
 *async
 *@brief notify the network status
 *
 *@param [in] ip_addr, ip address
 *@param [in] status, ip status
 *@param [in] io_looper, io looper
 *
 *@return， 抛到io线程后返回,  返回 ne_someip_error_code_ok 表示成功， 返回 ne_someip_error_code_failed 表示失败
 */
ne_someip_error_code_t
ne_someip_ipc_notify_network_status(ne_someip_endpoint_unix_t* endpoint, uint32_t ip_addr, ne_someip_network_states_t status);

/**
 *async
 *@brief register service status handler
 *
 *@param [in] instance, instance信息
 *
 *@return， 抛到io线程后返回,  返回 ne_someip_error_code_ok 表示成功， 返回 ne_someip_error_code_failed 表示失败
 */
ne_someip_error_code_t
ne_someip_ipc_reg_service_status_handler(const ne_someip_common_service_instance_t* instance,
	const ne_someip_service_instance_spec_t* instance_spec);

/**
 *async
 *@brief unregister service status handler
 *
 *@param [in] instance, instance信息
 *
 *@return， 抛到io线程后返回,  返回 ne_someip_error_code_ok 表示成功， 返回 ne_someip_error_code_failed 表示失败
 */
ne_someip_error_code_t
ne_someip_ipc_unreg_service_status_handler(const ne_someip_common_service_instance_t* instance,
	const ne_someip_service_instance_spec_t* instance_spec);

/*create endpoint for instance*/
ne_someip_endpoint_unix_t*
ne_someip_ipc_create_unix_endpoint(const ne_someip_looper_t* work_looper,
	const ne_someip_looper_t* io_looper, const ne_someip_endpoint_unix_addr_t* unix_addr);

/***************************************sd message*******************************************/

//offer service
ne_someip_error_code_t
ne_someip_ipc_behaviour_offer(const ne_someip_provided_instance_t* instance);

//stop offer service
ne_someip_error_code_t
ne_someip_ipc_behaviour_stop_offer(const ne_someip_provided_instance_t* instance, bool is_finished, pthread_t tid);

//find service
ne_someip_error_code_t
ne_someip_ipc_behaviour_find(const ne_someip_common_service_instance_t* instance,
	ne_someip_ipc_send_find_t* find_info);

//stop find service
ne_someip_error_code_t
ne_someip_ipc_behaviour_stop_find(const ne_someip_common_service_instance_t* instance,
	ne_someip_ipc_send_find_t* find_info);

//subscribe, if subscribe that is cached for recv offer, then is_delay = true
ne_someip_error_code_t
ne_someip_ipc_behaviour_subscribe(const ne_someip_required_service_instance_t* instance,
	ne_someip_list_t* eventgroup_id_list, bool is_delay);

//stop subscribe
ne_someip_error_code_t
ne_someip_ipc_behaviour_stop_subscribe(const ne_someip_required_service_instance_t* instance,
	ne_someip_eventgroup_id_t eventgroup_id, pthread_t tid);

//subscribe ack
ne_someip_error_code_t
ne_someip_ipc_behaviour_subscribe_ack(const ne_someip_provided_instance_t* instance,
	const ne_someip_list_t* ack_list, const ne_someip_list_t* nack_list);

ne_someip_error_code_t
ne_someip_ipc_behaviour_subscribe_nack_for_miss_subscriber(const ne_someip_provided_instance_t* instance,
	ne_someip_eventgroup_id_t eventgroup_id, const ne_someip_remote_client_info_t* remote_addr);

//get client id
ne_someip_error_code_t ne_someip_ipc_behaviour_get_client_id(const ne_someip_endpoint_unix_t* unix_endpoint,
	ne_someip_sequence_id_t seq_id, pthread_t tid);

//find remote service
ne_someip_error_code_t
ne_someip_ipc_behaviour_find_remote_service(const ne_someip_common_service_instance_t* instance,
	ne_someip_ipc_find_remote_svs_t* find_info);

#ifdef __cplusplus
}
#endif
#endif // MANAGER_NE_SOMEIP_IPC_BEHAVIOUR_H
/* EOF */
