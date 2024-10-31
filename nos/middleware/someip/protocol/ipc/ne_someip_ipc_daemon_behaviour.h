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
#ifndef MANAGER_NE_SOMEIP_IPC_DAEMON_BEHAVIOUR_H
#define MANAGER_NE_SOMEIP_IPC_DAEMON_BEHAVIOUR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_define.h"
#include "ne_someip_ipc_define.h"
#include "ne_someip_server_define.h"
#include "ne_someip_client_define.h"
#include "ne_someip_sd_define.h"

/*init, unix_data endpoint
注冊回調函數，unix_data ne_someip_ipc_daemon_behaviour_recv_proxy_msg*/
ne_someip_endpoint_unix_t* ne_someip_ipc_daemon_behaviour_init(ne_someip_looper_t* looper);

//deinit，取消注冊回調函數
void ne_someip_ipc_daemon_behaviour_deinit();

//notify service available, ne_someip_ipc_recv_offer_t
void ne_someip_ipc_deamon_behaviour_notify_remote_service_status(const ne_someip_sd_recv_offer_t* offer_info, bool is_stop,
	const ne_someip_endpoint_unix_addr_t* unix_path);

//notify subcribe, ne_someip_ipc_recv_subscribe_t
void ne_someip_ipc_deamon_behaviour_notify_subscribe(const ne_someip_list_t* subscribe_info, bool is_stop,
	const ne_someip_endpoint_net_addr_pair_t* net_addr, ne_someip_endpoint_unix_addr_t* proxy_unix_path);

//notify subcribe ack, ne_someip_ipc_recv_subscribe_ack_t
void ne_someip_ipc_deamon_behaviour_notify_subscribe_ack(const ne_someip_sd_recv_subscribe_ack_t* subscribe_ack_info,
	bool is_ack, ne_someip_list_t* addr_list);

//notify remote reboot
void ne_someip_ipc_deamon_behaviour_notify_remote_reboot(uint32_t remote_addr,
	const ne_someip_endpoint_unix_addr_t* unix_path);

//notify local service status
void ne_someip_ipc_deamon_behaviour_notify_local_offer_status(const ne_someip_ipc_send_offer_t* service,
	ne_someip_ser_offer_status_t status);

//notify local find service status
void ne_someip_ipc_deamon_behaviour_notify_local_find_status(const ne_someip_ipc_send_find_t* service,
	ne_someip_find_service_states_t status);

//notify local subscribe status
void ne_someip_ipc_deamon_behaviour_notify_local_subscribe_status(const ne_someip_ipc_send_subscribe_t* subscribe,
	ne_someip_eventgroup_subscribe_states_t status);

void ne_someip_ipc_daemon_behaviour_ser_and_send_service_handler_reply(
	const ne_someip_ipc_reg_unreg_service_handler_t* message, ne_someip_endpoint_unix_addr_t* unix_path, bool status);

//link status changed
void ne_someip_ipc_daemon_behaviour_link_changed(ne_someip_endpoint_link_state_t* state, void* user_data);

//收到来自proxy的消息, daemon init的時候調用注冊接口
void ne_someip_ipc_daemon_behaviour_recv_unix_msg(ne_someip_trans_buffer_struct_t* trans_buffer, void* addr_pair,
	void* user_data);

#ifdef __cplusplus
}
#endif
#endif // MANAGER_NE_SOMEIP_IPC_DAEMON_BEHAVIOUR_H
/* EOF */