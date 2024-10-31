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
#ifndef MANAGER_SERVER_NE_SOMEIP_SERVER_DEFINE_H
#define MANAGER_SERVER_NE_SOMEIP_SERVER_DEFINE_H

#include "ne_someip_define.h"
#include "ne_someip_looper.h"
#include "ne_someip_endpoint_define.h"
#include "ne_someip_map.h"
#include "ne_someip_handler.h"
#include "ne_someip_provided_eventgroup_behaviour.h"
#include "ne_someip_server_context.h"

#define NESOMEIP_WAIT_SET_EG_PRIORITY 5

typedef enum ne_someip_ser_offer_status
{
	ne_someip_offer_status_down = 0x00,
	ne_someip_offer_status_wait_network_up = 0x01,
	ne_someip_offer_status_wait_endpoint = 0x02,
	ne_someip_offer_status_initial = 0x05,
	ne_someip_offer_status_repetition = 0x06,
	ne_someip_offer_status_main = 0x07,
	ne_someip_offer_status_trigger_down = 0x08,
} ne_someip_ser_offer_status_t;

typedef enum ne_someip_ser_timer_type
{
    ne_someip_ser_timer_type_stop_offer = 0x0,
}ne_someip_ser_timer_type_t;

typedef struct ne_someip_saved_event_resp_seq_info {
    ne_someip_method_id_t method_id;
    ne_someip_session_id_t session_id;
}ne_someip_saved_event_resp_seq_info_t;

typedef struct ne_someip_server_forward_link {
    ne_someip_address_type_t addr_type;
    uint32_t remote_addr;
    uint16_t remote_port;
} ne_someip_server_forward_link_t;

typedef struct ne_someip_event_spec {
    void* seq_id;
    void* instance;
    ne_someip_header_t header;
    ne_someip_payload_t* payload;
} ne_someip_event_spec_t;

typedef struct ne_someip_response_spec
{
	void* seq_id;
	void* instance;
	ne_someip_header_t header;
	ne_someip_payload_t* payload;
	ne_someip_remote_client_info_t remote_addr;
} ne_someip_response_spec_t;

typedef struct ne_someip_send_spec
{
	void* seq_id;
	ne_someip_method_id_t method_id;
	bool is_notify;
	uint32_t send_num;
} ne_someip_send_spec_t;

typedef struct ne_someip_recv_req_handler_info
{
	ne_someip_recv_request_handler handler;
	const void* user_data;
} ne_someip_recv_req_handler_info_t;

typedef struct ne_someip_recv_sub_handler_info
{
	ne_someip_recv_subscribe_handler handler;
	const void* user_data;	
} ne_someip_recv_sub_handler_info_t;

typedef struct ne_someip_resp_handler_info
{
	ne_someip_send_resp_status_handler handler;
	const void* user_data;
} ne_someip_resp_handler_info_t;

typedef struct ne_someip_notify_handler_info
{
	ne_someip_send_event_status_handler handler;
	const void* user_data;
} ne_someip_notify_handler_info_t;

typedef struct ne_someip_ser_status_handler_info
{
	ne_someip_offer_status_handler handler;
	const void* user_data;
} ne_someip_ser_status_handler_info_t;

typedef struct ne_someip_ser_instance_sync
{
	ne_someip_provided_instance_t* instance;
	pthread_t tid;
}ne_someip_ser_instance_sync_t;

typedef struct ne_someip_subscriber_addr
{
	uint32_t ip_addr;
	uint16_t tcp_port;
	uint16_t udp_port;
	ne_someip_address_type_t type;
} ne_someip_subscriber_addr_t;

typedef struct ne_someip_subscriber_usr_data
{
	ne_someip_subscriber_addr_t sub_addr; 
	ne_someip_provided_eg_t* eg_behaviour;
} ne_someip_subscriber_usr_data_t;

typedef struct ne_someip_eg_behaviour_subscriber
{
	ne_someip_provided_eg_t* eg_behaviour;
	uint32_t ttl;
	uint32_t client_addr;
	ne_someip_address_type_t type;
	bool reliable_flag;
	bool unreliable_flag;
	uint16_t tcp_port;
	uint16_t udp_port;
	ne_someip_remote_client_info_t remote_addr;
} ne_someip_eg_behaviour_subscriber_t;

typedef struct ne_someip_ser_sync_wait_timer
{
    ne_someip_server_context_t* context;
    ne_someip_sequence_id_t seq_id;
    pthread_t tid;
    ne_someip_ser_timer_type_t type;
}ne_someip_ser_sync_wait_timer_t;

typedef struct ne_someip_server_internal_config
{
	ne_someip_service_id_t service_id;
	ne_someip_instance_id_t instance_id;
	ne_someip_major_version_t major_version;
	ne_someip_minor_version_t minor_version;
	bool reliable_flag;
	bool unreliable_flag;
	uint16_t tcp_port;
	uint16_t udp_port;
	uint32_t ip_addr;
	ne_someip_address_type_t addr_type;
	ne_someip_server_offer_time_config_t time;
	uint32_t multicast_addr;
	uint16_t multicast_port;
} ne_someip_server_internal_config_t;

#endif // MANAGER_SERVER_NE_SOMEIP_SERVER_DEFINE_H
/* EOF */
