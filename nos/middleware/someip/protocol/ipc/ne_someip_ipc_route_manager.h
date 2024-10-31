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
#ifndef SRC_PROTOCOL_IPC_NE_SOMEIP_IPC_ROUTE_MANAGER_H
#define SRC_PROTOCOL_IPC_NE_SOMEIP_IPC_ROUTE_MANAGER_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_endpoint_define.h"

typedef struct ne_someip_ipc_route_manager
{
    ne_someip_map_t* offer_service_map;
    ne_someip_map_t* reg_service_status_map;
    ne_someip_map_t* sub_eventgroup_map;
}ne_someip_ipc_route_manager_t;

ne_someip_ipc_route_manager_t* ne_someip_ipc_route_mgr_create();

void ne_someip_ipc_route_mgr_destroy();

ne_someip_endpoint_unix_addr_t*
ne_someip_ipc_route_mgr_find_unix_from_offer(ne_someip_service_instance_spec_t key);

bool ne_someip_ipc_route_mgr_save_unix_from_offer(ne_someip_service_instance_spec_t key,
	ne_someip_endpoint_unix_addr_t* unix_addr);

bool ne_someip_ipc_route_mgr_delete_unix_from_offer(ne_someip_service_instance_spec_t key);

ne_someip_endpoint_unix_addr_t*
ne_someip_ipc_route_mgr_find_unix_from_status_handle(ne_someip_service_instance_spec_t key);

bool ne_someip_ipc_route_mgr_save_unix_from_status_handle(ne_someip_service_instance_spec_t key,
	ne_someip_endpoint_unix_addr_t* unix_addr);

bool ne_someip_ipc_route_mgr_delete_unix_from_status_handle(ne_someip_service_instance_spec_t key);

ne_someip_endpoint_unix_addr_t*
ne_someip_ipc_route_mgr_find_unix_from_sub_eg(ne_someip_service_instance_spec_t key);

bool ne_someip_ipc_route_mgr_save_unix_from_sub_eg(ne_someip_service_instance_spec_t key,
	ne_someip_endpoint_unix_addr_t* unix_addr);

bool ne_someip_ipc_route_mgr_delete_unix_from_sub_eg(ne_someip_service_instance_spec_t key);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_IPC_NE_SOMEIP_IPC_ROUTE_MANAGER_H
/* EOF */