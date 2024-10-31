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
#include "ne_someip_ipc_route_manager.h"

static ne_someip_ipc_route_manager_t* g_someip_ipc_route_manager = NULL;

ne_someip_ipc_route_manager_t* ne_someip_ipc_route_mgr_create()
{
	return NULL;
}

void ne_someip_ipc_route_mgr_destroy()
{
	return;
}

ne_someip_endpoint_unix_addr_t*
ne_someip_ipc_route_mgr_find_unix_from_offer(ne_someip_service_instance_spec_t key)
{
	return NULL;
}

bool ne_someip_ipc_route_mgr_save_unix_from_offer(ne_someip_service_instance_spec_t key,
	ne_someip_endpoint_unix_addr_t* unix_addr)
{
	return true;
}

bool ne_someip_ipc_route_mgr_delete_unix_from_offer(ne_someip_service_instance_spec_t key)
{
	return true;
}

ne_someip_endpoint_unix_addr_t*
ne_someip_ipc_route_mgr_find_unix_from_status_handle(ne_someip_service_instance_spec_t key)
{
	return NULL;
}

bool ne_someip_ipc_route_mgr_save_unix_from_status_handle(ne_someip_service_instance_spec_t key,
	ne_someip_endpoint_unix_addr_t* unix_addr)
{
	return true;
}

bool ne_someip_ipc_route_mgr_delete_unix_from_status_handle(ne_someip_service_instance_spec_t key)
{
	return true;
}

ne_someip_endpoint_unix_addr_t*
ne_someip_ipc_route_mgr_find_unix_from_sub_eg(ne_someip_service_instance_spec_t key)
{
	return NULL;
}

bool ne_someip_ipc_route_mgr_save_unix_from_sub_eg(ne_someip_service_instance_spec_t key,
	ne_someip_endpoint_unix_addr_t* unix_addr)
{
	return true;
}

bool ne_someip_ipc_route_mgr_delete_unix_from_sub_eg(ne_someip_service_instance_spec_t key)
{
	return true;
}
