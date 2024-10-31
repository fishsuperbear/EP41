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
#ifndef SRC_PROTOCOL_RPC_COMMON_REUSE_NE_SOMEIP_COMMON_REUSE_TOOL_H
#define SRC_PROTOCOL_RPC_COMMON_REUSE_NE_SOMEIP_COMMON_REUSE_TOOL_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "ne_someip_map.h"
#include "ne_someip_list.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_common_reuse_define.h"

// hash function
uint32_t ne_someip_comm_reuse_tool_method_key_hash(const ne_someip_reg_method_key_t* method_key);
uint32_t ne_someip_comm_reuse_tool_event_key_hash(const ne_someip_reg_event_key_t* event_key);
uint32_t ne_someip_comm_reuse_tool_sub_eg_key_hash(const ne_someip_reg_sub_eg_key_t* sub_eg_key);
uint32_t ne_someip_comm_reuse_tool_net_addr_pair_key_hash(const ne_someip_endpoint_net_addr_pair_t* addr_pair_key);

// compare function
int ne_someip_comm_reuse_tool_method_key_compare(const ne_someip_reg_method_key_t* key1,
    const ne_someip_reg_method_key_t* key2);
int ne_someip_comm_reuse_tool_event_key_compare(const ne_someip_reg_event_key_t* key1,
    const ne_someip_reg_event_key_t* key2);
int ne_someip_comm_reuse_tool_sub_eg_key_compare(const ne_someip_reg_sub_eg_key_t* key1,
	const ne_someip_reg_sub_eg_key_t* key2);
int ne_someip_comm_reuse_tool_net_addr_pair_key_compare(const ne_someip_endpoint_net_addr_pair_t* key1,
    const ne_someip_endpoint_net_addr_pair_t* key2);

// free function
void ne_someip_comm_reuse_tool_method_key_free(ne_someip_reg_method_key_t* info);
void ne_someip_comm_reuse_tool_event_key_free(ne_someip_reg_event_key_t* info);
void ne_someip_comm_reuse_tool_sub_eg_key_free(ne_someip_reg_sub_eg_key_t* info);
void ne_someip_comm_reuse_tool_method_info_free(ne_someip_reg_method_info_t* info);
void ne_someip_comm_reuse_tool_event_info_free(ne_someip_reg_event_info_t* info);
void ne_someip_comm_reuse_tool_event_eg_free(ne_someip_event_eg_info_t* info);
void ne_someip_comm_reuse_tool_sub_eg_info_free(ne_someip_list_t* sub_eg_list);
void ne_someip_comm_reuse_tool_net_addr_pair_free(ne_someip_endpoint_net_addr_pair_t* info);
void ne_someip_comm_reuse_tool_ipc_endpoint_info_free(ne_someip_ipc_endpoint_info_t* info);

// find/save/delete endpoint
void* ne_someip_comm_reuse_tool_find_endpoint(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_t* addr);
bool ne_someip_comm_reuse_tool_save_endpoint(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_t* key,
    ne_someip_endpoint_unix_addr_t* proxy_path, void* endpoint);
bool ne_someip_comm_reuse_tool_delete_endpoint(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_t* addr,
    ne_someip_endpoint_unix_addr_t* proxy_path);
bool ne_someip_comm_reuse_tool_is_endpoint_saved(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_t* addr);
bool ne_someip_comm_reuse_tool_remove_socket_by_unix_path(ne_someip_map_t* save_map, ne_someip_endpoint_unix_addr_t* addr);

// find/save/delete tcp endpoint link
bool ne_someip_comm_reuse_tool_is_tcp_link_saved(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_pair_t* net_pair);
bool ne_someip_comm_reuse_tool_save_tcp_link_unix_path(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_pair_t* net_pair,
    ne_someip_endpoint_unix_addr_t* proxy_path);
bool ne_someip_comm_reuse_tool_delete_tcp_link_unix_path(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_pair_t* net_pair,
    ne_someip_endpoint_unix_addr_t* proxy_path);
bool ne_someip_comm_reuse_tool_remove_link_by_unix_path(ne_someip_map_t* save_map, ne_someip_endpoint_unix_addr_t* addr);

// find/save/delete udp_multicast
bool ne_someip_comm_reuse_tool_is_udp_multicast_saved(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_t* addr);
bool ne_someip_comm_reuse_tool_save_udp_multicast(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_t* addr,
    ne_someip_endpoint_unix_addr_t* proxy_path);
bool ne_someip_comm_reuse_tool_delete_udp_multicast(ne_someip_map_t* save_map, ne_someip_endpoint_net_addr_t* addr,
    ne_someip_endpoint_unix_addr_t* proxy_path);
bool ne_someip_comm_reuse_tool_remove_multi_by_unix_path(ne_someip_map_t* save_map, ne_someip_endpoint_unix_addr_t* addr);

// find/save/delete req info
ne_someip_endpoint_unix_addr_t* ne_someip_comm_reuse_tool_find_req_info(ne_someip_map_t* save_map, ne_someip_reg_method_key_t* key);
bool ne_someip_comm_reuse_tool_save_req_info(ne_someip_map_t* save_map, ne_someip_reg_method_key_t* key,
    ne_someip_endpoint_unix_addr_t* client_unix_path);
bool ne_someip_comm_reuse_tool_delete_req_info(ne_someip_map_t* save_map, ne_someip_reg_method_key_t* key, bool is_resize);
bool ne_someip_comm_reuse_tool_is_req_info_saved(ne_someip_map_t* save_map, ne_someip_reg_method_key_t* key);
bool ne_someip_comm_reuse_tool_remove_req_info_by_unix_path(ne_someip_map_t* save_map, ne_someip_endpoint_unix_addr_t* addr);

// find/save/delete resp info
ne_someip_reg_method_info_t* ne_someip_comm_reuse_tool_find_resp_info(ne_someip_map_t* save_map, ne_someip_reg_method_key_t* key);
bool ne_someip_comm_reuse_tool_save_resp_info(ne_someip_map_t* save_map, ne_someip_reg_method_key_t* key,
    ne_someip_endpoint_unix_addr_t* client_unix_path);
bool ne_someip_comm_reuse_tool_delete_resp_info(ne_someip_map_t* save_map, ne_someip_reg_method_key_t* key, bool is_resize);
bool ne_someip_comm_reuse_tool_is_resp_info_saved(ne_someip_map_t* save_map, ne_someip_reg_method_key_t* key);
bool ne_someip_comm_reuse_tool_remove_resp_info_by_unix_path(ne_someip_map_t* save_map, ne_someip_endpoint_unix_addr_t* addr);

// find/save/delete event info
ne_someip_reg_event_info_t* ne_someip_comm_reuse_tool_find_event_info(ne_someip_map_t* save_map, ne_someip_reg_event_key_t* key);
bool ne_someip_comm_reuse_tool_save_event_info(ne_someip_map_t* save_map, ne_someip_reg_event_key_t* key,
    ne_someip_endpoint_unix_addr_t* client_unix_path, ne_someip_list_t* eg_list);
bool ne_someip_comm_reuse_tool_delete_event_info(ne_someip_map_t* save_map, ne_someip_reg_event_key_t* key, bool is_resize);
bool ne_someip_comm_reuse_tool_is_event_info_saved(ne_someip_map_t* save_map, ne_someip_reg_event_key_t* key);
bool ne_someip_comm_reuse_tool_update_event_info(ne_someip_reg_event_info_t* event_info,
    ne_someip_endpoint_unix_addr_t* client_unix_path, ne_someip_list_t* eg_list);
bool ne_someip_comm_reuse_tool_save_sub_eg(ne_someip_map_t* sub_eg_map, ne_someip_eventgroup_id_t* eg_id);
bool ne_someip_comm_reuse_tool_save_reg_unix(ne_someip_map_t* reg_event_map, ne_someip_endpoint_unix_addr_t* addr,
    bool is_reg);
bool ne_someip_comm_reuse_tool_update_reg_unix(ne_someip_map_t* reg_event_map, ne_someip_endpoint_unix_addr_t* addr,
    bool is_reg);
bool ne_someip_comm_reuse_tool_save_sub_unix_path(ne_someip_map_t* sub_eg_path_map, ne_someip_reg_sub_eg_key_t* key,
    ne_someip_endpoint_unix_addr_t* unix_path);
bool ne_someip_comm_reuse_tool_delete_sub_unix_path(ne_someip_map_t* sub_eg_path_map, ne_someip_reg_sub_eg_key_t* key,
    ne_someip_endpoint_unix_addr_t* unix_path);
bool ne_someip_comm_reuse_tool_update_event_reg_status(ne_someip_map_t* sub_eg_map, ne_someip_map_t* reg_event_map,
    ne_someip_endpoint_unix_addr_t* unix_path);
bool ne_someip_comm_reuse_tool_remove_event_info_by_unix_path(ne_someip_map_t* save_map, ne_someip_endpoint_unix_addr_t* addr);

ne_someip_endpoint_buffer_t* ne_someip_comm_reuse_tool_create_ep_buffer(uint8_t* data, uint32_t size);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_RPC_COMMON_REUSE_NE_SOMEIP_COMMON_REUSE_TOOL_H
/* EOF */
