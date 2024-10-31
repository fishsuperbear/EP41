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
#ifndef BASE_TRANSMIT_NE_SOMEIP_TRANSMITIMPLTCP_H
#define BASE_TRANSMIT_NE_SOMEIP_TRANSMITIMPLTCP_H

#ifdef  __cplusplus
extern "C" {
#endif

#include "ne_someip_transmitimpl.h"
#include "ne_someip_map.h"

typedef struct {
    ne_someip_transmit_impl_t base;
    uint32_t listen_num;
    char if_name[128];
    ne_someip_map_t* map_connect_info;
} ne_someip_transmit_impl_tcp_t;

ne_someip_transmit_impl_tcp_t* ne_someip_transmit_impl_tcp_new();
void ne_someip_transmit_impl_tcp_free(ne_someip_transmit_impl_tcp_t *impl);

int ne_someip_transmit_impl_tcp_set_config(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core,
    ne_someip_transmit_config_type_t type, const void* config);
int ne_someip_transmit_impl_tcp_get_config(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core,
    ne_someip_transmit_config_type_t type, void* config);
int ne_someip_transmit_impl_tcp_prepare(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core);
int ne_someip_transmit_impl_tcp_start(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core);
int ne_someip_transmit_impl_tcp_stop(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core);
int ne_someip_transmit_impl_tcp_query(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_query_type_t type, void* data);
int ne_someip_transmit_impl_tcp_in_connection(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int listen_fd);
ne_someip_transmit_core_send_result_t ne_someip_transmit_impl_tcp_send(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_buffer_t* buffer, void* peer_address);
ne_someip_transmit_core_send_result_t ne_someip_transmit_impl_tcp_send_by_fd(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_buffer_t* buffer, int fd);
ne_someip_transmit_core_recv_result_t ne_someip_transmit_impl_tcp_recv(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int data_fd, ne_someip_transmit_buffer_t* buffer, void* peer_address, void* local_address);
void ne_someip_transmit_impl_tcp_on_error(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int fd, int error);
int32_t ne_someip_transmit_impl_tcp_address_equal(const void* addr1, const void* addr2);
int ne_someip_transmit_impl_tcp_join_group(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_group_type_t group_type,
    void* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);
int ne_someip_transmit_impl_tcp_leave_group(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_group_type_t group_type,
    void* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);
int ne_someip_transmit_impl_tcp_link_setup(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core,
    ne_someip_transmit_link_t* link, void* peer_address);
int ne_someip_transmit_impl_tcp_link_teardown(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int fd, void* peer_address);
bool ne_someip_transmit_impl_tcp_link_supported_chk(ne_someip_transmit_impl_t* impl);
void ne_someip_transmit_impl_tcp_fd_removed(ne_someip_transmit_impl_t* impl, int fd, void* peer_address);
void* ne_someip_transmit_impl_tcp_malloc_addr();
uint32_t ne_someip_transmit_impl_tcp_addr_hash(const void* addr);

#ifdef  __cplusplus
}
#endif
#endif  // BASE_TRANSMIT_NE_SOMEIP_TRANSMITIMPLTCP_H
/* EOF */
