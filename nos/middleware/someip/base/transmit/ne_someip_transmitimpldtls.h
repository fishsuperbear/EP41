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
// #ifndef BASE_TRANSMIT_NE_SOMEIP_TRANSMITIMPLDTLS_H
// #define BASE_TRANSMIT_NE_SOMEIP_TRANSMITIMPLDTLS_H

// #ifdef  __cplusplus
// extern "C" {
// #endif

// #include "ne_someip_transmitimpl.h"
// #include <openssl1_1/openssl/ssl.h>
// #include <openssl1_1/openssl/err.h>
// #include <openssl1_1/openssl/bio.h>

// typedef struct {
//     ne_someip_transmit_impl_t base;
//     int listen_fd;
//     char if_name[128];
//     int hops;
//     ne_someip_list_t* multicast_addr_list;   // 已经加入的组播列表,节点类型为 ne_someip_enpoint_multicast_addr_t
//     SSL_CTX* m_ctx;
//     ne_someip_map_t* udp_dtls_map; // key = (ne_someip_endpoint_net_addr_t)addr, value= (ne_someip_transmit_impl_dtls_ssl_info_t)info
// } ne_someip_transmit_impl_dtls_t;

// ne_someip_transmit_impl_dtls_t* ne_someip_transmit_impl_dtls_new();
// void ne_someip_transmit_impl_dtls_free(ne_someip_transmit_impl_dtls_t *impl);

// int ne_someip_transmit_impl_dtls_set_config(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core,
//     ne_someip_transmit_config_type_t type, const void* config);
// int ne_someip_transmit_impl_dtls_get_config(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core,
//     ne_someip_transmit_config_type_t type, void* config);
// int ne_someip_transmit_impl_dtls_prepare(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core);
// int ne_someip_transmit_impl_dtls_start(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core);
// int ne_someip_transmit_impl_dtls_stop(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core);
// int ne_someip_transmit_impl_dtls_query(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_query_type_t type, void* data);
// int ne_someip_transmit_impl_dtls_in_connection(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int listen_fd);
// ne_someip_transmit_core_send_result_t ne_someip_transmit_impl_dtls_send(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_buffer_t* buffer, void* peer_address);
// ne_someip_transmit_core_send_result_t ne_someip_transmit_impl_dtls_send_by_fd(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_buffer_t* buffer, int fd);
// ne_someip_transmit_core_recv_result_t ne_someip_transmit_impl_dtls_recv(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int data_fd, ne_someip_transmit_buffer_t* buffer, void* peer_address, void* local_address);
// void ne_someip_transmit_impl_dtls_on_error(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int fd, int error);
// int32_t ne_someip_transmit_impl_dtls_address_equal(void* addr1, void* addr2);
// int ne_someip_transmit_impl_dtls_join_group(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_group_type_t group_type,
//     void* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);
// int ne_someip_transmit_impl_dtls_leave_group(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_group_type_t group_type,
//     void* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);
// int ne_someip_transmit_impl_dtls_link_setup(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core,
//     ne_someip_transmit_link_t* link, void* peer_address);
// int ne_someip_transmit_impl_dtls_link_teardown(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int fd, void* peer_address);
// bool ne_someip_transmit_impl_dtls_link_supported_chk(ne_someip_transmit_impl_t* impl);
// void ne_someip_transmit_impl_dtls_fd_removed(ne_someip_transmit_impl_t* impl, int fd, void* peer_address);
// void* ne_someip_transmit_impl_dtls_address_create();
// uint32_t ne_someip_transmit_impl_dtls_addr_hash(void* addr);

// #ifdef  __cplusplus
// }
// #endif
// #endif  // BASE_TRANSMIT_NE_SOMEIP_TRANSMITIMPLDTLS_H
// /* EOF */
