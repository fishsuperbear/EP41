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
#ifndef SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_RECV_H
#define SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_RECV_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_endpoint_define.h"

ne_someip_transmit_buffer_t* ne_someip_ep_recv_alloc_memory_tcp(ne_someip_endpoint_tcp_data_t* endpoint,
    const ne_someip_endpoint_net_addr_t* peer_address, int avalible_size);

ne_someip_transmit_buffer_t* ne_someip_ep_recv_alloc_memory_udp(void* endpoint, int avalible_size);

ne_someip_transmit_buffer_t* ne_someip_ep_recv_alloc_memory_unix(ne_someip_endpoint_unix_t* endpoint,
    const ne_someip_endpoint_unix_addr_t* peer_address, int avalible_size);

ne_someip_error_code_t ne_someip_ep_recv_receive_tcp(ne_someip_endpoint_tcp_data_t* endpoint, ne_someip_transmit_normal_buffer_t* buffer,
    ne_someip_endpoint_net_addr_t* peer_addr, ne_someip_transmit_io_result_t result);

ne_someip_error_code_t ne_someip_ep_recv_receive_udp(void* endpoint, ne_someip_transmit_iov_buffer_t* buffer,
    ne_someip_endpoint_net_addr_t* peer_addr, ne_someip_transmit_io_result_t result);

ne_someip_error_code_t ne_someip_ep_recv_receive_unix(ne_someip_endpoint_unix_t* endpoint, ne_someip_transmit_normal_buffer_t* buffer,
    ne_someip_endpoint_unix_addr_t* peer_addr, ne_someip_transmit_io_result_t result);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_RECV_H
/* EOF */
