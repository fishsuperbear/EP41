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
#ifndef SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_UDP_SD_H
#define SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_UDP_SD_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_endpoint_define.h"
#include "ne_someip_looper.h"
#include "ne_someip_endpoint_common_func.h"

/**
 * @brief 创建ne_someip_endpoint_udp_sd_t对象，ne_someip_endpoint_udp_sd_t对象初始引用计数为1. (work线程中运行)
 *
 * @param [in] type : The identifier type of the endpoint.
 * @param [in] addr : The local unix addr of the unix endpoint.
 * @param [in] looper : The looper of the IO thread.
 *
 * @return 返回ne_someip_endpoint_udp_sd_t对象指针.
 *
 * @attention Synchronous I/F.
 */
ne_someip_endpoint_udp_sd_t* ne_someip_endpoint_udp_sd_create(
    ne_someip_endpoint_type_t type, ne_someip_endpoint_net_addr_t* addr, ne_someip_looper_t* looper);

/**
 * @brief 引用ne_someip_endpoint_udp_sd_t对象，ne_someip_endpoint_udp_sd_t对象引用计数加1. (work线程中运行)
 *        可能多线程访问，非多线程安全，是否需要加锁.
 *
 * @param [in] endpoint : ne_someip_endpoint_udp_sd_t对象指针.
 *
 * @return 返回ne_someip_endpoint_udp_sd_t对象指针.
 *
 * @attention Synchronous I/F.
 */
ne_someip_endpoint_udp_sd_t* ne_someip_endpoint_udp_sd_ref(ne_someip_endpoint_udp_sd_t* endpoint);

/**
 * @brief 释放ne_someip_endpoint_udp_sd_t对象引用，ne_someip_endpoint_udp_sd_t对象引用计数减1. (work线程中运行)
 *        可能多线程访问，非多线程安全，是否需要加锁.
 *
 * @param [in] endpoint : ne_someip_endpoint_udp_sd_t对象指针.
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_udp_sd_unref(ne_someip_endpoint_udp_sd_t* endpoint);

/**
 * @brief 保存加入组播地址,加入组播，io线程调用，io线程执行，不转线程 (收组播的时候，以组播的IP和port创建udp endpoint)
 *
 * @param [in] endpoint : ne_someip_endpoint_udp_sd_t对象指针.
 * @param [in] group_addr : 收组播的local ip地址
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_udp_sd_join_group(
    ne_someip_endpoint_udp_sd_t* endpoint, ne_someip_endpoint_net_addr_t* interface_addr);

/**
 * @brief 删除组播地址，移除组播，io线程调用，io线程执行，不转线程
 *
 * @param [in] endpoint : ne_someip_endpoint_udp_sd_t对象指针.
 * @param [in] group_addr : 收组播的local ip地址
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_udp_sd_leave_group(
    ne_someip_endpoint_udp_sd_t* endpoint, ne_someip_endpoint_net_addr_t* interface_addr);

/**
 * @brief endpoint发送数据的send异步接口。
 *
 * @param [in] endpoint : ne_someip_endpoint_udp_sd_t对象指针.
 * @param [in] trans_buffer : 发送的buffer list数据
 * @param [in] peer_addr : 发送对端udp的地址
 * @param [in] send_policy : 发送策略配置参数
 * @param [in] seq_data : defined and used by user
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_udp_sd_send_async(ne_someip_endpoint_udp_sd_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer, ne_someip_endpoint_net_addr_t* peer_addr,
    ne_someip_endpoint_send_policy_t* send_policy, const void* seq_data);

/***********************callback**************************/

/**
 * @brief transmit状态变化时通知。在io线程中调用执行，保存相应的状态
 *
 * @param [in] endpoint : ne_someip_endpoint_udp_sd_t对象指针
 * @param [in] state : tranmit变化后的状态
 * @param [in] peer_addr : 建立连接对端tcp的地址
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_udp_sd_transmit_state_change(ne_someip_endpoint_udp_sd_t* endpoint, ne_someip_endpoint_transmit_state_t state);

/**
 * @brief 加入组播状态变化时通知。在io线程中调用执行，保存相应的状态
 *
 * @param [in] endpoint : ne_someip_endpoint_udp_sd_t对象指针
 * @param [in] group_addr : 组播的地址
 * @param [in] state : 加入/移除组播变化的状态
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
// void ne_someip_endpoint_udp_sd_add_multicast_addr_state_change(ne_someip_endpoint_udp_sd_t* endpoint,
//     ne_someip_enpoint_multicast_addr_t* group_addr, ne_someip_endpoint_add_multicast_addr_state_t state);

void ne_someip_endpoint_udp_sd_async_send_reply(ne_someip_endpoint_udp_sd_t* endpoint,
    const void* seq_data, ne_someip_error_code_t result);

/**
 * @brief 收到数据后调用的接口，io线程中调用执行回调函数
 *
 * @param [in] endpoint : ne_someip_endpoint_udp_sd_t对象指针.
 * @param [in] trans_buffer : 接收到完整一包数据的buffer list
 * @param [in] size : 接收到数据的长度
 * @param [in] peer_addr : 接收数据对端udp的地址
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_udp_sd_on_receive(ne_someip_endpoint_udp_sd_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer, uint32_t size, ne_someip_endpoint_net_addr_t* peer_addr);

/**
 * @brief 收到数据后分发给不同instance注册的receiver. (work线程中运行) (proxy进程使用接口)
 *
 * @param [in] endpoint : ne_someip_endpoint_udp_sd_t对象指针.
 * @param [in] trans_buffer : 接收到完整一包数据的buffer list
 *
 * @return ne_someip_list_t <void*>
 *
 * @attention Synchronous I/F
 */
ne_someip_list_t* ne_someip_endpoint_udp_sd_dispatcher_instance(ne_someip_endpoint_udp_sd_t* endpoint,
	ne_someip_trans_buffer_struct_t* trans_buffer);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_UDP_SD_H
/* EOF */