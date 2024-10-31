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
#ifndef SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_TCP_DATA_H
#define SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_TCP_DATA_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_endpoint_define.h"
#include "ne_someip_looper.h"
#include "ne_someip_endpoint_common_func.h"

/* @注释：
 * proxy进程,transmit、transmit link状态的维护在work线程
 * daemon进程，直接在io线程进行转发，transmit、transmit link状态的维护直接在io线程
 *
 * 本文件函数，除去明确说明适用于proxy进程或daemon进程的接口，其他接口通用
 */

/**
 * @brief 创建ne_someip_endpoint_tcp_data_t对象，ne_someip_endpoint_tcp_data_t对象初始引用计数为1. (work线程中运行)
 *
 * @param [in] type : The identifier type of the endpoint.
 * @param [in] addr : The local unix addr of the unix endpoint.
 * @param [in] looper : The looper of the IO thread.
 *
 * @return 返回ne_someip_endpoint_tcp_data_t对象指针.
 *
 * @attention Synchronous I/F.
 */
ne_someip_endpoint_tcp_data_t* ne_someip_endpoint_tcp_data_create(
    ne_someip_endpoint_type_t type, ne_someip_endpoint_net_addr_t* addr, ne_someip_looper_t* looper,
    ne_someip_endpoint_role_type_t role_type, bool is_need_switch_thread, const ne_someip_ssl_key_info_t* key_info);

/**
 * @brief 引用 ne_someip_endpoint_tcp_data_t, ne_someip_endpoint_tcp_data_t 对象引用计数加1. (work线程中运行)
 *        可能多线程访问，非多线程安全，是否需要加锁.
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t 对象指针.
 *
 * @return 返回 ne_someip_endpoint_tcp_data_t 对象指针.
 *
 * @attention Synchronous I/F.
 */
ne_someip_endpoint_tcp_data_t* ne_someip_endpoint_tcp_data_ref(ne_someip_endpoint_tcp_data_t* endpoint);

/**
 * @brief 释放 ne_someip_endpoint_tcp_data_t 对象引用， ne_someip_endpoint_tcp_data_t 对象引用计数减1. (work线程中运行)
 *        可能多线程访问，非多线程安全，是否需要加锁.
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t 对象指针.
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_tcp_data_unref(ne_someip_endpoint_tcp_data_t* endpoint);

/**
 * @brief 调用transmit_stop，进行相应处理
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t 对象指针.
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_tcp_data_stop(ne_someip_endpoint_tcp_data_t* endpoint);

/**
 * @brief 创建tcp连接动作，建立连接
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t对象指针.
 * @param [in] peer_addr : 建立连接对端tcp的地址
 * @param [in] role : 作为连接link的role角色(client端或server端)
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_tcp_data_create_transmit_link(
    ne_someip_endpoint_tcp_data_t* endpoint, ne_someip_endpoint_net_addr_t* peer_addr, ne_someip_endpoint_link_role_t role);

/**
 * @brief 销毁tcp连接动作，断开连接
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t对象指针.
 * @param [in] peer_addr : 销毁连接对端tcp的地址
 * @param [in] role : 作为连接link的role角色(client端或server端)
 *
 * @return void.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_tcp_data_destroy_transmit_link(
    ne_someip_endpoint_tcp_data_t* endpoint, ne_someip_endpoint_net_addr_t* peer_addr, ne_someip_endpoint_link_role_t role);

/**
 * @brief endpoint发送数据的send接口，work线程调用，具体处理转到io线程进行 (proxy进程使用接口)
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t对象指针.
 * @param [in] trans_buffer : 发送的buffer list数据
 * @param [in] peer_addr : 发送对端tcp的地址
 * @param [in] send_policy : 发送策略
 * @param [in] seq_data : seq_data用户定义数据
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Asynchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_tcp_data_send_async(ne_someip_endpoint_tcp_data_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer, ne_someip_endpoint_net_addr_t* peer_addr,
    ne_someip_endpoint_send_policy_t* send_policy, const void* seq_data);

// check if the endpoint use tls
bool ne_someip_endpoint_tcp_data_is_tls_used(const ne_someip_endpoint_tcp_data_t* endpoint);

bool ne_someip_endpoint_tcp_data_get_link_remote_addr(const ne_someip_endpoint_tcp_data_t* endpoint,
    ne_someip_list_t* remote_addr_list);

/***********************internal func**************************/
ne_someip_error_code_t ne_someip_endpoint_tcp_data_create_trans_link_act(
    ne_someip_endpoint_tcp_data_t* endpoint, ne_someip_endpoint_net_addr_t* peer_addr, ne_someip_endpoint_link_role_t role);

ne_someip_error_code_t ne_someip_endpoint_tcp_data_destroy_trans_link_act(
    ne_someip_endpoint_tcp_data_t* endpoint, ne_someip_endpoint_net_addr_t* peer_addr, ne_someip_endpoint_link_role_t role);

/***********************callback**************************/
/**
 * @brief transmit状态变化时通知。proxy进程，在work线程中调用执行回调函数，保存相应的状态；daemon进程，在io线程中调用执行，保存相应的状态
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t对象指针
 * @param [in] state : tranmit变化后的状态
 * @param [in] peer_addr : 建立连接对端tcp的地址
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_tcp_data_transmit_state_change(ne_someip_endpoint_tcp_data_t* endpoint, ne_someip_endpoint_transmit_state_t state);

/**
 * @brief transmit link状态变化时通知。proxy进程，work线程中调用执行回调函数，保存相应的状态；daemon进程，在io线程中调用执行，保存相应的状态
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t对象指针
 * @param [in] link : ne_someip_transmit_link_t对象指针
 * @param [in] state : tranmit link变化后的状态
 * @param [in] peer_addr : 建立连接对端tcp的地址
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_tcp_data_transmit_link_state_change(ne_someip_endpoint_tcp_data_t* endpoint, ne_someip_transmit_link_t *link,
    ne_someip_endpoint_transmit_link_state_t state, ne_someip_endpoint_net_addr_t* peer_addr);

void ne_someip_endpoint_tcp_data_async_send_reply(ne_someip_endpoint_tcp_data_t* endpoint,
    const void* seq_data, ne_someip_error_code_t result);

/**
 * @brief 收到数据后调用的接口，work线程中调用执行回调函数 (proxy进程使用接口)
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t对象指针.
 * @param [in] trans_buffer : 接收到完整一包数据的buffer list
 * @param [in] size : 接收到数据的长度
 * @param [in] peer_addr : 接收数据对端tcp的地址
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_tcp_data_on_receive(ne_someip_endpoint_tcp_data_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer, uint32_t size, ne_someip_endpoint_net_addr_t* peer_addr);

/**
 * @brief server receive the peer found notify when client connect to server
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t object pointer.
 * @param [in] peer_addr : client net addr
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_tcp_data_peer_founded(ne_someip_endpoint_tcp_data_t* endpoint, ne_someip_endpoint_net_addr_t* peer_address);

/**
 * @brief 收到数据后分发给不同instance注册的receiver. (work线程中运行) (proxy进程使用接口)
 *
 * @param [in] endpoint : ne_someip_endpoint_tcp_data_t对象指针.
 * @param [in] trans_buffer : 接收到完整一包数据的buffer list
 *
 * @return ne_someip_list_t <void*>
 *
 * @attention Synchronous I/F
 */
ne_someip_list_t* ne_someip_endpoint_tcp_data_dispatcher_instance(ne_someip_endpoint_tcp_data_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_TCP_DATA_H
/* EOF */
