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
#ifndef SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_UNIX_H
#define SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_UNIX_H
#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_endpoint_define.h"
#include "ne_someip_endpoint_common_func.h"
#include "ne_someip_looper.h"
#include "ne_someip_thread.h"

/* @注释：
 * proxy进程,transmit、transmit link状态的维护在work线程
 * daemon进程，直接在io线程进行转发，transmit、transmit link状态的维护直接在io线程
 *
 * 本文件函数，除去明确说明适用于proxy进程或daemon进程的接口，其他接口通用
 */

/**
 * @brief 创建ne_someip_endpoint_unix_t对象，ne_someip_endpoint_unix_t对象初始引用计数为1. (work线程中运行)
 *
 * @param [in] endpoint_type : The identifier type of the endpoint.
 * @param [in] addr : The local unix addr of the unix endpoint.
 * @param [in] looper : The looper of the IO thread.
 * @param [in] role_type : The role type of the endpoint.(server or client)
 *
 * @return 返回ne_someip_endpoint_unix_t对象指针.
 *
 * @attention Synchronous I/F.
 */
ne_someip_endpoint_unix_t* ne_someip_endpoint_unix_create(
    ne_someip_endpoint_type_t endpoint_type, ne_someip_endpoint_unix_addr_t* local_addr,
    ne_someip_looper_t* work_looper, ne_someip_looper_t* io_looper,
    ne_someip_endpoint_role_type_t role_type, bool is_need_switch_thread);

/**
 * @brief 引用 ne_someip_endpoint_unix_t, ne_someip_endpoint_unix_t 对象引用计数加1. (work线程中运行)
 *        可能多线程访问，非多线程安全，是否需要加锁.
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t 对象指针.
 *
 * @return 返回 ne_someip_endpoint_unix_t 对象指针.
 *
 * @attention Synchronous I/F.
 */
ne_someip_endpoint_unix_t* ne_someip_endpoint_unix_ref(ne_someip_endpoint_unix_t* endpoint);

/**
 * @brief 释放 ne_someip_endpoint_unix_t, ne_someip_endpoint_unix_t 对象引用计数减1. (work线程中运行)
 *        可能多线程访问，非多线程安全，是否需要加锁.
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t 对象指针.
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_unix_unref(ne_someip_endpoint_unix_t* endpoint);

/**
 * @brief 调用transmit_stop，进行相应处理
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t 对象指针.
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_unix_stop(ne_someip_endpoint_unix_t* endpoint);

/**
 * @brief 创建unix domain连接动作，建立连接，work线程调用，具体处理转到io线程进行
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t对象指针.
 * @param [in] peer_addr : 建立连接对端unix domain的地址
 * @param [in] role : 作为连接link的role角色(client端或server端)
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_unix_create_transmit_link(ne_someip_endpoint_unix_t* endpoint,
    ne_someip_endpoint_unix_addr_t* peer_addr, ne_someip_endpoint_link_role_t role);

/**
 * @brief 销毁unix domain连接动作，断开连接，work线程调用，具体处理转到io线程进行
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t对象指针.
 * @param [in] peer_addr : 销毁连接对端unix domain的地址
 * @param [in] role : 作为连接link的role角色(client端或server端)
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_unix_destroy_transmit_link(
    ne_someip_endpoint_unix_t* endpoint, ne_someip_endpoint_unix_addr_t* peer_addr, ne_someip_endpoint_link_role_t role);

/**
 * @brief endpoint发送数据的send异步接口。work线程调用，具体处理转到io线程进行 (proxy进程使用接口)
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t对象指针.
 * @param [in] ipc_header : 发送完整一包数据的 ipc header
 * @param [in] someip_header : 发送完整一包数据的 someip header，非转发someip数据时，值为NULL
 * @param [in] payload : 发送完整一包数据的 payload，无payload时，值为NULL
 * @param [in] peer_addr : 发送对端unix domain的地址
 * @param [in] seq_data : user data
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Asynchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_unix_send_async(ne_someip_endpoint_unix_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer, ne_someip_endpoint_unix_addr_t* peer_addr, const void* seq_data);

/***********************internal func**************************/
ne_someip_error_code_t ne_someip_endpoint_unix_create_trans_link_act(
    ne_someip_endpoint_unix_t* endpoint, ne_someip_endpoint_unix_addr_t* peer_addr, ne_someip_endpoint_link_role_t role);

ne_someip_error_code_t ne_someip_endpoint_unix_destroy_trans_link_act(
    ne_someip_endpoint_unix_t* endpoint, ne_someip_endpoint_unix_addr_t* peer_addr, ne_someip_endpoint_link_role_t role);


/***********************callback**************************/

/**
 * @brief transmit状态变化时通知。proxy进程，在work线程中调用执行回调函数，保存相应的状态；daemon进程，在io线程中调用执行，保存相应的状态
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t对象指针
 * @param [in] state : tranmit变化后的状态
 * @param [in] peer_addr : 建立连接对端unix domain的地址
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_unix_transmit_state_change(ne_someip_endpoint_unix_t* endpoint, ne_someip_endpoint_transmit_state_t state);

/**
 * @brief transmit link状态变化时通知。proxy进程，work线程中调用执行回调函数，保存相应的状态；daemon进程，在io线程中调用执行，保存相应的状态
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t对象指针
 * @param [in] link : ne_someip_transmit_link_t对象指针
 * @param [in] state : tranmit link变化后的状态
 * @param [in] peer_addr : 建立连接对端unix domain的地址
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_unix_transmit_link_state_change(ne_someip_endpoint_unix_t* endpoint, ne_someip_transmit_link_t *link,
    ne_someip_endpoint_transmit_link_state_t state, ne_someip_endpoint_unix_addr_t* peer_addr);

void ne_someip_endpoint_unix_async_send_reply(ne_someip_endpoint_unix_t* endpoint,
    const void* seq_data, ne_someip_error_code_t result);

/**
 * @brief 收到数据后调用的接口，work线程中调用执行回调函数 (proxy进程使用接口)
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t对象指针.
 * @param [in] ipc_header : 接收到完整一包数据的 ipc header
 * @param [in] someip_header : 接收到完整一包数据的 someip header，非转发someip数据时，值为NULL
 * @param [in] payload : 接收到完整一包数据的 payload，无payload时，值为NULL
 * @param [in] transmit_link : 接收到数据的transmit link
 * @param [in] peer_addr : 接收数据对端unix domain的地址
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_unix_on_receive(ne_someip_endpoint_unix_t* endpoint, ne_someip_trans_buffer_struct_t* trans_buffer,
    uint32_t size, ne_someip_endpoint_unix_addr_t* peer_addr);

/**
 * @brief server receive the peer found notify when client connect to server
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t object pointer.
 * @param [in] peer_addr : client unix domain addr
 *
 * @return void
 *
 * @attention Synchronous I/F.
 */
void ne_someip_endpoint_unix_peer_founded(ne_someip_endpoint_unix_t* endpoint, ne_someip_endpoint_unix_addr_t* peer_address);

/**
 * @brief 收到数据后分发给不同的instance. (work线程中运行) (proxy进程使用接口)
 *
 * @param [in] endpoint : ne_someip_endpoint_unix_t对象指针.
 * @param [in] trans_buffer : 接收到完整一包数据的buffer list
 *
 * @return ne_someip_list_t <void*>
 *
 * @attention Synchronous I/F
 */
ne_someip_list_t* ne_someip_endpoint_unix_dispatcher_instance(ne_someip_endpoint_unix_t* endpoint,
    ne_someip_trans_buffer_struct_t* trans_buffer);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_UNIX_H
/* EOF */
