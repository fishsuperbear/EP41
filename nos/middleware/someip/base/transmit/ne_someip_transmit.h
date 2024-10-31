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
#ifndef BASE_TRANSMIT_NE_SOMEIP_TRANSMIT_H
#define BASE_TRANSMIT_NE_SOMEIP_TRANSMIT_H

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "ne_someip_looper.h"
#include "ne_someip_transmitdefines.h"
#include "ne_someip_transmitimpl.h"
#include "ne_someip_transmitcore.h"
#include "ne_someip_internal_define.h"

typedef struct ne_someip_transmit ne_someip_transmit_t;

/** ne_someip_transmit_new
 * @type: transmit类型，参考ne_someip_transmit_type_t定义
 * @local_address: transmit对象绑定的本地地址, 具体类型由type决定:
 *                 TCP/UDP: ne_someip_endpoint_net_addr_t, 若type为UDP,分unicast和multicast两种: ip_addr分别为单播和组播ip地址;
 *                 UNIX_DOMAIN: ne_someip_endpoint_unix_addr_t;
 * @is_listen_mode: true 监听模式(TCP/Unix: server)；false 非监听模式(TCP/Unix: client)
 *
 * 创建ne_someip_transmit_t对象,ne_someip_transmit_t对象初始引用计数为1
 *
 * Returns: 成功:返回ne_someip_transmit_t对象指针; 失败:返回NULL
 */
ne_someip_transmit_t* ne_someip_transmit_new(ne_someip_transmit_type_t type, const void* local_address, bool is_listen_mode);

/** ne_someip_transmit_ref
 * @handle:
 *
 * 增加ne_someip_transmit_t对象的引用计数
 *
 * Returns: 成功:返回ne_someip_transmit_t对象指针; 失败:返回NULL
 */
ne_someip_transmit_t* ne_someip_transmit_ref(ne_someip_transmit_t* handle);

/** ne_someip_transmit_unref
 * @handle:
 *
 * 减少ne_someip_transmit_t对象的引用计数
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_unref(ne_someip_transmit_t* handle);

/** ne_someip_transmit_set_looper
 * @handle:
 * @looper: NELooper对象，给transmit运行提供支持
 *
 * 设置给transmit运行提供支持的looper对象。必须在ne_someip_transmit_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_set_looper(ne_someip_transmit_t* handle, ne_someip_looper_t* looper);

/** ne_someip_transmit_get_looper
 * @handle:
 *
 * 获取给transmit运行提供支持的looper对象
 *
 * Returns: 返回looper对象指针；失败则返回NULL
 */
ne_someip_looper_t* ne_someip_transmit_get_looper(ne_someip_transmit_t* handle);

/** ne_someip_transmit_set_config
 * @handle:
 * @type: 属性类型,参考ne_someip_transmit_config_type_t定义
 * @config: 属性值。实际类型与type相关
 *
 * 配置transmit的属性。必须在ne_someip_transmit_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_set_config(ne_someip_transmit_t* handle, ne_someip_transmit_config_type_t type, const void* config);

/** ne_someip_transmit_get_config
 * @handle:
 * @type: 属性类型,参考ne_someip_transmit_config_type_t定义
 * @config: 输出参数，属性值。实际类型与type相关
 *
 * 获取transmit的type属性类型的属性值
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_get_config(ne_someip_transmit_t* handle, ne_someip_transmit_config_type_t type, void* config);

/** ne_someip_transmit_set_callback
 * @handle:
 * @callback: 回调函数。用户可以通过回调获取状态变化，peer端连接建立等通知，参考ne_someip_transmit_callback_t定义
 * @user_data: user data, 透传参数,ne_someip_transmit_callback_t相关的回调中,被传递
 *
 * 设置transmit的回调函数。必须在ne_someip_transmit_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_set_callback(ne_someip_transmit_t* handle, ne_someip_transmit_callback_t* callback, void* user_data);

/** ne_someip_transmit_set_source
 * @handle:
 * @source: source对象。当transmit被trigger（调用ne_someip_transmit_link_trigger_available）时，
 *        transmit将通过source对象获取要发送的数据，并由source对象指定发送目标地址。
 * @user_data: user data, 透传参数,ne_someip_transmit_source_t相关的回调中,被传递
 *
 * 设置transmit的source对象。必须在ne_someip_transmit_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_set_source(ne_someip_transmit_t* handle, ne_someip_transmit_source_t* source, void* user_data);

/** ne_someip_transmit_set_sink
 * @handle:
 * @sink: sink对象。当transmit有监听到新数据被接收时，transmit将通过sink对象将接收到的
 *        数据传递给用户
 * @user_data: user data, 透传参数,ne_someip_transmit_sink_t相关的回调中,被传递
 *
 * 设置transmit的sink对象。必须在ne_someip_transmit_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_set_sink(ne_someip_transmit_t* handle, ne_someip_transmit_sink_t* sink, void* user_data);

/** ne_someip_transmit_prepare
 * @handle:
 *
 * transmit使用用户配置的各种属性，进行工作前的准备工作。
 * 比如预分配buffer等
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_prepare(ne_someip_transmit_t* handle);

/** ne_someip_transmit_start
 * @handle:
 *
 * transmit开始工作
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_start(ne_someip_transmit_t* handle);

/** ne_someip_transmit_stop
 * @handle:
 *
 * transmit停止工作。perpare阶段申请的资源，比如预分配buffer等，都将被释放
 * ne_someip_transmit_t对象的引用计数减1
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_stop(ne_someip_transmit_t* handle);

/** ne_someip_transmit_get_state
 * @handle:
 *
 * 获取transmit的状态。
 *
 * Returns: state
 */
ne_someip_transmit_state_t ne_someip_transmit_get_state(ne_someip_transmit_t* handle);

/** ne_someip_transmit_query
 * @handle:
 * @type: 查询类型, transmit支持的查询类型，参考ne_someip_transmit_query_type_t定义
 * @data: 输出参数，查询结果。实际类型与type相关
 *
 * 查询
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_query(ne_someip_transmit_t* handle, ne_someip_transmit_query_type_t type, void* data);

/** ne_someip_transmit_join_group
 * @handle:
 * @group_type: 组播类型, 参考ne_someip_transmit_group_type_t定义
 * @unicast_addr: 绑定本地网卡的ip地址(单播)
 * @group_address: 组播地址
 *
 * 加入组播
 * 这种情况,ne_someip_transmit_new接口local_address传入的是组播ip和端口,需和group_address地址一致,
 * unicast_addr传入绑定本地网卡的ip地址(单播)
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_join_group(ne_someip_transmit_t* handle, ne_someip_transmit_group_type_t group_type,
    ne_someip_endpoint_net_addr_t* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);

/** ne_someip_transmit_leave_group
 * @handle:
 * @type: 组播类型, 参考ne_someip_transmit_group_type_t定义
 * @unicast_addr: 绑定本地网卡的ip地址(单播)
 * @group_address: 组播地址
 *
 * 离开组播
 * 这种情况,ne_someip_transmit_new接口local_address传入的是组播ip和端口,需和group_address地址一致,
 * unicast_addr传入绑定本地网卡的ip地址(单播)
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_leave_group(ne_someip_transmit_t* handle, ne_someip_transmit_group_type_t group_type,
    ne_someip_endpoint_net_addr_t* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);

/** ne_someip_transmit_link_trigger_available
 * @handle:
 * @type: 数据收发类型。NE_SOMEIP_TRANSMIT_LINK_TYPE_SOURCE：数据发送；NE_SOMEIP_TRANSMIT_LINK_TYPE_SINK：数据接收
 * @link: 指定收发数据link，对于无连接的数据收发，link指定为NULL
 *
 * 触发transmit进行数据收发
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_link_trigger_available(ne_someip_transmit_t* handle, ne_someip_transmit_link_type_t type, ne_someip_transmit_link_t* link);

/** ne_someip_transmit_link_setup
 * @handle:
 * @link_type: 建立的link类型。参考ne_someip_transmit_link_type_t定义
 * @peer_address: 绑定或建立连接所对应的peer address, ne_someip_endpoint_net_addr_t or ne_someip_endpoint_unix_addr_t
 * @user_data: user data，将在ne_someip_transmit_callback_t中和link相关的回调中，被传递
 *
 * 建立连接
 * 对于服务端，ne_someip_transmit_link_setup将创建一个ne_someip_transmit_link_t对象，并绑定到对应的连接
 * 对于客户端，ne_someip_transmit_link_setup将创建一个ne_someip_transmit_link_t对象，并执行connect开始建立连接
 * 对于无连接的trasmit对象执行ne_someip_transmit_link_setup，将返回NULL
 * 已经建立的ne_someip_transmit_link_t对象，将会在连接状态变化时收到link状态变化通知
 *
 * Returns: 成功:0 失败:-1
 */
int ne_someip_transmit_link_setup(ne_someip_transmit_t* handle, ne_someip_transmit_link_type_t link_type,
    void* peer_address, void* user_data);

/** ne_someip_transmit_link_teardown
 * @handle:
 * @link: ne_someip_transmit_link_t对象
 *
 * 拆除连接
 * 拆除连接过程中，将会收到link状态变化为INVALID通知，
 * 用户在收到INVALID状态通知回调后，ne_someip_transmit_link_t对象将不可靠，随时可能被销毁
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_link_teardown(ne_someip_transmit_t* handle, ne_someip_transmit_link_t* link);

/** ne_someip_transmit_link_get_state
 * @handle:
 * @link: ne_someip_transmit_link_t对象
 *
 * 获取ne_someip_transmit_link_t对象状态
 *
 * Returns: ne_someip_transmit_link_t对象状态
 */
ne_someip_transmit_link_state_t ne_someip_transmit_link_get_state(ne_someip_transmit_link_t* link);

/** ne_someip_transmit_link_get_type
 * @handle:
 * @link: ne_someip_transmit_link_t对象
 *
 * 获取ne_someip_transmit_link_t对象类型
 *
 * Returns: ne_someip_transmit_link_t对象类型
 */
ne_someip_transmit_link_type_t ne_someip_transmit_link_get_type(ne_someip_transmit_link_t* link);

/** ne_someip_transmit_link_get_peer_address
 * @handle:
 * @link: ne_someip_transmit_link_t对象
 * @peer_address: 输出参数，ne_someip_transmit_link_t对象对应连接的peer address,
 * ne_someip_endpoint_net_addr_t or ne_someip_endpoint_unix_addr_t
 *
 * 获取ne_someip_transmit_link_t对象的peer address
 *
 * Returns: 0:成功, -1: 失败
 */
int ne_someip_transmit_link_get_peer_address(ne_someip_transmit_link_t* link, void* peer_address);

/** ne_someip_transmit_link_get_userdata
 * @handle:
 * @link: ne_someip_transmit_link_t对象
 *
 * 获取ne_someip_transmit_link_t对象的user data
 *
 * Returns: ne_someip_transmit_link_t对象的user data
 */
void* ne_someip_transmit_link_get_userdata(ne_someip_transmit_link_t* link);

uint16_t ne_someip_transmit_get_port(const ne_someip_transmit_t* transmit);

#ifdef  __cplusplus
}
#endif
#endif  // BASE_TRANSMIT_NE_SOMEIP_TRANSMIT_H
/* EOF */
