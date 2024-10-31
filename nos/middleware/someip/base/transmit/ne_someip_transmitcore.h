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
#ifndef BASE_TRANSMIT_NE_SOMEIP_TRANSMITCORE_H
#define BASE_TRANSMIT_NE_SOMEIP_TRANSMITCORE_H

#ifdef  __cplusplus
extern "C" {
#endif

#include "ne_someip_list.h"
#include "ne_someip_map.h"
#include "ne_someip_object.h"
#include "ne_someip_looper.h"
#include "ne_someip_transmitdefines.h"
#include "ne_someip_internal_define.h"
// #include "ne_someip_transmitimpl.h"

typedef struct ne_someip_transmit_impl ne_someip_transmit_impl_t;
typedef struct ne_someip_wait_obj ne_someip_wait_obj_t;
typedef struct ne_someip_transmit_core ne_someip_transmit_core_t;


// ne_someip_transmit_core_t
struct ne_someip_transmit_core {
    ne_someip_transmit_type_t transmit_type;        // transmit类型
    bool is_listen_mode;                     // true:监听模式(TCP/Unix: server)；false:非监听模式(TCP/Unix: client); UDP: false.
    void* local_address;                     // transmit对象绑定的本地地址
    ne_someip_transmit_state_t transmit_state;      // transmit(core对象)的状态
    // listen系列：
    // 对于服务器，在创建后，start则启动全部监听。
    // 对于客户端，不存在监听fd
    // 对于没有link概念的（udp），创建listen_fd后启动error，在用户请求link_trigger时，启动read
    // prepar的时候创建
    int32_t listen_fd;                            // listen模式的fd（用于tcp服务器和udp，客户端没有listen概念，使用link）
    // start后创建并选择启动
    ne_someip_looper_io_source_t* listen_source;
    // 需要malloc
    ne_someip_map_t* connect_source_map;        // <fd, ne_someip_transmit_core_source_entry_t*>，通过fd与source进行绑定，并监听error
    // ne_someip_list_t* link_list;             // 已经建立的连接, 节点类型为 ne_someip_transmit_link_t
    ne_someip_map_t* link_map;                  // <peer_address, ne_someip_transmit_link_t*>，通过peer addr与link绑定
    // 用户设置数据
    ne_someip_transmit_impl_t* impl;                // impl对象
    ne_someip_transmit_callback_t* callback;        // callback对象
    ne_someip_transmit_source_t* source;            // source对象
    ne_someip_transmit_sink_t* sink;                // sink对象
    ne_someip_looper_t* looper;              // 给transmit运行提供支持的looper对象
    void* callback_user_data;                // callback user_data
    void* source_user_data;                  // source user_data
    void* sink_user_data;                    // sink user_data
    ne_someip_wait_obj_t* wait_obj;
    NEOBJECT_MEMBER
};

typedef enum {
    NE_SOMEIP_TRANSMIT_CORE_RECV_ERROR_NO_ERROR = 0,    // 正常完成，transmitcore循环进行下次接收
    NE_SOMEIP_TRANSMIT_CORE_RECV_ERROR_NOT_READY,       // 底层还未准备好，transmitcore退出本次接收
    NE_SOMEIP_TRANSMIT_CORE_RECV_ERROR_NO_DATA,         // 没有可以读取的数据，transmitcore退出本次接收
    NE_SOMEIP_TRANSMIT_CORE_RECV_ERROR_MSG_ERROR,       // 用户提供的buff有问题，transmitcore不动作，正常通知用户释放数据
    NE_SOMEIP_TRANSMIT_CORE_RECV_ERROR_DIS_CONNECT,     // 连接被断开，如果由连接transmitcore将link状态改为disconnect，如果无连接，则退出本次接收
    NE_SOMEIP_TRANSMIT_CORE_RECV_ERROR_RECONNECT,       // DTLS receive unexpected data, transmitcore should start send to reconnect.
    NE_SOMEIP_TRANSMIT_CORE_RECV_ERROR_MAJOR_ERROR,     // 发送严重错误，如果有连接transmitcore将断开连接，如果无连接，则退出本次接收
    NE_SOMEIP_TRANSMIT_CORE_RECV_ERROR_UNKNOW           // 未知问题，处理同严重错误
}ne_someip_transmit_core_recv_result_t;

typedef enum {
    NE_SOMEIP_TRANSMIT_CORE_SEND_ERROR_NO_ERROR = 0,    // 正常完成，transmitcore循环进行下次接收
    NE_SOMEIP_TRANSMIT_CORE_SEND_ERROR_NOT_READY,       // 底层还未准备好，transmitcore设置err_code:not_ready通知用户
    NE_SOMEIP_TRANSMIT_CORE_SEND_ERROR_BUFF_FULL,       // fd剩余buff不足，transmitcore将追加write监听，并退出本次发送
    NE_SOMEIP_TRANSMIT_CORE_SEND_ERROR_MSG_ERROR,       // 用户给出的数据有问题，transmitcore不动作，正常通知用户释放数据
    NE_SOMEIP_TRANSMIT_CORE_SEND_ERROR_DIS_CONNECT,     // 连接被断开，如果由连接transmitcore将link状态改为disconnect，如果无连接，则退出本次接收
    NE_SOMEIP_TRANSMIT_CORE_SEND_ERROR_MAJOR_ERROR,     // 发送严重错误，如果有连接transmitcore将断开连接，如果无连接，则退出本次接收
    NE_SOMEIP_TRANSMIT_CORE_SEND_ERROR_UNKNOW           // 未知问题，处理同严重错误
}ne_someip_transmit_core_send_result_t;

/** ne_someip_transmit_core_new
 * @impl: NETransmitImpl对象指针，负责真正的数据通信
 * @type: transmit类型，参考ne_someip_transmit_type_t定义
 * @local_address: transmit对象绑定的本地地址, ne_someip_endpoint_net_addr_t or ne_someip_endpoint_unix_addr_t
 * @is_listen_mode: true 监听模式；false 非监听模式
 *
 * 创建ne_someip_transmit_core_t对象
 * NETransmitCore职责：
 *    1. 响应NETransmit的请求，监听fd，管理数据连接
 *    2. 通过NETransmitImpl进行UDP、TCP、UNIX Domain通信
 *    3. 通过NETransmitSource和NETransmitSink与用户交互数据
 *
 * Returns: 返回ne_someip_transmit_core_t对象指针
 */
ne_someip_transmit_core_t*
ne_someip_transmit_core_new(ne_someip_transmit_impl_t* impl, ne_someip_transmit_type_t type, const void* local_address, bool is_listen_mode);

/** ne_someip_transmit_core_ref
 * @core:
 *
 * 增加ne_someip_transmit_core_t对象的引用计数
 *
 * Returns: 返回ne_someip_transmit_core_t对象指针
 */
ne_someip_transmit_core_t* ne_someip_transmit_core_ref(ne_someip_transmit_core_t* core);

/** ne_someip_transmit_core_unref
 * @core:
 *
 * 减少ne_someip_transmit_core_t对象的引用计数
 *
 * Returns: 0:成功, -1: 失败
 */
void ne_someip_transmit_core_unref(ne_someip_transmit_core_t* core);

/** ne_someip_transmit_core_set_looper
 * @core:
 * @looper: NELooper对象，给transmit运行提供支持
 *
 * 设置给transmit运行提供支持的looper对象。必须在ne_someip_transmit_core_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_set_looper(ne_someip_transmit_core_t* core, ne_someip_looper_t* looper);

/** ne_someip_transmit_core_get_looper
 * @core:
 *
 * 获取给transmit运行提供支持的looper对象
 *
 * Returns: 返回looper对象指针；失败则返回NULL
 */
ne_someip_looper_t* ne_someip_transmit_core_get_looper(ne_someip_transmit_core_t* core);

/** ne_someip_transmit_core_set_config
 * @core:
 * @type: 属性类型,参考ne_someip_transmit_config_type_t定义
 * @config: 属性值。实际类型与type相关
 *
 * 配置transmit的属性。必须在ne_someip_transmit_core_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_set_config(ne_someip_transmit_core_t* core, ne_someip_transmit_config_type_t type, const void* config);

/** ne_someip_transmit_core_get_config
 * @core:
 * @type: 属性类型,参考ne_someip_transmit_config_type_t定义
 * @config: 输出参数，属性值。实际类型与type相关
 *
 * 获取transmit的type属性类型的属性值
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_get_config(ne_someip_transmit_core_t* core, ne_someip_transmit_config_type_t type, void* config);

/** ne_someip_transmit_core_set_callback
 * @core:
 * @callback: 回调函数。用户可以通过回调获取状态变化，peer端连接建立等通知，参考ne_someip_transmit_callback_t定义
 * @user_data: user data
 *
 * 设置transmit的回调函数。必须在ne_someip_transmit_core_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_set_callback(ne_someip_transmit_core_t* core, ne_someip_transmit_callback_t* callback, void* user_data);

/** ne_someip_transmit_core_set_source
 * @core:
 * @source: source对象。当transmit被trigger（调用ne_someip_transmit_link_trigger_available）时，
 *        transmit将通过source对象获取要发送的数据，并由source对象指定发送目标地址。
 * @user_data: user data
 *
 * 设置transmit的source对象。必须在ne_someip_transmit_core_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_set_source(ne_someip_transmit_core_t* core, ne_someip_transmit_source_t* source, void* user_data);

/** ne_someip_transmit_core_set_sink
 * @core:
 * @sink: sink对象。当transmit有监听到新数据被接收时，transmit将通过sink对象将接收到的
 *        数据传递给用户
 * @user_data: user data
 *
 * 设置transmit的sink对象。必须在ne_someip_transmit_core_prepare前调用
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_set_sink(ne_someip_transmit_core_t* core, ne_someip_transmit_sink_t* sink, void* user_data);

/** ne_someip_transmit_core_prepare
 * @core:
 *
 * transmit使用用户配置的各种属性，进行工作前的准备工作。
 * 比如预分配buffer等
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_prepare(ne_someip_transmit_core_t* core);

/** ne_someip_transmit_core_start
 * @core:
 *
 * transmit开始工作
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_start(ne_someip_transmit_core_t* core);

/** ne_someip_transmit_core_stop
 * @core:
 *
 * transmit停止工作。perpare阶段申请的资源，比如预分配buffer等，都将被释放
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_stop(ne_someip_transmit_core_t* core);

/** ne_someip_transmit_core_get_state
 * @core:
 *
 * 获取transmit的状态。
 *
 * Returns: state
 */
ne_someip_transmit_state_t ne_someip_transmit_core_get_state(ne_someip_transmit_core_t* core);

/** ne_someip_transmit_core_query
 * @core:
 * @type: 查询类型, transmit支持的查询类型，参考ne_someip_transmit_query_type_t定义
 * @data: 输出参数，查询结果。实际类型与type相关
 *
 * 查询
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_query(ne_someip_transmit_core_t* core, ne_someip_transmit_query_type_t type, void* data);

/** ne_someip_transmit_core_join_group
 * @core:
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
int32_t ne_someip_transmit_core_join_group(ne_someip_transmit_core_t* core, ne_someip_transmit_group_type_t group_type,
    const ne_someip_endpoint_net_addr_t* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);

/** ne_someip_transmit_core_leave_group
 * @core:
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
int32_t ne_someip_transmit_core_leave_group(ne_someip_transmit_core_t* core, ne_someip_transmit_group_type_t group_type,
    const ne_someip_endpoint_net_addr_t* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);

/** ne_someip_transmit_core_link_setup
 * @core:
 * @link_type: 建立的link类型。参考ne_someip_transmit_link_t定义
 * @peer_address: 绑定或建立连接所对应的peer address, ne_someip_endpoint_net_addr_t or ne_someip_endpoint_unix_addr_t
 * @user_data: user data，将在ne_someip_transmit_callback_t中和link相关的回调中，被传递
 *
 * 建立连接
 * 对于服务端，ne_someip_transmit_core_link_setup将创建一个ne_someip_transmit_link_t对象，并绑定到对应的连接
 * 对于客户端，ne_someip_transmit_core_link_setup将创建一个ne_someip_transmit_link_t对象，并执行connect开始建立连接
 * 对于无连接的trasmit对象执行ne_someip_transmit_core_link_setup，将返回NULL
 * 已经建立的ne_someip_transmit_link_t对象，将会在连接状态变化时收到link状态变化通知
 *
 * Returns: 成功:0 失败:-1
 */
int ne_someip_transmit_core_link_setup(ne_someip_transmit_core_t* core, ne_someip_transmit_link_type_t link_type,
    const void* peer_address, void* user_data);

/** ne_someip_transmit_core_link_teardown
 * @core:
 * @link: ne_someip_transmit_link_t对象
 *
 * 拆除连接
 * 拆除连接过程中，将会收到link状态变化为INVALID通知，
 * 用户在收到INVALID状态通知回调后，ne_someip_transmit_link_t对象将不可靠，随时可能被销毁
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_link_teardown(ne_someip_transmit_core_t* core, ne_someip_transmit_link_t* link);

/** ne_someip_transmit_core_link_get_state
 * @core:
 * @link: ne_someip_transmit_link_t对象
 *
 * 获取ne_someip_transmit_link_t对象类型
 *
 * Returns: ne_someip_transmit_link_t对象状态
 */
ne_someip_transmit_link_state_t ne_someip_transmit_core_link_get_state(ne_someip_transmit_link_t* link);

/** ne_someip_transmit_core_link_get_type
 * @core:
 * @link: ne_someip_transmit_link_t对象
 *
 * 获取ne_someip_transmit_link_t对象类型
 *
 * Returns: ne_someip_transmit_link_t对象状态
 */
ne_someip_transmit_link_type_t ne_someip_transmit_core_link_get_type(ne_someip_transmit_link_t* link);

/** ne_someip_transmit_core_link_get_peer_address
 * @core:
 * @link: ne_someip_transmit_link_t对象
 * @peer_address: 输出参数，ne_someip_transmit_link_t对象对应连接的peer address,
 * ne_someip_endpoint_net_addr_t or ne_someip_endpoint_unix_addr_t
 *
 * 获取ne_someip_transmit_link_t对象的peer address
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_link_get_peer_address(ne_someip_transmit_link_t* link, void* peer_address);

/** ne_someip_transmit_core_link_get_userdata
 * @core:
 * @link: ne_someip_transmit_link_t对象
 *
 * 获取ne_someip_transmit_link_t对象的user data
 *
 * Returns: ne_someip_transmit_link_t对象的user data
 */
void* ne_someip_transmit_core_link_get_userdata(ne_someip_transmit_link_t* link);

/** ne_someip_transmit_core_link_trigger_available
 * @core:
 * @type: 数据收发类型。NE_SOMEIP_TRANSMIT_LINK_TYPE_SOURCE：数据发送；NE_SOMEIP_TRANSMIT_LINK_TYPE_SINK：数据接收
 * @link: 指定收发数据link，对于无连接的数据收发，link指定为NULL
 *
 * 触发transmit进行数据收发
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_link_trigger_available(ne_someip_transmit_core_t* core, ne_someip_transmit_link_type_t type, ne_someip_transmit_link_t* link);

/** ne_someip_transmit_core_register_listen_fd
 * @core:
 * @listen_fd: 连接监听fd
 *
 * (tcp/unixDomain server端)监听transmit的新连接。
 * 监听到listen_fd的事件，将调用NETransmitImpl的in_connection或on_error回调函数
 * 本函数必须在transmit的looper中调用。
 * 通常情况，将在NETransmitImpl的start回调函数中调用本函数
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_register_listen_fd(ne_someip_transmit_core_t* core, int32_t listen_fd);

/** ne_someip_transmit_core_register_connect_fd
 * @core:
 * @connect_fd: 数据监听fd，连接失败为-1
 * peer_address:绑定或建立连接所对应的peer address,
 * ne_someip_endpoint_net_addr_t or ne_someip_endpoint_unix_addr_t
 *
 * 监听transmit的connect_fd。
 * 监听到connect_fd的事件，将调用NETransmitImpl的recv或on_error回调函数
 * 本函数必须在transmit的looper中调用。
 * 通常情况，将在NETransmitImpl的start或in_connection回调函数中调用本函数
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_register_connect_fd(ne_someip_transmit_core_t* core, int32_t connect_fd, void* peer_address);

/** ne_someip_transmit_core_unregister_listen_fd
 * @core:
 * @listen_fd: 连接监听fd
 *
 * (tcp/unixDomain server端)停止监听transmit的新连接。
 * 本函数必须在transmit的looper中调用。
 * 通常情况，将在NETransmitImpl的stop或on_error回调函数中调用本函数
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_unregister_listen_fd(ne_someip_transmit_core_t* core, int32_t listen_fd);

/** ne_someip_transmit_core_unregister_connect_fd
 * @core:
 * @connect_fd: 数据监听fd
 *
 * 停止监听transmit的connect_fd。
 * 本函数必须在transmit的looper中调用。
 * 通常情况，将在NETransmitImpl的link_teardown或on_error回调函数中调用本函数
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_unregister_connect_fd(ne_someip_transmit_core_t* core, int32_t connect_fd);

/** ne_someip_transmit_core_source_available
 * @core:
 * @peer_address: 对端地址
 * @result: 底层是否准备完成
 *
 * 回调用户底层准备完毕
 * 本函数必须在transmit的looper中调用。
 * 仅适用于无连接，当底层还未准备好时用户就请求发送数据，会通知用户not_ready
 * 当底层准备完毕后，通过该函数回调通知用户可以发送
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_transmit_core_source_available(ne_someip_transmit_core_t* core, void* peer_address, bool result);


#ifdef  __cplusplus
}
#endif
#endif  // BASE_TRANSMIT_NE_SOMEIP_TRANSMITCORE_H
/* EOF */
