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
#ifndef BASE_TRANSMIT_NE_SOMEIP_TRANSMITIMPL_H
#define BASE_TRANSMIT_NE_SOMEIP_TRANSMITIMPL_H

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "ne_someip_transmitdefines.h"
#include "ne_someip_list.h"
#include "ne_someip_map.h"
#include "ne_someip_internal_define.h"
#include "ne_someip_transmitcore.h"
#include <sys/types.h>

typedef struct ne_someip_transmit_impl ne_someip_transmit_impl_t;

struct ne_someip_transmit_impl {
    const void* transmit_config[NE_SOMEIP_TRANSMIT_CONFIG_PREALLOC_MAX];    // 对应关系为: value = transmit_config[config_type]

    /** set_config
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     * @type: 属性类型,参考ne_someip_transmit_config_type_t定义
     * @config: 输出参数，属性值。实际类型与type相关
     *
     * 配置transmit的属性
     *
     * Returns: 0:成功, -1: 失败
     **/
    int (*set_config)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_config_type_t type, const void* config);

    /** get_config
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     * @type: 属性类型,参考ne_someip_transmit_config_type_t定义
     * @config: 输出参数，属性值。实际类型与type相关
     *
     * 获取transmit的属性
     *
     * Returns: 0:成功, -1: 失败
     */
    int (*get_config)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_config_type_t type, void* config);

    /** prepare
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     *
     * 工作准备，根据需要申请资源。
     * 将在transmit的looper中回调本函数
     *
     * Returns: 0:成功, -1: 失败
     */
    int (*prepare)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core);

    /** start
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     *
     * 开始工作
     * 将在transmit的looper中回调本函数
     * 对于面向连接的服务端，应该调用ne_someip_transmit_core_register_listen_fd监听新连接
     *
     * Returns: 服务器:对应启动的fd 客户端:0，失败:-1
     */
    int (*start)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core);

    /** stop
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     *
     * 停止工作。释放prepare阶段申请的资源。
     * 对于面向连接的服务端，应该调用ne_someip_transmit_core_unregister_listen_fd移除连接监听
     * 将在transmit的looper中回调本函数
     *
     * Returns: 0:成功, -1: 失败
     */
    int (*stop)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core);

    /** query
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     * @type: 查询类型, transmit支持的查询类型，参考ne_someip_transmit_query_type_t定义
     * @data: 输出参数，查询结果。实际类型与type相关
     *
     * 查询
     * 将在transmit的looper中回调本函数
     *
     * Returns: 0:成功, -1: 失败
     */
    int (*query)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_query_type_t type, void* data);

    /** in_connection
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     * @listen_fd: 
     *
     * listen_fd有新连接事件
     * 将在transmit的looper中回调本函数
     *
     * Returns: 0:成功, -1: 失败
     */
    int (*in_connection)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int listen_fd);

    /** send
     * @impl:
     * @buffer: 要发送的数据对象
     * @peer_address: 要发送的目标地址
     *
     * 发送数据buffer到peer_address
     * 将在transmit的looper中回调本函数
     *
     * Returns: 成功:发送数据的长度, 失败:-1
     */
    ne_someip_transmit_core_send_result_t (*send)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_buffer_t* buffer, void* peer_address);

    /** send_by_fd
     * @impl:
     * @buffer: 要发送的数据对象
     * @fd: 已经建立的连接fd
     *
     * 把数据buffer写入fd对应的连接
     * 将在transmit的looper中回调本函数
     *
     * Returns: 成功:发送数据的长度, 失败:-1
     */
    ne_someip_transmit_core_send_result_t (*send_by_fd)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_buffer_t* buffer, int fd);

    /** recv
     * @impl:
     * @data_fd: 已经建立的连接fd
     * @buffer: 输出参数。存储接收到的数据
     * @peer_address：输出参数。指明数据的发送方地址
     *
     * 从data_fd中接收数据
     * 将在transmit的looper中回调本函数
     *
     * Returns: 成功:接收数据的长度, 失败:-1
     */
    ne_someip_transmit_core_recv_result_t (*recv)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int data_fd, ne_someip_transmit_buffer_t* buffer, void* peer_address, void* local_address);

    /** on_error
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     * @fd: 发生error的fd
     * @error: error code
     *
     * error通知
     * 将在transmit的looper中回调本函数
     *
     * Returns: 
     */
    void (*on_error)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int fd, int error);

    /** address_equal
     * @addr1: 地址1
     * @addr2: 地址2
     *
     * 判断地址1和地址2是否为同一地址
     *
     * Returns: true:地址相同, false: 地址不同
     */
    int32_t (*address_equal)(const void* addr1, const void* addr2);

    /** join_group
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     * @group_type: 组播类型
     * @unicast_addr: 绑定本地网卡的ip地址(单播)
     * @group_address: 组播地址
     *
     * 加入组播
     * 将在transmit的looper中回调本函数
     *
     * Returns: 0:成功, -1: 失败
     */
    int (*join_group)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_group_type_t group_type,
        void* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);

    /** leave_group
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     * @group_type: 组播类型
     * @unicast_addr: 绑定本地网卡的ip地址(单播)
     * @group_address: 组播地址
     *
     * 离开组播
     * 将在transmit的looper中回调本函数
     *
     * Returns: 0:成功, -1: 失败
     */
    int (*leave_group)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_group_type_t group_type,
        void* unicast_addr, ne_someip_enpoint_multicast_addr_t* group_address);

    /** link_setup
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     * @link_type: 建立的link类开过。参考NETransmitLinkType定义 // [TBD]
     * @peer_address: 绑定或建立连接所对应的peer address
     *
     * 建立连接
     * 对于服务端，link_setup根据需要进行必要的处理
     * 对于客户端，link_setup应该执行connect处理
     * connect成功后应该调用ne_someip_transmit_core_register_data_fd，监听数据
     * 将在transmit的looper中回调本函数
     *
     * Returns: 0:成功, -1: 失败
     */
    int (*link_setup)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, ne_someip_transmit_link_t* link, void* peer_address);

    /** link_teardown
     * @impl:
     * @core: ne_someip_transmit_core_t对象指针
     * @fd:
     * @peer_address: 绑定或建立连接所对应的peer address
     *
     * 拆除连接
     * 对于客户端，link_teardown应该断开连接close fd
     * 断开连接后调用ne_someip_transmit_core_unregister_data_fd，移除数据监听
     *
     * Returns: 0:成功, -1: 失败
     */
    int (*link_teardown)(ne_someip_transmit_impl_t* impl, ne_someip_transmit_core_t* core, int fd, void* peer_address);

    /** link_supported_chk
     * @impl:
     *
     * 获取ne_someip_transmit_impl_t对象是否支持link
     *
     * Returns: true:支持link, false: 不支持link
     */
    bool (*link_supported_chk)(ne_someip_transmit_impl_t* impl);

    /** fd_removed
     * @impl:
     * @fd:
     * @peer_address: 绑定或建立连接所对应的peer address
     *
     * fd已经移除监听，ne_someip_transmit_impl_t可以关闭fd、释放对应的peer_address
     *
     * Returns:
     */
    void (*fd_removed)(ne_someip_transmit_impl_t* impl, int fd, void* peer_address);

    /** addr_create
     * 
     * 创建对应的addr内存地址
     * 
     */
    void* (*addr_create)();

    /** addr_create
     * 
     * 将对应的地址转为hash值
     * 
     */
    uint32_t (*addr_hash)(const void* addr);
};

#ifdef  __cplusplus
}
#endif
#endif  // BASE_TRANSMIT_NE_SOMEIP_TRANSMITIMPL_H
/* EOF */
