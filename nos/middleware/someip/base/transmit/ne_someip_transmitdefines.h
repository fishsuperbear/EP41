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
#ifndef BASE_TRANSMIT_NE_SOMEIP_TRANSMITDEFINES_H
#define BASE_TRANSMIT_NE_SOMEIP_TRANSMITDEFINES_H

#ifdef  __cplusplus
extern "C" {
#endif

# include <stdint.h>
# include <ne_someip_internal_define.h>
// define udp bind max retry num
#define NE_SOMEIP_TRANSMIT_UDP_BIND_NUM 5

// define udp bind retry time
#define NE_SOMEIP_TRANSMIT_UDP_BIND_TIME (200 * 1000)

typedef enum {
    NE_SOMEIP_TRANSMIT_STATE_STOPED = 0, // default
    NE_SOMEIP_TRANSMIT_STATE_PREPARING,
    NE_SOMEIP_TRANSMIT_STATE_PREPARED,
    NE_SOMEIP_TRANSMIT_STATE_STARTED,
    NE_SOMEIP_TRANSMIT_STATE_ERROR,
    NE_SOMEIP_TRANSMIT_STATE_MAX
} ne_someip_transmit_state_t;

typedef enum {
    // transmit的优先级，对应的值类型为int：0..7
    NE_SOMEIP_TRANSMIT_CONFIG_PRIORITY = 0, /* int: valid range 0..7҅ no default value */
    // transmit提供的预分配buffer的块数，对应的值类型为int
    NE_SOMEIP_TRANSMIT_CONFIG_PREALLOC_BUFFER_CNT,
    // transmit提供的预分配buffer的大小(bytes)，对应的值类型为int
    NE_SOMEIP_TRANSMIT_CONFIG_PREALLOC_BUFFER_SIZE,
    // ne_someip_ssl_key_info_t [3 files]
    NE_SOMEIP_TRANSMIT_CONFIG_PREALLOC_SSL_KEY_INFO,
    NE_SOMEIP_TRANSMIT_CONFIG_PREALLOC_MAX
} ne_someip_transmit_config_type_t;

typedef enum {
    // 查询当前transmit是否支持multi link。对应的值类型为int： 1 支持multi link；0 不支持multi link
    NE_SOMEIP_TRANSMIT_QUERY_TYPE_REQUIRED_MULTI_LINK = 0,/* int */
    // ne_someip_transmit_prealloc_buffer_t, Prepare时由transmit分配，用户可以查询、使用，最后由transmit在stop时释放，用户无需释放
    // NE_SOMEIP_TRANSMIT_QUERY_TYPE_PREALLOC_BUFFER_CNT，返回预分配的buffer数，对应的值类型为int
    NE_SOMEIP_TRANSMIT_QUERY_TYPE_PREALLOC_BUFFER_CNT,
    // NE_SOMEIP_TRANSMIT_QUERY_TYPE_PREALLOC_BUFFERS，返回预分配的buffer对象指针，对应的值类型ne_someip_transmit_prealloc_buffer_t**
    NE_SOMEIP_TRANSMIT_QUERY_TYPE_PREALLOC_BUFFERS,
    NE_SOMEIP_TRANSMIT_QUERY_TYPE_MAX
} ne_someip_transmit_query_type_t;

typedef enum {
    NE_SOMEIP_TRANSMIT_TYPE_TCP = 0,    // ne_someip_endpoint_net_addr_t: ip+port
    NE_SOMEIP_TRANSMIT_TYPE_UDP,        // ne_someip_endpoint_net_addr_t: ip+port
    NE_SOMEIP_TRANSMIT_TYPE_UNIX_DOMAIN,// ne_someip_endpoint_unix_addr_t: path
    NE_SOMEIP_TRANSMIT_TYPE_TLS,
    NE_SOMEIP_TRANSMIT_TYPE_DTLS,
    NE_SOMEIP_TRANSMIT_TYPE_MAX
} ne_someip_transmit_type_t;

typedef enum {
    NE_SOMEIP_TRANSMIT_GROUP_TYPE_UDP = 0,  // UDP组播
    NE_SOMEIP_TRANSMIT_GROUP_TYPE_RAW,      // 二层协议组播
    NE_SOMEIP_TRANSMIT_GROUP_TYPE_MAX
} ne_someip_transmit_group_type_t;

typedef enum {
    NE_SOMEIP_TRANSMIT_LINK_TYPE_SOURCE = 0,   // 数据发送
    NE_SOMEIP_TRANSMIT_LINK_TYPE_SINK,         // 数据接收
    NE_SOMEIP_TRANSMIT_LINK_TYPE_MAX
} ne_someip_transmit_link_type_t;

typedef enum {
    NE_SOMEIP_TRANSMIT_LINK_STATE_PENDING = 0,   // peer端还未连接
    NE_SOMEIP_TRANSMIT_LINK_STATE_RUNNING,       // peer端连接已建立，运行中状态
    NE_SOMEIP_TRANSMIT_LINK_STATE_DISCONNECTED,  // peer端连接已经断开，不可用，需要立即调用teardown
    NE_SOMEIP_TRANSMIT_LINK_STATE_INVALID,       // 已经teardown，随时会被销毁，不能再操作此link对象
    NE_SOMEIP_TRANSMIT_LINK_STATE_ERROR,
    NE_SOMEIP_TRANSMIT_LINK_STATE_MAX
} ne_someip_transmit_link_state_t;

typedef enum {
    NE_SOMEIP_TRANSMIT_BUFFER_TYPE_FD = 0,
    NE_SOMEIP_TRANSMIT_BUFFER_TYPE_NORMAL = 1,
    NE_SOMEIP_TRANSMIT_BUFFER_TYPE_IOV = 2,
    NE_SOMEIP_TRANSMIT_BUFFER_TYPE_PREALLOC = 3,
    NE_SOMEIP_TRANSMIT_BUFFER_TYPE_MAX
} ne_someip_transmit_buffer_type;

// 用户需要关心的结果，其他异常或错误会由core处理，不会通知到用户
typedef enum {
    NE_SOMEIP_TRANSMIT_ERROR_NO_ERROR = 0,     // 正常完成收发动作，没有问题
    NE_SOMEIP_TRANSMIT_ERROR_MSG_ERROR,        // 用户给出的数据buff有问题
    NE_SOMEIP_TRANSMIT_ERROR_NOT_READY,         // 底层还未准备好，等待后续底层准备好通知(link:link_source_available, 无link:source_available)
    NE_SOMEIP_TRANSMIT_ERROR_RECONNECT,         // DTLS receive unexpected data, should reconnect peer
    NE_SOMEIP_TRANSMIT_ERROR_UNKNOW_ERROR
}ne_someip_transmit_io_result_t;

typedef struct {
    ne_someip_transmit_buffer_type type;
} ne_someip_transmit_buffer_t;

typedef struct {
    ne_someip_transmit_buffer_t base;
    // buffer起始位置
    void* buffer;
    // 本次数据写入/读取起始位置偏移量
    int offset;
    // 期望写入/读取的总长度
    int length;
    // 实际写入/读取的长度
    ssize_t result;
    void* user_data;
} ne_someip_transmit_normal_buffer_t;

typedef struct {
    ne_someip_transmit_buffer_t base;
    int count;
    int fds[2];
    void* user_data;
} ne_someip_transmit_fd_buffer_t;

typedef struct{
    ne_someip_transmit_buffer_t base;
    // iov buffer数组
    void* iovBuffer;
    // buffer数组偏移位置
    int offset;
    // buffer数组发送长度
    int length;
    // 实际写入/读取的长度
    ssize_t result;
    size_t tp_separation_time_usec;
    void* user_data;
} ne_someip_transmit_iov_buffer_t;

typedef struct{
    ne_someip_transmit_buffer_t base;
    void* handle;       // only for transmit
    void* user_data;    // only for user
    void* buffer;
    int max_size;
    int offset;
    int length;
    int64_t timestamp;
} ne_someip_transmit_prealloc_buffer_t;

typedef struct ne_someip_transmit_link ne_someip_transmit_link_t;
struct ne_someip_transmit_link{
    ne_someip_transmit_link_state_t link_state;    // link对象状态
    ne_someip_transmit_link_type_t link_type;      // 数据收发类型
    int32_t data_fd;                            // 建立连接的socket_fd
    ne_someip_transmit_type_t transmit_type;       // transmit类型, 获取peer_address时需要判断类型
    void* local_addr;
    void* peer_address;                     // link的对端地址
    void* user_data;
    void (*link_state_changed)(void* user_data, ne_someip_transmit_link_t *link, int state);
};

typedef struct {
    ne_someip_transmit_buffer_t* (*require)(void* user_data, ne_someip_transmit_link_t* link, int avalible_size, void* peer_address);
    void (*release)(void* user_data, ne_someip_transmit_buffer_t* buffer, ne_someip_transmit_io_result_t result);
} ne_someip_transmit_source_t;

typedef struct {
    ne_someip_transmit_buffer_t* (*dequeue)(void* user_data, ne_someip_transmit_link_t* link, int avalible_size);
    void (*enqueue)(void* user_data, ne_someip_transmit_buffer_t* buffer, const void* peer_address, ne_someip_transmit_io_result_t result);
} ne_someip_transmit_sink_t;

typedef struct {
    void (*state_changed)(void* user_data, int state);
    // 对于面向连接的transmit对象，当peer的连接建立时会接收到此通知
    void (*peer_founded)(void* user_data, void* peer_address);
    void (*link_state_changed)(void* user_data, ne_someip_transmit_link_t *link, int state);
    // 接收到此通知，说明transmit通过NETransmitSource/NETransmitSource对象require/dequeue buffer失败，
    // transmit将暂停收发数据，直至用户再次调用ne_someip_transmit_link_trigger_available
    void (*link_buffer_exhausted)(void* user_data, ne_someip_transmit_link_t *link);
    // 之前link trigger发送数据时，发生了错误，transmit重新监听到了写可用
    void (*link_source_available)(void* user_data, ne_someip_transmit_link_t* link);
    void (*source_available)(void* user_data, void* peer_address, bool result);
} ne_someip_transmit_callback_t;


typedef struct {
    char ca_crt_path[256];
    char crt_path[256];
    char key_path[256];
}ne_someip_transmit_ssl_key_info_t;

#ifdef  __cplusplus
}
#endif
#endif  // BASE_TRANSMIT_NE_SOMEIP_TRANSMITDEFINES_H
/* EOF */
