/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: 提供错误码通用功能
 * Create: 2021-04-23
 */

#ifndef BSL_ENO_API_H
#define BSL_ENO_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 错误码的通用结构 */
/********************************************************/
/* |0         7|8        15|16         23|24        31| */
/* |reserve字段|  中间件    |  模块       |  错误码    | */
/* |           |           |             |            | */
/********************************************************/

/** @defgroup bsl bsl */
/**
 * @ingroup bsl
 * 错误码头保留字段
 */
#define BSL_ENO_REVERSED_FLAG   0U

/**
 * @ingroup bsl
 * 0x1 操作权限受限。
 */
#define AEN_ENO_PERM 1U /* Operation not permitted */

/**
 * @ingroup bsl
 * 0x2 文件或者文件夹不存在。
 */
#define AEN_ENO_NOENT 2U /* No such file or directory */

/**
 * @ingroup bsl
 * 0x4 不能在中断下调用。
 */
#define AEN_ENO_INTR 4U /* Interrupted system call */

/**
 * @ingroup bsl
 * 0x5 IO操作失败。
 */
#define AEN_ENO_IO 5U /* I/O error */

/**
 * @ingroup bsl
 * 0x9 文件数量错误。
 */
#define AEN_ENO_BADF 9U /* Bad file number */

/**
 * @ingroup bsl
 * 0xB 请稍后重试。
 */
#define AEN_ENO_AGAIN 11U /* Try again */

/**
 * @ingroup bsl
 * 0xC 内存不足。
 */
#define AEN_ENO_NOMEM 12U /* Out of memory */

/**
 * @ingroup bsl
 * 0xD 权限受限。
 */
#define AEN_ENO_ACCES 13U /* Permission denied */

/**
 * @ingroup bsl
 * 0xE 地址错误。
 */
#define AEN_ENO_FAULT 14U /* Bad address */

/**
 * @ingroup bsl
 * 0x10 设备或者资源繁忙。
 */
#define AEN_ENO_BUSY 16U /* Device or resource busy */

/**
 * @ingroup bsl
 * 0x11 文件已经存在。
 */
#define AEN_ENO_EXIST 17U /* File exists */

/**
 * @ingroup bsl
 * 0x15 目录文件。
 */
#define AEN_ENO_ISDIR 21U /* Is a directory */

/**
 * @ingroup bsl
 * 0x16 参数错误。
 */
#define AEN_ENO_INVAL 22U /* Invalid argument */

/**
 * @ingroup bsl
 * 0x1B 文件过大。
 */
#define AEN_ENO_FBIG 27U /* File too large */

/**
 * @ingroup bsl
 * 0x1C 设备上没有空间
 */
#define AEN_ENO_NOSPC 28U /* No space left on device */

/**
 * @ingroup bsl
 * 0x20 管道断开。
 */
#define AEN_ENO_PIPE 32U /* Broken pipe */

/**
 * @ingroup bsl
 * 0x21 变量超出范围。
 */
#define AEN_ENO_DOM 33U /* Math argument out of domain of func */

/**
 * @ingroup bsl
 * 0x22 数学计算结果超出描述范围。
 */
#define AEN_ENO_RANGE 34U /* Math result not representable */

/**
 * @ingroup bsl
 * 0x25 无记录锁可用。
 */
#define AEN_ENO_NOLCK 37U /* No record locks available */

/**
 * @ingroup bsl
 * 0x26 函数没有实现。
 */
#define AEN_ENO_NOSYS 38U /* Function not implemented */

/**
 * @ingroup bsl
 * 0x28 链接或者调用发生循环。
 */
#define AEN_ENO_LOOP 40U /* Too many symbolic links encountered */

/**
 * @ingroup bsl
 * 0x29 操作阻止。
 */
#define AEN_ENO_WOULDBLOCK 41U /* Operation would block */

/**
 * @ingroup bsl
 * 0x2A 没有期望类型的消息。
 */
#define AEN_ENO_NOMSG 42U /* No message of desired type */

/**
 * @ingroup bsl
 * 0x2C 通道数超过范围。
 */
#define AEN_ENO_CHRNG 44U /* Channel number out of range */

/**
 * @ingroup bsl
 * 0x3C 无可用的数据。
 */
#define AEN_ENO_NODATA 60U /* No data available */

/**
 * @ingroup bsl
 * 0x3D 定时器停止。
 */
#define AEN_ENO_TIME 61U /* Timer expired */

/**
 * @ingroup bsl
 * 0x45 发送失败。
 */
#define AEN_ENO_COMM 69U /* Communication error on send */

/**
 * @ingroup bsl
 * 0x49 消息数据内容无效。
 */
#define AEN_ENO_BADMSG 73U /* Not a data message */

/**
 * @ingroup bsl
 * 0x4A 赋值超出变量类型定义取值范围。
 */
#define AEN_ENO_OVERFLOW 74U /* Value too large for defined data type */

/**
 * @ingroup bsl
 * 0x4C 文件描述符发生错误。
 */
#define AEN_ENO_BADFD 76U /* File descriptor in bad state */

/**
 * @ingroup bsl
 * 0x57 没有套接字可供操作。
 */
#define AEN_ENO_NOTSOCK 87U /* Socket operation on non-socket */

/**
 * @ingroup bsl
 * 0x58 目的地址请求失败。
 */
#define AEN_ENO_DESTADDRREQ 88U /* Destination address required */

/**
 * @ingroup bsl
 * 0x59 消息过长。
 */
#define AEN_ENO_MSGSIZE 89U /* Message too long */

/**
 * @ingroup bsl
 * 0x5E 传输端不支持此操作。
 */
#define AEN_ENO_OPNOTSUPP 94U /* Operation not supported on transport endpoint */

/**
 * @ingroup bsl
 * 0x63 网络关闭。
 */
#define AEN_ENO_NETDOWN 99U /* Network is down */

/**
 * @ingroup bsl
 * 0x65 复位导致网络连接断开。
 */
#define AEN_ENO_NETRESET 101U /* Network dropped connection because of reset */

/**
 * @ingroup bsl
 * 0x66 软件原因导致连接终止。
 */
#define AEN_ENO_CONNABORTED 102U /* Software caused connection abort */

/**
 * @ingroup bsl
 * 0x67 同级重置连接。
 */
#define AEN_ENO_CONNRESET 103U /* Connection reset by peer */

/**
 * @ingroup bsl
 * 0x68 没有可用的缓存。
 */
#define AEN_ENO_NOBUFS 104U /* No buffer space available */

/**
 * @ingroup bsl
 * 0x6A 传输端点没有建立连接。
 */
#define AEN_ENO_NOTCONN 106U /* Transport endpoint is not connected */

/**
 * @ingroup bsl
 * 0x6B 传输端关闭后无法发送。
 */
#define AEN_ENO_SHUTDOWN 107U /* Cannot send after transport endpoint shutdown */

/**
 * @ingroup bsl
 * 0x6D 连接超时。
 */
#define AEN_ENO_TIMEDOUT 109U /* Connection timed out */

/**
 * @ingroup bsl
 * 0x70 与主机没有路由。
 */
#define AEN_ENO_HOSTUNREACH 112U /* No route to host */

/**
 * @ingroup bsl
 * 0x71 操作就绪。
 */
#define AEN_ENO_ALREADY 113U /* Operation already in progress */

/**
 * @ingroup bsl
 * 0x72 操作进行中。
 */
#define AEN_ENO_INPROGRESS 114U /* Operation now in progress */

/**
 * @ingroup bsl
 * 0x79 超过配额限制。
 */
#define AEN_ENO_DQUOT 121U /* Quota exceeded */

/**
 * @ingroup bsl
 * 0x7C 模块没有初始化。
 */
#define AEN_ENO_NOTINIT 124U /* module not init */

/**
 * @ingroup bsl
 * 0x83 不支持操作。
 */
#define AEN_ENO_NOTSUPPORT 131U /* Not support */

/**
 * @ingroup bsl
 * 0x84 主动退出指示。
 */
#define AEN_ENO_ABORT 132U  /* abort */

/**
 * @ingroup bsl
 * 0x85 资源达到上限。
 */
#define AEN_ENO_NRESOURCE 133U /* Out of resources */

/**
 * @ingroup bsl
 * 0xff 未知错误码。
 */
#define AEN_ENO_UNKNOWN 255U /* Unknow reason */

/**
 * 错误码统一生成宏
 * mid_ 表示中间件编号
 * module_ 表示模块编号
 * eno_ 表示具体错误码
 */
#define BSL_ENO_MAKE(mid_, module_, eno_)   (((uint32_t)BSL_ENO_REVERSED_FLAG << 24U) | ((uint32_t)(mid_) << 16U) | \
                                             ((uint32_t)(module_) << 8U) | (uint32_t)(eno_))

/**
 * @ingroup bsl
 * 0x2001 模块已初始化。保持兼容保留。完成错误码归一后消除。
 */
#define AEN_ENO_ALREADY_INIT 8193U /* module already init */

/**
 * @ingroup bsl
 * 0x2002 参数已配置。保持兼容保留。完成错误码归一后消除。
 */
#define AEN_ENO_PARAM_EXIST 8194U /* param already exist */

/**
 * @ingroup bsl
 * 0x1 操作失败。
 */
#ifndef BSL_ERR
#define BSL_ERR 1U
#endif

/**
 * @ingroup bsl
 * 0x0 操作成功。
 */
#ifndef BSL_OK
#define BSL_OK 0U /* OK */
#endif

#ifdef __cplusplus
}
#endif

#endif /* BSL_ENO_H */
