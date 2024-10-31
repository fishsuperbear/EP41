/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:
 * Create: 2020/9/6
 * Notes:
 * History:
 */

#ifndef CERT_VERIFY_API_H
#define CERT_VERIFY_API_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cert
 * CertVerifyParam 结构体，用于传递证书验证参数
 */
typedef struct CertVerifyParam_ {
    size_t hostNameSize;   /* < 主机名长度 */
    uint8_t *hostName;     /* < 对等证书中的主机名匹配 */
    uint64_t checkTime;    /* < 使用时间 */
    int32_t depth;         /* < 验证深度，-1不受限制 */
    size_t ipSize;         /* < IP地址长度 */
    uint8_t *ip;           /* < 如果不是空IP地址匹配 */
    bool checkCRL;         /* < 检查 crl 标志位 */
    bool checkExtKeyUsage; /* < 检查证书公钥用途是否包含 Server Authentication */
} CertVerifyParam;

#ifdef __cplusplus
}
#endif

#endif // CERT_VERIFY_API_H