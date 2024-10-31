/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:
 * Create: 2020/11/10
 * History:
 * 2020/8/15 第一次创建
 */

#ifndef CME_PKI_ERRORFUN_API_H
#define CME_PKI_ERRORFUN_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Function names Contains the function tag code for the All the PKI functions
#define CME_PKI_SYSTEM_ERR_BASE 30u
#define CME_PKI_CONTEXT_ERR_BASE 31u
#define CME_PKI_COMMON_ERR_BASE 32u
#define CME_PKI_CERT_ERR_BASE 33u
#define CME_PKI_CRL_ERR_BASE 34u
#define CME_PKI_OCSP_ERR_BASE 35u

#ifdef __cplusplus
}
#endif

#endif
