/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description:TLS Adaptor 适配层数据类型定义头文件
 * Create: 2021/1/28
 */
#ifndef TLS_API_H
#define TLS_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TlsCtx_ *TlsCtxHandle;
typedef struct TlsSsl_ *TlsSslHandle;
typedef struct TlsX509_ *TlsX509Handle;
typedef struct TlsStack_ *TlsX509StackHandle;

typedef unsigned int (*TlsPskClientCbFunc)(const TlsSslHandle ssl,
                                           const char *hint,
                                           char *identity,
                                           unsigned int maxIdentityLen,
                                           unsigned char *psk,
                                           unsigned int maxPskLen);

typedef unsigned int (*TlsPskServerCbFunc)(const TlsSslHandle ssl,
                                           const char *identity,
                                           unsigned char *psk,
                                           unsigned int maxPskLen);

#ifdef __cplusplus
}
#endif
#endif