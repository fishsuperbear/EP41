/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description:TLS 统一注册对外接口
 * Create: 2021/1/28
 */
#ifndef TLS_REG_API_H
#define TLS_REG_API_H

#include <stdint.h>
#include <stddef.h>
#include <tls/tls_api.h>
#include <keys/pkey_param_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup adaptor
 * @brief SSL 初始化
 *
 * @param void [IN]。
 *
 * @retval void
 */
typedef void (*TlsInitFunc)(void);

/**
* @ingroup adaptor
* @brief 创建 server ctx 对象
*
* @param void [IN]。
*
* @retval TlsCtxHandle 对象
*/
typedef TlsCtxHandle (*TlsServerCtxCreateFunc)(void);

/**
* @ingroup adaptor
* @brief 创建 client ctx 对象
*
* @param void [IN]。
*
* @retval TlsCtxHandle 对象
*/
typedef TlsCtxHandle (*TlsClientCtxCreateFunc)(void);

/**
* @ingroup adaptor
* @brief 释放 tls ctx 对象
*
* @param ctx [IN] 待释放的ctx
*
* @retval NULL
*/
typedef void (*TlsCtxFreeFunc)(TlsCtxHandle ctx);

/**
* @ingroup adaptor
* @brief 设置 cipher list 列表
*
* @param ctx [IN] 待配置的ctx
* @param str [IN] 配置的cipher list
*
* @retval 1 成功
*         0 失败
*/
typedef int32_t (*TlsCtxCipherListSetFunc)(TlsCtxHandle ctx, const uint8_t *str);

/**
* @ingroup adaptor
* @brief 设置 tls client 回调函数
*
* @param ctx [IN] 待配置的ctx
* @param cb [IN] client回调函数
*
* @retval NULL
*/
typedef void (*TlsCtxPskClientCbSetFunc)(TlsCtxHandle ctx, TlsPskClientCbFunc cb);

/**
* @ingroup adaptor
* @brief 设置 tls server 回调函数
*
* @param ctx [IN] 待配置的ctx
* @param cb [IN] server 回调函数
*
* @retval NULL
*/
typedef void (*TlsCtxPskServerCbSetFunc)(TlsCtxHandle ctx, TlsPskServerCbFunc cb);

/**
* @ingroup adaptor
* @brief 设置使用的客户端证书
*
* @param ctx [IN] 待配置的ctx
* @param x [IN] x509 证书
*
* @retval 1 成功
*         0 失败
*/
typedef int32_t (*TlsCtxCertificateUseFunc)(TlsCtxHandle ctx, TlsX509Handle x);

/**
* @ingroup adaptor
* @brief 上下文ctx添加可信任CA证书
*
* @param ctx [IN] 待配置的ctx
* @param x [IN] X509证书内容
*
* @retval 成功 SAL_SUCCESS
*         失败 非SAL_SUCCESS
*/
typedef int32_t (*TlsCtxAddCertFunc)(TlsCtxHandle ctx, TlsX509Handle x);

/**
* @ingroup adaptor
* @brief 获取最新 ssl 错误代码
*
* @param void [IN]
*
* @retval errcode
*/
typedef uint64_t (*TlsErrLastErrorPeekFunc)(void);

/**
* @ingroup adaptor
* @brief 清除上一次 ssl 错误缓存
*
* @param void [IN]
*
* @retval NULL
*/
typedef void (*TlsErrErrorClearFunc)(void);

/**
* @ingroup adaptor
* @brief 获取最新 ssl error string
*
* @param void [IN]
*
* @retval err string
*/
typedef char *(*TlsErrorStringGetFunc)(uint64_t e);

/**
* @ingroup adaptor
* @brief 获取 ssl connect/accpet/read/write等操作的错误
*
* @param void [IN]
*
* @retval errcode
*/
typedef int32_t (*TlsSslErrorGetFunc)(const TlsSslHandle ssl, int32_t retCode);

/**
* @ingroup adaptor
* @brief 从上下文ctx创建ssl
*
* @param ctx [IN] 上下文
*
* @retval TlsSslHandle ssl句柄
*/
typedef TlsSslHandle (*TlsSslNewFunc)(TlsCtxHandle ctx);

/**
* @ingroup adaptor
* @brief 释放ssl
*
* @param ssl [IN] 待释放的ssl句柄
*
* @retval NULL
*/
typedef void (*TlsSslFreeFunc)(TlsSslHandle ssl);

/**
* @ingroup adaptor
* @brief 关闭ssl
*
* @param ssl [IN] 待关闭的ssl句柄
*
* @retval void
*/
typedef void (*TlsSslShutDownFunc)(TlsSslHandle ssl);

/**
* @ingroup adaptor
* @brief 设置ssl mode 为auto retry
*
* @param ssl [IN] 待设置的ssl句柄
*
* @retval NULL
*/
typedef void (*TlsAutoRetryModeSetFunc)(TlsSslHandle ssl);

/**
* @ingroup adaptor
* @brief 设置 fd 句柄到 ssl
*
* @param ssl [IN] 待设置的ssl句柄
* @param fd [IN] 调用socket创建的文件描述符
*
* @retval 1 成功
*         0 失败
*/
typedef int32_t (*TlsFdSetFunc)(TlsSslHandle ssl, int32_t fd);

/**
* @ingroup adaptor
* @brief 设置 host 到 ssl
*
* @param ssl [IN] 待设置的ssl句柄
* @param url [IN] ssl连接设置需访问的url
*
* @retval 1 成功
*         0 失败
*/
typedef int32_t (*TlsTlsExtHostNameSetFunc)(TlsSslHandle ssl, const uint8_t *url);

/**
* @ingroup adaptor
* @brief ssl accept操作
*
* @param ssl [IN] 等待一个TLS/SSL客户端启动TLS/SSL握手的句柄
*
* @retval 1 成功
*         0 失败
*/
typedef int32_t (*TlsSslAcceptFunc)(TlsSslHandle ssl);

/**
* @ingroup adaptor
* @brief ssl connect操作
*
* @param ssl [IN] 进行 ssl connect 的句柄
*
* @retval 1 成功
*         0 失败
*/
typedef int32_t (*TlsConnectFunc)(TlsSslHandle ssl);

/**
* @ingroup adaptor
* @brief ssl 连接状态设置
*
* @param ssl [IN] 待配置的ssl句柄
*
* @retval NULL
*/
typedef void (*TlsConnectionStateSetFunc)(TlsSslHandle ssl);

/**
* @ingroup adaptor
* @brief 非阻塞ssl accept 状态设置
*
* @param ssl [IN] 待配置的ssl句柄
*
* @retval NULL
*/
typedef void (*TlsAcceptStateSetFunc)(TlsSslHandle ssl);

/**
* @ingroup adaptor
* @brief ssl read
*
* @param ssl [IN] ssl句柄
* @param buf [IN] 读取buf空间
* @param num [IN] buf长度
*
* @retval 读取内容长度
*/
typedef int32_t (*TlsReadFunc)(TlsSslHandle ssl, uint8_t *buf, int32_t num);

/**
* @ingroup adaptor
* @brief ssl write
*
* @param ssl [IN] ssl句柄
* @param buf [IN] 要写的内容
* @param num [IN] 要写的长度
*
* @retval 写成功内容长度
*/
typedef int32_t (*TlsWriteFunc)(TlsSslHandle ssl, const uint8_t *buf, int32_t num);

/**
* @ingroup adaptor
* @brief pem 格式转x509
*
* @param buf [IN] 要转换的内容
* @param bLen [IN] 要转换的长度
*
* @retval 成功 非NULL
*         失败 NULL
*/
typedef TlsX509Handle (*TlsPem2X509Func)(const uint8_t *buf, uint32_t bLen);

/**
* @ingroup adaptor
* @brief 释放x509格式报文句柄
*
* @param x [IN] 待释放的内容
*
* @retval NULL
*/
typedef void (*TlsX509CertFreeFunc)(TlsX509Handle x);

/**
* @ingroup adaptor
* @brief EccPriKey加入上下文ctx
*
* @param ctx [IN] 上下文
* @param key [IN] EccPriKey 内容
* @param keyLen [IN] key长度
*
* @retval 成功 1
*         失败 0
*/
typedef int32_t (*TlsCtxEccPrivKeyUseFunc)(TlsCtxHandle ctx, const uint8_t *key, uint32_t keyLen);

/**
* @ingroup adaptor
* @brief 获取对端证书链
*
* @param ssl [IN] ssl句柄
*
* @retval 成功 非NULL
*         失败 NULL
*/
typedef TlsX509StackHandle (*TlsPeerCertChainGetFunc)(TlsSslHandle ssl);

/**
* @ingroup adaptor
* @brief 获取对端证书链长度
*
* @param chain [IN] 证书链
*
* @retval 证书链长度
*/
typedef size_t (*TlsX509CertChainSizeGetFunc)(TlsX509StackHandle chain);

/**
* @ingroup adaptor
* @brief 根据索引获取X509证书链
*
* @param chain [IN] 证书链
* @param idx [IN] 证书链索引
*
* @retval 成功 非NULL
*         失败 NULL
*/
typedef TlsX509Handle (*TlsX509CertGetByIndexFunc)(TlsX509StackHandle chain, size_t idx);

/**
* @ingroup adaptor
* @brief X509证书转换成buffer
*
 * @param cert [IN] 证书句柄
 * @param buf [OUT] 缓存指针
 * @param bLen [IN] 缓存长度
*
* @retval 成功 1
*         失败 0
*/
typedef int32_t (*TlsX509CertWrite2BufferFunc)(TlsX509Handle cert, uint8_t *buf, uint32_t blen);

/**
* @ingroup adaptor
* @brief X509证书转换成buffer
*
* @param cert [IN] 证书
* @param buffer [IN] 缓存
*
* @retval 成功 0
*         失败 非0
*/
typedef int64_t (*TlsVerifyResultGetFunc)(TlsSslHandle ssl);

/**
* @ingroup adaptor
* @brief 添加扩展项
*
* @param ctx [IN] 上下文
* @param x [IN] 扩展项
*
* @retval 成功 0
*         失败 非0
*/
typedef int64_t (*TlsExtraChainCertAddFunc)(TlsCtxHandle ctx, TlsX509Handle x);

/**
* @ingroup adaptor
* @brief DER 格式转 X509格式
*
* @param buf [IN] DER格式证书内容
* @param bLen [IN] 证书长度
*
* @retval 成功 0
*         失败 非0
*/
typedef TlsX509Handle (*TlsDer2X509Func)(uint8_t *buf, uint32_t bLen);

/**
* @ingroup adaptor
* @brief X509 格式转 PEM 格式
*
* @param x [IN] X509
* @param pemBuf [OUT] pem buffer
* @param bLen [OUT] pem buffer len
*
* @retval 成功 0
*         失败 非0
*/
typedef void (*Tls509ToPemFunc)(TlsX509Handle x, uint8_t *pemBuf, uint32_t *pemBufSize);

/**
* @ingroup adaptor
* @brief 设置对端校验
*
* @param ctx [IN] 待设置的上下文
*
* @retval NULL
*/
typedef void (*TlsCtxPeerValidationSetFunc)(TlsCtxHandle ctx);

/**
* @ingroup adaptor
* @brief Csa签名
*
* @param ctx [IN] ctx 原文
* @param ctxLen [IN] ctx 原文长度
* @param sign [OUT] 签名
* @param signLen [OUT] 签名长度
* @retval SAL_SUCCESS 签名成功
*         其他 失败
*
*/
typedef uint32_t (*TlsSslSignFunc)(const uint8_t *ctx, uint32_t ctxLen, const char *keyPath,
                                   uint8_t *sign, uint32_t *signLen);
/**
* @ingroup adaptor
* @brief Csa验签
*
* @param pubKeyCert [IN] 公钥证书
* @param ctx [IN] 原文
* @param ctx [IN] ctx 原文长度
* @param sign [OUT] 签名
* @param signLen [OUT] 签名长度
* @retval SAL_SUCCESS 验签成功
*         其他 失败
*/
typedef uint32_t (*TlsSslVerifyFunc)(const uint8_t *pubKeyCert, const uint8_t *ctx, uint32_t ctxLen,
                                     const uint8_t *sign, uint32_t signLen);

typedef struct TagSslAdapterHandleFunc {
    TlsInitFunc tlsInitFunc;
    TlsServerCtxCreateFunc tlsServeCtxCreateFunc;
    TlsClientCtxCreateFunc tlsClientCtxCreateFunc;
    TlsCtxFreeFunc tlsCtxFreeFunc;
    TlsCtxCipherListSetFunc tlsCtxCipherListSetFunc;
    TlsCtxPskClientCbSetFunc tlsCtxPskClientCbSetFunc;
    TlsCtxPskServerCbSetFunc tlsCtxPskServerCbSetFunc;
    TlsCtxAddCertFunc tlsCtxCaCertAddFunc;
    TlsCtxAddCertFunc tlsCtxClientCertAddFunc;
    TlsErrLastErrorPeekFunc tlsErrLastErrorPeekFunc;
    TlsErrErrorClearFunc tlsErrErrorClearFunc;
    TlsErrorStringGetFunc tlsErrorStringGetFunc;
    TlsSslErrorGetFunc tlsSslErrorGetFunc;
    TlsSslNewFunc tlsSslNewFunc;
    TlsSslFreeFunc tlsSslFreeFunc;
    TlsSslShutDownFunc tlsSslShutDownFunc;
    TlsAutoRetryModeSetFunc tlsAutoRetryModeSetFunc;
    TlsFdSetFunc tlsFdSetFunc;
    TlsTlsExtHostNameSetFunc tlsSetTlsExtHostNameFunc;
    TlsSslAcceptFunc tlsSslAcceptFunc;
    TlsConnectFunc tlsConnectFunc;
    TlsConnectionStateSetFunc tlsConnectionStateSetFunc;
    TlsAcceptStateSetFunc tlsAcceptStateSetFunc;
    TlsReadFunc tlsReadFunc;
    TlsWriteFunc tlsWriteFunc;
    TlsPem2X509Func tlsPem2X509Func;
    TlsX509CertFreeFunc tlsX509CertFreeFunc;
    TlsCtxEccPrivKeyUseFunc tlsCtxEccPrivKeyUseFunc;
    TlsPeerCertChainGetFunc tlsPeerCertChainGetFunc;
    TlsX509CertChainSizeGetFunc tlsX509CertChainSizeGetFunc;
    TlsX509CertGetByIndexFunc tlsX509CertGetByIndexFunc;
    TlsX509CertWrite2BufferFunc tlsCertWrite2BufferFunc;
    TlsVerifyResultGetFunc tlsVerifyResultGetFunc;
    TlsExtraChainCertAddFunc tlsExtraChainCertAddFunc;
    TlsDer2X509Func tlsDer2X509Func;
    Tls509ToPemFunc tlsX509ToPemFunc;
    TlsCtxPeerValidationSetFunc tlsCtxPeerValidationSetFunc;
    TlsSslSignFunc tlsSslSignFunc;
    TlsSslVerifyFunc tlsSslVerifyFunc;
} TlsAdapterHandleFunc;

#ifdef __cplusplus
}
#endif
#endif