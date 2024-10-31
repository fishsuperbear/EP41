/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:   Authenticated Encryption with Associated Data,
 * Create: 2020/05/08
 * History:
 * 2020/05/08 第一次创建
 */
/** @defgroup hitls hitls */
/** @defgroup adaptor Adaptor_API
 * @ingroup hitls
 */

#ifndef SAL_ADAPTOR_REG_API_H
#define SAL_ADAPTOR_REG_API_H

#include "crypto_reg_api.h"
#include "keys_reg_api.h"
#include "pkey_reg_api.h"
#include "cert_reg_api.h"
#include "store_reg_api.h"
#include "comm_reg_api.h"
#include "zip_reg_api.h"
#include "tls_reg_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup adaptor
 * CertAdaptHandleFuncs 结构体，注册函数集合
 */
typedef struct AdaptorRegisterFunc_ {
    CryptoAdaptHandleFunc *cryptoFunc; /* < 所有加解密函数注册函数 */
    KeysAdaptHandleFunc *keysFunc;     /* < 所有密钥管理函数注册函数 */
    PKeyAdaptHandleFunc *pkeyFunc;     /* < 所有密钥对函数注册函数 */
    CertAdaptHandleFuncs *certFunc;    /* < 所有证书函数注册函数 */
    StoreAdaptorHandleFunc *storeFunc; /* < 所有存储管理函数注册函数 */
    CommAdaptorHandleFunc *httpFunc;   /* < 所有HTTP/HTTPS处理函数注册函数 */
    ZipAdaptorHandleFunc *zipFunc;     /* < 所有解压zip函数注册函数 */
    TlsAdapterHandleFunc *tlsFunc;  /* < 所有tls函数注册函数 */
} AdaptorRegisterFunc;

/**
 * @ingroup adaptor
 * @brief 注册Crypto\Keys\Cert\Os 适配层接口
 * @param adaptorRegisterFunc [IN] 注册函数集合
 * @retval void。
 * @see 无
 */
void SAL_AdaptorRegister(const AdaptorRegisterFunc *adaptorRegisterFunc);

#ifdef __cplusplus
}
#endif

#endif /* SAL_ADAPTOR_REG_API_H */
