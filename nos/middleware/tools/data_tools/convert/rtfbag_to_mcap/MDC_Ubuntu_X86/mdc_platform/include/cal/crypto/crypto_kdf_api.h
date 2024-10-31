/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: KDF注册接口对外头文件
 * Create: 2020/05/18
 * History:
 */
#ifndef CRYPTO_CRYPTO_KDF_API_H
#define CRYPTO_CRYPTO_KDF_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
* @ingroup crypto
* crypto KDF算法枚举
*/
typedef enum {
    CRYPTO_KDF_TLS_SHA_256, /* < KDF算法：SHA_256 */
    CRYPTO_KDF_TLS_SHA_384, /* < KDF算法：SHA_384 */
    CRYPTO_KDF_TLS_SHA_512, /* < KDF算法：SHA_512 */
    CRYPTO_KDF_TLS_SM3, /* < KDF算法：SM3 */
    CRYPTO_KDF_AES_MP,  /* < KDF算法：AES_MP(Miyaguchi-Preneel) */
    CRYPTO_KDF_BUTT,    /* < 最大枚举值 */
} CryptoKdfAlgorithm;

/**
 * @ingroup crypto
 * CryptoKeyDeriveParam_ 句柄，用于传递密钥派生信息
 */
typedef struct CryptoKeyDeriveParam_ *CryptoKeyDeriveParamHandle;

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_CRYPTO_KDF_API_H */
