/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: asycipher注册接口对外头文件
 * Create: 2021/01/04
 * History:
 */

/** @defgroup adaptor adaptor */
/**
 * @defgroup crypto crypto
 * @ingroup adaptor
 */

#ifndef CRYPTO_ASYCIPHER_API_H
#define CRYPTO_ASYCIPHER_API_H

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @ingroup crypto
 * asycrypto 密码算法枚举
 */
typedef enum {
    CRYPTO_ASYCIPHER_NULL, /* < 没有密码算法 */
    CRYPTO_ASYCIPHER_SM2,  /* < 密码算法：CRYPTO_CIPHER_SM2 */
    CRYPTO_ASYCIPHER_BUTT, /* < 最大枚举值 */
} CryptoAsycipherAlgorithm;


/**
 * @ingroup crypto
 * Crypto Asy Cipher 上下文
 */
typedef struct CryptoAsyCipherCtx_ *CryptoAsycipherHandle;

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_ASYCIPHER_API_H */
