/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: cipher注册接口对外头文件
 * Create: 2020/05/18
 * History:
 */

#ifndef CRYPTO_CRYPTO_CIPHER_API_H
#define CRYPTO_CRYPTO_CIPHER_API_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup crypto
 * crypto 密码算法枚举
 */
typedef enum {
    CRYPTO_CIPHER_NULL,              /* < 没有密码算法 */
    CRYPTO_CIPHER_AES_128_GCM,       /* < 密码算法：AES_128_GCM */
    CRYPTO_CIPHER_AES_256_GCM,       /* < 密码算法：AES_256_GCM */
    CRYPTO_CIPHER_CHACHA20_POLY1305, /* < 密码算法：CHACHA20_POLY1305 */
    CRYPTO_CIPHER_SM4_CCM,           /* < 密码算法：SM4_CCM */
    CRYPTO_CIPHER_SM4_ECB,           /* < 密码算法：SM4/ECB/NoPadding */
    CRYPTO_CIPHER_AES_128_ECB,       /* < 密码算法：AES_128_ECB */
    CRYPTO_CIPHER_AES_128_CBC,       /* < 密码算法：AES_128_CBC */
    CRYPTO_CIPHER_SM4_CBC,           /* < 密码算法：SM4/CBC/PKCS7 */
    CRYPTO_CIPHER_SM4_GCM,           /* < 密码算法：SM4_GCM */
    CRYPTO_CIPHER_BUTT,              /* < 最大枚举值 */
} CryptoCipherAlgorithm;

/**
 * @ingroup crypto
 * Crypto Cipher 上下文
 */
typedef struct CryptoCipherCtx_ *CryptoCipherHandle;

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_CRYPTO_CIPHER_API_H */
