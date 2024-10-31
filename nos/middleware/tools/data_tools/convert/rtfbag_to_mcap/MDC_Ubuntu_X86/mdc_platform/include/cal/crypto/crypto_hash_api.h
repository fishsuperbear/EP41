/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: HASH注册接口对外头文件
 * Create: 2020/05/18
 * History:
 */
#ifndef CRYPTO_CRYPTO_HASH_API_H
#define CRYPTO_CRYPTO_HASH_API_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup crypto
 * crypto 哈希算法枚举
 */
typedef enum {
    CRYPTO_HASH_MD5,     /* < 哈希算法：MD5 */
    CRYPTO_HASH_SHA1,    /* < 哈希算法：SHA1 */
    CRYPTO_HASH_SHA_256, /* < 哈希算法：SHA_256 */
    CRYPTO_HASH_SHA_384, /* < 哈希算法：SHA_384 */
    CRYPTO_HASH_SHA_512, /* < 哈希算法：SHA_512 */
    CRYPTO_HASH_SM3,     /* < 哈希算法：SM3 */
    CRYPTO_HASH_BUTT,    /* < 最大枚举值 */
} CryptoHashAlgorithm;

/**
 * @ingroup crypto
 * Crypto Hash 上下文
 */
typedef struct CryptoHashCtx_ *CryptoHashHandle;

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_CRYPTO_HASH_API_H */
