/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: MAC注册接口对外头文件
 * Create: 2020/05/18
 * History:
 */
#ifndef CRYPTO_CRYPTO_MAC_API_H
#define CRYPTO_CRYPTO_MAC_API_H

#include <stdint.h>
#include <stddef.h>
#include "keys/keys_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup crypto
 * crypto MAC算法枚举
 */
typedef enum {
    CRYPTO_MAC_256,  /* < MAC算法：MAC_256 */
    CRYPTO_MAC_384,  /* < MAC算法：MAC_384 */
    CRYPTO_MAC_SM3,  /* < MAC算法：MAC_SM3 */
    CRYPTO_CMAC_AES_128, /* < CMAC算法：AES_128 */
    CRYPTO_MAC_BUTT, /* < 最大枚举值 */
} CryptoMacAlgorithm;

/**
 * @ingroup crypto
 * Crypto Mac 上下文
 */
typedef struct CryptoMacCtx_ *CryptoMacHandle;

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_CRYPTO_MAC_API_H */
