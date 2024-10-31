/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: CryptoKeyDeriveParm结构体
 * Create: 2020/05/18
 * History:
 */

#ifndef CRYPTO_CRYPTO_KDF_DERIVE_API_H
#define CRYPTO_CRYPTO_KDF_DERIVE_API_H

#include <stdint.h>
#include <stddef.h>
#include "keys/keys_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup crypto
 * CryptoKeyDeriveInfo 结构体，用于传递密钥派生信息
 */
typedef struct CryptoKeyDeriveParam_ {
    KeysKeyHandle secret;          /* < prf 密钥 */
    size_t labelLen;               /* < 标签长度 */
    const uint8_t *label;          /* < 标签 */
    size_t seedLen;                /* < 种子长度 */
    const uint8_t *seed;           /* < 种子 */
    size_t outLen;                 /* < 需要派生密钥的长度 */
} CryptoKeyDeriveInfo;

#ifdef __cplusplus
}
#endif

#endif // CRYPTO_CRYPTO_KDF_DERIVE_API_H
