/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: dh算法注册接口对外头文件
 * Create: 2022/05/09
 * History:
 */

#ifndef CRYPTO_DH_API_H
#define CRYPTO_DH_API_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DH_PRIME_MODULUS_MAX_LEN 1024u

typedef struct DhBigInt_ {
    size_t len;                             /* < bigInt 有效字节数 */
    uint8_t val[DH_PRIME_MODULUS_MAX_LEN];  /* < MSB 方式保存 bigInt */
} DhBigInt;

typedef struct CryptoDhParam_ {
    DhBigInt p;     /* < dh参数p */
    DhBigInt g;     /* < dh参数g */
} CryptoDhParam;

typedef CryptoDhParam *CryptoDhParamHandle;
typedef const CryptoDhParam *CryptoDhParamRoHandle;

#ifdef __cplusplus
}
#endif

#endif // CRYPTO_DH_API_H
