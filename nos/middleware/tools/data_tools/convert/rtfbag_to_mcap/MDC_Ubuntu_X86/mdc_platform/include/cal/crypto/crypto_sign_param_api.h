/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:签名参数结构体
 * Create: 2020/05/18
 * History:
 */

#ifndef CRYPTO_CRYPTO_SIGN_PARAM_API_H
#define CRYPTO_CRYPTO_SIGN_PARAM_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "crypto_hash_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup crypto
 * CryptoSignParam 结构体， 用于传递签名参数
 */
typedef struct CryptoSignParam_ {
    CryptoHashAlgorithm hashAlgorithm; /* < 签名验签使用的Hash方法 */
    const uint8_t *data;               /* < 需要签名的数据 */
    size_t dataLen;                    /* < 签名数据长度，需要大于0 */
    void *params;                      /* < 特定签名时需要传入参数，无参数需求是，可以传NULL */
    size_t paramsSize;                 /* < 参数的长度，参数为NULL时，长度不判断 */
    /* 一般来说，签名数据有两种格式：
     * 1.裸数据的形式。如ECC就是两个点的拼接，RSA就是两个大数的拼接。（具体拼接形式，要看上层业务）
     * 2.ASN.1格式的签名数据，格式会在对应的标准中定义，ECC的标准为x.62，RSA的为PKCS#1或RFC4055。
     * 因为大数有正数和负数的区别，所以如果是裸数据的形式，不好在上层处理，希望底层能够处理这种差别。
     * 以ECC签名为例，对应的openssl的接口为：ECDSA_do_verify和ECDSA_verify。 */
    bool isSignDataPrimitive; /* < 是否为裸数据。 */
    uint8_t *sign;            /* < 签名后的结果数据。如果传入空指针，则返回签名长度。 */
    size_t signLen; /* < 出入参: 入参表示保存签名结果的空间大小。出参: 表示实际签名后的数据长度。 */
} CryptoSignParam;

#ifdef __cplusplus
}
#endif

#endif // CRYPTO_CRYPTO_SIGN_PARAM_API_H
