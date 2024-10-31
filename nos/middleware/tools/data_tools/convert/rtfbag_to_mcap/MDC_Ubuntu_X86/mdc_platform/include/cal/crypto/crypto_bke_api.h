/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: 蝴蝶算法注册接口对外头文件
 * Create: 2021/01/18
 * History:
 */

#ifndef CRYPTO_BKE_API_H
#define CRYPTO_BKE_API_H

#include "keys/keys_api.h"
#include "keys/pkey_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup keys
 * 扩展密钥最小的大小 256位 即32个字节
 */
#define BKE_DERIVATION_PRIVATE_KEY_SIZE 32u

/**
 * @ingroup keys
 * 蝴蝶算法非对称密钥类型为SM2
 */
#define BKE_ASYMMETRIC_KEY_TYPE PKEY_ALG_KEY_SM2

/**
 * @ingroup crypto
 * bke 蝴蝶算法密钥套根据国标(A a kS)为一套与签名相关，(P p kE)为一套与加密相关
 */
typedef struct BKESuitKeys_ {
    PKeyAsymmetricKeyHandle asyPublicKey; /* < 非对称公钥，对应A或P */
    KeysKeyHandle asyPrivateKey; /* < 非对称私钥，对应a或p */
    KeysKeyHandle sysKey; /* < 对称密钥，对应kS或kE */
} BkeSuitKeys;

/**
 * @ingroup crypto
 * asycrypto 蝴蝶算法密钥套件
 */
typedef BkeSuitKeys *BkeSuitKeysHandle;
typedef const BkeSuitKeys *BkeSuitKeysRoHandle;

/**
 * @ingroup crypto
 * bke 密钥扩展类型枚举
 */
typedef enum {
    BKE_DERIVATION_FS, /* < 扩展签名密钥 */
    BKE_DERIVATION_FE, /* < 扩展加密密钥 */
} BKEDerivationFuncType;

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_BKE_API_H */