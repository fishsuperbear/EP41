/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:签名注册接口对外头文件
 * Create: 2020/05/18
 * History:
 */
#ifndef CRYPTO_CRYPTO_SIGN_API_H
#define CRYPTO_CRYPTO_SIGN_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup crypto
 * crypto 签名算法枚举
 */
typedef enum {
    CRYPTO_SIGN_RSA_PKCS1_V15, /* < 签名算法：RSA_PKCS1_V15 */
    CRYPTO_SIGN_ECDSA,         /* < 签名算法：ECDSA */
    CRYPTO_SIGN_ED25519,       /* < 签名算法：ED25519 */
    CRYPTO_SIGN_SM2,           /* < 签名算法：SM2 */
    CRYPTO_SIGN_BUTT,          /* < 最大枚举值 */
} CryptoSignAlgorithm;

/**
 * @ingroup cert
 * 证书签名算法枚举
 */
typedef enum {
    CRYPTO_SIGN_SCHEME_RSA_MD5,          /* < 签名算法：RSA_MD5 */
    CRYPTO_SIGN_SCHEME_RSA_SHA1,         /* < 签名算法：RSA_SHA1 */
    CRYPTO_SIGN_SCHEME_RSA_PKCS1_SHA256, /* < 签名算法：RSA_PKCS1_SHA256 */
    CRYPTO_SIGN_SCHEME_RSA_PKCS1_SHA384, /* < 签名算法：RSA_PKCS1_SHA384 */
    CRYPTO_SIGN_SCHEME_RSA_PKCS1_SHA512, /* < 签名算法：RSA_PKCS1_SHA512 */
    CRYPTO_SIGN_SCHEME_ECDSA_SHA1,       /* < 签名算法：ECDSA_SECP256R1_SHA1 */
    CRYPTO_SIGN_SCHEME_ECDSA_SHA256,     /* < 签名算法：ECDSA_SECP256R1_SHA256 */
    CRYPTO_SIGN_SCHEME_ECDSA_SHA384,     /* < 签名算法：ECDSA_SECP384R1_SHA384 */
    CRYPTO_SIGN_SCHEME_ECDSA_SHA512,     /* < 签名算法：ECDSA_SECP521R1_SHA512 */
    CRYPTO_SIGN_SCHEME_SM2_SM3,          /* < 签名算法：CRYPTO_SIGN_SCHEME_SM2_SM3 */
    CRYPTO_SIGN_SCHEME_SM2_SHA1,         /* < 签名算法：CRYPTO_SIGN_SCHEME_SM2_SHA1 */
    CRYPTO_SIGN_SCHEME_SM2_SHA256,       /* < 签名算法：CRYPTO_SIGN_SCHEME_SM2_SHA256 */
    CRYPTO_SIGN_SCHEME_ED25519,          /* < 签名算法：ED25519 */
    CRYPTO_SIGN_SCHEME_BUTT              /* < 最大枚举值 */
} CryptoSignSchemes;

/**
 * @ingroup crypto
 * CryptoSignParam_ 句柄， 用于传递签名参数
 */
typedef struct CryptoSignParam_ *CryptoSignParamHandle;
typedef const struct CryptoSignParam_ *CryptoSignParamRoHandle;

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_CRYPTO_SIGN_API_H */
