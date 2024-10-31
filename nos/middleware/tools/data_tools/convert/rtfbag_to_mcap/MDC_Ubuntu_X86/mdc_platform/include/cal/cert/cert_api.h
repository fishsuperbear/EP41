/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:证书注册接口对外头文件
 * Create: 2020/6/10
 * History:
 */

/** @defgroup adaptor adaptor */
/** @defgroup cert cert
 * @ingroup adaptor
 */

#ifndef CERT_CERT_API_H
#define CERT_CERT_API_H

/** 密钥用途定义 */
#define CERT_KEY_USAGE_ENCIPHER_ONLY  0x0001u       /* < 仅用于加密 */
#define CERT_KEY_USAGE_CRL_SIGN  0x0002u            /* < CRL签名 */
#define CERT_KEY_USAGE_KEY_CERT_SIGN  0x0004u       /* < 证书签名 */
#define CERT_KEY_USAGE_KEY_AGREEMENT  0x0008u       /* < 密钥协商 */
#define CERT_KEY_USAGE_DATA_ENCIPHERMENT  0x0010u   /* < 数据加密 */
#define CERT_KEY_USAGE_KEY_ENCIPHERMENT  0x0020u    /* < 密钥加密 */
#define CERT_KEY_USAGE_NON_REPUDIATION  0x0040u     /* < 认可签名 */
#define CERT_KEY_USAGE_DIGITAL_SIGNATURE  0x0080u   /* < 数字签名 */
#define CERT_KEY_USAGE_DECIPHER_ONLY  0x8000u       /* < 仅用于解密 */

/**
 * @ingroup cert
 * cert 上下文
 */
typedef struct CertCtx_ *CertHandle;

/**
 * @ingroup cert
 * cert 只读上下文
 */
typedef const struct CertCtx_ *CertRoHandle;

/**
 * @ingroup cert
 * cert pool 上下文
 */
typedef struct CertPool_ *CertPoolHandle;

/**
 * @ingroup cert
 * 证书授权算法枚举
 */
typedef enum {
    CERT_AUTH_NULL,  /* < 没有授权算法 */
    CERT_AUTH_RSA,   /* < 授权算法使用 RSA */
    CERT_AUTH_ECDSA, /* < 授权算法使用 ECDSA */
    CERT_AUTH_SM2,   /* < 授权算法使用 SM2 */
    CERT_AUTH_BUTT,  /* < 最大枚举值 */
} CertAuthorization;

/**
 * @ingroup cert
 * 证书签名算法枚举
 */
typedef enum {
    CERT_SIG_SCHEME_RSA_PKCS1_SHA256 = 0X0401,       /* < 签名算法：RSA_PKCS1_SHA256 */
    CERT_SIG_SCHEME_RSA_PKCS1_SHA384 = 0X0501,       /* < 签名算法：RSA_PKCS1_SHA384 */
    CERT_SIG_SCHEME_RSA_PKCS1_SHA512 = 0X0601,       /* < 签名算法：RSA_PKCS1_SHA512 */
    CERT_SIG_SCHEME_ECDSA_SECP256R1_SHA256 = 0X0403, /* < 签名算法：ECDSA_SECP256R1_SHA256 */
    CERT_SIG_SCHEME_ECDSA_SECP384R1_SHA384 = 0X0503, /* < 签名算法：ECDSA_SECP384R1_SHA384 */
    CERT_SIG_SCHEME_ECDSA_SECP521R1_SHA512 = 0X0603, /* < 签名算法：ECDSA_SECP521R1_SHA512 */
    CERT_SIG_SCHEME_ED25519 = 0X0807,                /* < 签名算法：ED25519 */
    CERT_SIG_SCHEME_SM2_SM3 = 0X0707,                /* < 签名算法：SM2SM3 */
    CERT_SIG_SCHEME_UNKNOWN = 0xffff                 /* < 未知签名算法 */
} CertSignatureSchemes;

/**
 * @ingroup cert
 * 证书验证参数结构体
 */
typedef struct CertVerifyParam_ *CertVerifyParamHandle;

#endif // CERT_CERT_API_H
