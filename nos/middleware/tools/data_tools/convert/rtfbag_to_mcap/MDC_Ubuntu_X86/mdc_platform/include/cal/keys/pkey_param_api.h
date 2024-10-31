/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:定义公钥体系中的公钥结构
 * Create: 2020/8/10
 * History:
 * 2020/8/10 第一次创建
 */

#ifndef PKEY_PARAM_API_H
#define PKEY_PARAM_API_H

#include <stdint.h>
#include <stddef.h>
#include "pkey_api.h"
#include "crypto/crypto_ecc_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup keys
 * pkey 最大大数长度为 1024
 */
#define PKEY_BIGINT_MAX_LEN 1024u /* < pkey 最大大数长度为 1024 */

/**
 * @ingroup keys
 * ED25519最大密钥长度为 32
 */
#define PKEY_ED25519_MAX_KEY_SIZE 32u

/**
 * @ingroup keys
 * PKeyBigInt 结构体， 用于传递 PKey
 */
typedef struct PKeyBigInt_ {
    size_t len;                       /* < bigInt 有效字节数 */
    uint8_t val[PKEY_BIGINT_MAX_LEN]; /* < MSB 方式保存 bigInt */
} PKeyBigInt;

/**
 * @ingroup keys
 * PKeyHiddenKey 结构体，用于传递 keys unique ID
 */
typedef struct {
    size_t len;                       /* < 缓冲区长度 */
    uint8_t buf[PKEY_BIGINT_MAX_LEN]; /* < keys 的唯一标识，例如UUID/filename */
} PKeyHiddenKey;

/**
 * @ingroup keys
 * PKeyRsaPublicKey 结构体，用于传递 RSA 公钥
 */
typedef struct {
    PKeyBigInt n; /* < 模数 n */
    PKeyBigInt e; /* < 公共指数 e */
} PKeyRsaPublicKey;

/**
 * @ingroup keys
 * PKeyRsaPrivateKey 结构体，用于传递 RSA 私钥
 */
typedef struct {
    PKeyBigInt n;    /* < 模数 n */
    PKeyBigInt e;    /* < 公共指数 e */
    PKeyBigInt d;    /* < 私有指数 */
    PKeyBigInt p;    /* < 主要因子 p */
    PKeyBigInt q;    /* < 主要因子 q */
    PKeyBigInt dP;   /* < CRT的指数dP */
    PKeyBigInt dQ;   /* < CRT的指数dQ */
    PKeyBigInt qInv; /* < CRT co-efficient qInv */
} PKeyRsaPrivateKey;

/**
 * @ingroup keys
 * PKeyRsaParam 结构体， 用于传递 RSA 参数
 */
typedef struct {
    uint32_t bits;     /* < rsa modulus 长度 */
    PKeyBigInt pubExp; /* < rsa public exponent */
} PKeyRsaParam;

/**
 * @ingroup keys
 * PKeyRsaKey 结构体， 用于传递 RSA 密钥对
 */
typedef struct {
    PKeyKeyType keyType; /* < 密钥对类型 */
    PKeyRsaParam rsaParam;
    union {
        PKeyRsaPublicKey publicKey;   /* < 联合体成员：公钥 */
        PKeyRsaPrivateKey privateKey; /* < 联合体成员：私钥 */
        PKeyHiddenKey hiddenKey;      /* < 联合体成员：hidden Key */
        void *bypassKey;              /* < 联合体成员：by pass key */
    } key;
} PKeyRsaKey;

/**
 * @ingroup keys
 * PKeyEdDsaKeyData 结构体，用于传递 EDDSA 密钥数据
 */
typedef struct {
    size_t len;                             /* < EDDSA 数据长度 */
    uint8_t buf[PKEY_ED25519_MAX_KEY_SIZE]; /* < EDDSA 数据 buffer */
} PKeyEdDsaKeyData;

/**
 * @ingroup keys
 * PKeyEdDsaKey 结构体，用于传递 EDDSA 密钥数据
 */
typedef struct {
    PKeyKeyType keyType; /* < 密钥类型 */
    union {
        PKeyEdDsaKeyData publicKey;  /* < 联合体成员：公钥 */
        PKeyEdDsaKeyData privateKey; /* < 联合体成员：私钥 */
        PKeyHiddenKey hiddenKey;     /* < 联合体成员：hidden Key */
        void *bypassKey;             /* < 联合体成员：by pass key */
    } key;
} PKeyEdDsaKey;

/**
 * @ingroup keys
 * PKeyEcDsaParam 结构体， 用于传递 ECDSA 参数
 */
typedef struct {
    CryptoEcGroupId groupId;
} PKeyEcDsaParam;

/**
 * @ingroup keys
 * PKeyEcDsaPublicKey 结构体，用于传递 ECDSA 公钥
 */
typedef struct PKeyEcPoint_ {
    PKeyBigInt x; /* < 变量 x 的大数 */
    PKeyBigInt y; /* < 变量 y 的大数 */
} PKeyEcPoint;

typedef PKeyEcPoint PKeyEcDsaPublicKey;

/**
 * @ingroup keys
 * PKeyEcDsaPrivateKey 结构体，用于传递 ECDSA 私钥
 */
typedef struct {
    PKeyBigInt x; /* < 变量 x 的大数 */
    PKeyBigInt y; /* < 变量 y 的大数 */
    PKeyBigInt p; /* < 变量 p 的大数 */
} PKeyEcDsaPrivateKey;

/**
 * @ingroup keys
 * PKeyEcDsaKey 结构体，用于传递 ECDSA 密钥对
 */
typedef struct PKeyEcDsaKey_ {
    PKeyKeyType keyType;  /* < 密钥对类型 */
    PKeyEcDsaParam param; /* < ecdsa参数 */
    union {
        PKeyEcDsaPublicKey publicKey;   /* < 联合体成员：公钥 */
        PKeyEcDsaPrivateKey privateKey; /* < 联合体成员：私钥 */
        PKeyHiddenKey hiddenKey;        /* < 联合体成员：hidden Key */
        void *bypassKey;                /* < 联合体成员：by pass key */
    } key;
} PKeyEcDsaKey;

/**
 * @ingroup keys
 * PKeyAsymmetricKey 结构体，用于传递非对称密钥对
 */
typedef struct PKeyAsymmetricKey_ {
    PKeyAlgorithmKeyType algKeyType; /* < 密钥对类型 */
    union {
        PKeyRsaKey rsaKey;  /* < 联合体成员：RSA 密钥对 */
        PKeyEcDsaKey ecKey; /* < 联合体成员：ECDSA 密钥对 */
        PKeyEdDsaKey edKey; /* < 联合体成员：EDDSA 密钥对 */
    } key;
    int32_t references;
} PKeyAsymmetricKey;

/**
 * @ingroup keys
 * PKeyGenerateKeyParam 结构体，用于传递生成密钥对时的参数
 */
typedef struct PKeyGenerateKeyParam_ {
    PKeyAlgorithmKeyType algKeyType; /* < 生成密钥对使用的算法 */
    union {
        PKeyRsaParam rsaParam;  /* < 联合体成员：RSA 参数 */
        PKeyEcDsaParam ecParam; /* < 联合体成员：ECDSA 参数  */
    } param;
} PKeyGenerateKeyParam;

#ifdef __cplusplus
}
#endif

#endif // PKEY_PARAM_API_H
