/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:定义公钥体系中的公钥结构
 * Create: 2020/8/10
 * History:
 * 2020/8/10 第一次创建
 */
#ifndef PKEY_API_H
#define PKEY_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup keys
 * PKey 算法枚举
 */
typedef enum {
    PKEY_ALG_KEY_RSA,     /* < pkey算法：RSA */
    PKEY_ALG_KEY_ECDSA,   /* < pkey算法：ECDSA */
    PKEY_ALG_KEY_ED25519, /* < pkey算法：ED25519 */
    PKEY_ALG_KEY_SM2,     /* < pkey算法：SM2 */
    PKEY_ALG_KEY_END,     /* < 最大枚举值 */
} PKeyAlgorithmKeyType;

/**
 * @ingroup keys
 * PKey 类型枚举
 */
typedef enum {
    PKEY_KEY_TYPE_PUBLIC = 1,     /* < pkey类型：PUBLIC */
    PKEY_KEY_TYPE_PRIVATE = 2,    /* < pkey类型：PRIVATE */
    PKEY_KEY_TYPE_HIDDEN_PRIVATE, /* < pkey类型：HIDDEN_PRIVATE */
    PKEY_KEY_TYPE_HIDDEN_PUBLIC,  /* < pkey类型：HIDDEN_PUBLIC */
    PKEY_KEY_TYPE_BYPASS_PRIVATE, /* < pkey类型：BYPASS_PRIVATE */
    PKEY_KEY_TYPE_BYPASS_PUBLIC,  /* < pkey类型：BYPASS_PUBLIC */
    PKEY_KEY_TYPE_END,            /* < 最大枚举值 */
} PKeyKeyType;

/**
 * @ingroup keys
 * PKeyBigInt_ 句柄，用于传递大数
 */
typedef struct PKeyBigInt_ *PKeyBigIntHandle;

/**
 * @ingroup keys
 * PKeyBigInt_ 句柄，用于传递只读的大数
 */
typedef const struct PKeyBigInt_ *PKeyBigIntRoHandle;

/**
 * @ingroup keys
 * PKeyEcPoint_ 句柄，用于ECDSA公钥
 */
typedef struct PKeyEcPoint_ *PKeyEcPointHandle;
typedef const struct PKeyEcPoint_ *PKeyEcPointRoHandle;

/**
 * @ingroup keys
 * PKeyAsymmetricKey_ 句柄，用于传递非对称密钥对
 */
typedef struct PKeyAsymmetricKey_ *PKeyAsymmetricKeyHandle;

/**
 * @ingroup keys
 * PKeyAsymmetricKey_ 句柄，用于只读传递非对称密钥对
 */
typedef const struct PKeyAsymmetricKey_ *PKeyAsymmetricKeyRoHandle;

/**
 * @ingroup keys
 * PKeyGenerateKeyParam_ 句柄，用于传递生成密钥对时的参数
 */
typedef struct PKeyGenerateKeyParam_ *PKeyGenerateKeyParamHandle;

/**
 * @ingroup keys
 * PKeyEcDsaKey_ 句柄，用于传递ECDSA密钥对
 */
typedef struct PKeyEcDsaKey_ *PKeyEcDsaKeyHandle;

/**
 * @ingroup keys
 * @brief   创建一个公钥
 * @param   algKeyType [in] 公钥算法类型
 * @param   keyType [in] 公钥的类型，是公钥还是私钥
 * @param   key [out] 创建的公钥
 * @retval SAL_SUCCESS 创建成果
 * @retval SAL_ERR_NO_MEMORY 内存不足，创建失败
 */
int32_t PKEY_NewKey(PKeyAlgorithmKeyType algKeyType, PKeyKeyType keyType, PKeyAsymmetricKeyHandle *key);

/**
 * @ingroup keys
 * @brief   释放指定秘钥
 * @param   key [in] 秘钥句柄
 * @retval 无
 */
void PKEY_FreeKey(PKeyAsymmetricKeyHandle key);

#ifdef __cplusplus
}
#endif

#endif // PKEY_API_H
