/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:非对称秘钥管理功能对外接口
 * Create: 2020/08/28
 * History:
 */
#ifndef ADAPTOR_PKEY_REG_API_H
#define ADAPTOR_PKEY_REG_API_H

#include <stdbool.h>
#include <stddef.h>
#include "keys/pkey_api.h"
#include "crypto/crypto_ecc_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* keySize:
 * when RSA:The number of bits in the modulus. 256, 512, 768, 1024, 1536 and 2048 bit keys SHALL be supported.
 * when ECDSA:Between 160 and 521 bits.
 * when EDDSA:256 bits.
 */
/**
 * @ingroup adaptor
 * @brief 生成密钥对的钩子。
 *
 * @param algkeyType [IN] 密钥类型
 * @param param [IN] 生成密钥的参数
 * @param publicKey [OUT] 生成的公钥
 * @param privateKey [OUT] 生成的私钥
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*PKeyGenerateKeyPairFunc)(PKeyAlgorithmKeyType algKeyType, PKeyGenerateKeyParamHandle param,
                                           PKeyAsymmetricKeyHandle publicKey, PKeyAsymmetricKeyHandle privateKey);

/**
 * @ingroup adaptor
 * @brief 校验密钥对的钩子。
 *
 * @param publicKey [IN] 密钥对的公钥
 * @param privateKey [IN] 密钥对的私钥
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef bool (*PKeyCheckKeyPairedFunc)(PKeyAsymmetricKeyRoHandle publicKey, PKeyAsymmetricKeyRoHandle privateKey);

/**
 * @ingroup adaptor
 * @brief 移除密钥对的钩子，它和PKeyGenerateKeyPairFunc对应。
 * 如果GenerateKeyPair后的Key是永久保留，则需要调用此接口移除，否则不需要，移除Key并不会释放Key指针，只是从永久保留区删除而已。
 *
 * @param key [IN] 待移除的 KEY
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*KeyRemoveKeyFunc)(PKeyAsymmetricKeyHandle key);

/**
 * @ingroup adaptor
 * @brief ec公钥转换为数组函数。
 *
 * @param groupId [IN] 公钥类型
 * @param octs [OUT] 导出数组，数组为标识+内容, 标识在octs[0], 0x04标识未压缩, 0x02标识压缩y0, 0x03标识压缩y1
 * @param octsLen [IN/OUT] 导出数组长度, 返回使用长度
 * @param point [IN] 待转换的公钥
 */
typedef int32_t (*PKeyEcpointToOctsFunc)(CryptoEcGroupId groupId, uint8_t *octs, size_t *octsLen,
                                         PKeyEcPointRoHandle point);

/**
 * @ingroup adaptor
 * @brief ec公钥转换为数组函数。
 *
 * @param groupId [IN] 公钥类型
 * @param point [OUT] 导出公钥
 * @param octs [IN] 导入的数组 数组为标识+内容形式, 标识在octs[0], 0x04标识未压缩, 0x02标识压缩y0, 0x03标识压缩y1
 * @param octsLen [IN] 导入的数组长度
 */
typedef int32_t (*PKeyOctsToEcpointRFunc)(CryptoEcGroupId groupId, PKeyEcPointHandle point, const uint8_t *octs,
                                          size_t octsLen);

/**
 * @ingroup adaptor
 * PKeyAdaptHandleFunc 结构体，密钥对管理适配层功能钩子函数集
 */
typedef struct {
    PKeyGenerateKeyPairFunc pkeyGenerateKeyPairCb; /* < 生成密钥对的钩子 */
    PKeyCheckKeyPairedFunc pkeyCheckPaired;        /* < 校验密钥对钩子 */
    PKeyOctsToEcpointRFunc pkeyOctsToEcpoint;      /* < 数组转换为ec公钥的钩子 */
    PKeyEcpointToOctsFunc pkeyEcPointToOcts;       /* < ec公钥转换为数组的钩子 */
    KeyRemoveKeyFunc pkeyRemoveKeyCb;              /* < 移除密钥对的钩子 */
} PKeyAdaptHandleFunc;

#ifdef __cplusplus
}
#endif

#endif // ADAPTOR_PKEY_REG_API_H
