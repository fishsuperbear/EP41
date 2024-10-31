/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: pkey的编解码接口
 * Create: 2020/10/26
 * Notes:
 * History:
 * 2020/10/26 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_x509_pkey CME_X509_PKEY_API
 * @ingroup cme_x509
 */
#ifndef CME_X509_PKEY_API_H
#define CME_X509_PKEY_API_H

#include "cme_asn1_api.h"
#include "cme_x509v3_extn_algid_api.h"
#include "crypto/crypto_sign_param_api.h"
#include "keys/pkey_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509_pkey
 * @brief X509对象公钥信息
 */
typedef struct {
    X509AlgIdentifier *algorithm;   /* < 算法类型 */
    Asn1BitString subjectPublicKey; /* < 编码过的公钥 */
} X509SubjectPublicKeyInfo;

/**
 * @ingroup cme_x509_pkey
 * @brief   编码公钥。
 * @param   key [IN] 待编码的公钥。
 * @param   bufLen [IN] 编码后Buff长度。
 * @retval  uint8_t* 编码结果。
 */
uint8_t *CME_PubKeyEncode(PKeyAsymmetricKeyRoHandle key, size_t *bufLen);

/**
 * @ingroup cme_x509_pkey
 * @brief   公钥解码。
 * @param   key [OUT] 解码后的公钥。
 * @param   buff [IN] 存储待解码的DER数据buff。
 * @param   bufLen [IN] buff长度。
 * @param   usedLen [IN] buff已使用长度。
 * @retval  int32_t 编码结果。
 */
int32_t CME_PubKeyDecode(PKeyAsymmetricKeyHandle key, const uint8_t *buff, size_t bufLen, size_t *usedLen);

/**
 * @ingroup cme_x509_pkey
 * @brief   将PKeyAsymmetricKey编码为PublicKeyInfo。
 * @param   key [IN] 待编码转换的公钥。
 * @param   encodedLen [IN] 编码长度。
 * @retval  uint8_t* 编码结果。
 */
uint8_t *CME_PubKeyInfoEncode(PKeyAsymmetricKeyRoHandle key, size_t *encodedLen);

/**
 * @ingroup cme_x509_pkey
 * @brief   将PublicKeyInfo解码为PKeyAsymmetricKey。
 * @param   encodedBuff [IN] 待解码的PublicKeyInfo数据。
 * @param   encodedLen  [IN] PublicKeyInfo数据长度。
 * @param   decodedLen  [OUT] 已解码的数据长度。
 * @retval  PKeyAsymmetricKey* 解码结果。
 */
PKeyAsymmetricKeyHandle CME_PubKeyInfoDecode(const uint8_t *encodedBuff, size_t encodedLen, size_t *decodedLen);

/**
 * @ingroup cme_x509_pkey
 * @brief   给定公钥创建SubjectPublicKeyInfo。
 * @param   publicKey [IN] 给定公钥。
 * @retval  X509SubjectPublicKeyInfo* 创建结果。
 */
X509SubjectPublicKeyInfo *X509_PubKeyInfoCreate(PKeyAsymmetricKeyRoHandle publicKey);

/**
 * @ingroup cme_x509_pkey
 * @brief   给定SubjectPublicKeyInfo解析出公钥。
 * @param   keyInfo [IN] 给定SubjectPublicKeyInfo。
 * @retval  PKeyAsymmetricKey* 解析结果。
 */
PKeyAsymmetricKeyHandle X509_PubKeyInfoParse(const X509SubjectPublicKeyInfo *keyInfo);

/**
 * @ingroup cme_x509_pkey
 * @brief   释放SubjectPublicKeyInfo。
 * @param   pubKey [IN] 待释放的SubjectPublicKeyInfo。
 */
void X509_PubKeyInfoFree(X509SubjectPublicKeyInfo *pubKey);

/**
 * @ingroup cme_x509_pkey
 * @brief   复制SubjectPublicKeyInfo。
 * @param   src [IN] 待复制的SubjectPublicKeyInfo。
 * @retval  X509SubjectPublicKeyInfo* 复制结果。
 */
X509SubjectPublicKeyInfo *X509_PubKeyInfoDump(const X509SubjectPublicKeyInfo *src);

/**
 * @ingroup cme_x509_pkey
 * @brief   编码SubjectPublicKeyInfo。
 * @param   pubKeyInfo [IN] 待编码的SubjectPublicKeyInfo。
 * @param   encodedLen [OUT] 编码长度。
 * @retval  uint8_t* 编码结果。
 */
uint8_t *X509_PubKeyInfoEncode(const X509SubjectPublicKeyInfo *pubKeyInfo, size_t *encodedLen);

/**
 * @ingroup cme_x509_pkey
 * @brief   解码生成SubjectPublicKeyInfo。
 * @param   buffer [IN] 待解码的buff数据。
 * @param   encodedLength [IN] buff长度。
 * @param   decodedLen [OUT] 已解码数据长度。
 * @retval  X509SubjectPublicKeyInfo* 解码结果。
 */
X509SubjectPublicKeyInfo *X509_PubKeyInfoDecode(const uint8_t *buffer, size_t encodedLength, size_t *decodedLen);

/**
 * @ingroup cme_x509_pkey
 * @brief   给定公钥生成哈希。
 * @param   pubKey [IN] 给定公钥。
 * @param   hashAlg [IN] 哈希算法。
 * @param   hashLength [OUT] 生成哈希长度。
 * @retval  uint8_t* 哈希结果
 */
uint8_t *X509_PubKeyInfoCalcHash(const X509SubjectPublicKeyInfo *pubKey, CryptoHashAlgorithm hashAlg,
                                 size_t *hashLength);

/**
 * @ingroup cme_x509_pkey。
 * @brief   从SubjectPublicKeyInfo结构体中获取算法ID。
 * @param   subPKInfo [IN] 待获取的SubjectPublicKeyInfo结构体。
 * @retval  X509AlgIdentifier* 算法ID结构体。
 */
X509AlgIdentifier *X509_PubKeyInfoGetAlgId(const X509SubjectPublicKeyInfo *subPKInfo);

/**
 * @ingroup cme_x509_pkey
 * @brief   解码生成私钥对象。
 * @param   encBuf [IN] 待解码的buff数据。
 * @param   encLen [IN] buff长度。
 * @param   decLen [OUT] 已解码数据长度。
 * @retval  PKeyAsymmetricKeyHandle 解码的私钥对象。
 */
PKeyAsymmetricKeyHandle X509_PrivateKeyDecode(const uint8_t *encBuf, size_t encLen, size_t *decLen);

#ifdef __cplusplus
}
#endif

#endif // CME_X509_PKEY_API_H
