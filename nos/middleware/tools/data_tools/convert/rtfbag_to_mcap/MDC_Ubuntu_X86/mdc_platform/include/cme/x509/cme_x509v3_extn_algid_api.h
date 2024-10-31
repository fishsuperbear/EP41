/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: Algorithm Identifier 模块相关接口
 * Create: 2020/10/26
 * Notes:
 * History:
 * 2020/10/26 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_x509v3_extn_algid CME_X509V3_EXTN_ALG_ID_API
 * @ingroup cme_x509
 */
#ifndef CME_X509V3_EXTN_ALGID_API_H
#define CME_X509V3_EXTN_ALGID_API_H

#include "cme_cid_api.h"
#include "cme_asn1_api.h"
#include "keys/pkey_api.h"
#include "crypto/crypto_sign_api.h"
#include "crypto/crypto_hash_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509v3_extn_algid
 * @brief x509算法标识符。
 */
typedef struct {
    Asn1Oid algorithm;           /* < 算法OID */
    Asn1AnyDefinedBy parameters; /* < 算法使用到的参数 */
} X509AlgIdentifier;

/**
 * @ingroup cme_x509v3_extn_algid
 * @brief   获取指定CmeAlgIdentifier的CID。
 * @param   algorithmIdent [IN] 待获取CID的CmeAlgIdentifier结构体。
 * @retval  CmeCid CID.
 */
CmeCid X509_AlgIdGetCID(const X509AlgIdentifier *algorithmIdent);

/**
 * @ingroup cme_x509v3_extn_algid
 * @brief   获取指定CmeAlgIdentifier的签名算法。
 * @param   algorithmIdent [IN] 待获取的CmeAlgIdentifier结构体。
 * @retval  CryptoSignAlgorithm.
 */
CryptoSignAlgorithm X509_AlgIdGetSignAlg(const X509AlgIdentifier *algorithmIdent);

/**
 * @ingroup cme_x509v3_extn_algid
 * @brief   获取指定CmeAlgIdentifier的hash算法。
 * @param   algorithmIdent [IN] 待获取的CmeAlgIdentifier结构体。
 * @retval  CryptoHashAlgorithm.
 */
CryptoHashAlgorithm X509_AlgIdGetHashAlg(const X509AlgIdentifier *algorithmIdent);

/**
 * @ingroup cme_x509v3_extn_algid
 * @brief   获取指定CmeAlgIdentifier的参数结构体。
 * @param   algorithmIdent [IN] 待获取参数的CmeAlgIdentifier结构体。
 * @retval  void* 参数结构体。
 */
void *X509_AlgIdGetParam(const X509AlgIdentifier *algorithmIdent);

/**
 * @ingroup cme_x509v3_extn_algid
 * @brief   获取EcDsaKey的EC参数。
 * @param   algorithmIdent [IN] 输入key。
 * @param   ecKey [OUT] EC参数内容。
 */
int32_t X509_AlgIdGetEcParam(const X509AlgIdentifier *algorithmIdent, PKeyEcDsaKeyHandle ecKey);

/**
 * @ingroup cme_x509v3_extn_algid
 * @brief   复制CmeAlgIdentifier结构体。
 * @param   algorithmIdent [IN] 待复制的结构体。
 * @retval  X509AlgIdentifier* 复制成功指向的结构体指针。
 * @retval  NULL 复制失败。
 */
X509AlgIdentifier *X509_AlgIdDump(const X509AlgIdentifier *algorithmIdent);

/**
 * @ingroup cme_x509v3_extn_algid
 * @brief   释放CmeAlgIdentifier结构体。
 * @param   algId [IN] 指向待释放结构体的指针。
 */
void X509_AlgIdFree(X509AlgIdentifier *algId);

#ifdef __cplusplus
}
#endif

#endif // CME_X509V3_EXTN_ALGID_API_H
