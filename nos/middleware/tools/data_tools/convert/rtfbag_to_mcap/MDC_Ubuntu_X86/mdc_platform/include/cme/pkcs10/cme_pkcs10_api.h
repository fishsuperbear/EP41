/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:
 * Create: 2020/9/14
 * Notes:
 * History:
 * 2020/9/14 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_pkcs10 CME_PKCS10_API
 * @ingroup cme_x509
 */
#ifndef CME_PKCS10_API_H
#define CME_PKCS10_API_H

#include "crypto/crypto_sign_api.h"
#include "x509/cme_x509_api.h"
#include "x509/cme_x509v3_extn_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_pkcs10
 * @brief PKCS10版本号。
 */
#define PKCS10_VERSION 0

/**
 * @brief PKCS10请求内容结构体。
 */
typedef struct {
    uint32_t ver;                     /* < 版本信息 */
    X509ExtName *subject;             /* < 证书主体 */
    X509SubjectPublicKeyInfo *pubKey; /* < subject公钥信息 */
    ListHandle attrList;              /* < 属性列表 */
} PKCS10ReqInfo;

/**
 * @brief PKCS10请求（含签名信息）结构体。
 */
typedef struct {
    PKCS10ReqInfo *certReqInfo; /* < 证书请求信息 */
    X509AlgIdentifier *sigAlg;  /* < 签名算法 */
    Asn1BitString signature;    /* < 签名内容 */
} PKCS10Req;

/**
 * @ingroup cme_pkcs10
 * @brief 创建证书请求。
 *
 * @param ver [IN] 版本。
 * @param subject [IN] 证书subject name。
 * @param pubKey [IN] 公钥。
 *
 * @retval PKCS10Req* 成功，返回PKCS10请求信息。
 * @retval NULL 失败。
 */
PKCS10Req *PKCS10_CertReqCreate(uint32_t ver, const X509ExtName *subject, PKeyAsymmetricKeyRoHandle pubKey);

/**
 * @ingroup cme_pkcs10
 * @brief 对单个证书请求进行签名。
 *
 * @param certReq [IN] PKCS10请求信息。
 * @param privateKey [IN] 私钥。
 * @param hashAlgId [IN] hash算法类型。
 *
 * @retval CME_SUCCESS 成功。
 * @retval CME_ERR_INVALID_ARG 无效参数。
 */
int32_t PKCS10_CertReqSign(PKCS10Req *certReq, PKeyAsymmetricKeyRoHandle privateKey, CryptoHashAlgorithm hashAlgId);

/**
 * @ingroup cme_pkcs10
 * @brief 使用指定的签名算法对证书请求进行签名。
 *
 * @param certReq [IN] PKCS10请求信息。
 * @param privateKey [IN] 私钥。
 * @param signAlg [IN] 签名算法类型。
 * @param hashAlg [IN] hash算法类型。
 *
 * @retval CME_SUCCESS 成功。
 * @retval CME_ERR_INVALID_ALGID 无效算法类型。
 */
int32_t PKCS10_CertReqSignWithSpecifiedAlg(PKCS10Req *certReq, PKeyAsymmetricKeyRoHandle privateKey,
                                           CryptoSignAlgorithm signAlg, CryptoHashAlgorithm hashAlg);

/**
 * @ingroup cme_pkcs10
 * @brief 该函数验证PKCS10Req结构中的签名。
 *
 * @param certReq [IN] 待验证的PKCS10请求信息。
 *
 * @retval CME_SUCCESS 成功。
 * @retval CME_ERR_INVALID_ARG 无效参数。
 */
int32_t PKCS10_CertReqVerify(const PKCS10Req *certReq);

/**
 * @ingroup cme_pkcs10
 * @brief 获取PKCS10版本。
 *
 * @param certReq [IN] PKCS10请求信息。
 * @retval uint32_t PKCS10版本号
 */
uint32_t PKCS10_CertReqGetVersion(const PKCS10Req *certReq);

/**
 * @ingroup cme_pkcs10
 * @brief   获取CSR签名算法标识符
 *
 * @param   crl [IN] 指定CSR结构.
 *
 * @return  X509AlgIdentifier* CRL签名算法
 * @return  NULL 如果CSR结构或待签证书列表无效
 */
X509AlgIdentifier *PKCS10_CertReqGetSignAlgId(const PKCS10Req *certReq);

/**
 * @ingroup cme_pkcs10
 * @brief PKCS10获取签名信息。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 * @retval Asn1BitString* 签名信息。
 */
const Asn1BitString *PKCS10_CertReqGetSignature(const PKCS10Req *certReq);

/**
 * @ingroup cme_pkcs10
 * @brief PKCS10获取Subject Public Key信息。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 * @retval X509SubjectPublicKeyInfo* Subject Public Key信息。
 */
const X509SubjectPublicKeyInfo *PKCS10_CertReqGetSubjectPublicKeyInfo(const PKCS10Req *certReq);

/**
 * @ingroup cme_pkcs10
 * @brief 获取CSR公钥。
 * @param certReq [IN] CSR结构体。
 * @retval CSR公钥。
 */
PKeyAsymmetricKey *PKCS10_CertReqExtractPublicKey(const PKCS10Req *certReq);

/**
 * @ingroup cme_pkcs10
 * @brief 获取PKCS10 Subject Name。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 * @retval X509ExtName* PKCS10 Subject Name.
 */
const X509ExtName *PKCS10_CertReqGetSubjectName(const PKCS10Req *certReq);

/**
 * @ingroup cme_pkcs10
 * @brief 往证书请求添加扩展。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 * @param extn [IN] 扩展信息。
 * @retval CME_SUCCESS 成功。
 * @retval CME_ERR_LIST_OPERATION_FAILED 无效算法类型。
 */
int32_t PKCS10_CertReqAddExtn(const PKCS10Req *certReq, const X509Ext *extn);

/**
 * @ingroup cme_pkcs10
 * @brief 获取证书请求扩展列表。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 * @retval CmeList 证书请求扩展列表。
 * @retval NULL 没有扩展。
 */
CmeList *PKCS10_CertReqGetExtList(const PKCS10Req *certReq);

/**
 * @ingroup cme_pkcs10
 * @brief 根据CID获取证书请求。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 * @param cid [IN] 欲获取的CID。
 * @retval X509Ext* 证书请求扩展。
 * @retval NULL 没有找到扩展。
 */
X509Ext *PKCS10_CertReqGetExtByCID(const PKCS10Req *certReq, CmeCid cid);

/**
 * @ingroup cme_pkcs10
 * @brief 往证书请求增加挑战密码。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 * @param pwd [IN] 挑战密码。
 * @param pwdLen [IN] 挑战密码长度。
 * @retval CME_SUCCESS 成功。
 * @retval CME_ERR_INVALID_ARG 无效参数。
 */
int32_t PKCS10_CertReqAddChallengePwd(const PKCS10Req *certReq, uint8_t *pwd, size_t pwdLen);

/**
 * @ingroup cme_pkcs10
 * @brief 从证书请求中获取挑战密码。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 * @param pwdLen[OUT] 挑战密码长度。
 * @retval uint8_t* 成功，挑战密码缓冲区。
 * @retval NULL 获取失败。
 */
uint8_t *PKCS10_CertReqGetChallengePwd(const PKCS10Req *certReq, size_t *pwdLen);

/**
 * @ingroup cme_pkcs10
 * @brief 编码证书请求。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 * @param certReqLen [IN] 请求信息长度。
 * @retval uint8_t* 成功，编码结果缓存区。
 * @retval NULL 失败。
 */
uint8_t *PKCS10_CertReqEncode(const PKCS10Req *certReq, size_t *certReqLen);

/**
 * @ingroup cme_pkcs10
 * @brief 解码证书请求。
 *
 * @param derData [IN] csr 的DER编码数据。
 * @param len [IN] csr数据长度。
 * @param decodedLen [OUT] csr解码的长度。
 * @retval PKCS10Req* 成功，CSR上下文结构体。
 * @retval NULL 失败。
 */
PKCS10Req *PKCS10_CertReqDecode(const uint8_t *derData, size_t len, size_t *decodedLen);

/**
 * @ingroup cme_pkcs10
 * @brief 释放证书请求结构体。
 *
 * @param certReq [IN] PKCS10请求信息结构体。
 */
void PKCS10_CertReqFree(PKCS10Req *certReq);

#ifdef __cplusplus
}
#endif

#endif /* CME_PKCS10_API_H */
