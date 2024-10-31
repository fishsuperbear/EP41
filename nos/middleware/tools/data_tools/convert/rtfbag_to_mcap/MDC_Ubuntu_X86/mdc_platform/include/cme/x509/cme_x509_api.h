/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:X509对外定义
 * Create: 2020/8/24
 * History:
 * 2020/8/24 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_x509_cert CME_X509_CERT_API
 * @ingroup cme_x509
 */
#ifndef CME_X509_API_H
#define CME_X509_API_H

#include "cme_x509v3_extn_api.h"
#include "cme_x509_pkey_api.h"
#include "cme_errcode_api.h"
#include "cme_cid_api.h"
#include "bsl_time_api.h"
#include "cme_x509v3_extn_dn_api.h"
#include "keys/pkey_param_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 用于指定证书的有效期。
 */
typedef struct {
    X509ExtTime *notBefore; /* < 证书生效日期 */
    X509ExtTime *notAfter;  /* < 证书失效日期 */
} X509Validity;

/**
 * @brief 证书待签名信息
 * @par 描述：
 * 包含证书相关信息及签发证书的CA名称。\n
 * 包含subject和issuer名称, subject的公钥, 有效期, 版本号和序列号。
 * @attention 有些会包含可选独立的字段。
 */
typedef struct {
    Asn1Int *version;                        /* < 版本号, 固定为版本3(值为2). */
    PKeyBigInt serialNumber;                 /* < CA分配给证书的正整数序列号. */
    X509AlgIdentifier *signature;            /* < 签名算法ID. */
    X509ExtName *issuer;                     /* < 证书签发者. */
    X509Validity *validity;                  /* < 证书有效期. */
    X509ExtName *subject;                    /* < 证书主体. */
    X509SubjectPublicKeyInfo *subjectPubKey; /* < 公钥及其用途. */
    Asn1BitString issuerUID;                 /* < 签发者唯一码. */
    Asn1BitString subjectUID;                /* < 证书主题唯一码. */
    X509ExtList *extensions;                 /* < 证书扩展项. */
} X509CertInfo;

/**
 * @brief X509证书结构体。
 */
typedef struct {
    X509CertInfo *toBeSigned;     /* < 证书信息 */
    X509AlgIdentifier *algorithm; /* < 签名算法. */
    Asn1BitString signature;      /* < 签名. */
    int32_t ref;                  /* < 被引用次数 */
} X509Cert;

/**
 * @ingroup cme_x509_cert
 * @brief 证书版本号。
 */
typedef enum {
    X509_CERT_VERSION1 = 0,
    X509_CERT_VERSION2 = 1,
    X509_CERT_VERSION3 = 2,
} X509CertVersion;

/**
 * @ingroup cme_x509_cert
 * @brief 信息访问方式。
 */
typedef enum {
    X509_AUTHORITY_INFORMATION_ACCESS = 0, /* < 通过签发者访问信息 */
    X509_SUBJECT_INFO_ACCESS = 1,          /* < 自己创建subject信息 */
    X509_BUTT
} X509InfoAccessChoice;

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书版本号。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书版本号。
 */
Asn1Int X509_GetVersion(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书生效时间。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书生效时间。
 */
BslSysTime *X509_ExtractNotBefore(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书失效时间。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书失效时间。
 */
BslSysTime *X509_ExtractNotAfter(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书公钥。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书公钥。
 */
PKeyAsymmetricKey *X509_ExtractPublicKey(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书序列号。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书序列号。
 */
PKeyBigInt *X509_GetSN(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书签名算法。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书签名算法。
 */
X509AlgIdentifier *X509_GetSignAlgId(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书使用者公钥信息。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书使用者公钥信息。
 */
const X509SubjectPublicKeyInfo *X509_GetSubjectPublicKeyInfo(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书主体唯一识别码。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书主体唯一识别码。
 */
Asn1BitString *X509_GetSubjectUID(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书签发者唯一识别码。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书签发者唯一识别码。
 */
Asn1BitString *X509_GetIssuerUID(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书主体名称。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书主体名称。
 */
X509ExtName *X509_GetSubjectName(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书签发者名称。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书签发者名称。
 */
X509ExtName *X509_GetIssuerName(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief   根据CID获取证书扩展。
 * @param   cert [IN] 证书信息结构体。
 * @param   cid [IN] CID.
 * @retval  证书扩展
 */
X509Ext *X509_GetExtByCID(const X509Cert *cert, CmeCid cid);

/**
 * @ingroup cme_x509_cert
 * @brief   根据critical标识获取证书扩展列表。
 * @param   cert [IN] 证书信息结构体。
 * @param   cid [IN] critical标识。
 * @retval  证书扩展列表。
 */
X509ExtList *X509_GetExtByCriticalFlag(const X509Cert *cert, bool criticalFlag);

/**
 * @ingroup cme_x509_cert
 * @brief   获取证书扩展数量。
 * @param   cert [IN] 证书信息结构体。
 * @retval  证书扩展数量。
 */
int32_t X509_GetExtCount(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief 从给定的证书获取扩展列表。
 * @par 描述：
 * 返回一个指向给定证书中的扩展列表的指针。
 * @param cert [IN] 指向内存中Cert结构的指针。
 * @retval X509ExtList* 包含所有Cert扩展名的列表指针。
 * @retval NULL 如果输入为NULL或请求的数据不可用。
 * cme_x509_api.h.
 *
 * @par 内存操作：
 * cme不会为CmeList分配内存，因此不应该释放该内存。
 */
X509ExtList *X509_GetExtns(const X509Cert *cert);

/**
 * @brief 本结构体用于存放签发者的名称与序列号。
 */
typedef struct {
    X509ExtName *issuerName;       /* < 签发者名称 */
    PKeyBigInt issuerSerialNumber; /* < 序列号 */
} X509IssuerAndSerial;

/**
 * @ingroup cme_x509_cert
 * @brief   计算签发者的哈希值
 * @param   issuerAndSerial [IN] 签发者名称及序列号。
 * @param   hashAlgID [IN] 哈希算法。
 * @param   hashLen [OUT] 哈希值长度。
 * @retval  计算的哈希值。
 */
uint8_t *X509_IssuerAndSerialCalcHash(const X509IssuerAndSerial *issuerAndSerial, CryptoHashAlgorithm hashAlgID,
                                      size_t *hashLen);

/**
 * @ingroup cme_x509_cert
 * @brief 比较两个证书的签发者与SN号是否相等。
 * @param sourceCert [IN] 证书1。
 * @param destCert [IN] 证书2。
 * @retval CME_SUCCESS 两证书签发者与SN相等。
 *         CME_ERR_INVALID_ARG 入参错误。
 *         CME_ERR_ISSUER_MISMATCH 两证书签发者与SN不相等。
 */
bool X509_IssuerAndSerialCompare(const X509Cert *sourceCert, const X509Cert *destCert);

/**
 * @ingroup cme_x509_cert
 * @brief 释放签发者名称及序列号。
 * @param issuerAndSerial [IN] 签发者名称及序列号结构体。
 * @retval CME_SUCCESS 两SN相等。
 */
void X509_IssuerAndSerialFree(X509IssuerAndSerial *issuerAndSerial);

/**
 * @ingroup cme_x509_cert
 * @brief 检查证书是否为CA证书。
 * @param   cert [IN] 证书信息结构体。
 * @retval CME_SUCCESS 证书为CA证书。
 */
bool X509_IsCACert(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief 编码证书。
 * @param   cert [IN] 证书信息结构体。
 * @param   encodedLen [OUT]编码后的长度。
 * @retval  der编码的证书。
 */
uint8_t *X509_CertEncode(const X509Cert *cert, size_t *encodedLen);

/**
 * @ingroup cme_x509_cert
 * @brief 解码证书。
 * @param encodedCert [IN] DER编码证书。
 * @param encodedLength [IN] 编码长度。
 * @param decodedLength [OUT] 解码后的长度。
 * @retval 证书结构体。
 */
X509Cert *X509_CertDecode(const uint8_t *encodedCert, size_t encodedLength, size_t *decodedLength);

/**
 * @ingroup cme_x509_cert
 * @brief  生成证书指纹。
 * @param   cert [IN] 证书信息结构体。
 * @param   hashAlg [IN] 哈希算法。
 * @param   returnHashLength [OUT] 手印哈希值得长度。
 * @retval  证书指纹的哈希值。
 */
uint8_t *X509_CertGenFingerPrint(const X509Cert *cert, CryptoHashAlgorithm hashAlg, size_t *returnHashLength);

/**
 * @ingroup cme_x509_cert
 * @brief   检查证书签名。
 * @param   cert [IN] 证书信息结构体。
 * @param   publicKey [IN] 签发者公钥。
 * @retval  CME_SUCCESS 签名正确。
 * @retval  CME_ERR_INVALID_ARG 入参错误。
 * @retval  CME_ERR_ENCODE_FAILED 证书编码错误。
 */
int32_t X509_VerifySignature(const X509Cert *cert, PKeyAsymmetricKeyRoHandle publicKey);

/**
 * @ingroup cme_x509_cert
 * @brief 检查证书公钥与私钥是否匹配。
 * @param   cert [IN] 证书信息结构体。
 * @param   privateKey [IN] 私钥。
 * @retval  CME_SUCCESS 公钥与私钥匹配。
 * @retval  CME_ERR_INVALID_ARG 入参错误。
 * @retval  CME_ERR_KEYPAIR_MISMATCH 公钥与私钥不匹配。
 */
int32_t X509_CheckPrivateKey(const X509Cert *cert, PKeyAsymmetricKeyRoHandle privateKey);

/**
 * @ingroup cme_x509_cert
 * @brief 比较两个SN号是否相等。
 * @param srcSerialNum [IN] SN1.
 * @param trgSerialNum [IN] SN2.
 * @retval true 两SN相等。
 * @retval false 入参错误或两SN不相等。
 */
bool X509_SNCompare(const PKeyBigInt *srcSerialNum, const PKeyBigInt *trgSerialNum);

/**
 * @ingroup cme_x509_cert
 * @brief SN号解码。
 * @param encSN [IN] 编码后的SN。
 * @param encSNLen [IN] 编码长度。
 * @param retDecodedLen 解码后的长度。
 * @retval 解码后的SN。
 */
PKeyBigInt *X509_DecodeSN(const uint8_t *encSN, size_t encSNLen, size_t *retDecodedLen);

/**
 * @ingroup cme_x509_cert
 * @brief 证书深拷贝。
 * @param src [IN] 目标证书。
 * @retval 拷贝后的证书。
 */
X509Cert *X509_CertDump(const X509Cert *src);

/**
 * @ingroup cme_x509_cert
 * @brief 证书浅拷贝(引用)。
 * @param srcCert [IN] 目标证书。
 * @retval 拷贝后的证书。
 */
X509Cert *X509_CertRef(X509Cert *srcCert);

/**
 * @ingroup cme_x509_cert
 * @brief 释放证书。
 * @param cert [IN] 待释放的证书结构体。
 * @retval 无。
 */
void X509_CertFree(X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief 释放主体信息访问列表。
 * @param infoAccessList [IN] 列表首指针。
 * @retval 无。
 */
void X509_SubjInfoAccessFree(ListHandle infoAccessList);

/**
 * @ingroup cme_x509_cert
 * @brief 释放签发者信息访问列表。
 * @param infoAccessList [IN] 列表首指针。
 * @retval 无。
 */
void X509_AuthorityInfoAccessFree(ListHandle infoAccessList);

/**
 * @ingroup cme_x509_cert
 * @brief 深拷贝证书有效期结构。
 * @param src [IN] 目标结构体。
 * @retval 拷贝后的结构体。
 */
X509Validity *X509_ValidityDump(const X509Validity *src);

/**
 * @ingroup cme_x509_cert
 * @brief 释放证书有效期结构。
 * @param validity [IN] 目标结构体。
 * @retval 无。
 */
void X509_ValidityFree(X509Validity *validity);

/**
 * @ingroup cme_x509_cert
 * @brief X509 CRL Distribution Point List.
 * @par 描述：
 * SEQUENCE SIZE 1..MAX OF DistributionPoint
 */
typedef CmeList X509_CRLDistPointList;

/**
 * @ingroup cme_x509_cert
 * @brief 深拷贝CRL分发点列表。
 * @param src [IN] 列表首指针。
 * @retval 拷贝后的列表。
 */
X509_CRLDistPointList *X509_CRLDistPointDump(const X509_CRLDistPointList *src);

/**
 * @ingroup cme_x509_cert
 * @brief 释放CRL分发点列表。
 * @param crlDistPoint [IN] 列表首指针。
 * @retval 无。
 */
void X509_CRLDistPointFree(X509_CRLDistPointList *crlDistPoint);

/**
 * @ingroup cme_x509_cert
 * @brief 密钥用途。
 *
 */
#define X509_KU_ENCIPHER_ONLY 0x0001u /* < 仅用于加密 */
#define X509_KU_CRL_SIGN 0x0002u /* < CRL签名 */
#define X509_KU_KEY_CERT_SIGN 0x0004u /* < 证书签名 */
#define X509_KU_KEY_AGREEMENT 0x0008u /* < 密钥协商 */
#define X509_KU_DATA_ENCIPHERMENT 0x0010u /* < 数据加密 */
#define X509_KU_KEY_ENCIPHERMENT 0x0020u /* < 密钥加密 */
#define X509_KU_NON_REPUDIATION 0x0040u /* < 认可签名 */
#define X509_KU_DIGITAL_SIGNATURE 0x0080u /* < 数字签名 */
#define X509_KU_DECIPHER_ONLY 0x8000u /* < 仅用于解密 */

/**
 * @ingroup cme_x509_cert
 * @brief 获取证书的key used。
 * @param   cert [IN] 证书信息结构体。
 * @retval uint32_t X509_KU_ENCIPHER_ONLY|X509_KU_CRL_SIGN中的位运算。
 */
uint32_t X509_CertKeyUsageGet(const X509Cert *cert);

/**
 * @ingroup cme_x509_cert
 * @brief X509 Key Usage List.
 *
 */
typedef CmeList X509_KeyUsageList;

/**
 * @ingroup cme_x509_cert
 * @brief 密钥用途扩展（WWW 服务端身份认证）。
 */
#define X509_EXT_KU_SERVERAUTH 0x0001u /* < 密钥用途扩展（WWW 服务端身份认证） */

/**
 * @ingroup cme_x509_cert
 * @brief 密钥用途扩展（WWW 客户端身份认证）。
 */
#define X509_EXT_KU_CLIENTAUTH 0x0002u /* < 密钥用途扩展（WWW 客户端身份认证） */

/**
 * @ingroup cme_x509_cert
 * @brief 密钥用途扩展（可下载实现代码签名）。
 */
#define X509_EXT_KU_CODE_SINGING 0x0004u /* < 密钥用途扩展（可下载实现代码签名） */

/**
 * @ingroup cme_x509_cert
 * @brief 密钥用途扩展（E-mail保护）。
 */
#define X509_EXT_KU_EMAIL_PROTECTION 0x0008u /* < 密钥用途扩展（E-mail保护） */

/**
 * @ingroup cme_x509_cert
 * @brief 密钥用途扩展（时间戳）。
 */
#define X509_EXT_KU_TIME_STAMPING 0x0010u /* < 密钥用途扩展（时间戳） */

/**
 * @ingroup cme_x509_cert
 * @brief 密钥用途扩展（OCSP响应签名）。
 */
#define X509_EXT_KU_OCSP_SIGNING 0x0020u /* < 密钥用途扩展（OCSP响应签名） */

/**
 * @ingroup cme_x509_cert
 * @brief 密钥用途扩展（因特网密钥交换）。
 */
#define X509_EXT_KU_IPSECIKE 0x40u /* < 密钥用途扩展（因特网密钥交换） */

/**
 * @ingroup cme_x509_cert
 * @brief 密钥用途扩展（任意扩展密钥用法）。
 */
#define X509_EXT_KU_ANYEXTENDEDKEYUSAGE 0x100u /* < 密钥用途扩展（任意扩展密钥用法） */

/**
 * @ingroup cme_x509_cert
 * @brief 深拷贝密钥用途扩展列表。
 * @param src [IN] 列表首指针。
 * @retval 拷贝后的列表。
 */
X509_KeyUsageList *X509_ExtendedKeyUsageDump(const X509_KeyUsageList *src);

/**
 * @ingroup cme_x509_cert
 * @brief 获取有效的密钥用途。
 * @param extnKeyUsageList [IN] 密钥用途扩展列表。
 * @param keyUsage [OUT] 密钥用途。
 * @retval 错误码。
 */
int32_t X509_ExtendedKeyUsageGetEffective(const X509_KeyUsageList *extnKeyUsageList, uint32_t *keyUsage);

/* The structure that identifies the period within which the private key is valid */
typedef struct {
    /* Time before which the private key corresponding to the public key is invalid */
    Asn1GeneralizedTime notBefore; /* Optional */

    /* Time after which the private key corresponding to the public key is invalid */
    Asn1GeneralizedTime notAfter; /* Optional */
} X509ExtPriKeyValidity;

/**
 * @ingroup cme_x509_cert
 * @brief 获取私钥失效时间。
 * @param privkeyUsagePeriod [IN] X509ExtPriKeyValidity扩展项（私钥有效期）。
 * @retval 私钥失效时间
 */
BslSysTime *X509EXT_ExtractNotAfter(const X509ExtPriKeyValidity *privkeyUsagePeriod);
/**
 * @ingroup cme_x509_cert
 * @brief 获取私钥起效时间。
 * @param privkeyUsagePeriod [IN] X509ExtPriKeyValidity扩展项（私钥有效期）。
 * @retval 私钥起效时间
 */
BslSysTime *X509EXT_ExtractNotBefore(const X509ExtPriKeyValidity *privkeyUsagePeriod);
/**
 * @ingroup cme_x509_cert
 * @brief 释放X509ExtPriKeyValidity扩展项（私钥有效期）
 * @param priUsagePeriod [IN] X509ExtPriKeyValidity扩展项。。
 * @retval 无
 */
void X509EXT_FreePriKeyUsage(X509ExtPriKeyValidity *priUsagePeriod);
/**
 * @ingroup cme_x509_cert
 * @brief 创建KeyUsage扩展项（秘钥用途）
 * @param extData [IN] 秘钥用例数据（参考rfc5280中4/2/1/3章节）。
 * @param ext [OUT] X509Ext扩展项。
 * @retval 无
 */
int32_t X509EXT_CreateExtKeyUsage(const uint32_t *extData, X509Ext *ext);

/**
 * @ingroup cme_x509_cert
 * @brief 释放密钥用途扩展列表。
 * @param extKUList [IN] 列表首指针。
 * @retval 无
 */
void X509_ExtendedKeyUsageFree(X509_KeyUsageList *extKUList);
/**
 * @ingroup cme_x509_cert
 * @brief 用于描述访问信息。
 *
 */
typedef struct {
    Asn1Oid accessMethod;           /* < 访问方法 */
    X509ExtGenName *accessLocation; /* < 访问地址 */
} X509AccessDescription;

/**
 * @ingroup cme_x509_cert
 * @brief 释放访问描述相关结构。
 * @param accessDesc [IN] 目标结构体。
 * @retval 无
 */
void X509_AccessDescriptionFree(X509AccessDescription *accessDesc);

/**
 * @ingroup cme_x509_cert
 * @brief the element is X509 Cert.
 *
 */
typedef CmeList X509CertList; /* < the element is X509 Cert */

/**
 * @ingroup cme_x509_cert
 * @brief 往列表中添加证书。
 *
 * @param cert [IN] x509证书。
 * @param certList [IN] Certificates List.
 * @retval CME_SUCCESS 成功。
 * @retval CME_ERR_INVALID_ARG 无效参数。
 */
int32_t X509_CertListAdd(const X509Cert *cert, X509CertList *certList);

/**
 * @ingroup cme_x509_cert
 * @brief 根据名称删除列表中的证书。
 *
 * @param issuerName [IN] Issuer Name.
 * @param certList [IN] Certificates List.
 * @retval CME_SUCCESS 成功。
 * @retval CME_ERR_INVALID_ARG 无效参数。
 */
int32_t X509_CertListDelByIssuerName(const X509ExtName *issuerName, X509CertList *certList);

/**
 * @ingroup cme_x509_cert
 * @brief 根据SN号删除列表中的证书。
 *
 * @param sourceSerialNum [IN] SN Number.
 * @param certList [IN] Certificates List.
 * @retval CME_SUCCESS 成功。
 * @retval CME_ERR_INVALID_ARG 无效参数。
 */
int32_t X509_CertListDelBySN(const PKeyBigInt *sourceSerialNum, X509CertList *certList);

/**
 * @brief 根据证书删除列表中的证书
 */
int32_t X509_CertListDelByCert(const X509Cert *cert, X509CertList *certList);

/**
 * @ingroup cme_x509_cert
 * @brief 编码证书列表。
 *
 * @param certList [IN] Certificates List.
 * @param encodedLen [IN] Encode Len.
 * @retval uint8_t* 编码后的数据缓冲区
 * @retval NULL 编码失败。
 */
uint8_t *X509_CertListEncode(const X509CertList *certList, size_t *encodedLen);

/**
 * @ingroup cme_x509_cert
 * @brief 解码证书列表。
 *
 * @param encodedCertList [IN] 已编码的Certificates List。
 * @param encodedLength [IN] 已编码Certificates List长度。
 * @param decodedLength [OUT] 解码后Certificates List长度。
 * @retval X509CertList* 解码后Certificates List缓冲区。
 * @retval NULL 解码失败。
 */
X509CertList *X509_CertListDecode(const uint8_t *encodedCertList, size_t encodedLength, size_t *decodedLength);

/**
 * @ingroup cme_x509_cert
 * @brief 根据 issuer 名称寻找列表中的证书。
 *
 * @param certList [IN] Certificates List.
 * @param issuerName [IN] Issuer Name.
 * @retval X509Cert* 搜索到的X509证书。
 * @retval NULL 搜索失败。
 */
X509Cert *X509_CertListSearchByIssuerName(const X509CertList *certList, const X509ExtName *issuerName);

/**
 * @ingroup cme_x509_cert
 * @brief 根据 subject名称寻找列表中的证书。
 *
 * @param certList [IN] Certificates List.
 * @param subjectName [IN] subject Name.
 * @retval X509Cert* 搜索到的X509证书。
 * @retval NULL 搜索失败。
 */
X509Cert *X509_CertListSearchBySubjectName(const X509CertList *certList, const X509ExtName *subjectName);

/**
 * @ingroup cme_x509_cert
 * @brief 根据根据签发者及序列号寻找列表中的证书。
 *
 * @param certList [IN] Certificates List.
 * @param issuerName [IN] Issuer Name.
 * @param certSerialNum [IN] Certificates Serial Number.
 * @retval X509Cert* 搜索到的X509证书。
 * @retval NULL 搜索失败。
 */
X509Cert *X509_CertListSearchByIssuerAndSerial(const X509CertList *certList, const X509ExtName *issuerName,
                                               const PKeyBigInt *certSerialNum);

/**
 * @ingroup cme_x509_cert
 * @brief 根据issuer和subject寻找列表中的证书。
 *
 * @param certList [IN] Certificates List.
 * @param issuerName [IN] Issuer Name.
 * @param subjectName [IN] Subject Name.
 * @param currDatetime [IN] Current Datatime.
 * @param idx [IN] 要获取满足条件的第几个证书。
 * @retval X509Cert* 搜索到的X509证书。
 * @retval NULL 搜索失败。
 */
X509Cert *X509_CertListSearchByIssuerAndSubject(const X509CertList *certList, const X509ExtName *issuerName,
                                                const X509ExtName *subjectName, const BslSysTime *currDatetime,
                                                uint32_t idx);

/**
 * @ingroup cme_x509_cert
 * @brief 根据subject key ID 或 authority Key ID 寻找列表中的证书。
 *
 * @param certList Certificates List.
 * @param subjectKeyId [IN] Subject Key ID.
 * @param subjectKeyIdLen [IN] Subject Key ID Len.
 * @param authorityKeyId [IN] Authority Key ID.
 * @param authorityKeyIdLen [IN] Authority Key ID Len.
 * @retval X509Cert* 搜索到的X509证书。
 * @retval NULL 搜索失败。
 */
X509Cert *X509_CertListSearchByKeyId(const X509CertList *certList, const uint8_t *subjectKeyId, size_t subjectKeyIdLen,
                                     const uint8_t *authorityKeyId, size_t authorityKeyIdLen);

/**
 * @ingroup cme_x509_cert
 * @brief 释放证书列表。
 *
 * @param certList Certificates List.
 */
void X509_CertListFree(X509CertList *certList);

#ifdef __cplusplus
}
#endif

#endif
