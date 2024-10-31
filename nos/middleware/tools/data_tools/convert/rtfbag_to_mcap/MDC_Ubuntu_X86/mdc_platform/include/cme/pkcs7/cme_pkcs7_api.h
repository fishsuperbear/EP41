/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Create: 2022/07/01
 * Notes:
 * History:
 * 2022/07/01 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_pkcs7 CME_PKCS7_API
 * @ingroup cme_x509
 */
#ifndef CME_PKCS7_API_H
#define CME_PKCS7_API_H

#include "asn1_types_api.h"
#include "crypto/crypto_sign_api.h"
#include "x509/cme_x509_api.h"
#include "x509/cme_x509v3_extn_dn_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GMT 0009 SM2 加密数据格式
 */
typedef struct {
    PKeyBigInt xCoordinate; /* x 分量 */
    PKeyBigInt yCoordinate; /* y 分量 */
    Asn1OctetString hash;   /* 杂凑值 */
    Asn1OctetString cipherText; /* 密文 */
} P7Sm2Cipher;

typedef Asn1Int P7CMSVersion;

typedef enum {
    P7_RECIPIENT_ID_ISSUER_AND_SN, /* < issuerAndSerialNumber */
    P7_RECIPIENT_ID_SK_ID, /* < subjectKeyIdentifier */
} P7RecipientIdentifierChoiceId;

typedef struct {
    X509ExtName issuer;
    PKeyBigInt serialNumber; /* < Certificate Serial Number */
} P7IssuerAndSerialNumber;

typedef struct {
    P7RecipientIdentifierChoiceId choiceId;
    union {
        P7IssuerAndSerialNumber issuerAndSerialNumber; // 颁发者可辨别名和颁发序列号，GMT 0010固定是这个
        // 不支持 rfc subjectKeyIdentifier
    };
} P7RecipientIdentifier;

typedef X509AlgIdentifier P7KeyEncryptionAlgorithmIdentifier;

typedef Asn1OctetString P7EncryptedKey;

typedef struct {
    P7CMSVersion version; // rfc 限制为 0 或 2；GMT0010没有限制，因此解析没有限制，需要用户检查
    P7RecipientIdentifier rid; // 接收者标识
    P7KeyEncryptionAlgorithmIdentifier keyEncryptionAlgorithm; // 算法 oid和参数
    P7EncryptedKey encryptedKey; // 数据加密密钥密文，GMT0010对应 SM2cipher
} P7KeyTransRecipientInfo;

typedef enum {
    P7_RECIPIENT_KTRI, /* < key trans */
    P7_RECIPIENT_KARI, /* < key agree */
    P7_RECIPIENT_KEKRI, /* < key */
    P7_RECIPIENT_PWRI, /* < password */
    P7_RECIPIENT_ORI, /* < other */
} P7RecipientChoiceId;

typedef struct {
    P7RecipientChoiceId choiceId;
    union {
        P7KeyTransRecipientInfo ktri; // 密钥传输，GMT 0010固定是这个
        // 不支持 rfc KeyAgreeRecipientInfo kari;
        // 不支持 rfc KEKRecipientInfo kekri;
        // 不支持 rfc PasswordRecipientInfo pwri;
        // 不支持 rfc OtherRecipientInfo ori;
    };
} P7RecipientInfo;

/**
 * RecipientInfos ::= SET SIZE (1..MAX) OF RecipientInfo
 */
typedef CmeList P7SetOfRecipientInfos;

typedef Asn1Oid P7ContentType;

typedef X509AlgIdentifier P7ContentEncryptionAlgorithmIdentifier;

typedef Asn1OctetString P7EncryptedContent;

typedef struct {
    P7ContentType contentType; // 内容类型oid, 例如 PKCS7 数据 1.2.840.113549.1.7.1
    P7ContentEncryptionAlgorithmIdentifier contentEncryptionAlgorithm; // 内容加密算法（oid和对应参数）
    P7EncryptedContent encryptedContent; // 加密后内容，[0] IMPLICIT  OPTIONAL
    Asn1OctetString sharedInfo1; // 协商好的共享消息，[1] IMPLICIT  OPTIONAL，GMT 0010有，rfc 没有
    Asn1OctetString sharedInfo2; // 协商好的共享消息，[2] IMPLICIT  OPTIONAL，GMT 0010有，rfc 没有
} P7EncryptedContentInfo;

typedef struct {
    P7CMSVersion version; // 版本号
    // 不支持 rfc 可选字段 originatorInfo，GMT 0010没有这个
    P7SetOfRecipientInfos recipientInfos; // 接收者信息集合
    P7EncryptedContentInfo encryptedContentInfo; // 加密后的内容信息
    // 不支持 rfc 可选字段 unprotectedAttrs，GMT 0010没有这个
} P7EnvelopedData;

typedef struct {
    P7ContentType contentType;   /* < the type of the associated */
    Asn1AnyDefinedBy parameters; /* < contentType 对应参数，当前只支持 EnvelopedData */
} P7ContentInfo;

/**
 * @ingroup cme_pkcs7
 * @brief 解码加密密钥
 * @param encryptedKey [IN] 从 ContentInfo 中获取的 encryptedKey
 * @param cipher [OUT] sm2 加密数据格式
 * @retval CME_SUCCESS 成功
 * @retval 错误码
 */
int32_t PKCS7_EncryptedKeyDecode(P7EncryptedKey *encryptedKey, P7Sm2Cipher **cipher);

/**
 * @ingroup cme_pkcs7
 * @brief 释放 sm2 加密数据格式
 *
 * @param cipher [IN] sm2 加密数据格式
 */
void PKCS7_Sm2CipherFree(P7Sm2Cipher *cipher);

/**
 * @ingroup cme_pkcs7
 * @brief 释放数字信封结构体。
 *
 * @param certReq [IN] PKCS7请求信息结构体。
 */
void PKCS7_ContentInfoFree(P7ContentInfo *envelopedData);

#ifdef __cplusplus
}
#endif

#endif /* CME_PKCS7_API_H */
