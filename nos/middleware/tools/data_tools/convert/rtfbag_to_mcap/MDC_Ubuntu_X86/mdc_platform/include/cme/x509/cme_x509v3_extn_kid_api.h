/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:
 * Create: 2020/10/26
 * Notes:
 * History:
 * 2020/10/26 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_x509v3_extn_kid CME_X509V3_EXTN_KID_API
 * @ingroup cme_x509
 */
#ifndef CME_X509V3_EXTN_KID_API_H
#define CME_X509V3_EXTN_KID_API_H

#include "cme_x509v3_extn_dn_api.h"
#include "cme_asn1_api.h"
#include "keys/pkey_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509v3_extn_kid
 * @brief 用于指示密钥生成类型的枚举。
 */
typedef enum {
    X509_KID_SHA1_160 = 0, /* < 20 字节 hash */
    X509_KID_SHA1_60 = 1,  /* < 8 字节 hash */
    X509_KID_BUTT          /* < 最大 KIDGenType */
} X509ExtKidGenType;

/**
 * @ingroup cme_x509v3_extn_kid
 * @brief 授权密钥标识符扩展提供了一种方法来标识与用于签名CRL的私钥相对应的公钥。
 * 标识可以基于密钥标识符（CRL签名者证书中的主体密钥标识符），也可以基于颁发者名称和序列号。
 * 当一个颁发者有多个签名密钥时，这个扩展特别有用。
 * @attention 注意：CRL颁发者必须使用密钥标识符方法，并且应该出现在所有的CRL中。
 */
typedef struct {
    Asn1OctetString keyId;             /* < 公钥编码后的哈希 */
    X509ExtGenNameList *auzCertIssuer; /* < CRL颁发者的证书名称，如在CRL中用作扩展 */
    PKeyBigIntHandle auzCertSN;        /* < 证书颁发者序列号 */
} X509ExtAkid;

/**
 * @ingroup cme_x509v3_extn_kid
 * @brief   此函数根据公钥和生成密钥标识符的模式生成 Subject Key ID。模式可以是SHA1 160位或64位。
 * SKID的编码没有Tag和长度值。
 * @param   publicKey [IN] 用于编码的公钥。
 * @param   kidGenType [IN] SKID 生成类型的枚举值。
 * @param   enPubKeyIdBuff [IN] 公钥编码或公钥内容编码选项。
 * @retval  Asn1OctetString* 编码后的 SKID 指针。
 * @retval NULL 输入参数为NULL。
 */
Asn1OctetString *X509EXT_SkidCreate(PKeyAsymmetricKeyRoHandle publicKey, X509ExtKidGenType kidGenType);

/**
 * @ingroup cme_x509v3_extn_kid
 * @brief   复制一份 Authority Key ID
 * @param   src [IN] 源 AKID
 * @retval  指向复制后的AKID指针
 */
X509ExtAkid *X509EXT_AkidDup(const X509ExtAkid *src);
/**
 * @ingroup cme_x509v3_extn_kid
 * @brief   从 Authority Key ID 获取证书序列号。
 * @param   akid [IN] AKID.
 * @retval  PKeyBigIntHandle 指向序列号和序列号长度的指针。
 * @retval NULL 输入参数为NULL。
 */
PKeyBigIntHandle X509EXT_AkidGetSn(const X509ExtAkid *akid);

/**
 * @ingroup cme_x509v3_extn_kid
 * @brief   从 Authority Key ID 获取证书颁发者。
 * @param   akid [IN] AKID.
 * @retval  X509ExtGenNameList* 指向AlternateName列表的指针。
 * @retval NULL 输入参数为NULL。
 */
X509ExtGenNameList *X509EXT_AkidGetIssuer(const X509ExtAkid *akid);

/**
 * @ingroup cme_x509v3_extn_kid
 * @brief   从 Authority Key ID 获取 keyId。
 * @param   akid [IN] AKID.
 * @retval  Asn1OctetString* 指向AKID中的KeyIdentifier的指针。
 * @retval NULL 输入参数为NULL。
 */
Asn1OctetString *X509EXT_AkidGetKid(X509ExtAkid *akid);

/**
 * @ingroup cme_x509v3_extn_kid
 * @brief   释放 Authority Key ID 结构。
 * @param   akid [IN] 待释放的AKID。
 */
void X509EXT_AkidFree(X509ExtAkid *akid);

#ifdef __cplusplus
}
#endif

#endif // CME_X509V3_EXTN_KID_API_H
