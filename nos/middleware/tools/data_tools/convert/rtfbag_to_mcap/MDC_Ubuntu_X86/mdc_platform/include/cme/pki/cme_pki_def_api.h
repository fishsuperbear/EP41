/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:
 * Create: 2020/11/10
 * History:
 * 2020/8/15 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509_pki CME_X509_PKI_API
 *  @ingroup cme
 */
/** @defgroup pki_def PKI_DEF_API
 * @ingroup cme_x509_pki
 */
#ifndef CME_PKI_DEF_API_H
#define CME_PKI_DEF_API_H

#include "cme_asn1_api.h"
#include "cme_cid_api.h"
#include "x509/cme_x509v3_extn_api.h"
#include "x509/cme_x509_api.h"
#include "x509/cme_x509_crl_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CME_PKI_SUCCESS 0
#define CME_PKI_ERROR (-1)
#define CME_PKI_CB_SUCCESS 1
#define CME_PKI_CB_ERROR 0
#define CME_PKI_NAME_MAXLEN 256u
#define CME_PKI_DIGEST_LEN 32u
#define CME_PKI_DIGEST_ALG CRYPTO_HASH_SHA_256

/**
 * @ingroup pki_def
 * @brief 本节包含了PKI编码格式的所有枚举。
 */
typedef enum {
    CME_PKI_ENCODE_FORM_PEM = 1,  /* < PEM格式标志 */
    CME_PKI_ENCODE_FORM_ASN1 = 2, /* < ASN格式标志 */
} CmePkiEncodeForm;

/**
 * @ingroup pki_def
 * @brief PKI Cert Request.
 */
#define CME_PKI_CERT_REQUEST 0x01u

/**
 * @ingroup pki_def
 * @brief PKI Trust CA.
 */
#define CME_PKI_CERT_TRUST_CA 0x02u

/**
 * @ingroup pki_def
 * @brief PKI Trust Cross CA.
 */
#define CME_PKI_CERT_TRUST_CROSS_CA 0x04u

/**
 * @ingroup pki_def
 * @brief PKI Default Cert.
 */
#define CME_PKI_CERT_DEFAULT_CERT 0x08u

/**
 * @ingroup pki_def
 * @brief PKI Local Cert.
 */
#define CME_PKI_CERT_LOCAL_CERT 0x10u

/**
 * @ingroup pki_def
 * @brief 是否为Cert Request类型。
 */
#define CME_PKI_CERT_IS_REQUEST(x_) (((x_) & CME_PKI_CERT_REQUEST) == CME_PKI_CERT_REQUEST)

/**
 * @ingroup pki_def
 * @brief 是否为Trust CA类型。
 */
#define CME_PKI_CERT_IS_TRUST_CA(x_) (((x_) & CME_PKI_CERT_TRUST_CA) == CME_PKI_CERT_TRUST_CA)

/**
 * @ingroup pki_def
 * @brief 是否为Trust Cross CA类型。
 */
#define CME_PKI_CERT_IS_TRUST_CROSS_CA(x_) (((x_) & CME_PKI_CERT_TRUST_CROSS_CA) == CME_PKI_CERT_TRUST_CROSS_CA)

/**
 * @ingroup pki_def
 * @brief 是否为Default Cert类型。
 */
#define CME_PKI_CERT_IS_DEFAULT(x_) (((x_) & CME_PKI_CERT_DEFAULT_CERT) == CME_PKI_CERT_DEFAULT_CERT)

/**
 * @ingroup pki_def
 * @brief 是否为Local Cert类型。
 */
#define CME_PKI_CERT_IS_LOCAL(x_) (((x_) & CME_PKI_CERT_LOCAL_CERT) == CME_PKI_CERT_LOCAL_CERT)

/**
 * @ingroup pki_def
 * @brief PKI文件信息结构体。
 */
typedef struct {
    bool isFile;                 /* < 是否为File */
    uint8_t *buf;                /* < 信息缓冲区 */
    size_t bufLen;               /* < 信息缓冲区长度 */
    uint8_t *passwd;             /* < 密钥缓冲区 */
    size_t passwdLen;            /* < 密钥长度 */
    CmePkiEncodeForm encodeForm; /* < 编码格式 */
} CmePkiFileInfo;

/**
 * @ingroup pki_def
 * @brief 在verify参数中检查时间有效性，如果该设置该flag，则必须同时设置当前时间，如果未设置，则获取系统时间。
 */
#define CME_PKI_CHECK_TIME 0x00000001u

/**
 * @ingroup pki_def
 * @brief 仅对终端实体证书启用CRL验证。
 */
#define CME_PKI_CHECK_CRL 0x00000002u

/**
 * @ingroup pki_def
 * @brief 使能整个链的CRL验证功能
 */
#define CME_PKI_CHECK_CRL_ALL 0x00000004u

/**
 * @ingroup pki_def
 * @brief 使能增量CRL验证功能。
 */
#define CME_PKI_CHECK_DELTA_CRL 0x00000008u

/**
 * @ingroup pki_def
 * @brief 启用交叉CA验证与CRL。
 */
#define CME_PKI_CHECK_CRL_CROSS_CA 0x00000010u

/**
 * @ingroup pki_def
 * @brief 启用keyUsage扩展校验。
 */
#define CME_PKI_X509_CHECK_CERT_KEYUSAGE 0x00000020u

/**
 * @ingroup pki_def
 * @brief 忽略关键扩展。
 */
#define CME_PKI_X509_V_FLAG_IGNORE_CRITICAL 0x00000040u

/**
 * @ingroup pki_def
 * @brief 启用跨CA验证。
 */
#define CME_PKI_CHECK_CROSS_CA 0x00000080u

/**
 * @ingroup pki_def
 * @brief 使能间接CRL验证功能。
 */
#define CME_PKI_EXTENDED_CRL_SUPPORT 0x00000100u

/**
 * @ingroup pki_def
 * @brief 支持对象带内CRL
 */
#define CME_PKI_OBJ_CRL_SUPPORT 0x00000200u

/**
 * @ingroup pki_def
 * @brief 启用所有OCSP验证。
 */
#define CME_PKI_CHECK_OCSP_ALL 0x00000400u

/**
 * @ingroup pki_def
 * @brief 启用OCSP响应程序证书CRL验证
 */
#define CME_PKI_OCSP_RESPONDER_CHECK_CRL 0x00000800u

/**
 * @ingroup pki_def
 * @brief 启用OCSP响应程序证书Delta CRL验证
 */
#define CME_PKI_OCSP_RESPONDER_CHECK_DELTA_CRL 0x00001000u

/**
 * @ingroup pki_def
 * @brief 启用使用响应消息中可用的OCSP响应器证书
 */
#define CME_PKI_OCSP_TRUST_RESPONDER_CERTS_IN_MSG 0x00002000u

/**
 * @ingroup pki_def
 * @brief 支持对象带内OCSP。
 */
#define CME_PKI_OBJ_OCSP_SUPPORT 0x00004000u

/**
 * @ingroup pki_def
 * @brief 启用OCSP验证。
 */
#define CME_PKI_CHECK_OCSP 0x00008000u

/**
 * @ingroup pki_def
 * @brief 指示用于生成证书请求的信息类型。
 * @par 描述：
 * @li CME_PKI_REQ_CA_HASH，如果选中此标志，生成的证书请求将包含CA_public_keyHash。
 * @li CME_PKI_REQ_CA_SUBJECT，如果选中此标志，生成的证书请求将包含CA_public_keySubject。
 * @li CME_PKI_REQ_CA_CERT，如果选中此标志，生成的证书请求将包含CA_public_keyCert。
 */
typedef enum {
    CME_PKI_REQ_CA_HASH = 1, /* < 请求类型为CA公钥的SHA1哈希 */
    CME_PKI_REQ_CA_SUBJECT,  /* < 请求类型是CA的编码主题名称 */
    CME_PKI_REQ_CA_CERT      /* < 请求类型是CA的编码证书 */
} CmePkiReqParam;

/**
 * @ingroup pki_def
 * @brief 主题备选名称的类型。
 *
 * @par 描述：
 * @li CME_PKI_ALT_NAME_DNS，DNS名称作为主题备用名称。
 * @li CME_PKI_ALT_NAME_IPADDRESS，IP地址作为主题备用名称。
 * @li CME_PKI_ALT_NAME_EMAILID，RFC822名称作为主题备用名称。
 */
typedef enum {
    CME_PKI_ALT_NAME_DNS = GENNAME_DNSNAME,       /* < ID type FQDN */
    CME_PKI_ALT_NAME_IPADDRESS = GENNAME_IPADDR,  /* < ID type IPV4 address */
    CME_PKI_ALT_NAME_EMAILID = GENNAME_RFC822NAME /* < ID type  e-mail id */
} CmePkiAltName;

/**
 * @ingroup pki_def
 * @brief 指示从上下文到对象需要考虑的参数。
 */
typedef enum {
    CME_PKI_SYNC_TYPE_DFLT_LOCAL_CERT = 0x1,      /* <  要从上下文中获取的默认本地证书 */
    CME_PKI_SYNC_TYPE_VERIFY_PARAM = 0x2,         /* < 验证从上下文获取的参数 */
    CME_PKI_SYNC_TYPE_ALL_LOCAL_CERT = 0x4,       /* < 从上下文中获取的默认证书和本地证书 */
    CME_PKI_SYNC_TYPE_DFLT_PRE_SHARED_CERT = 0x8, /* < 要从上下文中获取的默认预共享证书 */
    CME_PKI_SYNC_TYPE_ALL_PRE_SHARED_CERT = 0x10  /* < 缺省证书和所有其他预共享证书从上下文获取 */
} CmePkiSyncType;

/**
 * @ingroup pki_def
 * @brief PKI选项标志。
 */
typedef enum {
    CME_PKI_OPTION_IGNORE_PVT_KEY = 0x00000001, /* < 只加载本地证书，忽略私钥。 */
} CmePkiOptions;

/**
 * @ingroup pki_def
 * @brief 此结构保存公共配置，是用于基于PKI的身份验证的公共存储库。
 */
typedef struct CmePkiCtx_ CmePkiCtx;

/**
 * @ingroup pki_def
 * @brief 这个结构保存了会话配置，并且是用于基于PKI的身份验证的附加存储库。
 */
typedef struct CmePkiObj_ CmePkiObj;

/**
 * @ingroup pki_def
 * @brief 此结构包含证书扩展的长度字段。
 * @attention 证书链验证必须从Basic约束扩展中获取的路径长度。
 */
typedef struct {
    Asn1Int pathLen; /* < 证书扩展的长度 */
} CmePkiX509CertExtLen;

/**
 * @ingroup pki_def
 * @brief 此结构包含证书扩展的标志字段。
 * @par 描述：
 * 用于存储标志，以指示证书是否为CA证书，以及用于证书链验证的其他标志值
 */
typedef struct {
    uint32_t extFlags;    /* < 链式验证 */
    uint32_t extKeyUsage; /* < 存储扩展密钥使用详细信息 */
    uint32_t keyUsage;    /* < 存储证书的密钥用法 */
} CmePkiX509CertExtFlag;

/**
 * @ingroup pki_def
 * @brief 此结构包含证书扩展的key id字段。
 */
typedef struct {
    Asn1OctetString *SKID; /* < 包含公钥哈希的主体密钥标识符扩展 */
    X509ExtAkid *AKID; /* < 授权密钥标识符扩展，包含颁发者公钥的散列以及证书颁发者的名称和颁发者的序列号 */
} CmePkiX509CertExtKeyid;

/**
 * @ingroup pki_def
 * @brief 此结构包含证书扩展中的加密相关字段。
 */
typedef struct {
    uint8_t certDigest[CME_PKI_DIGEST_LEN]; /* < 证书摘要 */
    PKeyAsymmetricKey *publicKey;           /* < 它用于保存证书的公钥 */
} CmePkiX509CertExtCrypt;

/**
 * @ingroup pki_def
 * @brief 此结构包含证书扩展的所有字段。
 */
typedef struct {
    CmePkiX509CertExtLen extLengths; /* < 此结构包含证书扩展的长度字段 */
    CmePkiX509CertExtFlag extFlags;  /* < 此结构包含证书扩展的标志字段 */
    CmePkiX509CertExtKeyid extKeyId; /* < 此结构包含证书扩展的key id字段 */
    CmePkiX509CertExtCrypt extCrypt; /* < 此结构包含证书扩展中的加密相关字段 */
} CmePkiX509CertExt;

/**
 * @ingroup pki_def
 * @brief 结构保存证书以及一些扩展字段。
 */
typedef struct {
    X509Cert *cert;                  /* < 必须从扩展字段中提取以下证书链信息的证书 */
    bool valid;                      /* < 指示证书是否有效的标志 */
    CmePkiX509CertExt extCertFields; /* < 包含证书扩展的结构 */
    int32_t ref;                     /* < 值指示引用此结构的对象数 */
} CmePkiX509Cert;

/**
 * @ingroup pki_def
 * @brief Issuing Distribution Point.
 */
typedef struct {
    X509CRLIssueDP *idp; /* < CRL IDP结构指针 */
    X509ExtAkid *akid;   /* < 授权密钥标识符扩展 */
    int32_t idpFlags;    /* < IDP标志 */
    int32_t idpReasons;  /* < IDP原因 */
} CmePkiCrlIdpAkidInfo;

/**
 * @ingroup pki_def
 * @brief Crl Number info.
 */
typedef struct {
    PKeyBigInt *crlNumber;     /* < CRL Number */
    PKeyBigInt *baseCrlNumber; /* < Base CRL Number */
} CmePkiCrlNumberInfo;

/**
 * @ingroup pki_def
 * @brief 此结构包含证书扩展的所有字段。
 */
typedef struct {
    uint32_t uiFlags;                       /* < Crl Number info */
    CmePkiCrlNumberInfo crlNumInfo;         /* < Crl Number info */
    CmePkiCrlIdpAkidInfo pkiCrlIdpAkidInfo; /* < Issuing Distribution Point */
    uint8_t crlHash[CME_PKI_DIGEST_LEN];    /* < CRL Hash Value */
} CmePkiX509CrlExt;

/**
 * @ingroup pki_def
 * @brief 保存证书以及一些扩展字段。
 */
typedef struct {
    X509CRLCertList *crl;        /* < CRL Cert List */
    bool valid;                  /* < Cert有效性 */
    CmePkiX509CrlExt crlExField; /* < CRL Extension */
    int32_t ref;                 /* < 引用 */
} CmePkiX509Crl;

/**
 * @ingroup pki_def
 * @brief OCSP配置参数
 *
 * OCSP配置类型标志。
 */
typedef enum {
    CME_PKI_OCSP_RESP_CHECK_TIME = 0x00000001,                  /* < 启用响应程序证书与时间验证 */
    CME_PKI_OCSP_RESP_CHECK_CRL = 0x00000002,                   /* < 仅对响应者证书启用CRL验证 */
    CME_PKI_OCSP_RESP_CHECK_DELTA_CRL = 0x00000008,             /* < 仅对响应者证书启用增量CRL验证 */
    CME_PKI_OCSP_RESP_X509_V_FLAG_IGNORE_CRITICAL = 0x00000040, /* < 忽略关键扩展 */
    CME_PKI_OCSP_SIGN_REQUEST = 0x00000400,                     /* < 启用对OCSP请求进行签名 */
    CME_PKI_OCSP_SIGN_REQ_WITH_CERT_CHAIN = 0x00000800,         /* < 启用发送带有OCSP请求的证书 */
    CME_PKI_OCSP_TRUST_RESP_CERTS_IN_MSG = 0x00001000, /* < 使能使用响应消息中的证书作为响应方证书的功能 */
    CME_PKI_OCSP_EXTN_NONCE = 0x00002000,              /* < 启用OCSP响应的Nonce验证 */
    CME_PKI_OCSP_EXTN_RESP_TYPES = 0x00004000          /* < 启用响应类型规范 */
} CmePkiOcspConfType;

/**
 * @ingroup pki_def
 * @brief PKI Verify回调函数。
 */
typedef int32_t (*CmePkiVerifyCbFunc)(int32_t verifyResult, void *storeCtx, const void *appData);

#ifdef __cplusplus
}
#endif

#endif
