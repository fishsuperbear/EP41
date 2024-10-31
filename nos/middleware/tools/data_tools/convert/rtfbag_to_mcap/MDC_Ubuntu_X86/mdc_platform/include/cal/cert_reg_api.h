/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:证书功能对外接口
 * Create: 2020/05/08
 * History:
 */

/** @defgroup adaptor adaptor */

#ifndef ADAPTOR_CERT_REG_API_H
#define ADAPTOR_CERT_REG_API_H

#include <stdint.h>
#include <stddef.h>
#include "keys/pkey_api.h"
#include "cert/cert_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup adaptor
 * @brief 加载 key 的钩子。
 *
 * @param key [OUT] 非对称密钥。
 * @param file [IN] 密钥文件。
 *
 * @retval int32_t, 由回调函数返回的错误码。
 */
typedef int32_t (*CertKeyLoadFromFileFunc)(PKeyAsymmetricKeyHandle *key, const char *file);

/**
 * @ingroup adaptor
 * @brief 加载证书的钩子。
 *
 * @param cert [OUT] 证书句柄。
 * @param certSize [OUT] 从文件中加载的证书个数。
 * @param file [IN] 证书文件
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertLoadFromFileFunc)(CertHandle *cert, size_t *certSize, const char *file);

/**
 * @ingroup adaptor
 * @brief 检查证书与私钥是否匹配的钩子。
 *
 * @param cert [IN] 证书句柄。
 * @param key [IN] 非对称密钥。
 *
 * @retval int32_t, 由回调函数返回的错误码。
 */
typedef int32_t (*CertCertMatchPrivateKeyFunc)(CertHandle cert, PKeyAsymmetricKeyHandle privateKey);

/**
 * @ingroup adaptor
 * @brief 解析证书的钩子。
 *
 * @param cert [OUT] 证书句柄。
 * @param data [IN] 证书缓存。
 * @param dataLen [IN] 缓存长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertParseFunc)(CertHandle *cert, const uint8_t *data, size_t dataLen);

/**
 * @ingroup adaptor
 * @brief 释放证书的钩子。
 *
 * @param cert [IN] 证书句柄。
 */
typedef void (*CertCertFreeFunc)(CertHandle cert);

/**
 * @ingroup adaptor
 * @brief 证书编码的钩子。
 *
 * @param cert [IN] 证书句柄。
 * @param data [OUT] 证书缓存。
 * @param dataLen [OUT] 缓存长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertEncodeFunc)(const CertHandle cert, uint8_t *data, size_t *dataLen);

/**
 * @ingroup adaptor
 * @brief 获取证书公钥的钩子。
 *
 * @param cert [IN] 证书句柄。
 * @param pubKeyHandle [OUT] 公钥句柄。
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertGetPubkeyFunc)(const CertHandle cert, PKeyAsymmetricKeyHandle *pubKeyHandle);

/**
 * @ingroup adaptor
 * @brief 获取证书授权算法的钩子。
 *
 * @param cert [IN] 证书句柄。
 * @param authAlg [OUT] 授权算法。
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertGetAuthAlgFunc)(const CertHandle cert, CertAuthorization *authAlg);

/**
 * @ingroup adaptor
 * @brief 获取证书签名算法的钩子。
 *
 * @param cert [IN] 证书句柄。
 * @param sig [OUT] 签名算法。
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertGetSignatureSchemeFunc)(const CertHandle cert, CertSignatureSchemes *sig);

/**
 * @ingroup adaptor
 * @brief 获取证书密钥用途的钩子。
 *
 * @param cert [IN] 证书句柄。
 * @param keyUsage [OUT] 密钥用途。
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertGetKeyUsageFunc)(const CertHandle cert, uint32_t *keyUsage);

/**
 * @ingroup adaptor
 * @brief 获取证书subjectname字段的钩子。
 *
 * @param cert [IN] 证书句柄。
 * @param subjectName [OUT] 证书subjectName字段。
 * @param len [IN] 输入字段最大长度。
 * @param outputLen [OUT] 输出证书subjectName字段长度。
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertGetSubjectNameFunc)(CertRoHandle cert, uint8_t *subjectName,
                                              size_t len, size_t *outputLen);

/**
 * @ingroup adaptor
 * @brief 新建证书池的钩子。
 *
 * @param rootCAsPool [IN] 证书池句柄。
 *
 * @retval CertPoolHandle, 返回创建的证书池句柄
 */
typedef CertPoolHandle (*CertCertPoolNewFunc)(const CertPoolHandle rootCAsPool);

/**
 * @ingroup adaptor
 * @brief 释放证书池的钩子。
 *
 * @param pool [IN] 证书池句柄。
 */
typedef void (*CertCertPoolFreeFunc)(CertPoolHandle pool);

/**
 * @ingroup adaptor
 * @brief 从文件或路径将根证书加载到证书池的钩子
 *
 * @param pool [IN] 证书池句柄。
 * @param file [IN] 根证书文件
 * @param path [IN] 根证书路径
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertPoolLoadLocationsFunc)(CertPoolHandle pool, const char *file, const char *path);

/**
 * @ingroup adaptor
 * @brief brief 从文件中将证书吊销列表加载到证书池的钩子。
 *
 * @param pool [IN] 证书池句柄。
 * @param file [IN] 证书吊销列表文件
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertPoolLoadCrlFunc)(CertPoolHandle pool, const char *file);

/**
 * @ingroup adaptor
 * @brief 设置证书池验证参数的钩子。
 *
 * @param pool [IN] 证书池句柄。
 * @param param [IN] 验证的参数
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertPoolSetVerifyParamFunc)(CertPoolHandle pool, const CertVerifyParamHandle param);

/**
 * @ingroup adaptor
 * @brief 证书池验证的钩子。
 *
 * @param pool [IN] 证书池句柄。
 * @param cert [IN] 证书句柄。
 * @param certChain [IN] 证书链
 * @param certChainSize [IN] 证书链长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertPoolVerifyFunc)(CertPoolHandle pool, const CertHandle cert, const CertHandle certChain[],
                                          size_t certChainSize);

/**
 * @ingroup adaptor
 * @brief 证书链排序的钩子。
 *
 * @param cert [IN] 证书句柄。
 * @param certChain [OUT] 证书链
 * @param certChainSize [IN] 证书链长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertSortChainFunc)(CertHandle cert, CertHandle *certChain, size_t certChainSize);

/**
 * @ingroup adaptor
 * @brief 检查证书链签发者DN的钩子。
 *
 * @param cert [IN] 证书句柄。
 * @param   DN [IN] 证书限定名
 * @param   DNSize [IN] 限定名长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertCheckDNFunc)(CertHandle cert, const uint8_t *DN, uint16_t DNSize);

/**
 * @ingroup adaptor
 * @brief 加载ocsp响应的钩子。
 *
 * @param pool [IN] 证书池句柄。
 * @param response [IN] ocsp响应
 * @param size [IN] ocsp响应长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CertCertLoadOcspRespFunc)(CertPoolHandle pool, const uint8_t *response, size_t size);

/**
 * @ingroup adaptor
 * @brief 克隆证书的钩子。
 *
 * @param cert [IN] 证书句柄。
 *
 * @retval 克隆证书的句柄
 */
typedef CertHandle (*CertCertDupFunc)(CertHandle cert);

/**
 * @ingroup adaptor
 * @brief 释放证书私钥的钩子。
 *
 * @param key [IN] 证书私钥句柄。
 *
 * @retval 无
 */
typedef void (*CertCertFreeKeyFunc)(PKeyAsymmetricKeyHandle key);

/**
 * @ingroup adaptor
 * CertAdaptHandleFuncs 结构体，证书适配层功能钩子函数集
 */
typedef struct {
    CertKeyLoadFromFileFunc keyLoadFromFile;               /* < 加载 key 的钩子 */
    CertCertFreeKeyFunc certFreeKey;                       /* < 释放 key 的钩子 */
    CertCertLoadFromFileFunc certLoadFromFile;             /* < 加载证书的钩子 */
    CertCertMatchPrivateKeyFunc certMatchPrivateKey;        /* < 检查证书与私钥是否匹配的钩子 */
    CertCertParseFunc certParse;                           /* < 解析证书的钩子 */
    CertCertFreeFunc certFree;                             /* < 释放证书的钩子 */
    CertCertEncodeFunc certEncode;                         /* < 证书编码的钩子 */
    CertCertGetPubkeyFunc certGetPubkey;                   /* < 获取证书公钥的钩子 */
    CertCertGetAuthAlgFunc certGetAuthAlg;                 /* < 获取证书授权算法的钩子 */
    CertCertGetSignatureSchemeFunc certGetSignatureScheme; /* < 获取证书签名算法的钩子 */
    CertCertGetKeyUsageFunc certGetKeyUsage;               /* < 获取证书密钥用途的钩子 */
    CertCertGetSubjectNameFunc certGetSubjectName;         /* < 获取证书subjectname字段的钩子 */
    CertCertPoolNewFunc certPoolNew;                       /* < 新建证书池的钩子 */
    CertCertPoolFreeFunc certPoolFree;                     /* < 释放证书池的钩子 */
    CertCertPoolLoadLocationsFunc certPoolLoadLocations;   /* < 加载本地证书池的钩子 */
    CertCertPoolLoadCrlFunc certPoolLoadCRL;               /* < 加载CRL的钩子 */
    CertCertPoolSetVerifyParamFunc certPoolSetVerifyParam; /* < 设置证书池验证参数的钩子 */
    CertCertPoolVerifyFunc certPoolVerify;                 /* < 证书池验证的钩子 */
    CertCertSortChainFunc certSortChain;                   /* < 证书链排序的钩子 */
    CertCertCheckDNFunc certCheckIssuerDN;                 /* < 检查证书链签发者DN的钩子 */
    CertCertLoadOcspRespFunc loadOcspResp;                 /* < 加载ocsp的钩子 */
    CertCertDupFunc certDup;                               /* < 克隆证书的钩子 */
} CertAdaptHandleFuncs;

#ifdef __cplusplus
}
#endif

#endif /* ADAPTOR_CERT_REG_API_H */
