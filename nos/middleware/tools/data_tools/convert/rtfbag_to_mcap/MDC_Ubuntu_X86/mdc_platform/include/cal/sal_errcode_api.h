/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 * Description: 统一定义adaptor中的错误码
 * Create: 2020/4/11
 * Note:
 * History: 2020/4/11 第一次创建
 */
#ifndef SAL_ERRCODE_API_H
#define SAL_ERRCODE_API_H

#include <stdint.h>
#include <stdbool.h>
#include "bsl_eno_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CAL_COMPONENT_ID 0x030Cu

/**
 * @ingroup adaptor
 * 无错误
 */
#define SAL_SUCCESS 0

/**
 * @ingroup adaptor
 * 未知错误
 */
#define SAL_ERR_UNKNOWN 1u

/**
 * @ingroup adaptor
 * 申请内存失败
 */
#define SAL_ERR_NO_MEMORY 2u

/**
 * @ingroup adaptor
 * 参数错误
 */
#define SAL_ERR_BAD_PARAM 3u

/**
 * @ingroup adaptor
 * 不支持的入参
 */
#define SAL_ERR_NOT_SUPPORT 4u

/**
 * @ingroup adaptor
 * 内存拷贝错误
 */
#define SAL_ERR_OVER_FLOW 5u

/**
 * @ingroup adaptor
 * 缓存长度不足
 */
#define SAL_ERR_BUFFER_NOT_ENOUGH 6u

/**
 * @ingroup adaptor
 * 证书格式错误
 */
#define SAL_ERR_BAD_CERTIFICATE 10u

/**
 * @ingroup adaptor
 * 读取证书失败
 */
#define SAL_ERR_LOAD_CERT 11u

/**
 * @ingroup adaptor
 * 读取私钥失败
 */
#define SAL_ERR_LOAD_KEY 12u

/**
 * @ingroup adaptor
 * 缓存初始化失败
 */
#define SAL_ERR_STORE_INIT 13u

/**
 * @ingroup adaptor
 * DN不相等
 */
#define SAL_ERR_DN_CMP 14u

/**
 * @ingroup adaptor
 * 不合法的OCSP响应
 */
#define SAL_ERR_BAD_OCSP_RESP 15u

/**
 * @ingroup adaptor
 * 读取CRL出错
 */
#define SAL_ERR_LOAD_CRL 16u

/**
 * @ingroup adaptor
 * 加载CRL错误
 */
#define SAL_ERR_ADD_CRL 17u

/**
 * @ingroup adaptor
 * 证书校验错误
 */
#define SAL_ERR_VERIFY_CERT 18u

/**
 * @ingroup adaptor
 * 未指明的证书校验错误
 */
#define SAL_ERR_VERIFY_UNSPECIFIED 19u

/**
 * @ingroup adaptor
 * 无法获取签发证书导致的证书校验失败
 */
#define SAL_ERR_VERIFY_UNABLE_TO_GET_ISSUER_CERT 20u

/**
 * @ingroup adaptor
 * 无法获取CRL导致的证书校验失败
 */
#define SAL_ERR_VERIFY_UNABLE_TO_GET_CRL 21u

/**
 * @ingroup adaptor
 * 无法解析证书签名导致证书校验失败
 */
#define SAL_ERR_VERIFY_UNABLE_TO_DECRYPT_CERT_SIGNATURE 22u

/**
 * @ingroup adaptor
 * 无法解析CRL签名导致证书校验失败
 */
#define SAL_ERR_VERIFY_UNABLE_TO_DECRYPT_CRL_SIGNATURE 23u

/**
 * @ingroup adaptor
 * 无法获取签发者公钥导致证书校验失败
 */
#define SAL_ERR_VERIFY_UNABLE_TO_DECODE_ISSUER_PUBLIC_KEY 24u

/**
 * @ingroup adaptor
 * 证书签名错误导致证书校验失败
 */
#define SAL_ERR_VERIFY_CERT_SIGNATURE_FAILURE 25u

/**
 * @ingroup adaptor
 * CRL签名错误导致证书校验失败
 */
#define SAL_ERR_VERIFY_CRL_SIGNATURE_FAILURE 26u

/**
 * @ingroup adaptor
 * 证书不在有效期内
 */
#define SAL_ERR_VERIFY_CERT_NOT_YET_VALID 27u

/**
 * @ingroup adaptor
 * 证书已过期
 */
#define SAL_ERR_VERIFY_CERT_HAS_EXPIRED 28u

/**
 * @ingroup adaptor
 * CRL未到有效期
 */
#define SAL_ERR_VERIFY_CRL_NOT_YET_VALID 29u

/**
 * @ingroup adaptor
 * CRL已过期
 */
#define SAL_ERR_VERIFY_CRL_HAS_EXPIRED 30u

/**
 * @ingroup adaptor
 * 校验证书生效日期错误
 */
#define SAL_ERR_VERIFY_ERROR_IN_CERT_NOT_BEFORE_FIELD 31u

/**
 * @ingroup adaptor
 * 证书失效日期错误
 */
#define SAL_ERR_VERIFY_ERROR_IN_CERT_NOT_AFTER_FIELD 32u

/**
 * @ingroup adaptor
 * CRL最近更新时间错误
 */
#define SAL_ERR_VERIFY_ERROR_IN_CRL_LAST_UPDATE_FIELD 33u

/**
 * @ingroup adaptor
 * CRL下次更新时间错误
 */
#define SAL_ERR_VERIFY_ERROR_IN_CRL_NEXT_UPDATE_FIELD 34u

/**
 * @ingroup adaptor
 * 内存错误导致证书校验失败
 */
#define SAL_ERR_VERIFY_OUT_OF_MEM 35u

/**
 * @ingroup adaptor
 * 零长自签名证书链
 */
#define SAL_ERR_VERIFY_DEPTH_ZERO_SELF_SIGNED_CERT 36u

/**
 * @ingroup adaptor
 * 证书链中存在自签名证书
 */
#define SAL_ERR_VERIFY_SELF_SIGNED_CERT_IN_CHAIN 37u

/**
 * @ingroup adaptor
 * 无法在本地获取签发者证书
 */
#define SAL_ERR_VERIFY_UNABLE_TO_GET_ISSUER_CERT_LOCALLY 38u

/**
 * @ingroup adaptor
 * 无法校验叶子证书签名
 */
#define SAL_ERR_VERIFY_UNABLE_TO_VERIFY_LEAF_SIGNATURE 39u

/**
 * @ingroup adaptor
 * 证书链超长
 */
#define SAL_ERR_VERIFY_CERT_CHAIN_TOO_LONG 40u

/**
 * @ingroup adaptor
 * 证书已废弃
 */
#define SAL_ERR_VERIFY_CERT_REVOKED 41u

/**
 * @ingroup adaptor
 * 无效的CA
 */
#define SAL_ERR_VERIFY_INVALID_CA 42u

/**
 * @ingroup adaptor
 * 路径长度过长
 */
#define SAL_ERR_VERIFY_PATH_LENGTH_EXCEEDED 43u

/**
 * @ingroup adaptor
 * 无效的purpose
 */
#define SAL_ERR_VERIFY_INVALID_PURPOSE 44u

/**
 * @ingroup adaptor
 * 非信任证书
 */
#define SAL_ERR_VERIFY_CERT_UNTRUSTED 45u

/**
 * @ingroup adaptor
 * 操作请求被拒绝
 */
#define SAL_ERR_VERIFY_CERT_REJECTED 46u

/**
 * @ingroup adaptor
 * 证书主体与签发者不匹配
 */
#define SAL_ERR_VERIFY_SUBJECT_ISSUER_MISMATCH 47u

/**
 * @ingroup adaptor
 * AKID与SKID不匹配
 */
#define SAL_ERR_VERIFY_AKID_SKID_MISMATCH 48u

/**
 * @ingroup adaptor
 * AKID与签发者序列号不匹配
 */
#define SAL_ERR_VERIFY_AKID_ISSUER_SERIAL_MISMATCH 49u

/**
 * @ingroup adaptor
 * 密钥用途不含证书签名
 */
#define SAL_ERR_VERIFY_KEYUSAGE_NO_CERTSIGN 50u

/**
 * @ingroup adaptor
 * 无法获取CRL签发者
 */
#define SAL_ERR_VERIFY_UNABLE_TO_GET_CRL_ISSUER 51u

/**
 * @ingroup adaptor
 * 未处理的critical扩展
 */
#define SAL_ERR_VERIFY_UNHANDLED_CRITICAL_EXTENSION 52u

/**
 * @ingroup adaptor
 * 密钥用途不含CRL签名
 */
#define SAL_ERR_VERIFY_KEYUSAGE_NO_CRL_SIGN 53u

/**
 * @ingroup adaptor
 * 未处理的CRL critical扩展
 */
#define SAL_ERR_VERIFY_UNHANDLED_CRITICAL_CRL_EXTENSION 54u

/**
 * @ingroup adaptor
 * 无效的非CA
 */
#define SAL_ERR_VERIFY_INVALID_NON_CA 55u

/**
 * @ingroup adaptor
 * 代理路径过长
 */
#define SAL_ERR_VERIFY_PROXY_PATH_LENGTH_EXCEEDED 56u

/**
 * @ingroup adaptor
 * 密钥用途不含数字签名
 */
#define SAL_ERR_VERIFY_KEYUSAGE_NO_DIGITAL_SIGNATURE 57u

/**
 * @ingroup adaptor
 * 不支持代理证书
 */
#define SAL_ERR_VERIFY_PROXY_CERTIFICATES_NOT_ALLOWED 58u

/**
 * @ingroup adaptor
 * 无效扩展
 */
#define SAL_ERR_VERIFY_INVALID_EXTENSION 59u

/**
 * @ingroup adaptor
 * 无效的规则扩展
 */
#define SAL_ERR_VERIFY_INVALID_POLICY_EXTENSION 60u

/**
 * @ingroup adaptor
 * 无明确校验规则
 */
#define SAL_ERR_VERIFY_NO_EXPLICIT_POLICY 61u

/**
 * @ingroup adaptor
 * CRL范围不一致
 */
#define SAL_ERR_VERIFY_DIFFERENT_CRL_SCOPE 62u

/**
 * @ingroup adaptor
 * 不支持的扩展特性
 */
#define SAL_ERR_VERIFY_UNSUPPORTED_EXTENSION_FEATURE 63u

/**
 * @ingroup adaptor
 * 资源非父类的子集
 */
#define SAL_ERR_VERIFY_UNNESTED_RESOURCE 64u

/**
 * @ingroup adaptor
 * 允许的子树冲突
 */
#define SAL_ERR_VERIFY_PERMITTED_VIOLATION 65u

/**
 * @ingroup adaptor
 * 排查的子树冲突
 */
#define SAL_ERR_VERIFY_EXCLUDED_VIOLATION 66u

/**
 * @ingroup adaptor
 * 最大最小子树错误
 */
#define SAL_ERR_VERIFY_SUBTREE_MINMAX 67u

/**
 * @ingroup adaptor
 * 计算AEAD时更新auth出错
 */
#define SAL_ERR_UPDATE_CIPHER 70u

/**
 * @ingroup adaptor
 * 算法初始化失败
 */
#define SAL_ERR_CIPHER_INIT 71u

/**
 * @ingroup adaptor
 * 加密失败
 */
#define SAL_ERR_CIPHER_ENCRYPT 72u

/**
 * @ingroup adaptor
 * 解密失败
 */
#define SAL_ERR_CIPHER_DECRYPT 73u

/**
 * @ingroup adaptor
 * 生成椭圆曲线密钥对失败
 */
#define SAL_ERR_EC_KEY_GEN 74u

/**
 * @ingroup adaptor
 * 哈希初始化失败
 */
#define SAL_ERR_HASH_INIT 75u

/**
 * @ingroup adaptor
 * 更新哈希数据失败
 */
#define SAL_ERR_HASH_UPDATE 76u

/**
 * @ingroup adaptor
 * 获取哈希值失败
 */
#define SAL_ERR_HASH_FINAL 77u

/**
 * @ingroup adaptor
 * 密钥派生失败
 */
#define SAL_ERR_KDF_KEY_DERIVE 78u

/**
 * @ingroup adaptor
 * 密钥分割失败
 */
#define SAL_ERR_KDF_KEY_DIVERSIFIER 79u

/**
 * @ingroup adaptor
 * MAC初始化失败
 */
#define SAL_ERR_MAC_INIT 80u

/**
 * @ingroup adaptor
 * MAC更新数据失败
 */
#define SAL_ERR_MAC_UPDATE 81u

/**
 * @ingroup adaptor
 * 计算MAC值失败
 */
#define SAL_ERR_MAC_FINAL 82u

/**
 * @ingroup adaptor
 * 获取随机数失败
 */
#define SAL_ERR_RAND_BYTE 83u

/**
 * @ingroup adaptor
 * 签名失败
 */
#define SAL_ERR_SIGN_SIGN 84u

/**
 * @ingroup adaptor
 * 验签失败
 */
#define SAL_ERR_SIGN_VERIFY 85u

/**
 * @ingroup adaptor
 * 蝴蝶算法密钥派生失败
 */
#define SAL_ERR_BKE_KEY_DERIVE 86u

/**
 * @ingroup adaptor
 * 获取PSK ID失败
 */
#define SAL_ERR_GET_PSK_KEYID 90u

/**
 * @ingroup adaptor
 * 错误码最大值
 */
#define SAL_ERR_END 0xffu

/**
 * @ingroup adaptor
 * adaptor 通用模块ID
 */
#define SAL_M_ADAPTOR_PUB 0u

/**
 * @ingroup adaptor
 * adaptor cert模块ID
 */
#define SAL_M_ADAPTOR_CERT 1u

/**
 * @ingroup adaptor
 * adaptor crypto模块ID
 */
#define SAL_M_ADAPTOR_CRYPTO 2u

/**
 * @ingroup adaptor
 * adaptor keys模块ID
 */
#define SAL_M_ADAPTOR_KEYS 3u

/**
 * @ingroup adaptor
 * adaptor comm模块ID
 */
#define SAL_M_ADAPTOR_COMMON 4u

/**
 * @ingroup adaptor
 * adaptor SSL模块ID
 */
#define SAL_M_ADAPTOR_TLS 5u

/**
 * @ingroup adaptor
 * adaptor 模块ID最大值
 */
#define SAL_M_END 6u

/**
 * @ingroup adaptor
 * 未知错误
 */
#define SAL_ERRNO_UNKNOWN \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_PUB, SAL_ERR_UNKNOWN)))
/**
 * @ingroup adaptor
 * 申请内存失败
 */
#define SAL_ERRNO_NO_MEMORY \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_PUB, SAL_ERR_NO_MEMORY)))

/**
 * @ingroup adaptor
 * 参数错误
 */
#define SAL_ERRNO_BAD_PARAM \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_PUB, SAL_ERR_BAD_PARAM)))

/**
 * @ingroup adaptor
 * 不支持的入参
 */
#define SAL_ERRNO_NOT_SUPPORT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_PUB, SAL_ERR_NOT_SUPPORT)))

/**
 * @ingroup adaptor
 * 内存拷贝错误
 */
#define SAL_ERRNO_OVER_FLOW \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_PUB, SAL_ERR_OVER_FLOW)))

/**
 * @ingroup adaptor
 * 缓存长度不足
 */
#define SAL_ERRNO_BUFFER_NOT_ENOUGH \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_PUB, SAL_ERR_BUFFER_NOT_ENOUGH)))

/**
 * @ingroup adaptor
 * 证书格式错误
 */
#define SAL_ERRNO_BAD_CERTIFICATE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_BAD_CERTIFICATE)))

/**
 * @ingroup adaptor
 * 读取证书失败
 */
#define SAL_ERRNO_LOAD_CERT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_LOAD_CERT)))

/**
 * @ingroup adaptor
 * 读取私钥失败
 */
#define SAL_ERRNO_LOAD_KEY \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_LOAD_KEY)))

/**
 * @ingroup adaptor
 * 缓存初始化失败
 */
#define SAL_ERRNO_STORE_INIT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_STORE_INIT)))

/**
 * @ingroup adaptor
 * DN不相等
 */
#define SAL_ERRNO_DN_CMP \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_DN_CMP)))

/**
 * @ingroup adaptor
 * 不合法的OCSP响应
 */
#define SAL_ERRNO_BAD_OCSP_RESP \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_BAD_OCSP_RESP)))

/**
 * @ingroup adaptor
 * 读取CRL出错
 */
#define SAL_ERRNO_LOAD_CRL \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_LOAD_CRL)))

/**
 * @ingroup adaptor
 * 加载CRL错误
 */
#define SAL_ERRNO_ADD_CRL \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_ADD_CRL)))

/**
 * @ingroup adaptor
 * 证书校验错误
 */
#define SAL_ERRNO_VERIFY_CERT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CERT)))

/**
 * @ingroup adaptor
 * 未指明的证书校验错误
 */
#define SAL_ERRNO_VERIFY_UNSPECIFIED \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNSPECIFIED)))

/**
 * @ingroup adaptor
 * 无法获取签发证书导致的证书校验失败
 */
#define SAL_ERRNO_VERIFY_UNABLE_TO_GET_ISSUER_CERT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNABLE_TO_GET_ISSUER_CERT)))

/**
 * @ingroup adaptor
 * 无法获取CRL导致的证书校验失败
 */
#define SAL_ERRNO_VERIFY_UNABLE_TO_GET_CRL \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNABLE_TO_GET_CRL)))

/**
 * @ingroup adaptor
 * 无法解析证书签名导致证书校验失败
 */
#define SAL_ERRNO_VERIFY_UNABLE_TO_DECRYPT_CERT_SIGNATURE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNABLE_TO_DECRYPT_CERT_SIGNATURE)))

/**
 * @ingroup adaptor
 * 无法解析CRL签名导致证书校验失败
 */
#define SAL_ERRNO_VERIFY_UNABLE_TO_DECRYPT_CRL_SIGNATURE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNABLE_TO_DECRYPT_CRL_SIGNATURE)))

/**
 * @ingroup adaptor
 * 无法获取签发者公钥导致证书校验失败
 */
#define SAL_ERRNO_VERIFY_UNABLE_TO_DECODE_ISSUER_PUBLIC_KEY \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNABLE_TO_DECODE_ISSUER_PUBLIC_KEY)))

/**
 * @ingroup adaptor
 * 证书签名错误导致证书校验失败
 */
#define SAL_ERRNO_VERIFY_CERT_SIGNATURE_FAILURE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CERT_SIGNATURE_FAILURE)))

/**
 * @ingroup adaptor
 * CRL签名错误导致证书校验失败
 */
#define SAL_ERRNO_VERIFY_CRL_SIGNATURE_FAILURE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CRL_SIGNATURE_FAILURE)))

/**
 * @ingroup adaptor
 * 证书不在有效期内
 */
#define SAL_ERRNO_VERIFY_CERT_NOT_YET_VALID \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CERT_NOT_YET_VALID)))

/**
 * @ingroup adaptor
 * 证书已过期
 */
#define SAL_ERRNO_VERIFY_CERT_HAS_EXPIRED \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CERT_HAS_EXPIRED)))

/**
 * @ingroup adaptor
 * CRL未到有效期
 */
#define SAL_ERRNO_VERIFY_CRL_NOT_YET_VALID \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CRL_NOT_YET_VALID)))

/**
 * @ingroup adaptor
 * CRL已过期
 */
#define SAL_ERRNO_VERIFY_CRL_HAS_EXPIRED \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CRL_HAS_EXPIRED)))

/**
 * @ingroup adaptor
 * 校验证书生效日期错误
 */
#define SAL_ERRNO_VERIFY_ERROR_IN_CERT_NOT_BEFORE_FIELD \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_ERROR_IN_CERT_NOT_BEFORE_FIELD)))

/**
 * @ingroup adaptor
 * 证书失效日期错误
 */
#define SAL_ERRNO_VERIFY_ERROR_IN_CERT_NOT_AFTER_FIELD \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_ERROR_IN_CERT_NOT_AFTER_FIELD)))

/**
 * @ingroup adaptor
 * CRL最近更新时间错误
 */
#define SAL_ERRNO_VERIFY_ERROR_IN_CRL_LAST_UPDATE_FIELD \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_ERROR_IN_CRL_LAST_UPDATE_FIELD)))

/**
 * @ingroup adaptor
 * CRL下次更新时间错误
 */
#define SAL_ERRNO_VERIFY_ERROR_IN_CRL_NEXT_UPDATE_FIELD \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_ERROR_IN_CRL_NEXT_UPDATE_FIELD)))

/**
 * @ingroup adaptor
 * 内存错误导致证书校验失败
 */
#define SAL_ERRNO_VERIFY_OUT_OF_MEM \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_OUT_OF_MEM)))

/**
 * @ingroup adaptor
 * 零长自签名证书链
 */
#define SAL_ERRNO_VERIFY_DEPTH_ZERO_SELF_SIGNED_CERT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_DEPTH_ZERO_SELF_SIGNED_CERT)))

/**
 * @ingroup adaptor
 * 证书链中存在自签名证书
 */
#define SAL_ERRNO_VERIFY_SELF_SIGNED_CERT_IN_CHAIN \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_SELF_SIGNED_CERT_IN_CHAIN)))

/**
 * @ingroup adaptor
 * 无法在本地获取签发者证书
 */
#define SAL_ERRNO_VERIFY_UNABLE_TO_GET_ISSUER_CERT_LOCALLY \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNABLE_TO_GET_ISSUER_CERT_LOCALLY)))

/**
 * @ingroup adaptor
 * 无法校验叶子证书签名
 */
#define SAL_ERRNO_VERIFY_UNABLE_TO_VERIFY_LEAF_SIGNATURE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNABLE_TO_VERIFY_LEAF_SIGNATURE)))

/**
 * @ingroup adaptor
 * 证书链超长
 */
#define SAL_ERRNO_VERIFY_CERT_CHAIN_TOO_LONG \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CERT_CHAIN_TOO_LONG)))

/**
 * @ingroup adaptor
 * 证书已废弃
 */
#define SAL_ERRNO_VERIFY_CERT_REVOKED \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CERT_REVOKED)))

/**
 * @ingroup adaptor
 * 无效的CA
 */
#define SAL_ERRNO_VERIFY_INVALID_CA \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_INVALID_CA)))

/**
 * @ingroup adaptor
 * 路径长度过长
 */
#define SAL_ERRNO_VERIFY_PATH_LENGTH_EXCEEDED \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_PATH_LENGTH_EXCEEDED)))

/**
 * @ingroup adaptor
 * 无效的purpose
 */
#define SAL_ERRNO_VERIFY_INVALID_PURPOSE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_INVALID_PURPOSE)))

/**
 * @ingroup adaptor
 * 非信任证书
 */
#define SAL_ERRNO_VERIFY_CERT_UNTRUSTED \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CERT_UNTRUSTED)))

/**
 * @ingroup adaptor
 * 操作请求被拒绝
 */
#define SAL_ERRNO_VERIFY_CERT_REJECTED \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_CERT_REJECTED)))

/**
 * @ingroup adaptor
 * 证书主体与签发者不匹配
 */
#define SAL_ERRNO_VERIFY_SUBJECT_ISSUER_MISMATCH \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_SUBJECT_ISSUER_MISMATCH)))

/**
 * @ingroup adaptor
 * AKID与SKID不匹配
 */
#define SAL_ERRNO_VERIFY_AKID_SKID_MISMATCH \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_AKID_SKID_MISMATCH)))

/**
 * @ingroup adaptor
 * AKID与签发者序列号不匹配
 */
#define SAL_ERRNO_VERIFY_AKID_ISSUER_SERIAL_MISMATCH \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_AKID_ISSUER_SERIAL_MISMATCH)))

/**
 * @ingroup adaptor
 * 密钥用途不含证书签名
 */
#define SAL_ERRNO_VERIFY_KEYUSAGE_NO_CERTSIGN \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_KEYUSAGE_NO_CERTSIGN)))

/**
 * @ingroup adaptor
 * 无法获取CRL签发者
 */
#define SAL_ERRNO_VERIFY_UNABLE_TO_GET_CRL_ISSUER \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNABLE_TO_GET_CRL_ISSUER)))

/**
 * @ingroup adaptor
 * 未处理的critical扩展
 */
#define SAL_ERRNO_VERIFY_UNHANDLED_CRITICAL_EXTENSION \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNHANDLED_CRITICAL_EXTENSION)))

/**
 * @ingroup adaptor
 * 密钥用途不含CRL签名
 */
#define SAL_ERRNO_VERIFY_KEYUSAGE_NO_CRL_SIGN \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_KEYUSAGE_NO_CRL_SIGN)))

/**
 * @ingroup adaptor
 * 未处理的CRL critical扩展
 */
#define SAL_ERRNO_VERIFY_UNHANDLED_CRITICAL_CRL_EXTENSION \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNHANDLED_CRITICAL_CRL_EXTENSION)))

/**
 * @ingroup adaptor
 * 无效的非CA
 */
#define SAL_ERRNO_VERIFY_INVALID_NON_CA \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_INVALID_NON_CA)))

/**
 * @ingroup adaptor
 * 代理路径过长
 */
#define SAL_ERRNO_VERIFY_PROXY_PATH_LENGTH_EXCEEDED \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_PROXY_PATH_LENGTH_EXCEEDED)))

/**
 * @ingroup adaptor
 * 密钥用途不含数字签名
 */
#define SAL_ERRNO_VERIFY_KEYUSAGE_NO_DIGITAL_SIGNATURE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_KEYUSAGE_NO_DIGITAL_SIGNATURE)))

/**
 * @ingroup adaptor
 * 不支持代理证书
 */
#define SAL_ERRNO_VERIFY_PROXY_CERTIFICATES_NOT_ALLOWED \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_PROXY_CERTIFICATES_NOT_ALLOWED)))

/**
 * @ingroup adaptor
 * 无效扩展
 */
#define SAL_ERRNO_VERIFY_INVALID_EXTENSION \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_INVALID_EXTENSION)))

/**
 * @ingroup adaptor
 * 无效的规则扩展
 */
#define SAL_ERRNO_VERIFY_INVALID_POLICY_EXTENSION \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_INVALID_POLICY_EXTENSION)))

/**
 * @ingroup adaptor
 * 无明确校验规则
 */
#define SAL_ERRNO_VERIFY_NO_EXPLICIT_POLICY \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_NO_EXPLICIT_POLICY)))

/**
 * @ingroup adaptor
 * CRL范围不一致
 */
#define SAL_ERRNO_VERIFY_DIFFERENT_CRL_SCOPE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_DIFFERENT_CRL_SCOPE)))

/**
 * @ingroup adaptor
 * 不支持的扩展特性
 */
#define SAL_ERRNO_VERIFY_UNSUPPORTED_EXTENSION_FEATURE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNSUPPORTED_EXTENSION_FEATURE)))

/**
 * @ingroup adaptor
 * 资源非父类的子集
 */
#define SAL_ERRNO_VERIFY_UNNESTED_RESOURCE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_UNNESTED_RESOURCE)))

/**
 * @ingroup adaptor
 * 允许的子树冲突
 */
#define SAL_ERRNO_VERIFY_PERMITTED_VIOLATION \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_PERMITTED_VIOLATION)))

/**
 * @ingroup adaptor
 * 排查的子树冲突
 */
#define SAL_ERRNO_VERIFY_EXCLUDED_VIOLATION \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_EXCLUDED_VIOLATION)))

/**
 * @ingroup adaptor
 * 最大最小子树错误
 */
#define SAL_ERRNO_VERIFY_SUBTREE_MINMAX \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CERT, SAL_ERR_VERIFY_SUBTREE_MINMAX)))

/**
 * @ingroup adaptor
 * 计算AEAD时更新auth出错
 */
#define SAL_ERRNO_UPDATE_CIPHER \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_UPDATE_CIPHER)))

/**
 * @ingroup adaptor
 * 算法初始化失败
 */
#define SAL_ERRNO_CIPHER_INIT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_CIPHER_INIT)))

/**
 * @ingroup adaptor
 * 加密失败
 */
#define SAL_ERRNO_CIPHER_ENCRYPT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_CIPHER_ENCRYPT)))

/**
 * @ingroup adaptor
 * 解密失败
 */
#define SAL_ERRNO_CIPHER_DECRYPT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_CIPHER_DECRYPT)))

/**
 * @ingroup adaptor
 * 生成椭圆曲线密钥对失败
 */
#define SAL_ERRNO_EC_KEY_GEN \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_EC_KEY_GEN)))

/**
 * @ingroup adaptor
 * 哈希初始化失败
 */
#define SAL_ERRNO_HASH_INIT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_HASH_INIT)))

/**
 * @ingroup adaptor
 * 更新哈希数据失败
 */
#define SAL_ERRNO_HASH_UPDATE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_HASH_UPDATE)))

/**
 * @ingroup adaptor
 * 获取哈希值失败
 */
#define SAL_ERRNO_HASH_FINAL \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_HASH_FINAL)))

/**
 * @ingroup adaptor
 * 密钥派生失败
 */
#define SAL_ERRNO_KDF_KEY_DERIVE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_KDF_KEY_DERIVE)))

/**
 * @ingroup adaptor
 * 密钥分割失败
 */
#define SAL_ERRNO_KDF_KEY_DIVERSIFIER \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_KDF_KEY_DIVERSIFIER)))

/**
 * @ingroup adaptor
 * MAC初始化失败
 */
#define SAL_ERRNO_MAC_INIT \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_MAC_INIT)))

/**
 * @ingroup adaptor
 * MAC更新数据失败
 */
#define SAL_ERRNO_MAC_UPDATE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_MAC_UPDATE)))

/**
 * @ingroup adaptor
 * 计算MAC值失败
 */
#define SAL_ERRNO_MAC_FINAL \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_MAC_FINAL)))

/**
 * @ingroup adaptor
 * 获取随机数失败
 */
#define SAL_ERRNO_RAND_BYTE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_RAND_BYTE)))

/**
 * @ingroup adaptor
 * 签名失败
 */
#define SAL_ERRNO_SIGN_SIGN \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_SIGN_SIGN)))

/**
 * @ingroup adaptor
 * 验签失败
 */
#define SAL_ERRNO_SIGN_VERIFY \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_SIGN_VERIFY)))

/**
 * @ingroup adaptor
 * 蝴蝶算法密钥派生失败
 */
#define SAL_ERRNO_BKE_KEY_DERIVE \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_CRYPTO, SAL_ERR_BKE_KEY_DERIVE)))

/**
 * @ingroup adaptor
 * 获取PSK ID失败
 */
#define SAL_ERRNO_GET_PSK_KEYID \
    (-((int32_t)BSL_ENO_MAKE(CAL_COMPONENT_ID, SAL_M_ADAPTOR_KEYS, SAL_ERR_GET_PSK_KEYID)))

/**
 * @ingroup adaptor
 * @brief 生成错误码函数封装。
 * @attention 无须直接调用，一般通过宏SAL_GenError调用。
 *
 * @param m [IN] 模块名ID，@see SAL_M_CONN
 * @param e [IN] 错误值，@see SAL_SUCCESS
 * @param fileName [IN] 错误产生的文件名。
 * @param lineNo [IN] 错误产生的行号。
 *
 * @retval error code。
 *
 * @see SAL_GenError
 */
int32_t SAL_GenErrorWrapper(uint16_t m, uint16_t e, const char *fileName, int32_t lineNo);

/**
 * @ingroup adaptor
 * @brief 生成错误码的宏。
 *
 * @param m [IN] 模块名ID，@see SAL_M_CONN
 * @param e [IN] 错误值，@see SAL_SUCCESS
 *
 * @retval error code。
 *
 * @see SAL_GenErrorWrapper
 */
#define SAL_GenError(m_, e_) SAL_GenErrorWrapper((m_), (e_), __FILENAME__, __LINE__)

/**
 * @ingroup adaptor
 * @brief 根据错误码获取错误值。
 *
 * @param e [IN] 错误码，@see SAL_SUCCESS。
 *
 * @retval 错误值。
 *
 * @see SAL_GenErrorWrapper
 */
uint16_t SAL_GetErrorCode(int32_t e);

/**
 * @ingroup adaptor
 * @brief 根据错误码获取模块ID。
 *
 * @param e [IN] 错误码，@see SAL_SUCCESS。
 *
 * @retval 模块号。
 *
 * @see SAL_GenErrorWrapper
 */
uint16_t SAL_GetErrorModule(int32_t e);

/**
 * @ingroup adaptor
 * @brief 检查错误码是否与预期一致
 *
 * @param err [IN] 错误码，@see SAL_SUCCESS。
 * @param module [IN] 模块名ID，@see SAL_M_CONN
 * @param errorCode [IN] 错误值，@see SAL_SUCCESS
 *
 * @retval true 错误码与预期一致
 *         false 错误码与预期不一致
 */
bool SAL_CheckErrorCode(int32_t err, uint16_t module, uint16_t errorCode);

#ifdef __cplusplus
}
#endif

#endif /* SAL_ERRCODE_API_H */
