/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 统一定义CME中的错误码
 * Create: 2020/08/03
 * Note:
 * 错误码体制：CME 的错误码由两部分组成，module ID + error code，其中module ID每个子模块一个。
 * error code是全局唯一的值，为了防止重复，分为通用区域和私有区域。通用区域即每个模块都有可能出现的错误，如no memory；
 * 私有区域只定义只有该区域会出现的错误，如record中特有的错误码，私有错误码通过唯一的前缀来区分。
 * 由于MISRA C中不能对枚举做运算，所以所有的错误码均采用宏管理，而不是枚举。
 * 失败的错误码永远是负值，即 -(module id + error code)。
 */

/** @defgroup cme cme */
/** @defgroup cme_common CME通用接口
 *  @ingroup cme
 */
/** @defgroup cme_errcode CME_Errcode_API
 * @ingroup cme_common
 */
#ifndef CME_ERRCODE_API_H
#define CME_ERRCODE_API_H

#include <stdint.h>
#include <stddef.h>
#include "bsl_eno_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup CME 组件码
 * CME 组件码
 */
#define CME_ENO_MID 0x310u
#define CME_ENO_MODULE_BASE 10u

/**
 * @ingroup cme_errcode
 * 执行成功
 */
#define CME_SUCCESS 0

/**
 * @ingroup cme_errcode
 * 未知错误
 */
#define CME_ERR_UNKNOWN (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 1u)))

/**
 * @ingroup cme_errcode
 * 内存申请失败
 */
#define CME_ERR_NO_MEMORY (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 2u)))

/**
 * @ingroup cme_errcode
 * 参数错误
 */
#define CME_ERR_BAD_PARAM (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 3u)))

/**
 * @ingroup cme_errcode
 * 不支持的类型
 */
#define CME_ERR_NOT_SUPPORT (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 4u)))

/**
 * @ingroup cme_errcode
 * 缓冲区溢出
 */
#define CME_ERR_OVER_FLOW (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 5u)))

/**
 * @ingroup cme_errcode
 * 缓冲区不足
 */
#define CME_ERR_BUFFER_NOT_ENOUGH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 6u)))

/**
 * @ingroup cme_errcode
 * 获取锁失败
 */
#define CME_ERR_LOCK_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 7u)))

/**
 * @ingroup cme_errcode
 * 接口未定义
 */
#define CME_ERR_INTERFACE_NOT_DEFINE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 8u)))

/**
 * @ingroup cme_errcode
 * 增加签名失败
 */
#define CME_ERR_ADD_SIGNER_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 9u)))

/**
 * @ingroup cme_errcode
 * 属性不可用
 */
#define CME_ERR_ATTR_NOT_AVAILABLE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 10u)))

/**
 * @ingroup cme_errcode
 * 内存拷贝失败
 */
#define CME_ERR_BUF_COPY_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 11u)))

/**
 * @ingroup cme_errcode
 * 证书无效
 */
#define CME_ERR_CERT_NOT_AVAILABLE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 12u)))

/**
 * @ingroup cme_errcode
 * CID不匹配
 */
#define CME_ERR_CID_MISMATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 13u)))

/**
 * @ingroup cme_errcode
 * 转换失败
 */
#define CME_ERR_CONVERT_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 14u)))

/**
 * @ingroup cme_errcode
 * 创建摘要失败
 */
#define CME_ERR_CREATE_DIGEST_FAIL (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 15u)))

/**
 * @ingroup cme_errcode
 * 封装数据创建失败
 */
#define CME_ERR_CREATE_ENVDATA_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 16u)))

/**
 * @ingroup cme_errcode
 * 创建请求失败
 */
#define CME_ERR_CREATE_REQUEST_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 17u)))

/**
 * @ingroup cme_errcode
 * 签名数据创建失败
 */
#define CME_ERR_CREATE_SIGNDATA_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 18u)))

/**
 * @ingroup cme_errcode
 * 创建签名者信息失败
 */
#define CME_ERR_CREATE_SIGNER_INFO_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 19u)))

/**
 * @ingroup cme_errcode
 * 创建简易数据失败
 */
#define CME_ERR_CREATE_SIMPLEDATA_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 20u)))

/**
 * @ingroup cme_errcode
 * 数据拷贝失败
 */
#define CME_ERR_DATA_COPY_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 21u)))

/**
 * @ingroup cme_errcode
 * 数据无效
 */
#define CME_ERR_DATA_NOT_AVAILABLE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 22u)))

/**
 * @ingroup cme_errcode
 * 解码Base64失败
 */
#define CME_ERR_DECODE_BASE64_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 23u)))

/**
 * @ingroup cme_errcode
 * 解码失败
 */
#define CME_ERR_DECODE_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 24u)))

/**
 * @ingroup cme_errcode
 * 解密失败
 */
#define CME_ERR_DECRYPT_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 25u)))

/**
 * @ingroup cme_errcode
 * 摘要验证失败
 */
#define CME_ERR_DIGEST_VERIFY_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 26u)))

/**
 * @ingroup cme_errcode
 * 扩展重复
 */
#define CME_ERR_DUPLICATE_EXTN (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 27u)))

/**
 * @ingroup cme_errcode
 * Base64编码失败
 */
#define CME_ERR_ENCODE_BASE64_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 28u)))

/**
 * @ingroup cme_errcode
 * 编码失败
 */
#define CME_ERR_ENCODE_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 29u)))

/**
 * @ingroup cme_errcode
 * 加密失败
 */
#define CME_ERR_ENCRYPT_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 30u)))

/**
 * @ingroup cme_errcode
 * 扩展无效
 */
#define CME_ERR_EXTN_NOT_AVAILABLE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 31u)))

/**
 * @ingroup cme_errcode
 * 打开文件失败
 */
#define CME_ERR_FILE_OPEN_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 32u)))

/**
 * @ingroup cme_errcode
 * 读文件失败
 */
#define CME_ERR_FILE_READ_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 33u)))

/**
 * @ingroup cme_errcode
 * 写文件失败
 */
#define CME_ERR_FILE_WRITE_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 34u)))

/**
 * @ingroup cme_errcode
 * 获取OID失败
 */
#define CME_ERR_GET_OID_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 35u)))

/**
 * @ingroup cme_errcode
 * 计算hash失败
 */
#define CME_ERR_HASH_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 36u)))

/**
 * @ingroup cme_errcode
 * 无效的算法ID
 */
#define CME_ERR_INVALID_ALGID (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 37u)))

/**
 * @ingroup cme_errcode
 * 无效的参数
 */
#define CME_ERR_INVALID_ARG (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 38u)))

/**
 * @ingroup cme_errcode
 * 无效的属性
 */
#define CME_ERR_INVALID_ATTR (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 39u)))

/**
 * @ingroup cme_errcode
 * 无效的Base64
 */
#define CME_ERR_INVALID_BASE64 (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 40u)))

/**
 * @ingroup cme_errcode
 * 无效的CID
 */
#define CME_ERR_INVALID_CID (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 41u)))

/**
 * @ingroup cme_errcode
 * 无效的数据长度
 */
#define CME_ERR_INVALID_DATA_LEN (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 42u)))

/**
 * @ingroup cme_errcode
 * 无效的日期或时间
 */
#define CME_ERR_INVALID_DATETIME (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 43u)))

/**
 * @ingroup cme_errcode
 * 无效的扩展
 */
#define CME_ERR_INVALID_EXTN (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 44u)))

/**
 * @ingroup cme_errcode
 * 无效的文件大小
 */
#define CME_ERR_INVALID_FILE_SIZE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 45u)))

/**
 * @ingroup cme_errcode
 * 无效的颁发者名称
 */
#define CME_ERR_INVALID_ISSUERNAME (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 46u)))

/**
 * @ingroup cme_errcode
 * 无效的IV
 */
#define CME_ERR_INVALID_IV (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 47u)))

/**
 * @ingroup cme_errcode
 * IV长度非法
 */
#define CME_ERR_INVALID_IV_LEN (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 48u)))

/**
 * @ingroup cme_errcode
 * key长度非法
 */
#define CME_ERR_INVALID_KEY_LEN (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 49u)))

/**
 * @ingroup cme_errcode
 * 无效的MAC
 */
#define CME_ERR_INVALID_MAC (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 50u)))

/**
 * @ingroup cme_errcode
 * 无效的模式
 */
#define CME_ERR_INVALID_MODE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 51u)))

/**
 * @ingroup cme_errcode
 * 无效的OID
 */
#define CME_ERR_INVALID_OID (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 52u)))

/**
 * @ingroup cme_errcode
 * 可打印字符串无效
 */
#define CME_ERR_INVALID_PRINTABLE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 53u)))

/**
 * @ingroup cme_errcode
 * 盐值长度非法
 */
#define CME_ERR_INVALID_SALT_LEN (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 54u)))

/**
 * @ingroup cme_errcode
 * 序列号无效
 */
#define CME_ERR_INVALID_SERIALNUMBER (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 55u)))

/**
 * @ingroup cme_errcode
 * 版本无效
 */
#define CME_ERR_INVALID_VERSION (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 56u)))

/**
 * @ingroup cme_errcode
 * 颁发者名称和序列号不匹配
 */
#define CME_ERR_ISSUER_MISMATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 57u)))

/**
 * @ingroup cme_errcode
 * 密钥派生失败
 */
#define CME_ERR_KEY_DERIV_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 58u)))

/**
 * @ingroup cme_errcode
 * 密钥生成失败
 */
#define CME_ERR_KEY_GEN_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 59u)))

/**
 * @ingroup cme_errcode
 * 公钥不匹配
 */
#define CME_ERR_KEY_MISMATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 60u)))

/**
 * @ingroup cme_errcode
 * 非对称密钥对不匹配
 */
#define CME_ERR_KEYPAIR_MISMATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 61u)))

/**
 * @ingroup cme_errcode
 * 名称不匹配
 */
#define CME_ERR_NAME_MISMATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 62u)))

/**
 * @ingroup cme_errcode
 * Subject Name无效
 */
#define CME_ERR_NAME_NOT_AVAILABLE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 63u)))

/**
 * @ingroup cme_errcode
 * Nonce不匹配
 */
#define CME_ERR_NONCE_MISMATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 64u)))

/**
 * @ingroup cme_errcode
 * OID不匹配
 */
#define CME_ERR_OID_MISMATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 65u)))

/**
 * @ingroup cme_errcode
 * 打开封装数据失败
 */
#define CME_ERR_OPEN_ENVELOP_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 66u)))

/**
 * @ingroup cme_errcode
 * 请求响应不匹配
 */
#define CME_ERR_REQ_RESP_MISMATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 67u)))

/**
 * @ingroup cme_errcode
 * 签名验证失败
 */
#define CME_ERR_SIGN_VERIFY_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 68u)))

/**
 * @ingroup cme_errcode
 * 签名失败
 */
#define CME_ERR_SIGNING_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 69u)))

/**
 * @ingroup cme_errcode
 * 字符串不匹配
 */
#define CME_ERR_STRING_MISMATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 70u)))

/**
 * @ingroup cme_errcode
 * malloc失败
 */
#define CME_ERR_MALLOC_FAIL (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 71u)))

/**
 * @ingroup cme_errcode
 * 访问空指针
 */
#define CME_ERR_NULL_PTR (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 72u)))

/**
 * @ingroup cme_errcode
 * 验证失败
 */
#define CME_ERR_VERIFY_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 73u)))

/**
 * @ingroup cme_errcode
 * LIB未初始化
 */
#define CME_ERR_INITLIB (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 74u)))

/**
 * @ingroup cme_errcode
 * 无效的私钥
 */
#define CME_ERR_INVALID_PRIVATE_KEY (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 75u)))

/**
 * @ingroup cme_errcode
 * 无效的私有OID类型代码
 */
#define CME_ERR_INVALID_PRV_OID_TYPE_CODE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 76u)))

/**
 * @ingroup cme_errcode
 * 输入中存在重复的OID
 */
#define CME_ERR_DUPLICATE_PRV_OID (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 77u)))

/**
 * @ingroup cme_errcode
 * 私有OID名称的长度无效
 */
#define CME_ERR_INVALID_PRV_OID_NAME_LENGTH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 78u)))

/**
 * @ingroup cme_errcode
 * 无效的list
 */
#define CME_ERR_INVALID_LIST (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 79u)))

/**
 * @ingroup cme_errcode
 * list中存在重复数据
 */
#define CME_ERR_DUPLICATE_LIST_DATA (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 81u)))

/**
 * @ingroup cme_errcode
 * list中存在多余的节点
 */
#define CME_ERR_EXCESS_NODES (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 82u)))

/**
 * @ingroup cme_errcode
 * 无效的list索引
 */
#define CME_ERR_INVALID_LIST_INDEX (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 84u)))

/**
 * @ingroup cme_errcode
 * 无效的list位置
 */
#define CME_ERR_INVALID_LIST_POSITION (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 85u)))

/**
 * @ingroup cme_errcode
 * 操作list失败
 */
#define CME_ERR_LIST_OPERATION_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 86u)))

/**
 * @ingroup cme_errcode
 * list对比失败
 */
#define CME_ERR_LIST_CMP_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 87u)))

/**
 * @ingroup cme_errcode
 * list不匹配
 */
#define CME_ERR_LIST_NO_MATCH (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 88u)))

/**
 * @ingroup cme_errcode
 * 未到有效时间
 */
#define CME_ERR_BEFORE_VALID_DATETIME (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 89u)))

/**
 * @ingroup cme_errcode
 * 超过有效时间
 */
#define CME_ERR_AFTER_VALID_DATETIME (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 90u)))

/**
 * @ingroup cme_errcode
 * pop创建失败
 */
#define CME_ERR_POP_CREATE_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_ENO_MODULE_BASE, 91u)))


/* error codes of x509 modules */
/**
 * @ingroup cme_errcode
 * X509起始错误码
 */
#define CME_X509_ERR_BASE 11u

/**
 * @ingroup cme_errcode
 * SN号不相等
 */
#define CME_X509_ERR_SN_NOT_EQUAL (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_X509_ERR_BASE, 1u)))

/* error codes of pkcs modules */
/**
 * @ingroup cme_errcode
 * PKCS10起始错误码
 */
#define CME_PKCS10_ERR_BASE 12u

/**
 * @ingroup cme_errcode
 * 产生公钥失败
 */
#define CME_PKCS10_ERR_CREATE_PUBKEY_FAILED (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_PKCS10_ERR_BASE, 1u)))

/* error codes of ocsp modules */
/**
 * @ingroup cme_errcode
 * OCSP起始错误码
 */
#define CME_OCSP_ERR_BASE 13u

/**
 * @ingroup cme_errcode
 * 无效的状态请求
 */
#define CME_OCSP_ERR_INVALID_REQUEST (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_OCSP_ERR_BASE, 1u)))

/**
 * @ingroup cme_errcode
 * 无效的应答
 */
#define CME_OCSP_ERR_INVALID_RESPONSE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_OCSP_ERR_BASE, 2u)))

/**
 * @ingroup cme_errcode
 * 状态请求已存在
 */
#define CME_OCSP_ERR_REQ_EXISTS (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_OCSP_ERR_BASE, 3u)))

/**
 * @ingroup cme_errcode
 * 应答已存在
 */
#define CME_OCSP_ERR_RESP_EXISTS (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_OCSP_ERR_BASE, 4u)))

/* error codes of pem modules */
/**
 * @ingroup cme_errcode
 * PEM起始错误码
 */
#define CME_PEM_ERR_BASE 14u

/**
 * @ingroup cme_errcode
 * 无效的对象类型
 */
#define CME_PEM_ERR_INVALID_OBJECT_TYPE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_PEM_ERR_BASE, 1u)))

/**
 * @ingroup cme_errcode
 * 无效的PEM进程类型
 */
#define CME_PEM_ERR_INVALID_PROC_TYPE (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_PEM_ERR_BASE, 2u)))

/**
 * @ingroup cme_errcode
 * 无效的PEM格式
 */
#define CME_PEM_ERR_INVALID_PEM_FORMAT (-((int32_t)BSL_ENO_MAKE(CME_ENO_MID, CME_PEM_ERR_BASE, 3u)))

/**
 * @ingroup cme_errcode
 * @brief 初始化errcode模块。
 *
 * @par 描述：
 * 初始化errcode模块。
 *
 * @param void.
 *
 * @retval CME_SUCCESS 成功。
 * @retval SAL_ERR_BAD_PARAM 失败。
 */
int32_t CME_InitErrorCode(void);

/**
 * @ingroup cme_errcode
 * @brief 生成错误码函数封装。
 *
 * @par 描述：
 * 无须直接调用，一般通过宏CME_GenError调用。
 *
 * @param errorCode [IN] 错误值码。
 * @param fileName [IN] 错误产生的文件名。
 * @param lineNo [IN] 错误产生的行号。
 *
 * @retval int32_t 错误码。
 *
 * @see CME_GenError
 */
int32_t CME_GenErrorWrapper(uint32_t errorCode, const char *fileName, int32_t lineNo);

/**
 * @ingroup cme_errcode
 * @brief 推错误码入栈函数封装。
 *
 * @par 描述：
 * 无须直接调用，一般通过宏CME_PushError调用。
 *
 * @param errorCode [IN] 错误码。
 * @param fileName [IN] 错误产生的文件名。
 * @param lineNo [IN] 错误产生的行号。
 *
 * @retval void.
 *
 * @see CME_PushError
 */
void CME_PushErrorWrapper(uint32_t errorCode, const char *fileName, int32_t lineNo);

/**
 * @ingroup cme_errcode
 * @brief 生成错误码的宏。
 *
 * @param e_ [IN] 错误值码。
 *
 * @retval int32_t 错误码。
 *
 * @see CME_GenErrorWrapper
 */
#define CME_GenError(e_) CME_GenErrorWrapper(CME_GetError(e_), __FILENAME__, __LINE__)

/**
 * @ingroup cme_errcode
 * @brief 获取错误码的值。
 *
 * @param error [IN] 错误码。
 *
 * @retval uint32_t 错误码。
 *
 * @see CME_GenErrorWrapper
 */
uint32_t CME_GetError(int32_t error);

/**
 * @ingroup cme_errcode
 * @brief 推错误码入栈的宏。
 *
 * @param e_ [IN] 错误码
 *
 * @retval void.
 *
 * @see CME_PushErrorWrapper
 */
#define CME_PushError(e_) CME_PushErrorWrapper(CME_GetError(e_), __FILENAME__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif
