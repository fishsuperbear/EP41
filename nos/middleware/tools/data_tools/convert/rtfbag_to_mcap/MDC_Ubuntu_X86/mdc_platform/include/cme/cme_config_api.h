/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 统一管理CME的相关配置，如堆栈大小，是否支持多线程等。
 * Create: 2020/10/26
 * Notes:
 * History:
 * 2020/10/26 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_common CME通用接口
 *  @ingroup cme
 */
/** @defgroup cme_config CME_Config_API
 * @ingroup cme_common
 */
#ifndef CME_CONFIG_API_H
#define CME_CONFIG_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_config
 * @brief   设置调用CME_ReadDERCodeFromFile接口时文件的最大大小, 默认为256MB
 * @param   maxFileSize [IN] 文件最大大小
 * @return  设置结果 CME_SUCCESS/CME_ERR_BAD_PARAM
 */
int32_t CME_MaxReadFileSizeSet(size_t maxFileSize);

/**
 * @ingroup cme_config
 * @brief   设置最大栈深度, 默认值为100
 * @param   maxDepth [IN] 最大栈深度
 * @return  设置结果 CME_SUCCESS/CME_ERR_BAD_PARAM
 */
int32_t CME_MaxStackDepthSet(size_t maxDepth);

/**
 * @ingroup cme_config
 * @brief   设置ASN Buff最大大小, 默认值为500MB
 * @param   buffSize [IN] buff大小
 * @return  设置结果 CME_SUCCESS/CME_ERR_BAD_PARAM
 */
int32_t CME_MaxASNBuffSizeSet(size_t buffSize);

/**
 * @ingroup cme_config
 * @brief   设置PSE Buff最大大小, 默认值为10MB
 * @param   buffSize [IN] buff大小
 * @return  设置结果 CME_SUCCESS/CME_ERR_BAD_PARAM
 */
int32_t CME_MaxPSEBuffSizeSet(size_t buffSize);

/**
 * @ingroup cme_config
 * @brief 设置或重置证书签名AlgorithmIdentifier检查。
 * @par 目的：
 * 用于设置或重置TBS内部和证书签名AlgorithmIdentifier检查。\n
 * 默认情况下启用检查。
 * @par 描述：
 * 设置或重置TBS内部和证书签名中AlgorithmIdentifier的匹配检查。
 *
 * @param enable [IN] 为1使能检查，为0禁能检查。
 *
 * @retval void.
 *
 * @par 依赖：
 * x509.h.
 * @par 注释：
 * 主线程使用。\n
 * 应用程序的开始时作为init的一部分调用。
 */
void CME_CertAlgIdCheckSet(bool enable);

/**
 * @ingroup cme_config
 * @brief 设置或重置强制CA“基本约束参数”扩展校验。
 * @par 目的：
 * 设置或重置强制CA“基本约束参数”扩展校验。对于V3证书，“基本约束”参数为必选参数。\n
 * 默认情况下启用此检查。如果“基本约束”字段不存在，则加载的证书会失败。\n
 * 如果禁用此检查，则不会对CA证书验证“基本约束”参数。
 *
 * @par 描述：
 * 用于设置或重置CA证书的必选CA“基本约束”扩展校验。
 *
 * @param enable [IN] 为1使能检查，为0禁能检查。
 *
 * @retval void.
 *
 * @par 依赖：
 * x509.h.
 *
 * @par 注释：
 * 如果启用了检查，要求证书中设置“基本约束”扩展名，默认启用。\n
 * 此API应该在主线程中使用，在库初始化后立即调用。
 */
void CME_CABasicConstraintCheckSet(bool enable);

/**
 * @ingroup cme_config
 * @brief 设置或重置密钥用法扩展检查。
 * @par 目的：
 * 用于设置或重置密钥用法扩展检查。
 *
 * @par 描述：
 * 用于设置或重置证书的密钥用法扩展检查。
 *
 * @param enable [IN] 为1使能检查，为0禁能检查。
 *
 * @retval void.
 *
 * @par 注释：
 * 如果启用了该检查，则将检查密钥用法扩展。\n
 * 如果证书中存在此扩展名，该设置是该应用程序的PKI的全局设置。\n
 * 这是一个配置API，默认值是disable。\n
 * 这不是线程安全API。它应该作为init的一部分在开头调用。
 *
 * @see CME_CAKeyUsageCheckSet
 */
void CME_KeyUsageCheckSet(bool enable);

/**
 * @ingroup cme_config
 * @brief 设置或重置强制密钥用法扩展检查。
 * @par 目的：
 * 设置或重置强制密钥用法扩展检查。
 *
 * @par 描述：
 * 设置或重置CA证书的强制密钥用法扩展检查。\n
 * 不是线程安全API。应该在应用程序的初始阶段调用。
 *
 * @param enable [IN] 为1使能检查，为0禁能检查。
 *
 * @retval void.
 */
void CME_CAKeyUsageCheckSet(bool enable);

#ifndef CME_PKI_NO_KEYCHECK_FOR_V1CERT

/**
 * @ingroup cme_config
 * @brief 使用v1证书时，启用密钥用法扩展检查。
 * @par 目的：
 * 使用v1证书时，启用密钥用法扩展检查。\n
 * 在应用程序没有使用CME_PKI_SetCAKeyUsageMandatoryCheck禁用密钥用法扩展检查的情况下，
 * 该接口可以通过启用密钥用法扩展检查来拒绝V1证书。
 *
 * @par 描述：
 * 用于设置或重置V1证书的密钥用法扩展检查。
 * 默认情况下，该检查将关闭。这不是线程安全API。它应该作为init的一部分在开头调用。
 *
 * @param enable [IN] 为1使能检查，为0禁能检查。
 *
 * @retval void.
 */
void CME_KeyUsageCheckForV1CertSet(bool enable);

#endif // CME_PKI_NO_KEYCHECK_FOR_V1CERT

#ifdef __cplusplus
}
#endif

#endif // CME_CONFIG_API_H
