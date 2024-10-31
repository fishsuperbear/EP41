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
/** @defgroup pki_error PKI_ERROR_API
 * @ingroup cme_x509_pki
 */
#ifndef CME_PKI_ERROR_API_H
#define CME_PKI_ERROR_API_H

#include <stdint.h>
#include <stdbool.h>
#include "cme_pki_errorfun_api.h"
#include "cme_pki_errordef_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup pki_error
 * @brief 错误信息加入错误信息栈。
 *
 * @param func [IN] 模块名称。
 * @param error [IN] 错误编号。
 * @param fileName [IN] 错误所在文件名称。
 * @param lineNo [IN] 行号。
 */
void CME_PKI_PushErrorWrapper(uint16_t func, uint16_t error, const char *fileName, int32_t lineNo);

/**
 * @ingroup pki_error
 * @brief 错误信息加入错误信息栈。
 */
#define CME_PKI_PUSH_ERR(func, error) \
    CME_PKI_PushErrorWrapper((uint16_t)(func), (uint16_t)(error), __FILENAME__, __LINE__)

/**
 * @ingroup pki_error
 * @brief 获取错误信息。
 *
 * @param errorCode 错误ID。
 * @retval uint16_t 返回的错误信息。
 */
uint16_t CME_PKI_GetErrorCode(int32_t errorCode);

/**
 * @ingroup pki_error
 * @brief 获取产生错误的模块。
 *
 * @param errorCode 错误ID。
 * @return uint16_t 返回产生错误的模块编号。
 */
uint16_t CME_PKI_GetErrorModule(int32_t errorCode);

/**
 * @ingroup pki_error
 * @brief 获取最近的一次错误信息。
 *
 * @retval int32_t 返回错误ID。
 */
int32_t CME_PKI_GetLastError(void);

/**
 * @ingroup pki_error
 * @brief 根据错误ID获取错误描述。
 *
 * @param errorCode 错误ID。
 * @return const char* 返回错误描述字符串。
 */
const char *CME_PKI_GetErrorStr(int32_t errorCode);

/**
 * @ingroup pki_error
 * @brief 清除错误信息。
 */
void CME_PKI_ClearError(void);

/**
 * @ingroup pki_error
 * @brief 强制清除错误信息。
 */
void CME_PKI_ClearErrorForce(void);

/**
 * @ingroup pki_error
 * @brief 此函数删除特定线程的错误栈或者删除所有错误栈。
 *
 * @param isRemoveAll 是否删除所有的错误栈。
 */
void CME_PKI_RemoveErrorStack(bool isRemoveAll);

/**
 * @ingroup pki_error
 * @brief 使能多线程模式。
 *
 * @return CME_SUCCESS 操作成功。
 */
int32_t CME_PKI_EnableMultiThread(void);

/**
 * @ingroup pki_error
 * @brief 清除多线程模式标志。
 */
void CME_PKI_CleanupThread(void);

/**
 * @ingroup pki_error
 * @brief 获取Verify结果描述。
 *
 * @param verifyResult [IN] Verify结果ID。
 * @return const char* 返回获取到的Verify描述。
 */
const char *CME_PKI_GetVerifyResultStr(CME_PKI_VerifyResult verifyResult);

#ifdef __cplusplus
}
#endif

#endif
