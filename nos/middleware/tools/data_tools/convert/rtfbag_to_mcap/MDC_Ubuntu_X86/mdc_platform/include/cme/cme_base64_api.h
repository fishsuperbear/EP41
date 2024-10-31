/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: base64 模块头文件
 * Create: 2022-07-18
 */

/** @defgroup cme cme */
/** @defgroup cme_common CME通用接口
 *  @ingroup cme_base64
 */
/** @defgroup cme_base64 CME_BASE64_API
 * @ingroup cme_common
 */
#ifndef CME_BASE64_API_H
#define CME_BASE64_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * cme_base64
 * @brief 将指定的缓冲区做编码成base 64格式
 * @par 描述：函数将给定的 buf 转换成 base64 格式，编码成功的情况下，dstBuf返回的空间使用完后需要由用户调用 OSAL_Free 释放掉。
 * @param srcBuf         [IN] 传入的buff缓冲区。
 * @param srcBufLen      [IN] 传入的buff缓冲区长度。
 * @param dstBuf        [OUT] 传出的buff缓冲区。
 * @retval 成功情况下，返回 CME_SUCCESS，否则返回失败错误码。
 */
int32_t CME_Base64Encode(const uint8_t *srcBuf, const size_t srcBufLen, char **dstBuf);

/**
 * cme_base64
 * @brief 将指定的缓冲区解码码成 16 进制数据流
 * @par 描述：函数将给定的 base64 格式转换成数据流，解码成功的情况下，dstBuf返回的空间使用完后需要由用户调用 OSAL_Free 释放掉。
 * @param srcBuf         [IN] 传入的buff缓冲区。
 * @param dstBufLen     [OUT] 解码后得到的编码长度。
 * @param dstBuf      [OUT] 解码后得到的编码字符串。
 * @retval 成功情况下，返回BSL_OK，否则返回失败错误码。
 */
int32_t CME_Base64Decode(const char *srcBuf, uint8_t **dstBuf, size_t *dstBufLen);

#ifdef __cplusplus
}
#endif

#endif // CME_BASE64_API_H
