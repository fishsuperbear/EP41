/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: DER数据读写接口
 * Create: 2020/10/26
 * Notes:
 * History:
 * 2020/10/26 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_common CME通用接口
 *  @ingroup cme
 */
/** @defgroup cme_util CME_UTIL_API
 * @ingroup cme_common
 */
#ifndef CME_UTIL_API_H
#define CME_UTIL_API_H

#include <stdint.h>
#include <stddef.h>
#include "asn1_types_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef Asn1OctetString CmeBuf;
typedef CmeBuf *CmeBufHandle;

/**
 * @ingroup cme_util
 *
 * @brief   从文件中读取DER编码数据到指定buffer
 *
 * @param   derData [IN] 存储der数据的buffer
 * @param   fileName [IN] 文件名
 * @retval  size_t 成功读取的字节数
 */
size_t CME_FileRead(uint8_t **derData, const char *fileName);

/**
 * @ingroup cme_util
 *
 * @brief   将给定buffer中的DER编码数据写入文件中
 *
 * @param   derData [IN] 存储DER编码数据的buffer
 * @param   dataLen [IN] buffer大小
 * @param   fileName [IN] 待写入的文件
 * @retval  size_t 成功写入文件的字节数
 */
size_t CME_FileWrite(const uint8_t *derData, size_t dataLen, const char *fileName);

#ifdef __cplusplus
}
#endif

#endif // CME_UTIL_API_H
