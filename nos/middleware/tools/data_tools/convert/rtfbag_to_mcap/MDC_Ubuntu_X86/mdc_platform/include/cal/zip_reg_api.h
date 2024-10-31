/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: V2X PKI unzip regist api
 * Create: 2020/12/28
 */
#ifndef V2X_ZIP_REG_API_H
#define V2X_ZIP_REG_API_H

#include <stdint.h>
#include "zip/zip_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup adaptor
 * @brief 解压zip数据钩子。
 *
 * @param zipBuf [IN] zip压缩数据
 * @param zipBufLen [IN] zip压缩数据长度
 * @param filesHandle [OUT] 解压后数据句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*UnzipBufferFunc)(const uint8_t *zipBuf, size_t zipBufLen, UnzipFilesHandle *filesHandle);

/**
 * @ingroup adaptor
 * @brief 释放解压后数据句柄。
 *
 * @param filesHandle [IN] 解压后数据句柄
 *
 * @retval void
 */
typedef void (*UnzipFilesHandleFreeFunc)(UnzipFilesHandle filesHandle);

typedef struct {
    UnzipBufferFunc unzipBufferFunc;
    UnzipFilesHandleFreeFunc unzipFilesHandleFreeFunc;
} ZipAdaptorHandleFunc;

#ifdef __cplusplus
}
#endif

#endif // V2X_ZIP_REG_API_H