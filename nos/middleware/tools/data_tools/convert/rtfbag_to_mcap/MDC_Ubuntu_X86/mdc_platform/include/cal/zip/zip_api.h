/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: V2X PKI unzip regist api
 * Create: 2020/12/28
 */

/** @defgroup adaptor adaptor */
/** @defgroup unzip unzip
 * @ingroup adaptor
 */

#ifndef CME_ZIP_API_H
#define CME_ZIP_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup unzip
 * UnzipFileInfo 结构体 用于存储解压后文件数据及文件名
 */
typedef struct {
    char *fileName;
    uint8_t *fileBuf;
    size_t fileBufLen;
} UnzipFileInfo;

/**
 * @ingroup unzip
 * UnzipFiles 结构体 用于存储解压后文件及文件个数
 */
typedef struct {
    UnzipFileInfo *fileInfos;
    size_t fileNum;
} UnzipFiles;

/**
 * @ingroup unzip
 * UnzipFilesHandle 解压后数据上下文
 */
typedef UnzipFiles *UnzipFilesHandle;

#ifdef __cplusplus
}
#endif

#endif // CME_ZIP_API_H
