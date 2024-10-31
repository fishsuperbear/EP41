/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 存储功能对外接口
 * Create: 2020/07/30
 * History:
 */
#ifndef ADAPTOR_STORE_REG_API_H
#define ADAPTOR_STORE_REG_API_H

#include <stdint.h>
#include <unistd.h>
#include "store/store_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup adaptor
 * @brief 打开文件的钩子。
 *
 * @param path [IN] 要打开的文件路径。
 * @param mode [IN] 模式。
 *
 * @retval StoreHandle 文件句柄。
 */
typedef StoreHandle (*StoreOpenFunc)(const char *path, const char *mode);

/**
 * @ingroup adaptor
 * @brief 关闭文件的钩子。
 *
 * @param handle [IN] 文件的句柄。
 *
 * @retval 无
 */
typedef int32_t (*StoreCloseFunc)(StoreHandle handle);

/**
 * @ingroup adaptor
 * @brief 读取文件数据的钩子。
 *
 * @param handle [IN]  文件的句柄。
 * @param buffer      [OUT] 存放数据的缓存。
 * @param size        [IN]  缓存的大小。
 *
 * @retval size_t 读取的数据长度
 */
typedef size_t (*StoreReadFunc)(StoreHandle handle, void *buffer, size_t size);

/**
 * @ingroup adaptor
 * @brief 写文件数据的钩子。
 *
 * @param handle [IN] 文件的句柄。
 * @param buffer      [IN] 待写入数据的缓存。
 * @param size        [IN] 待写入数据的的大小。
 *
 * @retval size_t 写入成功的数据长度
 */
typedef size_t (*StoreWriteFunc)(StoreHandle handle, const void *buffer, size_t size);

/**
 * @ingroup adaptor
 * @brief 获取文件数据长度的钩子。
 *
 * @param handle [IN] 文件的句柄。
 *
 * @retval ssize_t 文件数据长度
 */
typedef ssize_t (*StoreGetLengthFunc)(StoreHandle handle);

typedef struct {
    StoreOpenFunc storeOpenFuncCb;
    StoreCloseFunc storeCloseFuncCb;
    StoreReadFunc storeReadFuncCb;
    StoreWriteFunc storeWriteFuncCb;
    StoreGetLengthFunc storeGetLengthFuncCb;
} StoreAdaptorHandleFunc;

#ifdef __cplusplus
}
#endif

#endif /* ADAPTOR_STORE_REG_API_H */
