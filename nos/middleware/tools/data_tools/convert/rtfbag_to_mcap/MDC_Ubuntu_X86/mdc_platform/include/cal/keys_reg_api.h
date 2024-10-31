/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:秘钥管理功能对外接口
 * Create: 2020/05/08
 * History:
 */
#ifndef ADAPTOR_KEYS_REG_API_H
#define ADAPTOR_KEYS_REG_API_H

#include <stdint.h>
#include <stddef.h>
#include "keys/keys_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup adaptor
 * @brief 导入密钥的钩子。
 *
 * @param keyType [IN] 密钥类型
 * @param keyBuf [IN] 密钥 buffer
 * @param bufLen [IN] 密钥 buffer 长度
 * @param keyHandle [OUT] 密钥句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*KeysKeyImportFunc)(KeysKeyType keyType, const uint8_t *keyBuf, size_t bufLen,
                                     KeysKeyHandle *keyHandle);

/**
 * @ingroup adaptor
 * @brief 导出密钥的钩子。
 *
 * @param key [IN] 密钥句柄
 * @param keyData [OUT] 密钥数据
 * @param size [IN/OUT] 入参：buffer 大小，出参：密钥数据长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*KeysKeyExportFunc)(KeysKeyRoHandle key, uint8_t *keyData, size_t *size);

/**
* @ingroup adaptor
* @brief 生成密钥函数，用于直接在安全存储区生成密钥，而不需要外部导入
*
* @param keyType [IN] 密钥类型
* @param keyLen [IN] 密钥数据长度
* @param id [IN] 密钥id
* @param idLen [IN] id 长度
* @param keyHandle [OUT] 密钥句柄
*
* @retval int32_t, 由回调函数返回的错误码
*/
typedef int32_t (*KeysKeyGenerateFunc)(KeysKeyType keyType, size_t keyLen, const uint8_t *id, size_t idLen,
                                       KeysKeyHandle *keyHandle);

/**
 * @ingroup adaptor
 * @brief 获取密钥类型的钩子。
 *
 * @param key [IN] 密钥句柄
 * @param keyType [OUT] 密钥类型
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*KeysKeyGetTypeFunc)(KeysKeyRoHandle key, KeysKeyType *keyType);

/**
 * @ingroup adaptor
 * @brief 通过 ID 查找密钥的钩子。
 *
 * @param keyType [IN] 密钥类型
 * @param id [IN] 密钥id
 * @param idlen [IN] 密钥id 长度
 * @param handle [OUT] 密钥句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*KeysKeyLookupByIdFunc)(KeysKeyType keyType, const uint8_t *id, size_t idLen, KeysKeyHandle *handle);

/**
 * @ingroup adaptor
 * @brief 组装 PSK 预主秘钥的钩子
 *
 * @param pskHandle [IN] PSK密钥句柄
 * @param otherSecretHandle [IN] other secret 句柄
 * @param pmsHandle [OUT] 预主密钥句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*KeysKeyCombineFunc)(KeysKeyRoHandle pskHandle, KeysKeyRoHandle otherSecretHandle,
                                      KeysKeyHandle *pmsHandle);

/**
 * @ingroup adaptor
 * @brief 保存密钥到安全区域。
 *
 * @param keyType [IN] 密钥类型
 * @param id [OUT] 密钥id
 * @param idLen [OUT] 密钥id 长度
 * @param handle [IN]  密钥句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*KeysKeySaveByIdFunc)(KeysKeyType keyType, uint8_t *id, size_t *idLen, KeysKeyHandle handle);

/**
 * @ingroup adaptor
 * @brief 释放密钥的钩子。
 *
 * @param keyHandle [IN] 密钥句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*KeysKeyFreeFunc)(KeysKeyHandle keyHandle);

/**
 * @ingroup adaptor
 * KeysAdaptHandleFunc 结构体，密钥管理适配层功能钩子函数集
 */
typedef struct {
    /* Key management */
    KeysKeyImportFunc keyImportCb;         /* < 导入密钥的钩子 */
    KeysKeyExportFunc keyExportCb;         /* < 导出密钥的钩子 */
    KeysKeyGenerateFunc keyGenerateCb;     /* < 生成密钥的钩子 */
    KeysKeyGetTypeFunc keyGetTypeCb;       /* < 获取密钥类型的钩子 */
    KeysKeyLookupByIdFunc keyLookupByIdCb; /* < 通过 ID 查找密钥的钩子 */
    KeysKeyCombineFunc keyCombineCb;       /* < 组装PSK预主秘钥的钩子 */
    KeysKeySaveByIdFunc keySaveByIdCb;     /* < 通过 ID 保存密钥到安全区域的钩子 */
    KeysKeyFreeFunc keyFreeCb;             /* < 释放密钥的钩子 */
} KeysAdaptHandleFunc;

#ifdef __cplusplus
}
#endif

#endif /* ADAPTOR_KEYS_REG_API_H */
