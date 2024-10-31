/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:key注册接口对外头文件
 * Create: 2020/6/18
 * History:
 */

/** @defgroup adaptor adaptor */
/**
 * @defgroup keys keys
 * @ingroup adaptor
 */

#ifndef KEYS_KEYS_API_H
#define KEYS_KEYS_API_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup keys
 * key 类型枚举
 */
typedef enum {
    KEYS_KEY_TYPE_PSK, /* < key类型：PSK */
    KEYS_KEY_TYPE_DH,  /* < key类型：DH */
    KEYS_KEY_TYPE_RAW, /* < 原始数据，只需存储缓冲区 */
    KEYS_KEY_TYPE_END, /* < 最大枚举值 */
} KeysKeyType;

/**
 * @ingroup keys
 * Keys Key 上下文
 */
typedef struct KeysKeyCtx_ *KeysKeyHandle;
typedef const struct KeysKeyCtx_ *KeysKeyRoHandle;

#ifdef __cplusplus
}
#endif

#endif // KEYS_KEYS_API_H
