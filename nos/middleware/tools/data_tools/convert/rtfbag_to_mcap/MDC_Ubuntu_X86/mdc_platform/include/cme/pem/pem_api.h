/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: PEM编解码接口
 * Create: 2022/07/18
 * Notes:
 * History:
 */

#ifndef PEM_API_H
#define PEM_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief PEM参数
 */
typedef struct {
    char *keyName;  /* < 参数名 */
    char *keyValue; /* < 参数值 */
} PemParameter;

/**
 * @brief PEM上下文句柄
 */
typedef struct PemCtx *PemCtxHandle;
typedef const struct PemCtx *PemCtxRoHandle;

/**
 * @brief PEM对象句柄
 */
typedef struct PemObj *PemObjHandle;
typedef const struct PemObj *PemObjRoHandle;

/**
 * @brief   设置PEM的数据最大长度，PEM字符串的默认最大长度为256KB,可设置的最大长度范围为(256KB,500MB)
 * @return  错误码
 */
int32_t PEM_MaxLengthSet(size_t maxLength);

/**
 * @brief   创建PEM对象
 * @param   type [IN] PEM类型字符串
 * @param   buf [IN] 内容数据缓冲区
 * @param   bufLen [IN] 内容数据的长度
 * @param   obj [OUT] PEM对象
 * @return  错误码
 */
int32_t PEM_ObjCreate(const char *type, const uint8_t *buf, size_t bufLen, PemObjHandle *obj);

/**
 * @brief   设置PEM对象的参数
 * @param   ctx [IN] PEM结构
 * @param   param [IN] 参数数组，如果为空指针，则忽略参数个数并返回成功
 * @param   paramCnt [IN] 参数数量，如果为0，则忽略参数并返回成功
 * @return  错误码
 */
int32_t PEM_ObjParamSet(PemObjHandle obj, const PemParameter *param, size_t paramCnt);

/**
 * @brief   创建PEM上下文
 * @param   obj [IN] PEM对象数组
 * @param   objCnt [IN] PEM对象数量
 * @param   ctx [OUT] PEM上下文
 * @return  错误码
 */
int32_t PEM_CtxCreate(PemObjHandle obj[], size_t objCnt, PemCtxHandle *ctx);

/**
 * @brief   编码PEM数据
 * @par     注意: PEM数据必须以-----BEGIN开始，-----END XXXX-----结尾
 * @param   ctx [IN] PEM结构
 * @param   encodeString [OUT] 编码后的字符串
 * @return  错误码
 */
int32_t PEM_CtxEncode(PemCtxHandle ctx, const char **encodeString);

/**
 * @brief   解码PEM数据
 * @par     注意: PEM数据必须以-----BEGIN开始，-----END XXXX-----结尾
 * @param   srcPem [IN] 源PEM数据
 * @param   ctx [OUT] PEM结构
 * @return  错误码
 */
int32_t PEM_CtxDecode(const char *srcPem, PemCtxHandle *ctx);

/**
 * @brief   获取PEM上下文的对象数量
 * @param   ctx [IN] PEM上线
 * @return  对象数量
 */
size_t PEM_ObjCntGet(PemCtxRoHandle ctx);

/**
 * @brief   获取指定索引的PEM对象
 * @param   ctx [IN] PEM上下文
 * @param   idx [IN] 对象索引
 * @return  对象
 */
PemObjRoHandle PEM_ObjGet(PemCtxRoHandle ctx, size_t idx);

/**
 * @brief   获取PEM对象的类型
 * @param   obj [IN] PEM对象
 * @return  PEM对象的类型
 */
const char *PEM_ObjTypeGet(PemObjRoHandle obj);

/**
 * @brief   获取PEM对象的参数数量
 * @param   obj [IN] PEM对象
 * @return  参数数量
 */
size_t PEM_ObjParamCntGet(PemObjRoHandle obj);

/**
 * @brief   获取PEM对象指定索引的参数
 * @param   obj [IN] PEM对象
 * @param   idx [IN] 参数索引
 * @return  参数
 */
const PemParameter *PEM_ObjParamGet(PemObjRoHandle obj, size_t idx);

/**
 * @brief   获取PEM对象的内容数据
 * @param   obj [IN] PEM对象
 * @param   bufLen [OUT] 内容数据的长度
 * @return  内容数据
 */
const uint8_t *PEM_ObjBufGet(PemObjRoHandle obj, size_t *bufLen);

/**
 * @brief   清空PEM对象
 * @param   obj [IN] PEM对象
 * @return  void
 */
void PEM_ObjFree(PemObjHandle obj);

/**
 * @brief   清空PEM上下文
 * @param   ctx [IN] PEM上下文
 * @return  void
 */
void PEM_CtxFree(PemCtxHandle ctx);

#ifdef __cplusplus
}
#endif

#endif // PEM_API_H