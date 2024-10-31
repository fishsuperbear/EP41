/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: ASN1功能接口定义
 * Create: 2020/8/20
 * Notes:
 * History:
 * 2020/8/20 第一次创建
 */
#ifndef ASN1_API_H
#define ASN1_API_H

#include "asn1_template_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup asn1_api ASN1编解码模块
 * @ingroup asn1
 */
/**
 * @ingroup asn1_api
 * @brief   DER编码格式的解码接口
 * @param   buf [IN] DER编码内容缓冲区。
 * @param   encLen [IN] DER编码内容缓冲区的长度。
 * @param   decLen [OUT] 已解码的长度。
 * @param   funcGenItem [IN] 指定解码模板项
 * @retval  解码的数据结构。
 */
void *ASN1_DerDecode(const uint8_t *buf, size_t encLen, size_t *decLen, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   DER编码格式的编码接口
 * @param   asn1Value [IN] 指定数据结构。
 * @param   encLen [OUT] 已编码的长度。
 * @param   funcGenItem [IN] 指定编码模板项
 * @retval  已编码的缓冲区
 */
uint8_t *ASN1_DerEncode(const void *asn1Value, size_t *encLen, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   OER编码格式的解码接口
 * @param   buf [IN] OER编码内容缓冲区。
 * @param   encLen [IN] OER编码内容缓冲区的长度。
 * @param   decLen [OUT] 已解码的长度。
 * @param   funcGenItem [IN] 指定解码模板项
 * @retval  解码的数据结构。
 */
void *ASN1_OerDecode(const uint8_t *buf, size_t encLen, size_t *decLen, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   OER编码格式的编码接口
 * @param   asn1Value [IN] 指定数据结构。
 * @param   encLen [OUT] 已编码的长度。
 * @param   funcGenItem [IN] 指定编码模板项
 * @retval  已编码的缓冲区
 */
uint8_t *ASN1_OerEncode(const void *asn1Value, size_t *encLen, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   OER编码格式的编码接口（指定缓冲区）
 * @param   asn1Value [IN] 指定数据结构。
 * @param   buf [IN] 指定内容缓冲区。
 * @param   bufLen [OUT] 指定内容缓冲区的长度。
 * @param   funcGenItem [IN] 指定编码模板项
 * @retval  已编码长度
 */
size_t ASN1_OerBufEncode(const void *asn1Value, uint8_t *buf, size_t bufLen, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   UPER编码格式的解码接口
 * @param   buf [IN] UPER编码内容缓冲区。
 * @param   encLen [IN] UPER编码内容缓冲区的长度。
 * @param   decLen [OUT] 已解码的长度。
 * @param   funcGenItem [IN] 指定解码模板项
 * @retval  解码的数据结构。
 */
void *ASN1_UperDecode(const uint8_t *buf, size_t encLen, size_t *decLen, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   UPER编码格式的编码接口
 * @param   asn1Value [IN] 指定数据结构。
 * @param   encLen [OUT] 已编码的长度。
 * @param   funcGenItem [IN] 指定编码模板项
 * @retval  已编码的缓冲区
 */
uint8_t *ASN1_UperEncode(const void *asn1Value, size_t *encLen, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   UPER编码格式的编码接口（指定缓冲区）
 * @param   asn1Value [IN] 指定数据结构。
 * @param   buf [IN] 指定内容缓冲区。
 * @param   bufLen [OUT] 指定内容缓冲区的长度。
 * @param   funcGenItem [IN] 指定编码模板项
 * @retval  已编码长度
 */
size_t ASN1_UperBufEncode(const void *asn1Value, uint8_t *buf, size_t bufLen, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   ASN1统一清空接口（不释放asn1Value本身）
 * @param   asn1Value [IN] 指定源数据结构。
 * @param   funcGenItem [IN] 指定编码模板项
 * @retval  void
 */
void ASN1_Clear(void *asn1Value, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   ASN1统一释放接口（释放asn1Value本身）
 * @param   asn1Value [IN] 指定源数据结构。
 * @param   funcGenItem [IN] 指定编码模板项，当该参数为空时，仅释放asn1Value本身。
 * @retval  void
 */
void ASN1_Free(void *asn1Value, Asn1GetItemFunc funcGenItem);

/**
 * @ingroup asn1_api
 * @brief   ASN1数据结构以人类友好的可读数据转储到指定缓冲区
 * @param   buf [IN] 指定内容缓冲区。
 * @param   bufLen [OUT] 指定内容缓冲区的长度。
 * @param   asn1Value [IN] 指定数据结构。
 * @param   funcGenItem [IN] 指定编码模板项
 * @retval  是否转储成功
 */
bool ASN1_Dump(char *buf, size_t bufLen, const void *asn1Value, Asn1GetItemFunc funcGenItem);

#ifdef __cplusplus
}
#endif

#endif
