/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: ASN1功能辅助工具函数的接口
 * Create: 2020/8/17
 * Notes:
 * History:
 * 2020/8/17 第一次创建
 */

#ifndef ASN1_UTIL_API_H
#define ASN1_UTIL_API_H

#include <stdbool.h>
#include "asn1_types_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup asn1_util ASN1通用接口
 * @ingroup asn1
 */
/**
 * @ingroup asn1_util
 * @brief   复制OID内容
 * @param   dst [IN] 目标OID。
 * @param   src [IN] 源OID。
 * @retval  bool 是否成功
 */
bool ASN1_OidCopy(Asn1Oid *dst, const Asn1Oid *src);
/**
 * @ingroup asn1_util
 * @brief   拷贝OID对象
 * @param   src [IN] 源OID。
 * @retval  拷贝的OID对象
 */
Asn1Oid *ASN1_OidDup(const Asn1Oid *src);
/**
 * @ingroup asn1_util
 * @brief   创建新的BitString对象
 * @param   bitLength [IN] 位长度。
 * @retval  创建的BitString对象
 */
Asn1BitString *ASN1_BitStringNew(size_t bitLength);
/**
 * @ingroup asn1_util
 * @brief   根据数据缓冲区生成新的BitString对象
 * @param   data [IN] 数据缓冲区
 * @param   bitLength [IN] 位长度。
 * @retval  创建的BitString对象
 */
Asn1BitString *ASN1_BitStringGen(const uint8_t *data, size_t bitLength);
/**
 * @ingroup asn1_util
 * @brief   复制BitString内容
 * @param   dst [IN] 目标BitString对象。
 * @param   src [IN] 源BitString对象。
 * @retval  bool 是否成功
 */
bool ASN1_BitStringCopy(Asn1BitString *dst, const Asn1BitString *src);
/**
 * @ingroup asn1_util
 * @brief   拷贝BitString对象
 * @param   src [IN] 源BitString对象。
 * @retval  拷贝的BitString对象
 */
Asn1BitString *ASN1_BitStringDup(const Asn1BitString *src);
/**
 * @ingroup asn1_util
 * @brief   对比BitString对象
 * @param   dst [IN] 目标BitString对象。
 * @param   src [IN] 源BitString对象。
 * @retval  拷贝的BitString对象
 */
bool ASN1_BitStringCmp(const Asn1BitString *dst, const Asn1BitString *src);
/**
 * @ingroup asn1_util
 * @brief   拷贝BitString对象
 * @param   asn1Value [IN] 指定BitString对象。
 * @retval  void
 */
void ASN1_BitStringClear(Asn1BitString *asn1Value);
/**
 * @ingroup asn1_util
 * @brief   释放BitString对象
 * @param   src [IN] 源BitString对象。
 * @retval  void
 */
void ASN1_BitStringFree(Asn1BitString *asn1Value);
/**
 * @ingroup asn1_util
 * @brief   创建新的OctetString对象
 * @param   bitLength [IN] 位长度。
 * @retval  创建的OctetString对象
 */
Asn1OctetString *ASN1_OctetStringNew(size_t len);
/**
 * @ingroup asn1_util
 * @brief   根据数据缓冲区生成新的OctetString对象
 * @param   data [IN] 数据缓冲区
 * @param   bitLength [IN] 位长度。
 * @retval  创建的OctetString对象
 */
Asn1OctetString *ASN1_OctetStringGen(const uint8_t *data, size_t len);
/**
 * @ingroup asn1_util
 * @brief   复制OctetString内容
 * @param   dst [IN] 目标OctetString对象。
 * @param   src [IN] 源OctetString对象。
 * @retval  bool 服饰是否成功
 */
bool ASN1_OctetStringCopy(Asn1OctetString *dst, const Asn1OctetString *src);
/**
 * @ingroup asn1_util
 * @brief   拷贝OctetString对象
 * @param   src [IN] 源OctetString对象。
 * @retval  拷贝的OctetString对象
 */
Asn1OctetString *ASN1_OctetStringDup(const Asn1OctetString *src);
/**
 * @ingroup asn1_util
 * @brief   对比OctetString对象
 * @param   dst [IN] 目标OctetString对象。
 * @param   src [IN] 源OctetString对象。
 * @retval  对比是否一致
 */
bool ASN1_OctetStringCmp(const Asn1OctetString *dst, const Asn1OctetString *src);
/**
 * @ingroup asn1_util
 * @brief   对比OctetString对象与指定缓冲区
 * @param   dst [IN] 目标OctetString对象。
 * @param   data [IN] 缓冲区
 * @param   dataLen [IN] 缓冲区长度
 * @retval  bool 对比是否一致
 */
bool ASN1_OctetStringCmpWithBuf(const Asn1OctetString *dst, const uint8_t *data, size_t dataLen);
/**
 * @ingroup asn1_util
 * @brief   清空OctetString对象
 * @param   asn1Value [IN] 指定OctetString对象。
 * @retval  void
 */
void ASN1_OctetStringClear(Asn1OctetString *asn1Value);
/**
 * @ingroup asn1_util
 * @brief   释放OctetString对象
 * @param   src [IN] 源OctetString对象。
 * @retval  void
 */
void ASN1_OctetStringFree(Asn1OctetString *asn1Value);
/**
 * @ingroup asn1_util
 * @brief   设置ASN1模块中可分配的缓冲区的最大大小
 * @param   buffSize [IN] 可分配缓冲区的最大大小值。
 * @retval  bool 设置是否成功
 */
bool ASN1_MaxBuffSizeSet(size_t buffSize);
/**
 * @ingroup asn1_util
 * @brief   获取ASN1模块中可分配的缓冲区的最大大小
 * @retval  size_t 目前设置的最大缓冲区大小
 */
size_t ASN1_MaxBuffSizeGet(void);
/**
 * @ingroup asn1_util
 * @brief   设置ASN1模块中可编解码最大深度
 * @param   maxDepth [IN] 可可编解码最大深度。
 * @retval  bool 设置是否成功
 */
bool ASN1_MaxStackDepthSet(size_t maxDepth);
/**
 * @ingroup asn1_util
 * @brief   获取ASN1模块中可编解码的最大深度
 * @retval  size_t 目前设置的可编解码最大深度
 */
size_t ASN1_MaxStackDepthGet(void);
/**
 * @ingroup asn1_util
 * @brief   检查OctetString是否为IA5字符集
 * @retval  true为IA5字符集，false为不是IA5字符集
 */
bool ASN1_IA5StringCheck(const Asn1OctetString *asn1Value);
/**
 * @ingroup asn1_util
 * @brief   检查OctetString是否为Printable字符集
 * @retval  true为Printable字符集，false为不是Printable字符集
 */
bool ASN1_PrintableStringCheck(const Asn1OctetString *asn1Value);
/**
 * @ingroup asn1_util
 * @brief   检查OctetString是否为Numeric字符集
 * @retval  true为Numeric字符集，false为不是Numeric字符集
 */
bool ASN1_NumericStringCheck(const Asn1OctetString *asn1Value);
/**
 * @ingroup asn1_util
 * @brief   检查OctetString是否为Visible字符集
 * @retval  true为Visible字符集，false为不是Visible字符集
 */
bool ASN1_VisibleStringCheck(const Asn1OctetString *asn1Value);
/**
 * @ingroup asn1_util
 * @brief   检查OctetString是否为UTF8字符集
 * @retval  true为UTF8字符集，false为不是UTF8字符集
 */
bool ASN1_UTF8StringCheck(const Asn1OctetString *asn1Value);

#ifdef __cplusplus
}
#endif

#endif // ASN1_UTIL_API_H
