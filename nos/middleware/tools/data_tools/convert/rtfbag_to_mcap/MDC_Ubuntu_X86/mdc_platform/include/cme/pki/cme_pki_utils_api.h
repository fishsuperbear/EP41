/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:
 * Create: 2020/11/10
 * History:
 * 2020/8/15 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509_pki CME_X509_PKI_API
 *  @ingroup cme
 */
/** @defgroup pki_utils PKI_UTILS_API
 * @ingroup cme_x509_pki
 */
#ifndef CME_PKI_UTILS_API_H
#define CME_PKI_UTILS_API_H

#include "x509/cme_x509_api.h"
#include "cme_cid_api.h"
#include "cme_pki_def_api.h"
#include "pem/cme_pem_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup pki_utils
 * @brief 从输入中提供的证书列表中获取输入证书的证书链。
 *
 * @par 描述：
 * CME_PKI_GetCertChainByCert函数从输入中提供的证书列表中形成输入证书的证书链，
 * 并在输出列表中返回对应的链。
 *
 * @param [IN] 输入要形成证书链的证书列表。
 * certList是一个包含X509Cert类型的元素的列表。
 * @param [IN] 输入的证书。
 * @param [OUT]  输出证书链。
 * outCertChain是一个包含X509Cert[NA/NA]类型的元素的列表
 *
 * @retval int32_t On successful execution. [CME_PKI_SUCCESS|NA]
 * @retval int32_t On all failure conditions. [CME_PKI_ERROR|NA]
 *
 * @par 内存操作：
 * 如果成功，该函数将为outCertChain分配内存（列表可能包含一个或多个条目），
 * 要释放内存应用程序必须调用LIST_FREE，使用free函数为X509_FreeCert。
 *
 * @attention
 * @li 输出证书链中的第一个元素是输入证书。
 * @li 可为输入证书构建的可能证书链。如果输入证书列表中未找到输入证书的颁发者证书，
 * 则在输出链中将考虑输入证书列表。因此，在输出中获得的链条为输入证书，可以不完整。
 * @li 构建证书链时，只做基本的验证。
 */
int32_t CME_PKI_GetCertChainByCert(ListRoHandle certList, X509Cert *cert, ListHandle *outCertChain);

/**
 * @ingroup pki_utils
 * @brief 从输入中提供的证书列表中排序证书。
 *
 * @par 描述：找到第一个end entity的证书，并在输入的证书列表中排序对应的证书链
 *
 * @param [IN] 输入要形成证书链的证书列表。
 * @param outCertChain [OUT] 一个包含X509Cert[NA/NA]类型的元素的列表。
 *
 * @retval int32_t On successful execution. [CME_PKI_SUCCESS|NA]
 * @retval int32_t On all failure conditions. [CME_PKI_ERROR|NA]
 *
 * @par 内存操作：
 * 如果成功，该函数将为outCertChain分配内存（列表可能包含一个或多个条目），
 * 要释放内存应用程序必须调用LIST_FREE，使用free函数为X509_FreeCert。
 *
 * @attention
 * @li 输出证书链中的第一个元素是end entity。
 * @li 可为输入证书构建的可能证书链。如果输入证书列表中未找到输入证书的颁发者证书，
 * 则在输出链中将考虑输入证书列表。因此，在输出中获得的链条为输入证书，可以不完整。
 * @li 构建证书链时，只做基本的验证。
 */
int32_t CME_PKI_SortCertChain(ListRoHandle pstCertChain, ListHandle *pstOutChain);

/**
 * @ingroup pki_utils
 * @brief 释放证书链列表。
 *
 * @par Description
 * CME_PKI_FreeCertChainList函数用于释放使用CmeList形成的列表，
 * 该列表包含CmeList类型的元素，而CmeList类型的元素又包含X509Cert类型的元素。
 *
 * @param certChainList [IN] 包含CmeList类型的元素的输入列表，
 * CmeList类型的元素又包含X509Cert类型的元素。
 *
 * @retval void 不返回任何值。
 */
void CME_PKI_FreeCertChainList(ListHandle certChainList);

#ifdef __cplusplus
}
#endif

#endif /* __CME_PKI_UTILS_API_H__ */
