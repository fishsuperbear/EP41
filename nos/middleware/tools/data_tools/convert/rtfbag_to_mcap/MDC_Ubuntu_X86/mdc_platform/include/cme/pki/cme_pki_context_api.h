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
/** @defgroup pki_context PKI_CONTEXT_API
 * @ingroup cme_x509_pki
 */
#ifndef CME_PKI_CONTEXT_API_H
#define CME_PKI_CONTEXT_API_H

#include <stdint.h>
#include "cme_cid_api.h"
#include "cme_pki_def_api.h"
#include "pem/cme_pem_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup pki_context
 * @brief 创建新的PKI上下文对象。
 *
 * @par 描述：
 * CME_PKI_CtxNew创建新的CmePkiCtx对象。
 *
 * @retval CmePkiCtx* 如果创建新的CmePkiCtx上下文成功。
 * @retval NULL 创建新的CmePkiCtx上下文失败。
 *
 * @par 内存操作：
 * cme会为CmePkiCtx分配内存，并返回应用程序。
 * 要释放该内存，用户必须调用CME_PKI_CtxFree函数。
 */
CmePkiCtx *CME_PKI_CtxNew(void);

/**
 * @ingroup pki_context
 * @brief 设置上下文名称
 *
 * @par 描述：
 * CME_PKI_CtxSetName函数用于设置对象名称。
 * 如果输入的名称是NULL终止的，则用户应该传递包括NULL在内的名称的长度。
 *
 * @param [IN,OUT] 指向CmePkiCtx结构的指针。
 * @param [IN] 上下文名称。
 * @param [IN] 名称的长度。长度不能大于PKI_NAME_MAXLEN(256)。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 */
int32_t CME_PKI_CtxSetName(CmePkiCtx *ctx, const char *name, size_t len);

/**
 * @ingroup pki_context
 * @brief 从上下文中获取名称。
 *
 * @par 描述：
 * CME_PKI_CtxGetName用于获取上下文关联的名称。
 *
 * @param ctx [IN] 指向CmePkiCtx结构的指针。
 * @param name [IN] 指向上下文名称的缓冲区指针。
 * @param len [IN] 上下文名称缓冲区的最大长度。
 * @param nameLen [OUT] 获取到的上下文名称的实际长度。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @par 内存操作：
 * 应用程序应该为名称分配和释放内存。
 */
int32_t CME_PKI_CtxGetName(const CmePkiCtx *ctx, char *name, size_t len, size_t *nameLen);

/**
 * @ingroup pki_context
 * @brief 释放PKI上下文对象。
 *
 * @par 描述：
 * CME_PKI_CtxFree函数检查context的引用计数，删除指向ctx的CmePkiCtx对象，释放申请的内存。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 *
 * @retval void 不返回任何值。
 */
void CME_PKI_CtxFree(CmePkiCtx *ctx);

/**
 * @ingroup pki_context
 * @brief 此函数用于将CA证书从文件加载到PKI上下文的信任存储中。
 *
 * @par 描述：
 * CME_PKI_CtxLoadTrustCACertificateFile函数读取包含“PEM”、“PFX”或“DER”格式的证书文件，
 * 并将证书添加到信任库中。密码为可选输入，只有PFX格式的证书才需要密码。
 * 若要使此CA成为“CA请求列表”的一部分，用户需要将bAddToCAReqList传递为true。
 *
 * @param ctx [IN,OUT] 指向CmePkiCtx结构指针。
 * @param certInfo [IN] 指向CmePkiFileInfo结构的指针。
 * @param certType [IN] 证书类型。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 * 如果输入无效（空指针），则失败一般发生在文件无法打开或证书解码失败的情况。
 *
 * @attention
 * @li 同一个文件可以有一个终端实体证书、密钥文件和多个中间CA证书，
 * 在这种情况下，只加载CA证书。
 * @li 对于PFX文件类型，不支持零密码长度。
 * @li 如果文件类型为PFX，则PFX文件只能使用加密密码模式创建。
 * @li 对于PEM文件类型，我们将不支持MIC_CLEAR模式。
 * @li 对于PEM文件，默认数据大小限制为256KB。如果应用程序需要设置大小超过256KB，
 * 则需要调用API CME_PKI_SetPemMaxSize。
 */
int32_t CME_PKI_CtxLoadTrustCA(CmePkiCtx *ctx, const CmePkiFileInfo *certInfo, uint32_t certType);

/**
 * @ingroup pki_context
 * @brief 将CRL从文件加载到存储库中。
 *
 * @par 描述：
 * CME_PKI_CtxLoadCrlFile函数读取包含PEM或DER格式的CRL(Certificate Revocation List)的文件，
 * 并将CRL添加到上下文中。
 *
 * @param ctx [IN,OUT] 指向CmePkiCtx结构的指针。
 * @param crlInfo [IN] CRL信息。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 所有故障情况。如果输入非法（空指针），文件无法打开，或者CRL解码失败
 *
 * @attention
 * @li 建议在加载crl日期前对crl日期进行校验。
 * @li 不支持CRL的PEM列表。
 * @li 如果存储中已经存在相同的crl，则返回错误。
 * @li 对于PEM文件，默认数据大小限制为256KB。如果应用程序需要设置大小超过256KB，
 * 则需要调用API CME_PKI_SetPemMaxSize。
 */
int32_t CME_PKI_CtxLoadCrl(CmePkiCtx *ctx, const CmePkiFileInfo *crlInfo);

/**
 * @ingroup pki_context
 * @brief 将OCSP响应消息加载到上下文中。OCSP响应将被解码并存储在上下文存储中。
 *
 * @par 描述：
 * CME_PKI_CtxLoadOCSPRespBuffer函数加载Context中的OCSP响应。
 * OCSP响应在加载到上下文中之前不会进行验证。
 *
 * @param ctx [IN] 指向PKI上下文的指针。
 * @param ocspResp [IN] pcOCSPRespMsg OCSP Response buffer。
 * @param ocspRespLen [IN] OCSP响应缓冲区长度[NA/NA]
 * @param encodeForm [IN] iBuffType OCSP response buffer type [CME_PKI_ENCODE_FORM_ASN1/NA]
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @attention
 * @li OCSP响应被解码并存储在上下文存储中。
 * @li 不会加载状态不为OCSP_SUCCESSFUL的OCSP响应。
 * @li 如果用户尝试加载相同的OCSP响应，则API返回失败。
 */
int32_t CME_PKI_CtxLoadOCSPRespBuffer(CmePkiCtx *ctx, const uint8_t *ocspResp, size_t ocspRespLen,
                                      CmePkiEncodeForm encodeForm);

/**
* @ingroup pki_context
* @brief 该功能用于将证书和密钥从文件中加载到PKI上下文的默认证书和密钥中。
*
* @par 描述：
* CME_PKI_CtxLoadDfltLocalCertAndKeyFile函数读取包含PEM、PFX或DER格式的证书和密钥文件。
* 并将证书和密钥添加到PKI Context的默认证书和密钥中。证书密码是一个可选的输入，
* 但仅适用于PFX格式。密钥文件密码也是可选的。将此证书集url的URL关联到相应的URL，
* 但它是一个可选的输入。如果在上下文中设置了CME_PKI_OPTION_IGNORE_PVT_KEY选项，
* 则只加载localcertificate，不加载私钥。私钥参数将被完全忽略。
*
* @param ctx [IN,OUT] 指向CmePkiCtx结构指针。
* @param certInfo [IN] 需要加载证书的文件名。
* @param keyInfo [IN] 包含证书私钥的文件
* @param urlBuf [IN] 可以下载此证书的URL。
* @param certType [IN] 证书类型。
*
* @retval CME_PKI_SUCCESS 执行成功。
* @retval CME_PKI_ERROR 执行失败。
* 如果输入无效（空指针），则可能会失败。\n
* 文件无法打开，或证书解码。
* @attention
* 一个文件可以同时包含证书和密钥，在这种情况下，certInfo和keyInfo的输入应使用相同的文件名。\n
* 一个文件不能包含多个终端实体证书。\n
* 只支持一个url。\n
* 当文件有多个证书时，第一个证书被视为终端实体证书。\n
* 默认本地证书是一种特殊的本地证书，默认证书是首选证书。本地证书的所有搜索都从默认证书开始。\n
* PFX格式中使用的术语有：
* @li Bag - 内含证书/ crl /钥匙
* @li Baglist - Bag清单
* @li Authsafe - Baglists列表。
- @li PFX - 编码的authsafe
* @li Mac password - 用于对PFX中存在的编码数据进行完整性检查。
* @li Enc password - 用于加密baglist或密钥之前存储在bag。
* 证书只能通过加密baglist进行加密。但是，密钥可以通过对baglist进行加密，
* 也可以将加密后的密钥存储在bag中。
* 在PFX格式中，支持以下类型的文件：
* @li 同时包含证书和私钥的文件，其加密和mac密码相同。
* @li 证书和keyfile不同，mac和加密密码不同的文件。
* @li 有多个加密或未加密的Baglist的文件。
* 然后，从第一个加密的Baglist中提取第一个证书进行检查。\n
* 如果证书不在第一个加密的baglist中，则API返回失败。\n
* 如果一个文件有多个加密或未加密的baglist，则取第一个加密或未加密的baglist中的第一个私钥。
* 如果私钥不在第一个baglist中，则在其它baglist中搜索私钥。如果私钥不存在于任何baglist中，则API返回失败。\n
* 对于PFX文件类型，不支持零密码长度。\n
* 如果文件类型为PFX，则PFX文件只能使用加密密码模式创建。
* 对于PEM文件类型，我们将不支持MIC_CLEAR模式。
* 支持的密钥大小范围为512~4096bits。\n
* 对于PEM文件，默认数据大小限制为256KB。如果应用程序需要设置大小超过256KB，则需要调用API CME_PKI_SetPemMaxSize。\n
* 如果应用程序没有在频繁的间隔内设置种子和熵，则可能导致易受攻击的密钥或随机asn1Val。\n
* 建议在频繁间隔设置种子和熵值，或者更新基于DRBG的随机实现移植版本。\n
* 调用该接口前，请确保已经通过cme_CRYPT_rand_init初始化DRBG生成随机数。
* 如果应用程序不想使用DRBG，那么应用程序应该调用cme_CRYPT_enable_drbg(0)来禁用DRBG随机功能。
* DRBG默认使能。
*/
int32_t CME_PKI_CtxLoadLocalCertAndKey(CmePkiCtx *ctx, CmePkiFileInfo *certInfo, const CmePkiFileInfo *keyInfo,
                                       const Asn1OctetString *urlBuf, uint32_t certType);

/**
 * @ingroup pki_context
 * @brief 设置上下文中的时间。
 *
 * @par 描述：
 * CME_PKI_CtxSetTime函数设置上下文中的时间，该时间用于检查证书和CRL的有效期。
 *
 * @param ctx [IN,OUT] 指向CmePkiCtx结构指针。
 * @param  dateTime [IN] 设置的时间
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @attention
 * @li BslSysTime结构的ucUTCSign、ucUTCHour、ucUTCMinute和millSecond字段不会被我们的功能使用，
 * 因此不会考虑和验证这些字段。
 * @li 在windows上，如果输入时间可以使用mktime转换为日历asn1Val，则认为输入时间是有效的。
 * @li 对于Dopra,BslSysTime结构中的年份asn1Val字段取值范围为1970~2134。
 * @li 如果校验时没有使用setTime，则使用系统时间。
 */
int32_t CME_PKI_CtxSetTime(CmePkiCtx *ctx, const BslSysTime *dateTime);

/**
 * @ingroup pki_context
 * @brief 获取上下文中设置的时间。
 *
 * @par 描述：
 * CME_PKI_CtxGetTime函数返回本次上下文中设置的时间，用于检查证书和CRL有效期。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param dateTime [OUT] 上下文中设置的时间。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @attention
 * @li 如果没有设置时间，则函数将返回错误。
 * @li BslSysTime结构的ucUTCSign、ucUTCHour、ucUTCMinute和millSecond字段没有被我们的功能使用，
 * 因此，已经设置的时间检查将作为输出提供，而不重视这些字段是有效的。
 * @par 内存操作：
 * 该函数内部为时间结构分配内存，分配的内存可以使用cme_free函数释放。
 */
int32_t CME_PKI_CtxGetTime(const CmePkiCtx *ctx, BslSysTime **dateTime);

/**
 * @ingroup pki_context
 * @brief 设置上下文中的校验深度。
 *
 * @par 描述：
 * CME_PKI_CtxSetVerifyDepth函数设置上下文中的校验深度。
 * 必须执行证书验证，直到此深度。
 * 验证深度[depth] asn1Val应大于或等于0，否则API返回错误。
 *
 * @param ctx [IN,OUT] 指向CmePkiCtx结构指针。
 * @param depth [IN] Deep证书验证的深度。深度应在[0,65535]的范围内
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @attention
 * 如果未设置校验深度（上下文和对象中未设置Depth），
 * 调用CME_PKI_ObjValidateCertChain接口时，会考虑深度asn1Val为0。
 */
int32_t CME_PKI_CtxSetVerifyDepth(CmePkiCtx *ctx, int32_t depth);

/**
 * @ingroup pki_context
 * @brief 获取上下文的校验深度。
 *
 * @par 描述：
 * CME_PKI_CtxGetVerifyDepth函数返回上下文的校验深度。
 * 必须执行证书验证，直到此深度。
 * 如果在调用设置深度之前调用此API，则API返回错误。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param depth [OUT] Deep包含证书验证的深度。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 */
int32_t CME_PKI_CtxGetVerifyDepth(const CmePkiCtx *ctx, int32_t *depth);

/**
 * @ingroup pki_context
 * @brief 将中的verify参数设置为context。
 *
 * @par 描述：
 * CME_PKI_CtxSetVerifyParam函数将中的verify参数设置为上下文。传递的每个新标志与现有标志或成或。
 *
 * @param ctx [IN,OUT] 指向CmePkiCtx结构指针。
 * @param verifyFlags [IN] 校验需要设置的参数标志位[NA/NA]
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @attention
 * 设置CRL标志和OCSP标志导致接口失败。
 * 证书撤销列表的CRL标志(CME_PKI_CHECK_CRL | CME_PKI_CHECK_CRL_ALL |
 * CME_PKI_CHECK_DELTA_CRL | CME_ PKI_EXTENDED_CRL_SUPPORT | CME_PKI_OBJ_CRL_SUPPORT)
 * 与OCSP标志位(CME_PKI_CHECK_OCSP | CME_PKI_CHECK_OCSP_ALL | CME_PKI_OCSP_RESPONDER_CHECK_CRL |
 * CME_PKI_OCSP_RESPONDER_CHECK_DELTA_CRL | CME_PKI_OCSP_TRUST_RESPONDER_CERTS_IN_MSG |
 *  CME_PKI_OBJ_OCSP_SUPPORT)存在冲突。
 */
int32_t CME_PKI_CtxSetVerifyParam(CmePkiCtx *ctx, uint32_t verifyFlags);

/**
 * @ingroup pki_context
 * @brief 获取上下文中设置的verify参数。
 *
 * @par 描述：
 * CME_PKI_CtxGetVerifyParam函数返回上下文中设置的verify参数。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param verifyFlags [OUT] verifyFlags包含校验参数标志。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @attention
 * 如果没有设置标志，则函数将返回错误。
 */
int32_t CME_PKI_CtxGetVerifyParam(const CmePkiCtx *ctx, uint32_t *verifyFlags);

/**
 * @ingroup pki_context
 * @brief 清除上下文中设置的verify参数。
 *
 * @par 描述：
 * CME_PKI_CtxClearVerifyParam函数清除Context中设置的校验参数标志位。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param verifyFlags [IN] 校验需要重置的参数标志位。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 */
int32_t CME_PKI_CtxClearVerifyParam(CmePkiCtx *ctx, uint32_t verifyFlags);

/**
 * @ingroup pki_context
 * @brief 此函数用于使用上下文指针从CA请求列表中获取CA的哈希或编码主题名称或证书列表。
 *
 * @par 描述：
 * CME_PKI_CtxGetCACertReq函数根据入参类型enId返回CA Certificate(s)或CA Certificate(s)信息
 * （根据传入的Subject Name进行哈希/编码后的Subject Name）。在dpOutList中给出需要输出的列表。
 * 请求的数据将从CA证书的信任存储中提取，这些证书在加载CA证书时被指示为证书请求列表的一部分。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param enId [IN] 参数类型。
 * [CME_PKI_CAPUBLICKEY/CME_PKI_REQ_CA_SUBJECT/CME_PKI_REQ_CA_CERT/NA]
 * @param dpOutList [OUT] 包含输出列表的列表。
 * @li 如果CmePkiReqParam为CME_PKI_CAPUBLICKEY或CME_PKI_REQ_CA_SUBJECT，
 * 则dpOutList包含Asn1OctetString类型的元素。
 * @li 如果CmePkiReqParam为CME_PKI_REQ_CA_CERT，
 * 则dpOutList中包含X509Cert类型的元素。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行成功。
 *
 * @par 内存操作：
 * 该函数会为dpOutList分配内存（该列表可能包含一个或多个条目），
 * 为了释放该内存应用程序必须调用LIST_FREE，
 * 其中free函数为ASN1_FreeOctetString。
 * 如果CmePkiReqParam为CME_PKI_CAPUBLICKEY或CME_PKI_REQ_CA_SUBJECT，
 * 则使用释放函数X509_FreeCert，如果CmePkiReqParam为CME_PKI_REQ_CA_CERT，
 * 则使用释放函数X509_FreeCert。
 * @attention
 * 输出主题名称将以编码DER格式提供。\n
 * 输出哈希类型为SHA1。
 */
int32_t CME_PKI_CtxGetCACertReq(CmePkiCtx *ctx, CmePkiReqParam enId, ListHandle *dpOutList);

/**
 * @ingroup pki_context
 * @brief 获取所有CA证书。
 *
 * @par 描述：
 * CME_PKI_CtxGetAllTrustCA函数返回所有CA证书
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param outCertList [OUT] 包含所有证书的列表。outCertList是一个包含X509Cert[NA/NA]类型的元素的列表
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @par 内存操作：
 * cme会为outCertList分配内存（该列表可能包含一个或多个条目），
 * 要释放这个内存应用程序必须调用LIST_FREE，其free函数为X509_FreeCert。
 * @attention
 * 检索顺序可能与CA的加载顺序不同。
 */
int32_t CME_PKI_CtxGetAllTrustCA(CmePkiCtx *ctx, uint32_t certType, ListHandle *outCertList);

/**
 * @ingroup pki_context
 * @brief 此函数用于从存储区获取所有CRL。
 *
 * @par 描述：
 * CME_PKI_CtxGetAllCRL函数从存储区返回所有CRL。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param outCRLList [OUT] 包含所有CRL列表。
 * outCRLList是一个包含X509CRLCertList类型的元素的列表。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @par 内存操作：
 * cme会为outCRLList（该列表可能包含一个或多个条目）分配内存，
 * 以释放该内存应用程序，就像调用LIST_FREE一样，释放函数为X509CRL_Free。
 */
int32_t CME_PKI_CtxGetAllCRL(CmePkiCtx *ctx, ListHandle *outCRLList);

/**
 * @ingroup pki_context
 * @brief 此函数用于从传递的上下文的本地存储区获取所有证书。
 *
 * @par 描述：
 * CME_PKI_CtxGetAllLocalCert函数返回本地存储中的所有证书，默认本地证书也包含在输出列表中。
 *
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param certType [IN] 证书类型。
 * @param outCertList [OUT] outCertList 包含所有证书的列表。
 * outCertList是一个包含X509Cert类型的元素的列表。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @par 内存操作：
 * 成功时，该函数将为outCertList分配内存（该列表可能包含一个或多个条目），
 * 以释放该内存应用程序必须使用释放函数X509_FreeCert调用LIST_FREE。
 *
 * @attention
 * outCertList只包含本地存储中的证书。如果将多个本地证书裁剪为不支持，
 * 则它将给出一个仅包含默认证书的列表。
 */
int32_t CME_PKI_CtxGetAllLocalCert(CmePkiCtx *ctx, uint32_t certType, ListHandle *outCertList);

/**
 * @ingroup pki_context
 * @brief 获取输入证书对应的私钥。
 *
 * @par 描述：
 * CME_PKI_CtxGetPvtKeyByCert函数从本地存储返回与输入证书对应的私钥。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param cert [IN] 输入证书。
 * @param privateKey [OUT] privKey输入证书关联的私钥。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @par 内存操作：
 * cme会为privKey分配内存，为了释放这个内存，应用程序必须调用“PKEY_FreeKey”。
 */
int32_t CME_PKI_CtxGetPvtKeyByCert(CmePkiCtx *ctx, const X509Cert *cert, PKeyAsymmetricKey **privateKey);

/**
 * @ingroup pki_context
 * @brief 该函数用于从本地存储获取与输入证书对应的URL。
 *
 * @par 描述：
 * CME_PKI_CtxGetURLByCert函数返回输入证书对应的URL。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param cert [IN] 输入证书。
 * @param urlBuf [OUT] 输入证书关联的urlBuf URL。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @par 内存操作：
 * cme会为“urlBuf”分配内存，为了释放这个内存，应用程序必须调用ASN1_FreeOctetString。
 */
int32_t CME_PKI_CtxGetURLByCert(CmePkiCtx *ctx, const X509Cert *cert, Asn1OctetString **urlBuf);

/**
 * @ingroup pki_context
 * @brief 从受信任存储中删除CA证书。
 *
 * @par 描述：
 * CME_PKI_CtxRemoveTrustedCA函数将输入的颁发者名称和序列号匹配的CA从信任存储中移除。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param certType [IN] 证书类型。
 * @param issuerName [IN] CA证书的Issuer Name。
 * @param serialNum [IN] CA证书中的Serial Number。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 */
int32_t CME_PKI_CtxRemoveTrustedCA(CmePkiCtx *ctx, uint32_t certType, const X509ExtName *issuerName,
                                   const PKeyBigInt *serialNum);

/**
 * @ingroup pki_context
 * @brief 从本地存储中删除本地证书和密钥。
 *
 * @par 描述：
 * CME_PKI_CtxRemoveLocalCert删除本地证书，删除与本地证书相关的所有信息，如私钥、URL信息等。
 * 也可以使用此接口删除默认本地证书详细信息。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param issuerName [IN] 证书的Issuer Name。
 * @param serialNum [IN] 证书的Serial Number。
 * @param certType [IN] 证书类型。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 */
int32_t CME_PKI_CtxRemoveLocalCert(CmePkiCtx *ctx, const X509ExtName *issuerName, const PKeyBigInt *serialNum,
                                   uint32_t certType);

/**
 * @ingroup pki_context
 * @brief 从存储库中删除CRL。
 *
 * @par 描述：
 * CME_PKI_CtxRemoveCrl函数将输入的颁发者名称和CRL编号匹配的CRL从信任存储中移除。
 *
 * @param ctx [IN] 指向CmePkiCtx结构指针。
 * @param issuerName [IN] 证书的Issuer Name。
 * @param extnCRLNum [IN] 证书的CRL Number。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @attention
 * @li 如果用户只输入颁发者名称，则所有匹配的CRL(S)和Delta CRL(S)都将被删除。
 * @li 如果用户同时输入颁发者名称和CRL编号，则所有匹配的CRL(S)和Delta CRL(S)都将被删除。
 */
int32_t CME_PKI_CtxRemoveCrl(CmePkiCtx *ctx, const X509ExtName *issuerName, const PKeyBigInt *extnCRLNum);

/**
 * @ingroup pki_context
 * @brief 从上下文中移除所有的OCSP响应。
 *
 * @par 描述：
 * CME_PKI_CtxRemoveAllOCSPResp函数从Context中移除所有可信的OCSP响应。
 *
 * @param ctx [IN] 指向PKI上下文的指针。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 *
 * @attention 如果上下文中没有加载OCSP响应，则API返回失败。
 */
int32_t CME_PKI_CtxRemoveAllOCSPResp(CmePkiCtx *ctx);

/**
 * @ingroup pki_context
 * @brief 设置PKI上下文配置选项。
 *
 * @par 描述：
 * CME_PKI_CtxSetOptions设置上下文选项，如CME_PKI_OPTION_IGNORE_PVT_KEY。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 */
int32_t CME_PKI_CtxSetOptions(CmePkiCtx *ctx, uint32_t options);

/**
 * @ingroup pki_context
 * @brief 获取PKI上下文选项。
 *
 * @par 描述：
 * CME_PKI_CtxGetOptions获取上下文中的配置选项。
 *
 * @retval CME_PKI_SUCCESS 执行成功。
 * @retval CME_PKI_ERROR 执行失败。
 */
int32_t CME_PKI_CtxGetOptions(CmePkiCtx *ctx, uint32_t *options);

/**
 * @ingroup pki_context
 * @brief 初始化PKI库。
 *
 * @par 描述：
 * CME_PKI_LibraryInit函数初始化错误字符串和锁。
 *
 * @retval CME_PKI_SUCCESS 初始化成功。
 * @attention
 * @li 此函数只能在应用程序启动时在主线程中调用。
 * @li CME_PKI_LibraryInit时分配的资源在进程退出时释放。
 */
int32_t CME_PKI_LibraryInit(void);

/**
 * @ingroup pki_context
 * @brief 不建议使用该功能。
 *
 * @par 描述：
 * 不建议使用该功能。
 *
 * @retval void 不返回任何值。
 * @attention
 * CME_PKI_LibraryInit时分配的资源在进程退出时释放。
 */
void CME_PKI_LibraryFini(void);

#ifdef __cplusplus
}
#endif

#endif
