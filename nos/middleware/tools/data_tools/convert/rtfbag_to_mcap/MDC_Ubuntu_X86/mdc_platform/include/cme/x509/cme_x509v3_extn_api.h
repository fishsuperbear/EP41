/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2021. All rights reserved.
 * Description: This Header file contains the x509v3_extn Structures, Functions and Enum
 * Create: 2012/10/20
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_x509v3_extn CME_X509V3_EXTN_API
 * @ingroup cme_x509
 */
#ifndef CME_X509V3_EXTN_API_H
#define CME_X509V3_EXTN_API_H

#include "cme_asn1_api.h"

#include "cme_x509v3_extn_dn_api.h"
#include "cme_x509v3_extn_kid_api.h"
#include "cme_x509v3_extn_dt_api.h"
#include "cme_x509v3_extn_algid_api.h"
#include "cme_x509v3_extn_constraint_api.h"
#include "cme_x509v3_extn_policy_api.h"
#include "cme_x509v3_extn_dp_api.h"
#include "cme_x509v3_extn_attr_api.h"
#include "keys/pkey_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509v3_extn
 * @brief 以下结构将用于创建扩展。
 * @par 描述：
 * 不同的扩展具有不同的类型临界。\n
 * 临界值需要始终作为布尔值输入。\n
 * 扩展ID字段是唯一标识扩展的对象标识符。\n
 * 扩展值是编码给定输入字段的DER值。\n
 *
 */
typedef struct {
    Asn1Oid extId;          /* < 扩展的OID */
    bool *critical;         /* < 扩展关键性级别 */
    Asn1OctetString extVal; /* < 扩展结构的编码值。 */
} X509Ext;

/**
 * @ingroup cme_x509v3_extn
 * @brief 证书扩展列表
 *
 */
typedef CmeList X509ExtList; /* < 元素为X509Ext */

/**
 * @ingroup cme_x509v3_extn
 * @brief 创建X509Ext结构。
 * @par 描述：
 * 该函数用于创建扩展名。创建扩展结构，将CID映射到OID并复制到扩展中。
 * Extension中必须设置的数据是被编码的，编码后的数据被连同Critical标志一起复制到扩展结构的extVal中。
 * 某些扩展结构应始终设置为预定义的临界值。如果Critical标志不是上述值，则不创建扩展。对于CA证书，
 * 如果在密钥用法扩展和基本约束扩展中设置了数字签名位，则应用程序必须将基本约束扩展的Critical标志设置为TRUE。
 * 下面列出了一些扩展Critical标志必须始终为false的一些情况：\n
 * @li 对于X509 EXTENSIONS:\n
 * Authority Key Identifier, Subject Key Identifier, Private Key Usage Period,\n
 * Subject Directory Attributes, Freshest CRL, Authority Info Access,\n
 * Subject Info Access, Issuer Alternate Name, CRL Distribution Point,\n
 * CRL Number.\n
 * @li 对于X509 CRL ENTRY EXTENSIONS：\n
 * CRL Reasons, Hold Instruction Code, Invalidity Date\n
 * 某些扩展关Critical标志始终为true:
 * @li 对于X509 EXTENSIONS：
 * Name Constraints, Inhibit Any Policy, Delta CRL Indicator,
 * Issuing Distribution Point, Policy Mapping, Policy Constraints.\n
 * 要创建CRL Reason Extension，请始终使用enum X509CRLReason。\n
 * 创建Reason Code扩展，提供指向enum的指针作为create函数的输入。\n
 * 创建Hold指令代码时，提供OID指针作为create函数的输入。\n
 * 若要创建无效日期扩展，请提供指向BslSysTime结构的指针。在内部，日期将被转换为广义的时间。只支持有效日期格式。\n
 * 要创建KEY USAGE扩展，提供的宏将用于指定证书的密钥用法。\n
 * 要创建ANY POLICYextension，必须使用指针变量将必须设置的数字传递给函数。
 *
 * @param cid [IN] cid 必须创建的扩展。
 * @param criticalFlag [IN] 必须设置的Critical标志。
 * @param extData [IN] 需要编码并复制到扩展的扩展数据。
 *
 * @retval X509Ext* 成功执行时返回X509Ext结构的指针
 * @retval NULL 创娘失败情况下，返回NULL。失败情况可能是以下情况之一：
 * @li 输入参数为NULL。
 * @li 内存申请失败。
 * @li 数据编码失败。
 *
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为X509Ext结构分配内存，为了释放这个内存应用程序必须调用X509EXT_Free API。
 */
X509Ext *X509EXT_Create(CmeCid cid, bool criticalFlag, const void *extData);

/**
 * @ingroup cme_x509v3_extn
 * @brief 设置扩展的Critical标志。
 *
 * @par 描述：
 * 该函数用于设置扩展的Critical标志。
 * @param criticalFlag [IN] 需要设置的Critical标志。
 * @param ext [OUT] 要设置的Critical标志所在的Extension。
 * @retval CME_SUCCESS 设置成功。
 * @retval CME_ERR_INVALID_ARG 输入参数为NULL。
 * @retval CME_ERR_DATA_COPY_FAILED 深拷贝Issuer Name失败。
 * @retval CME_ERR_MALLOC_FAIL 内存申请失败。
 * @par 依赖：
 * x509.h
 */
int32_t X509EXT_SetCritical(bool criticalFlag, X509Ext *ext);

/**
 * @ingroup cme_x509v3_extn
 * @brief 获取扩展的Critical标志。
 *
 * @par 描述：
 * 返回扩展的Critical标志。
 * @param ext [IN]  从中提取Critical标志的扩展。
 *
 * @retval bool 成功执行时返回扩展的Critical标志。
 * @retval false 对于失败情况，返回此值。
 * @retval false 输入参数为NULL。
 * @par 依赖：
 * x509.h.
 *
 */
bool X509EXT_GetCritical(const X509Ext *ext);

/**
 * @ingroup cme_x509v3_extn
 * @brief 此函数用于检查对应CID扩展的Critical标志。
 *
 * @param cid [IN] CID.
 * @param criticality [IN] Critical标志
 * @retval bool 返回检查结果。
 */
bool X509EXT_CriticalityCheck(CmeCid cid, bool criticality);

/**
 * @ingroup cme_x509v3_extn
 * @brief 从给定的扩展中获取CID。
 *
 * @par 描述：
 * 该函数从扩展返回Common ID。调用CID_GetCidByOid函数获取Extension中OID的整数映射。
 * @param ext [IN] 需要获取CID的扩展。
 *
 * @retval int32_t 成功执行时返回扩展中的OID的整数映射。
 * @retval -1 输入参数为不正确或为NULL。
 * @par 依赖：
 * x509.h.
 */
CmeCid X509EXT_GetCID(const X509Ext *ext);

/**
 * @ingroup cme_x509v3_extn
 * @brief 从给定的扩展中获取解码数据。
 *
 * @par 描述：
 * 从扩展中获取解码后的数据。根据扩展中的OID对传入的扩展结构进行解码，
 * 函数返回从扩展中解码出来的数据的指针。
 * @param ext [IN] 必须解码的扩展。
 *
 * @retval void* 成功执行时，解码扩展后返回void指针。
 * void*必须根据扩展中的OID映射到相应的结构。
 * @retval NULL 操作失败时返回NULL，可能的原因为：
 * @li 输入参数为NULL。
 * @li 内存申请失败。
 * @li 解码失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会分配内存为Extension值，要释放此内存应用程序必须调用相应的释放函数。
 */
void *X509EXT_ExtractContent(const X509Ext *ext);

/**
 * @ingroup cme_x509v3_extn
 * @brief 深拷贝X509Ext结构。
 *
 * @par 描述：
 * 深拷贝X509Ext结构。
 * @param src [IN] 指向X509Ext结构的源指针。
 *
 * @retval X509Ext* 指向X509Ext的目的指针。
 * @retval NULL 输入参数为NULL。
 * @retval NULL 内存申请失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为X509Ext结构分配内存，为了释放这个内存应用程序必须调用X509EXT_Free函数。
 *
 */
X509Ext *X509EXT_Dump(const X509Ext *src);

/**
 * @ingroup cme_x509v3_extn
 * @brief 释放X509Ext结构内存。
 *
 * @par 描述：
 * 释放X509Ext结构内存。
 * @param ext [IN] 指向X509Ext结构的指针。
 *
 * @retval void 不返回任何值。
 * @par 依赖：
 * x509.h.
 *
 */
void X509EXT_Free(X509Ext *ext);

/**
 * @ingroup cme_x509v3_extn
 * @brief  解码扩展列表。
 *
 * @par 描述：
 * 从编码的x509扩展列表生成x509扩展的解码列表。
 * @param encodedExtList [IN] 需要进行解码操作的x509扩展列表。
 * @param encLen [IN] 需要进行解码操作的x509扩展列表长度。
 * @param decLen [OUT] 解码后的列表长度。
 * @retval CmeList* 指向解码后内存的指针。
 * @retval NULL 输入参数无效、内存申请失败或解码失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为SEC_List结构分配内存，为了释放这个内存应用程序必须调用X509EXT_Free作为释放函数的LIST_FREE宏。
 *
 * @attention 扩展解码时分配内存。使用完列表后，请删除该列表。
 */
X509ExtList *X509EXT_ExtListDecode(const uint8_t *encodedExtList, size_t encLen, size_t *decLen);

/**
 * @ingroup cme_x509v3_extn
 * @brief 对给定的x509扩展进行编码。
 *
 * @par 描述：
 * 该函数用于对给定的X509扩展列表进行编码。
 * @param extList [IN] 需要进行编码的X509ExtList。
 * @param encLen [OUT] 编码后数据缓存长度。
 * @retval uint8_t* 指向编码后的数据缓存指针。
 * @retval NULL 输入参数无效或编码失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为编码的ExtnList分配内存，为了释放这个内存应用程序必须调用cme_free函数。
 *
 * @attention 扩展解码时分配内存。使用完列表后，请删除列表。
 */
uint8_t *X509EXT_ExtListEncode(const X509ExtList *extList, size_t *encLen);

/**
 * @ingroup cme_x509v3_extn
 * @par Prototype
 * @brief 深拷贝x509扩展List。
 *
 * @par 描述：
 * 深拷贝x509扩展List。
 * @param src [IN] 指向ListHandle结构源指针。
 * @param [OUT] N/A N/A [N/A]
 * @retval CmeList* 指向ListHandle结构目的指针。
 * @retval NULL 输入参数为NULL。
 * @retval NULL 内存申请失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为SEC_List结构分配内存，为了释放这个内存，
 * 应用程序必须调用LIST_FREE宏，并使用X509EXT_Free作为空闲函数。
 *
 */
ListHandle X509EXT_ExtListDump(ListRoHandle src);

/**
 * @ingroup cme_x509v3_extn
 * @brief 释放CmeList（扩展列表）结构。
 *
 * @par 描述：
 * 释放CmeList（扩展列表）结构。
 * @param extList [IN] 指向需要被free的CmeList结构。
 *
 * @retval void 不返回任何值。
 * @par 依赖：
 * x509.h.
 */
void X509EXT_ExtListFree(ListHandle extList);

/**
 * @ingroup cme_x509v3_extn
 * @brief 从给定的列表中获取指定Cid的扩展
 *
 * @par 描述：
 * 从给定的列表中获取指定Cid的扩展
 * @param list [IN] X509Ext列表.
 * @param extensionID [IN] 扩展Cid.
 *
 * @retval X509Ext* 返回扩展。
 * @retval NULL 如果输入参数为NULL或扩展列表为空。
 * @par 依赖：
 * x509.h.
 */
X509Ext *X509EXT_ListGetExtn(const X509ExtList *list, CmeCid extensionID);

#ifdef __cplusplus
}
#endif

#endif /* _CME_X509V3_EXTN_API_H_ */
