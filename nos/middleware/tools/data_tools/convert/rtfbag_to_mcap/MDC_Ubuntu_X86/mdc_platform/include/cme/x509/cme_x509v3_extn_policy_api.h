/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:
 * Create: 2020/10/26
 * Notes:
 * History:
 * 2020/10/26 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_x509v3_extn_policy CME_X509V3_EXTN_POLICY_API
 * @ingroup cme_x509
 */
#ifndef CME_X509V3_EXTN_POLICY_API_H
#define CME_X509V3_EXTN_POLICY_API_H

#include "cme_asn1_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 通知引用序列列表。
 *
 */
typedef CmeList NoticeReferenceSeqOf;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 表示各种字符串格式。
 *
 */
typedef enum {
    DISPLAYTEXT_IA5STRING,     /* < 表示IA5字符串格式 */
    DISPLAYTEXT_VISIBLESTRING, /* < 表示VISIBLE字符串格式 */
    DISPLAYTEXT_BMPSTRING,     /* < 表示BMP字符串格式 */
    DISPLAYTEXT_UTF8STRING     /* < 表示UTF8字符串格式 */
} DispTxtId;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 该结构在创建证书扩展时使用。结构体支持不同的字符串格式。这两种字符串格式都可以存在。
 *
 */
typedef struct {
    DispTxtId choiceId; /* < 可以表示字符串的各种字符串格式 */
    union DisplayText {
        Asn1IA5String *ia5Str;         /* < 包含字符串及其长度的结构 */
        Asn1VisibleString *visibleStr; /* < 包含字符串及其长度的结构 */
        Asn1BMPString *bmpStr;         /* < 包含字符串及其长度的结构 */
        Asn1UTF8String *utf8Str;       /* < 包含字符串及其长度的结构 */
    } a; /* < 联合体保存字符串表示形式中的任一个，字符串最大长度为200 */
} X509ExtDispTxt;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 如果使用NoticeReference策略限定符，则为组织命名并按编号标识该组织准备的特定文本语句。
 *
 */
typedef struct {
    X509ExtDispTxt *organization; /* < 由组织准备的通知文本，在证书路径验证期间必须显示。
                                    如果在UserNotice结构的explicitText字段中也提到显式
                                    通知文本，则通知文本必须首先显示 */
    NoticeReferenceSeqOf *noticeNumbers; /* < 引用组织文本中的语句的数字列表 */
} X509ExtNoticeRef;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief UserNotice结构是证书策略中的限定符之一。
 * @attention 使用证书时向依赖方显示用户通知。
 * 应用软件应显示所用证书路径的所有证书中的所有用户通知，
 * 除非通知是重复的，那么只需要显示一份副本。
 * 如果设置了这两个选项，应用程序必须首先使用noticeRef字段中的通知。
 */
typedef struct {
    X509ExtNoticeRef *noticeRef;  /* < 可选，组织发出的通知文本，
                                     并按编号标明该组织编写的特定文本声明 */
    X509ExtDispTxt *explicitText; /* < 可选，explicitText字段包括证书中直接显示文本语句，
                                    当使用证书时必须向依赖方显示 */
} X509ExtUsrNotice;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 从X509ExtUsrNotice证书策略限定符中提取显式文本数据。
 *
 * @par 描述：
 * 从X509ExtUsrNotice证书策略限定符中提取Explicit Text。此Explicit Text在证书路径验证过程中使用。
 * @param userNotice [IN] 从中检索文本的X509ExtUsrNotice结构指针。
 * @param strType [OUT] 字符串格式。
 * @retval Asn1OctetString* 指向Explicit Text的指针。
 * @retval NULL 如果输入参数为NULL。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme不会为Asn1OctetString结构分配内存，因此不应该释放该内存。
 *
 */
Asn1OctetString *X509EXT_PolicyExplicitTextGet(const X509ExtUsrNotice *userNotice, DispTxtId *strType);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 从X509ExtUsrNotice证书Policy Qualifier中获取组织文本数据。
 *
 * @par 描述：
 * 从X509ExtUsrNotice证书Policy Qualifier返回Organization文本。
 * Organization文本说明了Organization可以接受的政策。此文本在证书路径验证过程中使用
 * @param [IN] 从中检索文本的X509ExtUsrNotice结构指针。
 * @param [OUT] 字符串格式。
 * @retval Asn1OctetString* 返回Organization文本。
 * @retval NULL 输入参数为NULL。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme不会为Asn1OctetString结构分配内存，因此不应该释放该内存。
 *
 */
Asn1OctetString *X509EXT_PolicyOrganizationTextGet(const X509ExtUsrNotice *userNotice, DispTxtId *strType);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 从给定的X509ExtUsrNotice结构中获取一个Notice Numbers。
 * @par 描述：
 * 返回证书策略的X509ExtUsrNotice限定符中的NoticeReference结构中的NoticeNumbers列表。
 * 编号列表指向证书策略验证过程中必须引用的Organization文本。
 * @param [IN] 从中获取数据的X509ExtUsrNotice结构指针。
 *
 * @retval CmeList* 返回Notice reference number列表指针。
 * @retval CmeList* 如果入参为空或者没有剩余Notice Number List。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme不会为CmeList结构分配内存，因此不应该释放该内存。
 *
 */
ListHandle X509EXT_PolicyNoticeNumbersGet(const X509ExtUsrNotice *userNotice);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 包含策略限定符信息列表策略限定符是cps url和用户通知。
 *
 */
typedef CmeList PolicyInformationSeqOf;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 该结构用于保存策略限定符信息。
 *
 * @par 描述：
 * 此结构在为证书创建证书策略时使用。该结构体包含一个策略OID。
 * 如果仅仅一个OID是不够的，则建议使用限定符。\n
 * 可以使用两种类型的Policy Qualifiers:
 * @li CPS Pointers.
 * @li UserNotices.
 * CPS Pointer Qualifier包含指向CA发布的Certification Practice Statement(CPS)的指针。指针形式为URI。\n
 * 使用证书时，用户通知旨在向依赖方显示。
 */
typedef struct {
    Asn1Oid policyQualifierId;  /* < 定义策略的OID */
    Asn1AnyDefinedBy qualifier; /* < 该字段包含策略限定符中的任何一个。
                                    限定符可以是CPS URL或UserNotice结构。 */
} X509ExtPolicyQual;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief Policy信息条款指明了签发证书所依据的Policy以及该证书可能用于的目的。
 * 在CA证书中，这些Policy信息条款限制了包含此证书的证书路径的所有Policy。
 */
typedef struct {                              /* < SEQUENCE */
    Asn1Oid policyIdentifier;                 /* < Policy Identifier */
    PolicyInformationSeqOf *policyQualifiers; /* < 可选，Policy Information Sequence，
                                                大小区间[1, MAX] */
} X509ExtPolicyInfo;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 深拷贝X509ExtPolicyInfo结构。
 *
 * @par 描述：
 * 深拷贝X509ExtPolicyInfo结构。
 * @param src [IN] 指向X509ExtPolicyInfo结构源指针。
 *
 * @retval X509ExtPolicyInfo* 指向X509ExtPolicyInfo结构目的指针。
 * @retval NULL 如果输入参数为NULL。
 * @retval NULL 如果内存申请失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为X509ExtPolicyInfo结构分配内存，为了释放这个内存应用程序必须调用X509EXT_FreePolicyInfo函数。
 */
X509ExtPolicyInfo *X509EXT_PolicyInformationDump(const X509ExtPolicyInfo *src);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 释放X509ExtPolicyInfo结构体。
 *
 * @par 描述：
 * 改函数用于释放X509ExtPolicyInfo结构体。
 * @param policyInfo [IN] 指向待释放的X509ExtPolicyInfo结构体指针。
 *
 * @retval void 不返回任何值。
 * @par 依赖：
 * x509.h.
 */
void X509EXT_PolicyInfoFree(X509ExtPolicyInfo *policyInfo);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 包含“Issuing and Subject Domain policy”的策略映射的结构。
 * @par 描述：
 * SEQUENCE SIZE 1..MAX OF X509ExtPolicyMapping
 */
typedef CmeList X509ExtPolicyMapList;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 该结构包含一对OID，
 * 用于映射Issuer Certificate Policy 与Subject Domain Policy等价。
 */
typedef struct {
    Asn1Oid issuerDomainPolicy;  /* < 颁发者CA的OID */
    Asn1Oid subjectDomainPolicy; /* < OID等同于isserDomainPolicy中提到的OID */
} X509ExtPolicyMapping;

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 从给定的策略映射结构中获取Issuer Domain Policy。
 *
 * @par 描述：
 * 此函数从给定的策略映射结构返回颁发者域策略。
 * 返回的CID可用于将CA的Issuer策略与其他CA域策略进行映射。
 * @param policyMappings [IN] X509ExtPolicyMapping结构指针。
 *
 * @retval int32_t 返回Issuer Domain Policy等效CID作为输出。
 * @retval -1 输入参数为NULL。
 * @par 依赖：
 * x509.h.
 */
int32_t X509EXT_PolicyIssuerDomainPolicyGet(const X509ExtPolicyMapping *policyMappings);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 从给定的策略映射结构中获取Subject Domain Policy。
 *
 * @par 描述：
 * 该函数从给定的策略映射结构返回主题域策略。
 * 返回的CID是必须与颁发的CA策略映射的策略。
 * @param [IN] X509ExtPolicyMapping结构指针。
 *
 * @retval int32_t 返回Subject Domain Policy等效CID作为输出。
 * @retval int32_t 输入参数为NULL。
 * @par 依赖：
 * x509.h.
 */
int32_t X509EXT_PolicySubjectDomainPolicyGet(const X509ExtPolicyMapping *policyMappings);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 深拷贝X509ExtPolicyMapList结构体。
 *
 * @par 描述：
 * 深拷贝X509ExtPolicyMapList结构体。
 * @param src [IN] 指向X509ExtPolicyMapList结构的源指针。
 *
 * @retval X509ExtPolicyMapList* 指向X509ExtPolicyMapList的目的指针。
 * @retval NULL 输入参数为NULL。
 * @retval NULL 内存申请失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为X509ExtPolicyMapList结构分配内存，
 * 为了释放这个内存应用程序必须调用X509EXT_FreePolicyMappings函数。
 */
X509ExtPolicyMapList *X509EXT_PolicyMappingsDump(const X509ExtPolicyMapList *src);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 释放CmeList结构-Policy映射列表。
 *
 * @par 描述：
 * 用于释放PolicyMappings列表的内存。
 * @param policyMappingList [IN] 需要释放的List指针。
 *
 * @retval void 不返回任何值。
 * @par 依赖：
 * x509.h.
 */
void X509EXT_PolicyMappingsFree(ListHandle policyMappingList);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 释放CmeList结构-证书策略列表。
 *
 * @par 描述：
 * 这用于释放证书策略列表的内存。
 * @param certPolicyList [IN] 需要释放的List指针。
 *
 * @retval void 不返回任何值。
 * @par 依赖：
 * x509.h.
 */
void X509EXT_CertPoliciesFree(ListHandle certPolicyList);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 释放X509ExtPolicyQual结构体内存。
 *
 * @param policyQualInfo [IN] 需要释放的X509ExtPolicyQual结构体指针。
 */
void X509EXT_PolicyQualifierInfoFree(X509ExtPolicyQual *policyQualInfo);

/**
 * @ingroup cme_x509v3_extn_policy
 * @brief 深拷贝X509ExtPolicyQual结构体。
 *
 * @param src [IN] 指向X509ExtPolicyQual结构体源指针。
 * @retval X509ExtPolicyQual* 指向X509ExtPolicyQual结构体目的指针。
 * @retval NULL 申请内存失败。
 */
X509ExtPolicyQual *X509EXT_PolicyQualifierInfoDump(const X509ExtPolicyQual *src);

#ifdef __cplusplus
}
#endif

#endif // CME_X509V3_EXTN_POLICY_API_H
