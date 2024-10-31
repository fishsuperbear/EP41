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
/** @defgroup cme_x509v3_extn_constraint CME_X509V3_EXTN_CONSTRAINT_API
 * @ingroup cme_x509
 */
#ifndef CME_CME_X509V3_EXTN_CONSTRAINT_API_H
#define CME_CME_X509V3_EXTN_CONSTRAINT_API_H

#include "cme_asn1_api.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief Basic Constraint Extension标识证书是否为CA证书以及包含
 * 此证书的有效证书路径的最大深度。
 * @par 描述：
 * @li cA boolean表示认证公钥是否属于CA。
 * 如果未断言cA布尔值，则密钥用法扩展中的keyCertSign位不能被断言。
 * @li pathLenConstraint字段只有在cA boolean被断言且密钥用法扩展断言keyCertSign位时才有意义。
 * 它给出了在有效证书路径中可能跟随此证书的非自签发中间证书的最大数量。
 */
typedef struct {
    bool *isCA;                 /* < 表示认证公钥是否属于CA */
    Asn1Int *pathLenConstraint; /* < 可选，有效证书路径中可能跟随此证书的
                                非自签发中间证书的最大数量，范围(0...MAX) */
} X509ExtBasicConstraint;

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 创建X509ExtBasicConstraint扩展结构体。
 *
 * @par 描述：
 * 该函数用于创建BasicConstraint扩展。BasicConstraint扩展仅用于CA证书。
 * 结构中的路径长度字段表示出现在证书路径中的证书数量。此字段在证书路径验证期间使用。
 * Pathlength在BasicConstraint扩展中的使用必须由应用程序决定。
 * 如果要选择此字段，则必须提供负值作为输入。
 *
 * @param caFlag [IN] 用于表示是否为CA证书。
 * @param pathLen [IN] 校验时使用的证书路径中的证书数量。
 * 如果不创建路径长度，那么函数的输入必须是负值。
 * 零和任何正值表示创建pathLength可选字段。
 *
 * @retval X509ExtBasicConstraint* 已创建的X509ExtBasicConstraint结构体指针。
 * @retval NULL 内存申请失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为X509ExtBasicConstraint结构分配内存，为了释放这个内存应用程序必须调用X509EXT_FreeBasicConstraint函数。
 *
 * @attention pathlength的值不会被验证。因此，应用程序必须注意，
 * 在终端实体证书的情况下，不要将pathlength值设置为大于零。
 *
 */
X509ExtBasicConstraint *X509EXT_BasicConstraintCreate(bool caFlag, int32_t pathLen);

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 从中提取路径长度。
 *
 * @par 描述：
 * 如果证书是CA证书，则返回证书的路径长度。
 * 此函数在X509_IsCACert之后使用。
 * 证书路径校验过程中使用。
 * @param basicConstraint [IN] 必须从中返回路径长度的结构。
 *
 * @retval int32_t 返回的路径长度。
 * @retval -1 如果输入参数为NULL。
 * @par 依赖：
 * x509.h.
 */
Asn1Int X509EXT_BasicConstraintGetPathLen(const X509ExtBasicConstraint *basicConstraint);

/* To duplicate BasicConstraint structure */
/**
 * @ingroup cme_x509v3_extn_constraint
 *
 * @brief 深拷贝X509ExtBasicConstraint结构。
 *
 * @par 描述：
 * 深拷贝X509ExtBasicConstraint结构。
 *
 * @param pSrc [IN] 指向X509ExtBasicConstraint结构源指针。
 *
 * @retval X509ExtBasicConstraint* 指向X509ExtBasicConstraint结构目的指针。
 * @retval NULL 输入参数为NULL。
 * @retval NULL 内存申请失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为X509ExtBasicConstraint结构分配内存，为了释放这个内存应用程序必须调用X509EXT_FreeBasicConstraint函数。
 *
 */
X509ExtBasicConstraint *X509EXT_BasicConstraintsDump(const X509ExtBasicConstraint *src);

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 释放X509ExtBasicConstraint结构体。
 *
 * @par 描述：
 * 释放X509ExtBasicConstraint结构体。
 * @param basicConstr [IN] 指向需要被释放的X509ExtBasicConstraint结构指针。
 * @retval void 不返回任何值。
 * @par 依赖：
 * x509.h.
 */
void X509EXT_BasicConstraintsFree(X509ExtBasicConstraint *basicConstr);

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief Sub Trees List
 *
 */
typedef CmeList GeneralSubtrees;

/**
 * @ingroup x509v3_extnStructures
 * @brief Name Constraints Extension.
 *
 * @par 描述：
 * 名称约束扩展，它只能在CA证书中使用，并且必须设置为Critical。
 * 限制适用于Subject Distinguished Name，并适用于Subject Alternative Names。
 * 限制由允许或排除的Name Subtrees来定义。
 * 任何与excludedSubtrees字段中的限制匹配的名称都是无效的，
 * 而不管permitSubtrees中出现什么信息。
 * @li permittedSubtrees，此字段可选。如果出现，则包含在证书路径验证期间有效的允许名称列表。
 * @li excludedSubtrees，此字段也是可选字段。如果出现，则包含路径验证期间不允许的排除名称列表。
 * 如果名称存在于排除列表中，则即使该名称存在于允许列表中，验证也应该停止。
 */
typedef struct {
    GeneralSubtrees *permittedSubtrees; /* < 可选，允许名称列表 */
    GeneralSubtrees *excludedSubtrees;  /* < 可选，排除名称列表 */
} X509ExtNameConstraint;

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 从Name约束结构中获取允许的名称列表。
 *
 * @par 描述：
 * 返回结构中允许的名称列表。该列表在证书路径验证过程中使用。
 * @param nameConstr[IN] nameConstr The structure from which the permitted list has to be extracted [N/A]
 *
 * @retval CmeList* 指向证书路径中允许使用的名称列表的指针。
 * @retval NULL 如果输入参数为NULL。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme不会为CmeList结构分配内存，因此不应该释放该内存。
 *
 */
ListHandle X509EXT_NameConstraintGetPermittedSubTree(const X509ExtNameConstraint *nameConstr);

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 从Name约束结构中获取排除名称的列表。
 *
 * @par 描述：
 * 返回结构中排除的名称列表。该列表在证书路径验证过程中使用。
 *
 * @param nameConstr [IN] 从中提取排除列表的结构。
 *
 * @retval CmeList* 指向证书路径中排除名列表的指针。
 * @retval NULL 输入参数为NULL。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme不会为CmeList结构分配内存，因此不应该释放该内存。
 *
 */
ListHandle X509EXT_NameConstraintGetExcludedSubTree(const X509ExtNameConstraint *nameConstr);

/* To duplicate NameConstraint structure */
/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 深拷贝X509ExtNameConstraint结构。
 *
 * @par 描述：
 * 深拷贝X509ExtNameConstraint结构。
 * @param pSrc [IN] 指向X509ExtNameConstraint结构的源指针。
 *
 * @retval X509ExtNameConstraint* 指向X509ExtNameConstraint结构的目的指针。
 * @retval NULL 输入参数为NULL。
 * @retval NULL 内存申请失败。
 *
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为X509ExtNameConstraint结构分配内存，
 * 为了释放这个内存应用程序必须调用X509EXT_FreeNameConstraint函数。
 *
 */
X509ExtNameConstraint *X509EXT_NameConstraintsDump(const X509ExtNameConstraint *src);

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 释放X509ExtNameConstraint结构体。
 *
 * @par 描述：
 * 用于释放X509ExtNameConstraint结构体。
 *
 * @param nameConstr [IN] 指向需要释放的X509ExtNameConstraint结构指针。
 *
 * @retval void 不返回任何值。
 * @par 依赖：
 * x509.h.
 *
 */
void X509EXT_NameConstraintFree(X509ExtNameConstraint *nameConstr);

/**
 *
 * @brief X509 Policy Constraint Extension.
 * @par 描述：
 * Policy Constraint 扩展可用于颁发给CA的证书。
 * 它可用于禁止策略映射或要求路径中的每个证书都包含可接受的策略标识符。
 * Policy Constraint扩展以两种方式约束路径验证：
 * @li requireExplicitPolicy，可选字段。如果requireExplicitPolicy字段存在，
 * 则其值指示在对整个路径要求显式策略之前可能出现在路径中的附加证书的数量。
 * @li inhibitPolicyMapping，此字段也是可选字段。如果inhibitionPolicyMapping字段存在，
 * 则其值指示在不再允许策略映射之前可能出现在路径中的附加证书的数量。
 *
 */
typedef struct {
    Asn1Int *requireExplicitPolicy; /* < 可选，对整个路径要求显式策略
                                    之前可能出现在路径中的附加证书的数量 */
    Asn1Int *inhibitPolicyMapping;  /* < 可选，不再允许策略映射之前可
                                    能出现在路径中的附加证书的数量 */
} X509ExtPolicyConstraint;

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 获取给定Policy constraint扩展的Explicit Policy。
 *
 * @par 描述：
 * 返回证书中X509ExtPolicyConstraint扩展的Explicit Policy。证书路径校验时使用。
 * @param poilcyConstr [IN] 从中提取Explicit Policy的Policy Constraint结构[N/A]
 *
 * @retval int32_t 返回Explicit Policy。
 * @retval -1 输入参数为NULL。
 * @par 依赖：
 * x509.h.
 *
 */
Asn1Int X509EXT_PolicyConstraintGetExplicitPolicy(const X509ExtPolicyConstraint *poilcyConstr);

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 从给定的Policy constraint扩展中获取Inhibit Policy。
 *
 * @par 描述：
 * 返回证书中X509ExtPolicyConstraint扩展的Inhibit Policy。
 * 证书路径校验时使用。
 * 返回值指示在策略映射不再允许之前可能出现在路径中的附加证书的数量。
 *
 * @param poilcyConstr [IN] 从中提取Inhibit Policy的Policy Constraint结构。
 *
 *
 * @retval int32_t 成功时返回Inhibit Policy值。
 * @retval -1 输入参数为NULL。
 *
 * @par 依赖：
 * x509.h.
 *
 */
Asn1Int X509EXT_PolicyConstraintGetInhibitPolicy(const X509ExtPolicyConstraint *poilcyConstr);

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 深拷贝X509ExtPolicyConstraintj结构。
 *
 * @par 描述：
 * 深拷贝X509ExtPolicyConstraintj结构。
 *
 * @param pSrc [IN] 指向X509ExtPolicyConstraint结构的源指针。
 *
 * @retval X509ExtPolicyConstraint* 指向X509ExtPolicyConstraint结构的目的指针。
 * @retval NULL 如果输入参数为NULL。
 * @retval NULL 如果内存申请失败。
 * @par 依赖：
 * x509.h.
 *
 * @par 内存操作：
 * cme会为X509ExtPolicyConstraint结构分配内存，
 * 为了释放这个内存应用程序必须调用X509EXT_FreePolicyConstraint函数。
 *
 */
X509ExtPolicyConstraint *X509EXT_PolicyConstraintsDump(const X509ExtPolicyConstraint *src);

/**
 * @ingroup cme_x509v3_extn_constraint
 * @brief 释放X509ExtPolicyConstraint结构体。
 *
 * @par 描述：
 * 该函数用于释放X509ExtPolicyConstraint结构体。
 * @param polConstr [IN] 指向需要释放的X509ExtPolicyConstraint结构体指针。
 *
 * @retval void 不返回任何值。
 * @par 依赖：
 * x509.h.
 */
void X509EXT_PolicyConstraintsFree(X509ExtPolicyConstraint *polConstr);

#ifdef __cplusplus
}
#endif

#endif // CME_CME_X509V3_EXTN_CONSTRAINT_API_H
