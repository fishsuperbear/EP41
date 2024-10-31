/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:Distribution Point扩展对外API
 * Create: 2020/10/26
 * Notes:
 * History:
 * 2020/10/26 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_x509v3_extn_dp CME_X509V3_EXTN_DP_API
 * @ingroup cme_x509
 */
#ifndef CME_X509V3_EXTN_DP_API_H
#define CME_X509V3_EXTN_DP_API_H

#include "cme_x509v3_extn_dn_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief 包含所有证书撤销原因，有效位只取撤销原因中的前2字节。
 *
 */
#define X509EXTN_DP_ALL_REASONS 0x807Fu

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief 证书撤销原因，对应于 DP 中的 reasons 类型，DP 中的 reasons 是 bitString 类型，因此需要按位设置。
 */
typedef enum {
    X509EXT_RVK_AA_COMPROMISE = 0x8000u,   /* < 签发者受质疑 */
    X509EXT_RVK_PRIV_WITHDRAWN = 0x01u,    /* < 权利被使用者撤回 */
    X509EXT_RVK_CERT_HOLD = 0x02u,         /* < 该证书的状态是有问题的,可以撤销 */
    X509EXT_RVK_CESSATION_OF_OPER = 0x04u, /* < 证书不再需要用作发布时的用途，但密钥未泄露 */
    X509EXT_RVK_SUPER_SEDED = 0x08u,       /* < 证书被废弃但没有证据表明密钥已被泄露 */
    X509EXT_RVK_AFFL_CHG = 0x10u,          /* < 签发者名称或其他信息被篡改 */
    X509EXT_RVK_CA_COMPROMISE = 0x20u,     /* < CA密钥泄露 */
    X509EXT_RVK_KEY_COMPROMISE = 0x40u,    /* < 证书私钥泄露 */
    X509EXT_RVK_UNUSED_CERT = 0x0080u,     /* < 证书不再被使用 */
    X509EXT_NO_DP_REASONS_FIELD = 0x0000u  /* < 不创建 Reason 字段 */
} X509ExtDPRvkReasons;

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief 下面的枚举表示名称选择类型。
 *
 */
typedef enum {
    X509EXT_CRL_ISSUER_FULL_NAME = 0, /* < 传递给 X509_createDistPointName 函数的输入，是CRL issuer备用名称的选项 */
    X509EXT_CRL_ISSUER_RELATIVE_NAME = 1 /* < 传递给 X509_createDistPointName 函数的输入，是CRL RDN的选项 */
} X509ExtDPNId;

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief X509ExtDPN， 用于传递 Distribution Point Name 内容。
 */
typedef struct {
    X509ExtDPNId choiceId; /* < DPN 枚举类型 */
    union DPChoice {
        X509ExtGenNameList *fullName; /* < issuer 全称 */
        X509ExtRDNSeq *nameRelativeToCRLIssuer; /* < DistinguishedName，可附加到CRLIssure的DN中，以获得分发点名称 */
    } a;                                        /* < DP Choice Union */
    X509ExtName *dpname;                        /* < DPN 内容 */
} X509ExtDPN;

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief   复制 Distribution Point Name 结构。
 * @param   src [IN] 待复制的 DPN。
 * @retval  X509ExtDPN，已复制的 DPN 结构体指针。
 */
X509ExtDPN *X509EXT_DPNDump(const X509ExtDPN *src);

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief   释放 Distribution Point Name。
 * @param   distPointName [IN] 待释放的 DPN。
 */
void X509EXT_DPNFree(X509ExtDPN *distPointName);

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief X509ExtDP 结构体，用于传递 Distribution Point 信息。
 * 此结构作为 CRL DistributionPoint 扩展的一部分。结构中的信息指定谁生成了 CRL。
 * 证书撤销原因参考 X509ExtDPRvkReasons。
 */
typedef struct {
    X509ExtDPN *dpn; /* < 可选项，证书颁发者与CRL颁发者相同时设置，若此字段设置了，reasons 也应被设置。 */
    Asn1BitString reasons;         /* < 可选项，指示生成CRL的原因 */
    X509ExtGenNameList *crlIssuer; /* < 可选项，证书颁发者不是CRL颁发者时设置该字段 */
} X509ExtDP;

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief   从 Distribution Point 结构中获取 CRL 颁发者。
 * @param   distPoint [IN] Distribution Point.
 * @retval  ListHandle, CRL 颁发者链表。
 */
X509ExtGenNameList *X509EXT_DPGetCRLIssuer(const X509ExtDP *distPoint);

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief   从 Distribution Point 结构中获取 Distribution Point Name。
 * @param   distPoint [IN] DP 结构。
 * @retval  X509ExtDPN，Distribution Point Name 结构指针。
 */
X509ExtDPN *X509EXT_DPGetDPN(const X509ExtDP *distPoint);

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief   从 Distribution Point 结构中获取 CRL 吊销原因。
 * @par 描述：Distribution Point 中 reason 字段是指 DP 中的生成 CRL 的原因可以使用的原因有哪些。
 * reasons 参数输出一个 uint32 类型，为当前 DP 支持的吊销原因的组合，通过存储于 DP 结构体中的 Asn1BitString
 * 取出， 由于有效位只使用了2位，因此将高 2 位取出，构成撤销原因组合后返回。 调用者可通过读出的 “reasons” 和
 * “X509EXTN_DP_ALL_REASONS” 进行 ”与运算“，得到 DP 中实际支持的撤销原因。 证书撤销原因参考 X509ExtDPRvkReasons。
 *
 * @param   distPoint [IN] DP 结构。
 * @param   reasons [OUT] 获取的 CRL 吊销原因。
 * @retval  int32_t，CRL 吊销原因。
 */
int32_t X509EXT_DPGetReason(const X509ExtDP *distPoint, uint32_t *reasons);

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief   复制 Distribution Point 结构。
 * @param   distPoint [IN] 源 DP。
 * @retval  X509ExtDP，已复制的 DPN 结构体指针。
 */
X509ExtDP *X509EXT_DPDump(const X509ExtDP *distPoint);

/**
 * @ingroup cme_x509v3_extn_dp
 * @brief   Distribution Point 结构。
 * @param   distPoint [IN] 待释放的DP。
 */
void X509EXT_DPFree(X509ExtDP *distPoint);

#ifdef __cplusplus
}
#endif

#endif // CME_X509V3_EXTN_DP_API_H
