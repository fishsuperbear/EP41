/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: X509 date time 扩展
 * Create: 2020/10/26
 * Notes:
 * History:
 * 2020/10/26 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_x509v3_extn_dt CME_X509V3_EXTN_DT_API
 * @ingroup cme_x509
 */
#ifndef CME_X509V3_EXTN_DT_API_H
#define CME_X509V3_EXTN_DT_API_H

#include "cme_asn1_api.h"
#include "bsl_time_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509v3_extn_dt
 * @brief x509 Extension Time ID.
 *
 */
typedef enum {
    TIME_UTCTIME, /* < UTC 时间格式 */
    TIME_GENTIME  /* < Generalized 时间格式 */
} X509ExtTimeId;

/**
 * @ingroup cme_x509v3_extn_dt
 * @brief X509ExtTime 结构体，用于创建 UTC 时间或 Generalized 时间。
 *
 */
typedef struct {
    X509ExtTimeId choiceId; /* < 表示时间的格式 */

    union TimeChoice {
        Asn1UTCTime *utcTime;         /* < UTC 时间 */
        Asn1GeneralizedTime *genTime; /* < Generalized 时间 */
    } a;                              /* < 可以包含UTC时间或Generalized时间的联合 */
} X509ExtTime;

/**
 * @ingroup cme_x509v3_extn_dt
 * @brief   此函数用于将 BslSysTime 结构转换为 UTC 时间或 Generalized 时间。
 * @details 当1970 <= sysTime->year < 2050 时，返回的是 UTC 时间格式，其他年份返回 Generalized 时间格式。
 * @param   sysTime [IN] BslSysTime 时间结构。
 * @return  X509ExtTime *，指向创建后的时间结构体指针。若失败，返回NULL。
 */
X509ExtTime *X509EXT_TimeCreate(const BslSysTime *sysTime);

/**
 * @ingroup cme_x509v3_extn_dt
 * @brief   复制一份 X509ExtTime
 * @param   srcTime [IN] 待复制的时间结构。
 * @return  X509ExtTime *，指向已复制的时间结构体指针，若失败，返回NULL。
 */
X509ExtTime *X509EXT_TimeDump(const X509ExtTime *srcTime);

/**
 * @ingroup cme_x509v3_extn_dt
 * @brief   释放 X509ExtTime 结构。
 * @param   time [IN] 待释放的时间。
 */
void X509EXT_TimeFree(X509ExtTime *extTime);

#ifdef __cplusplus
}
#endif

#endif // CME_X509V3_EXTN_DT_API_H
