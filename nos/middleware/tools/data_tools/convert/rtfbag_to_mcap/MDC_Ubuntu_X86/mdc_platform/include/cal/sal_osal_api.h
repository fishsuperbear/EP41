/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 操作系统抽象层，抽象需要对外暴露的操作系统定义和接口。
 * Create: 2020/11/6
 * Notes:
 * History:
 * 2020/11/6 第一次创建
 */

#ifndef SAL_OSAL_API_H
#define SAL_OSAL_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t OsalTime;

/**
 * @ingroup adaptor
 * 存储utc时间的结构体
 */
typedef struct {
    uint16_t year;       /* 年的范围 从 0 到 65536 */
    uint8_t month;       /* 月的范围 从 1 到 12 */
    uint8_t day;         /* 日的范围 从 1 到 31 */
    uint8_t hour;        /* 时的范围 从 0 到 23 */
    uint8_t minute;      /* 分的范围 从 0 到 59 */
    uint16_t millSecond; /* 毫秒的范围 从 0 到 999 */
    uint8_t second;      /* 秒的范围 从 0 到 59 */
    uint8_t utcSign;     /* utc正负 0为正, 1为负 */
    uint8_t utcHour;     /* utc时, 范围为0~11. */
    uint8_t utcMinute;   /* utc分, 范围为0~59. */
} OsalDateTime;

/**
 * @ingroup adaptor
 * @brief   释放AUTOTBP组件中分配的内存
 * @param   value [IN] 内存地址
 * @retval 无
 */
void OSAL_Free(void *value);

#ifdef __cplusplus
}
#endif

#endif // SAL_OSAL_API_H
