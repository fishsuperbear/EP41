/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description:Linux Time External Interface
 * Create: 2021/06/04
 */

#ifndef BSL_TIME_API_H
#define BSL_TIME_API_H

#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup bsl
 * @brief 基础时间数据结构定义。
 */
typedef struct {
    uint16_t year;      /**< 年，取值范围为[0, 65535] */
    uint8_t  month;     /**< 月，取值范围为[1, 12] */
    uint8_t  day;       /**< 日，取值范围为[1, 31] */
    uint8_t  hour;      /**< 时, 取值范围为[0, 23] */
    uint8_t  minute;    /**< 分，取值范围为[0, 59] */
    uint8_t  second;    /**< 秒，取值范围为[0, 59] */
    uint8_t  reserved;  /**< 保留位 */
    uint32_t millSec;   /**< 毫秒，取值范围为[0, 999] */
    uint32_t microSec;  /**< 微秒，取值范围为[0, 999] */
} BslSysTime;

/**
 * @ingroup bsl
 * @brief unix时间结构定义。
 */
typedef int64_t BslUnixTime;

/**
 * @ingroup bsl
 * @brief 获取时间回调函数原型
 * @par 描述：
 * 获取时间回调函数原型定义
 * @attention 无。
 * @par 依赖: 如下
 * @li bsl：该接口所属的开发包。
 * @li bsl_time_api.h：该接口声明所在的头文件。
 * @since AutoTBP V100R021C00
 * @see 无。
 */
typedef BslUnixTime (*BslTimeFunc)(void);

/**
 * @ingroup bsl
 * @brief   获取系统时间函数注册接口
 * @par 描述:
 * 用户可以使用该接口注册系统时间获取函数。
 * @attention
 * 该接口注册可以多次注册 注册成功之后，不能再次注册为空.
 * 时间范围的说明:
 * 用户使用Linux系统年上限为2038;
 * 时间的下限为1970-1-1 0:0:0.
 * 建议用户使用这个最小的交集，即年份的界限为 1970-1-1 0:0:0 ~ 2038-01-19 03:14:08.
 * @param func [IN] 注册系统时间获取函数
 * @retval void  无。
 * @par 依赖: 如下
 * @li bsl：该接口所属的开发包。
 * @li bsl_time_api.h：该接口声明所在的头文件。
 * @since AutoTBP V100R021C10
 * @see 无
 */
void BSL_SysTimeFuncReg(BslTimeFunc func);

#ifdef __cplusplus
}
#endif

#endif
