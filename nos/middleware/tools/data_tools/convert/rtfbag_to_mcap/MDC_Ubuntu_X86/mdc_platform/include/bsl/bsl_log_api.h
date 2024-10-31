/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: bsl_log模块对外头文件
 * Create: 2021-06-04
 */

#ifndef BSL_LOG_API_H
#define BSL_LOG_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup bsl
 *
 * 0x0 审计日志级别。
 */
#define BSL_LOG_LEVEL_SEC         0U

/**
 * @ingroup bsl
 *
 * 0x1 紧急日志级别。
 */
#define BSL_LOG_LEVEL_FATAL       1U

/**
 * @ingroup bsl
 *
 * 0x2 错误日志级别。
 */
#define BSL_LOG_LEVEL_ERR         2U

/**
 * @ingroup bsl
 *
 * 0x3 警告日志级别。
 */
#define BSL_LOG_LEVEL_WARN        3U

/**
 * @ingroup bsl
 *
 * 0x4 信息日志级别。
 */
#define BSL_LOG_LEVEL_INFO        4U

/**
 * @ingroup bsl
 *
 * 0x5 调试日志级别。
 */
#define BSL_LOG_LEVEL_DEBUG       5U

/**
 * @ingroup bsl
 * @brief   log 回调函数原型。
 * @par 描述:
 * log 回调函数原型。
 * @attention 无。
 * @param eno        [IN] 错误码。
 * @param level      [IN] log的级别。
 * @param fileName   [IN] log所在的文件。
 * @param lineNum    [IN] log所在的行号。
 * @param logStr     [IN] log字符串。
 * @par 依赖: 如下
 * @li bsl：该接口所属的开发包。
 * @li bsl_log_api.h：该接口声明所在的头文件。
 * @since AutoTBP V100R021C00
 * @see 无。
 */
typedef void (*BslLogFunc)(uint32_t eno, uint32_t level, const char *fileName, uint32_t lineNum, const char *logStr);

/**
 * @ingroup bsl
 * @brief   注册log回调函数。
 * @par 描述:
 * 注册log回调函数。
 * @attention 1、该接口为进程级，注册后可接管该进程内所有bsl日志。
 *            2、该接口为进程日志总接口，使用本接口后，BSL_LogOpsReg功能将不生效。
 * @param hook   [IN] log回调函数。
 * @retval BSL_OK  注册成功。
 * @retval AEN_ENO_INVAL 注册失败。
 * @par 依赖: 如下
 * @li bsl：该接口所属的开发包。
 * @li bsl_log_api.h：该接口声明所在的头文件。
 * @since AutoTBP V100R021C00
 * @see BSL_LogOpsReg。
 */
uint32_t BSL_LogHookReg(BslLogFunc hook);

/**
 * @ingroup bsl
 * @brief   去注册log回调函数。
 * @par 描述:
 * 去注册log回调函数。
 * @attention
 * @param 无
 * @retval BSL_OK  去注册成功。
 * @retval BSL_ERR 去注册失败。
 * @par 依赖: 如下
 * @li bsl：该接口所属的开发包。
 * @li bsl_log_api.h：该接口声明所在的头文件。
 * @since AutoTBP V100R021C00
 * @see 无。
 */
uint32_t BSL_LogHookUnReg(void);

/**
 * @ingroup bsl
 * @brief   日志过滤等级设置。
 * @par 描述:
 * 日志过滤等级设置。。
 * @attention 无。
 * @param level      [IN] log的过滤级别。
 * @par 依赖: 如下
 * @li bsl：该接口所属的开发包。
 * @li bsl_log_api.h：该接口声明所在的头文件。
 * @since AutoTBP V100R022C10
 * @see 无。
 */
void BSL_LogFilter(uint32_t level);

#ifdef __cplusplus
}
#endif

#endif /* BSL_LOG_API_H */
