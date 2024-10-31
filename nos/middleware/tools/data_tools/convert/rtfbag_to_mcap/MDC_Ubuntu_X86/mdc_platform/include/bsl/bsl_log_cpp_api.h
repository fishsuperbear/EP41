/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: c++日志对外头文件
 * Create: 2022-03-28
 */

#ifndef BSL_LOG_CPP_API_H
#define BSL_LOG_CPP_API_H

#include <cstdint>
#include <string>
#include "bsl_log_api.h"

namespace Atp {
namespace Log {

/** @defgroup bsl bsl */
/**
 * @ingroup bsl
 * @brief   模块日志接管回调函数注册。
 * @par 描述: 进程可能包含多个模块，注册对应模块后，该模块的日志会传入回调函数中处理。
 * @attention 1、该接口不能在日志接管回调函数hook中被再次调用，可能导致死锁。
 *            2、日志接管回调函数hook中不能再次调用BSL日志记录接口。
 *            3、本接口对使用bsl log模块接口记录的日志生效（包括C++和C日志记录接口）
 *            4、若用户注册BSL_LogHookReg接口，BSL_LogOpsReg将不生效。
 * @param mdlName      [IN] 模块名。
 * @param hook         [IN] 日志接管回调函数。
 * @retval BSL_OK  注册成功。
 * @retval AEN_ENO_NOTINIT 日志模块未初始化。
 * @retval AEN_ENO_INVAL 参数为空。
 * @retval AEN_ENO_ALREADY 模块重复注册。
 * @retval AEN_ENO_NOMEM 内存不足。
 * @par 依赖: 如下
 * @li bsl：该接口所属的开发包。
 * @li bsl_log_cpp_api.h：该接口声明所在的头文件。
 * @since AutoTBP V100R022C00
 * @see MdlLogUnreg、BSL_LogHookReg。
 */
std::uint32_t MdlLogReg(std::string mdlName, BslLogFunc hook);

/**
 * @ingroup bsl
 * @brief   日志接管回调函数去注册。
 * @par 描述: 去注册对应模块的日志接管回调函数。
 * @attention 1、该接口不能在日志接管回调函数hook中被再次调用，可能导致死锁。
 *            2、本接口对使用bsl log模块接口记录的日志生效（包括C++和C日志记录接口）
 * @param mdlName      [IN] 模块名。
 * @retval BSL_OK  注册成功。
 * @retval AEN_ENO_NOTINIT 日志模块未初始化。
 * @retval AEN_ENO_NODATA 未匹配到模块名。
 * @par 依赖: 如下
 * @li bsl：该接口所属的开发包。
 * @li bsl_log_cpp_api.h：该接口声明所在的头文件。
 * @since AutoTBP V100R022C00
 * @see MdlLogReg
 */
std::uint32_t MdlLogUnreg(std::string mdlName);
} /* namespace Log */
} /* namespace Atp */

#endif /* BSL_LOG_CPP_API_H */
