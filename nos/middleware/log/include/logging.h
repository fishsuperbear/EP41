/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     logging.h                                                     *
*  @brief    Define of fuctions                                          *
*  Details.                                                                         *
*                                                                                   *
*  @version  0.0.0.1                                                                *
*                                                                                   *
*-----------------------------------------------------------------------------------*
*  Change History :                                                                 *
*  <Date>     | <Version> | <Author>       | <Description>                          *
*-----------------------------------------------------------------------------------*
*  2022/06/15 | 0.0.0.1   | YangPeng      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/

#ifndef __HZ_LOGGING_H__
#define __HZ_LOGGING_H__
#include "logger.h"
#include "logstream.h"

namespace hozon {
namespace netaos {
namespace log {


void InitLogging(
    std::string appId,  // the log id of application
    std::string appDescription, // the log id of application
    LogLevel appLogLevel, //the log level of application
    std::uint32_t outputMode, //the output log mode
    std::string directoryPath, //the log file directory, active when output log to file
    std::uint32_t maxLogFileNum, //the max number log file , active when output log to file, MAX:20, MIN:0
    std::uint64_t maxSizeOfLogFile, //the max size[M] of each  log file , active when output log to file, MAX:100M, MIN:2M
    bool isMain = false, // if flag == true, replace appID and others params, or do nothing
    bool pureLogFormat = false   //true : only output raw log data；false：output log as special type, type：[Y-M-D H:M:S.MS] [log level]   [AppID]  [ctxID]      [log aw data]
) noexcept;

void InitLogging(
    std::string logCfgFile  // the log config file
) noexcept;

std::shared_ptr<Logger> CreateLogger(
    const std::string ctxId,
    const std::string ctxDescription,
    const LogLevel ctxDefLogLevel = LogLevel::kInfo
) noexcept;

std::shared_ptr<Logger> CreateOperationLogger(
    const std::string ctxId,
    const std::string ctxDescription,
    const LogLevel ctxDefLogLevel = LogLevel::kInfo
) noexcept;

void InitMcuLogging(
    const std::string appId
) noexcept;

std::shared_ptr<Logger> CreateMcuLogger(
    const std::string appId,
    const std::string ctxId,
    const LogLevel ctxDefLogLevel = LogLevel::kInfo
) noexcept;

}
}
}
#endif //__HZ_LOGGING_H__
