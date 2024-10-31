/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     global_log_config_manager.hpp                                          *
*  @brief    Define of class GlobalLogConfigManager                                 *
*  Details.                                                                         *
*                                                                                   *
*  @version  0.0.0.1                                                                *
*                                                                                   *
*-----------------------------------------------------------------------------------*
*  Change History :                                                                 *
*  <Date>     | <Version> | <Author>       | <Description>                          *
*-----------------------------------------------------------------------------------*
*  2023/12/06 | 0.0.0.1   | XuMengJun      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/

#ifndef __GLOBAL_LOG_CONFIG_MANAGER_H__
#define __GLOBAL_LOG_CONFIG_MANAGER_H__

#include <string>
#include <unordered_map>

#include "logger.h"

namespace hozon {
namespace netaos {
namespace log {

/**
* @brief HzGbloalLogConfigManager class
* hozon global log config class
*/
class HzGlobalLogConfigManager {
public:
    /** 
     * @brief GetInstance function
     * 
     * @return Return an object from the singleton pattern of HzGlobalLogConfigManager
     *
     */
    static HzGlobalLogConfigManager& GetInstance();

    /** 
        * @brief Destructor function of class HzGlobalLogConfigManager
        * 
    */
    ~HzGlobalLogConfigManager();

public:
    struct LogConfig {
        std::string appId;                                              // Log application identification
        LogLevel appLogLevel;                                           // Log level of application
        std::uint32_t logMode;                                          // Log mode
        std::string logPath;                                            // Log stored directory path
        std::uint32_t maxLogFileNum;                                    // max log file stored number
        std::uint64_t maxSizeOfEachLogFile;                             // max size of each log file
        std::unordered_map<std::string, LogLevel> ctxIdLogLevelMap_;    // ctxID <===> LogLevel
        bool hasAppId;
        bool hasLogLevel;
        bool hasLogMode;
        bool hasLogPath;
        bool hasMaxLogFileNum;
        bool hasMaxSizeOfEachLogFile;
        LogConfig() : hasAppId(false), hasLogLevel(false), hasLogMode(false), hasLogPath(false),
                    hasMaxLogFileNum(false), hasMaxSizeOfEachLogFile(false) {}
    };
    using LogConfigPtr = std::shared_ptr<LogConfig>;

public:
    /** 
     * @brief LoadConfig function, it can be called mutilple times
     *
     */
    bool LoadConfig();

    /** 
     * @brief GetAppLogConfig function, only the LoadConfig function returns true, this function returns data correctly.
     * 
     * @param appId          appid 
     *
     */
    const HzGlobalLogConfigManager::LogConfigPtr& GetAppLogConfig(const std::string &appId);

private:
    /** 
    * @brief Constructor function of class HzGlobalLogConfigManager
    *
    */
    HzGlobalLogConfigManager();

private:
    bool hasLoadFile = false;

    const std::string global_log_config_file = "/app/conf/log_daemon_storage.json";

    std::unordered_map<std::string, LogConfigPtr> appIdLogConfigMap_;
};

} // namespace log
} // namespace netaos
} // namespace hozon

#endif // __GLOBAL_LOG_CONFIG_MANAGER_H__
