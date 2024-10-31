/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     hz_log_manager.hpp                                                     *
*  @brief    Define of class HzLogManager                                          *
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

#ifndef __LOG_MANAGER_HPP__
#define __LOG_MANAGER_HPP__

#include <iostream>
#include <memory>
#include <mutex>
#include "hz_logger.hpp"
#include "hz_operation_logger.hpp"
#include "hz_mcu_logger.hpp"
#include <vector>
#include <map>
#include "adapter/log_device_adapter.hpp"


namespace hozon {
namespace netaos {
namespace log {

/**
* @brief HzLogManager class
* hozon log manager class
*/
class HzLogManager
{
public:
    /** 
    * @brief Constructor function of class HzLogManager
    *
    */
    HzLogManager()
    : appId_("INVALID")
    , appDescription_("invalid application")
    , appLogLevel_(LogLevel::kOff)
    , mode_(0)
    , directoryPath_("./")
    , logFileBaseName_(appId_)
    , maxLogFileStoredNum_(20)
    , maxSizeOfEachLogFile_(20 * 1024 * 1024)
    ,pureLogFormat_(false){
        OperationLogger_.clear();
    }

    /** 
        * @brief Destructor function of class HzLogManager
        * 
    */
    ~HzLogManager() {

    };

    /** 
        * @brief GetInstance function
        * 
        * @return Return an object from the singleton pattern of HzLogManager.
        *
    */
    static HzLogManager* GetInstance();

    /** 
        * @brief InitLogging function
        * 
        * @param appId                  application identification
        * @param appDescription         application description
        * @param appDefLogLevel         application defined log level
        * @param mode                   Log mode
        * @param directoryPath          Log stored directory path
        * @param maxLogFileNum          Max number to store log file
        * @param maxSizeOfLogFile       Max size of each log backup file
        * @param pureLogFormat          pure log or special format log
        *
    */
    void InitLogging(std::string appId, std::string appDescription, LogLevel appDefLogLevel,
                     std::uint32_t mode, std::string directoryPath, std::uint32_t maxLogFileNum,
                     std::uint64_t maxSizeOfLogFile, bool pureLogFormat);


    /** 
        * @brief creatLogger function
        * 
        * @param logId                  Log identification
        * @param logDescription         Log description
        * @param moduleDefLevel         Log level of module
        * @param loggerCount            Logger count
        *
    */
    std::shared_ptr<Logger> creatLogger(std::string logId, std::string logDescription,
                                        LogLevel moduleDefLevel, uint32_t& loggerCount);

    
    /** 
        * @brief creatLogger function
        * 
        * @param logId                  Log identification
        * @param logDescription         Log description
        * @param moduleDefLevel         Log level of module
        * @param loggerCount            Logger count
        *
    */
    std::shared_ptr<Logger> CreateOperationLogger(std::string logId, std::string logDescription,
                                        LogLevel moduleDefLevel, uint32_t& loggerCount);


    // /** 
    //     * @brief logout function
    //     * 
    //     * @param logId         Log identification
    //     * @param logDescription         Log description
    //     * @param level The log level to logout
    //     * @param message The message to be output
    //     *
    // */
    void logout(LogLevel level, const std::string& message);

    /** 
        * @brief setAllCtxLogLevel function
        * 
        * @param level The log level to logout
        *
    */
    void setAllCtxLogLevel(LogLevel level);

    /** 
        * @brief setAppLogLevel function
        * 
        * @param appId         application id
        * @param level The log level to logout
        *
    */
    void setAppLogLevel(std::string appId, LogLevel level);

    /** 
        * @brief setCtxLogLevel function
        * 
        * @param ctxId         ctx id
        * @param level The log level to logout
        *
    */
    void setCtxLogLevel(std::string ctxId, LogLevel level);

    /** 
        * @brief setSpecifiedLogLevel function
        * 
        * @param appId         application id
        * @param ctxId         ctx id
        * @param level The log level to logout
        *
    */
    void setSpecifiedLogLevel(std::string appId, std::string ctxId, LogLevel level);


    /** 
        * @brief getAppId function
        * 
        * @return Return the identification of application
        *
    */
    std::string getAppId()
    {
        return appId_;
    }

    /** 
        * @brief getAppDescription function
        * 
        * @return Return the Description of application
        *
    */
    std::string getAppDescription()
    {
        return appDescription_;
    }

     /** 
        * @brief getAppLogLevel function
        * 
        * @return Return the log level of application
        *
    */
    LogLevel getAppLogLevel()
    {
        return appLogLevel_;
    }

    /** 
        * @brief getLogMode function
        * 
        * @return Return the log mode
        *
    */
    std::uint32_t getLogMode()
    {
        return mode_;
    }

    /** 
        * @brief getLogFileName function
        * 
        * @return Return the log file name
        *
    */
    std::string getLogFileBaseName()
    {
        return logFileBaseName_;
    }

    /** 
        * @brief getMaxLogFileStoredNum function
        * 
        * @return Return the max store file number of log
        *
    */
    std::uint32_t getMaxLogFileStoredNum()
    {
        return maxLogFileStoredNum_;
    }

    /** 
        * @brief getMaxSizeOfEachLogFile function
        * 
        * @return Return the max size of each log file
        *
    */
    std::uint32_t getMaxSizeOfEachLogFile()
    {
        return maxSizeOfEachLogFile_;
    }

    /** 
        * @brief getRawLogFormat function
        * 
        * @return Return the raw log format
        *
    */
    bool getPureLogFormat()
    {
        return pureLogFormat_;
    }

    // for mcu log
    void initMcuLogging(const std::string& appId);
    std::shared_ptr<Logger> createMcuLogger(const std::string& appId, const std::string& ctxId, const LogLevel& level);
    void mcuLogout(const std::string& appId, const LogLevel& level, const std::string& message);

private:
    std::string fixPath(const std::string& path);

private:
    std::string appId_; //Log application identification
    std::string appDescription_;    //Log application description
    LogLevel appLogLevel_;  //Log level of application
    std::uint32_t mode_;    //Log mode
    std::string directoryPath_; //Log stored directory path
    std::string logFileBaseName_; //Log file base name
	std::uint32_t  maxLogFileStoredNum_; //max log file stored number
	std::uint64_t  maxSizeOfEachLogFile_;  //max size of each log file
    bool pureLogFormat_;
    static std::mutex mtx_;
    static HzLogManager* instance_;

    std::vector<std::shared_ptr<HzLogger>> Logger_;
    std::vector<std::shared_ptr<HzOperationLogger>> OperationLogger_;
    std::vector<std::shared_ptr<HzMcuLogger>> mcuLogger_;
    std::vector<std::string> mcuAppId_;
    std::shared_ptr<ILogDeviceAdapter> LoggerTrace_ {nullptr};
    std::map<std::string, std::shared_ptr<ILogDeviceAdapter>> mcuLogTraceMap_{};
};



}
}
}
#endif  // __HZ_LOG_MANAGER_HPP__ 
