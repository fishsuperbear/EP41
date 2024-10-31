/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     ara_log_impl.hpp                                                       *
*  @brief    Define of class logger::Impl                                 *
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


#ifndef __LOG_CTX_IMPL_HPP__
#define __LOG_CTX_IMPL_HPP__

#include <iostream>
#include <memory>

#include "logger.h"

namespace hozon {
namespace netaos {
namespace log {

/**
* @brief Logger CtxImpl class
* This Imol class is required by ara::log.
*/
class CtxImpl
{
 public:
        /** 
        * @brief Constructor function of class Logger::CtxImpl
        * 
        * @param selfLogger    Logger handler
        * @param ctxLogId         ctx Log identification
        * @param ctxLogDescription         Log description
        * @param ctxDefLogLevel         Log level
        * @param logTo         Log to where
        * @param logFile         Log file path
        * @param maxLogFileNum         Max number to store log file
        * @param maxSizeOfLogFile         Max size of each log backup file
        *
        */
        CtxImpl(Logger &selfLogger, std::string  ctxLogId, std::string ctxLogDescription, LogLevel ctxDefLogLevel);

        /** 
        * @brief Destructor function of class Logger::CtxImpl
        * 
        */
        ~CtxImpl(){};

        /** 
        * @brief getCtxLogId function
        * 
        * @return Return the log id
        *
        */
        std::string  getCtxLogId();

        /** 
        * @brief getCtxLogDescription function
        * 
        * @return Return the log description
        *
        */
        std::string  getCtxLogDescription();

        /** 
        * @brief getLogLevel function
        * 
        * @return Return the log level
        *
        */
        LogLevel  getCtxLogLevel();

        /** 
        * @brief getOutputLogLevel function
        * 
        * @return Return the log level
        *
        */
        LogLevel  getOutputLogLevel();

        /** 
        * @brief IsEnabled function
        * 
        * @param level The log level of app log level
        * 
        * @return bool
        *
        */
        bool IsEnabled(LogLevel level);

        /** 
        * @brief UpdateAppLogLevel function
        * 
        * @param level The log level of app log level
        * 
        * @return void
        *
        */
        void UpdateAppLogLevel(const LogLevel appLogLevel);

        /** 
        * @brief normalSetCtxLogLevel function
        * 
        * @param level The log level to be seted
        * 
        * @return void
        *
        */
        void normalSetCtxLogLevel(const LogLevel level);


        /** 
        * @brief forceSetCtxLogLevel function
        * 
        * @param level The log level to be seted
        * 
        * @return void
        *
        */
        void forceSetCtxLogLevel(const LogLevel level);

        /** 
        * @brief SetOutputLogLevel function
        * 
        * @param level The log level to be seted
        * 
        * @return void
        *
        */
        void setOutputLogLevel(const LogLevel level);

        /** 
        * @brief logout function
        * 
        * @param level The log level to logout
        * @param message The message to be output
        *
        */
        void LogOut(LogLevel level, const std::string& message);

 private:
        std::string ctxLogId_;  // log id, input from function CreateLogger
        std::string ctxLogDescription_;  // log description, input from function CreateLogger
        LogLevel ctxDefLogLevel_; // ctx log level
        bool forceOutput_;
        LogLevel finalOutputLogLevel_; // output log level
        Logger &m_logger;   // logger handler, created after called function CreateLogger
};


}
}
}
#endif  // __LOG_CTX_IMPL_HPP__ 